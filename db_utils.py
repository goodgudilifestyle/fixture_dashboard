import sqlite3
from typing import List, Optional, Tuple, Dict
import hashlib
from datetime import datetime

DB_PATH = "gogigi.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_step3_schema():
    conn = get_conn()
    try:
        # SKU mapping table (Step 3A)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS fixture_sku_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store TEXT NOT NULL,
            fixture_code TEXT NOT NULL,
            sku TEXT NOT NULL,
            product_name TEXT,
            category TEXT,
            priority INTEGER,
            created_at TEXT NOT NULL,
            UNIQUE(store, fixture_code, sku)
        );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_fsm_store_sku ON fixture_sku_map(store, sku);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fsm_store_fixture ON fixture_sku_map(store, fixture_code);")

        # Sales raw (Step 3B ingestion)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sales_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT NOT NULL,
            sale_date TEXT NOT NULL,           -- normalized YYYY-MM-DD
            store TEXT NOT NULL,               -- POS Outlet / store name
            sku TEXT NOT NULL,
            qty REAL NOT NULL,
            amount REAL NOT NULL,
            invoice_no TEXT,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_raw_store_date ON sales_raw(store, sale_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_raw_store_sku ON sales_raw(store, sku);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_raw_filehash ON sales_raw(file_hash);")

        # Pre-aggregations for speed
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sku_sales_daily (
            store TEXT NOT NULL,
            sale_date TEXT NOT NULL,
            sku TEXT NOT NULL,
            qty REAL NOT NULL,
            amount REAL NOT NULL,
            PRIMARY KEY (store, sale_date, sku)
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ssd_store_date ON sku_sales_daily(store, sale_date);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS fixture_sales_daily (
            store TEXT NOT NULL,
            sale_date TEXT NOT NULL,
            fixture_code TEXT NOT NULL,
            qty REAL NOT NULL,
            amount REAL NOT NULL,
            PRIMARY KEY (store, sale_date, fixture_code)
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fsd_store_date ON fixture_sales_daily(store, sale_date);")

        conn.commit()
    finally:
        conn.close()


def list_stores_from_fixtures() -> List[str]:
    cols = table_columns("fixtures")
    store_col = _pick_col(cols, ["store", "pos_outlet", "posOutlet", "outlet", "store_name"])

    if not store_col:
        return []

    conn = get_conn()
    try:
        rows = conn.execute(f"SELECT DISTINCT {store_col} AS store FROM fixtures ORDER BY 1;").fetchall()
        stores = []
        for r in rows:
            s = str(r["store"]).strip() if r["store"] is not None else ""
            if s:
                stores.append(s)
        return stores
    finally:
        conn.close()



def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def table_columns(table: str) -> List[str]:
    conn = get_conn()
    try:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()


def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def get_fixture_codes_for_store(store: str) -> List[str]:
    """
    Reads fixture codes from your existing fixtures table.
    We try to find a store column + a fixture_code column.
    If your fixtures table uses different names, add them to candidates below.
    """
    cols = table_columns("fixtures")

    store_col = _pick_col(cols, ["store", "pos_outlet", "posOutlet", "outlet", "store_name"])
    code_col = _pick_col(cols, ["fixture_code", "code", "fixtureCode", "fixture_id_code", "name", "label", "title"])

    if not code_col:
        raise RuntimeError("Could not find fixture code column in fixtures table. Add your column name in db_utils.py")

    conn = get_conn()
    try:
        if store_col:
            rows = conn.execute(
                f"SELECT DISTINCT {code_col} AS code FROM fixtures WHERE {store_col} = ? ORDER BY 1;",
                (store,),
            ).fetchall()
        else:
            # If fixtures table doesn't store store, we cannot validate per-store. We'll validate globally.
            rows = conn.execute(
                f"SELECT DISTINCT {code_col} AS code FROM fixtures ORDER BY 1;"
            ).fetchall()

        return [str(r["code"]).strip() for r in rows if r["code"] is not None and str(r["code"]).strip() != ""]
    finally:
        conn.close()


def list_stores_from_sales_or_fixtures() -> List[str]:
    """
    Build store list from sales_raw and/or fixtures.
    """
    stores = set()
    conn = get_conn()
    try:
        # from sales_raw
        rows = conn.execute("SELECT DISTINCT store FROM sales_raw ORDER BY store;").fetchall()
        for r in rows:
            if r["store"]:
                stores.add(str(r["store"]).strip())

        # from fixtures if possible
        cols = table_columns("fixtures")
        store_col = _pick_col(cols, ["store", "pos_outlet", "posOutlet", "outlet", "store_name"])
        if store_col:
            rows2 = conn.execute(f"SELECT DISTINCT {store_col} AS store FROM fixtures ORDER BY 1;").fetchall()
            for r in rows2:
                if r["store"]:
                    stores.add(str(r["store"]).strip())
    finally:
        conn.close()

    return sorted([s for s in stores if s])


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def upsert_fixture_sku_rows(rows: List[Dict], mode: str):
    """
    mode = 'append' or 'replace'
    replace => delete store's existing rows for fixture_code+sku collisions? or full store replace.
    We'll implement:
      - replace: delete all mappings for (store) present in upload, then insert fresh
      - append: insert unique rows; ignore duplicates via UNIQUE constraint
    """
    if not rows:
        return {"inserted": 0, "skipped": 0, "stores_replaced": 0}

    conn = get_conn()
    inserted = 0
    skipped = 0
    stores_replaced = 0
    try:
        if mode == "replace":
            upload_stores = sorted({r["store"] for r in rows})
            for s in upload_stores:
                conn.execute("DELETE FROM fixture_sku_map WHERE store = ?;", (s,))
                stores_replaced += 1
            conn.commit()

        for r in rows:
            try:
                conn.execute(
                    """
                    INSERT INTO fixture_sku_map
                    (store, fixture_code, sku, product_name, category, priority, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["store"],
                        r["fixture_code"],
                        r["sku"],
                        r.get("product_name"),
                        r.get("category"),
                        r.get("priority"),
                        now_iso(),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                skipped += 1

        conn.commit()
    finally:
        conn.close()

    return {"inserted": inserted, "skipped": skipped, "stores_replaced": stores_replaced}


def export_fixture_sku_map(store: Optional[str] = None) -> List[Dict]:
    conn = get_conn()
    try:
        if store:
            rows = conn.execute(
                """
                SELECT store, fixture_code, sku, product_name, category, priority, created_at
                FROM fixture_sku_map
                WHERE store = ?
                ORDER BY fixture_code, sku;
                """,
                (store,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT store, fixture_code, sku, product_name, category, priority, created_at
                FROM fixture_sku_map
                ORDER BY store, fixture_code, sku;
                """
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_store_fixture_sku_map(store: str) -> List[Dict]:
    """
    Returns all mapping rows for a store (for manual mapping UI display).
    """
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT store, fixture_code, sku, created_at
            FROM fixture_sku_map
            WHERE store = ?
            ORDER BY fixture_code, sku;
            """,
            (store,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_fixture_sku_rows(store: str, skus: List[str]) -> int:
    """
    Deletes mapping rows for given SKUs in the selected store.
    Returns number of deleted rows.
    """
    if not skus:
        return 0

    # normalize + unique
    skus = [str(s).strip() for s in skus if str(s).strip()]
    if not skus:
        return 0

    conn = get_conn()
    try:
        qmarks = ",".join(["?"] * len(skus))
        cur = conn.execute(
            f"DELETE FROM fixture_sku_map WHERE store = ? AND sku IN ({qmarks});",
            [store] + skus,
        )
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()



def get_active_layout_id(store: str):
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT id
            FROM layouts
            WHERE store = ?
              AND is_active = 1
            ORDER BY id DESC
            LIMIT 1;
            """,
            (store,),
        ).fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def get_fixtures_for_store(store: str):
    """
    ✅ Return fixtures ONLY from the ACTIVE layout of the store.
    Prevents old layouts from merging into heatmap.
    """
    layout_id = get_active_layout_id(store)
    if not layout_id:
        return []

    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT fixture_code, fixture_type, display_name, x, y, w, h
            FROM fixtures
            WHERE layout_id = ?
            ORDER BY id ASC;
            """,
            (layout_id,),
        ).fetchall()

        out = []
        for r in rows:
            out.append({
                "fixture_code": r[0],
                "fixture_type": r[1],
                "display_name": r[2],
                "x": float(r[3]),
                "y": float(r[4]),
                "w": float(r[5]),
                "h": float(r[6]),
            })
        return out
    finally:
        conn.close()

def delete_fixture_sku_rows_for_fixture(store: str, fixture_code: str, skus: List[str]) -> int:
    """
    Deletes mapping rows only for the selected fixture.
    """
    if not skus:
        return 0
    skus = [str(s).strip() for s in skus if str(s).strip()]
    if not skus:
        return 0

    conn = get_conn()
    try:
        qmarks = ",".join(["?"] * len(skus))
        cur = conn.execute(
            f"""
            DELETE FROM fixture_sku_map
            WHERE store = ? AND fixture_code = ? AND sku IN ({qmarks});
            """,
            [store, fixture_code] + skus,
        )
        conn.commit()
        return cur.rowcount or 0
    finally:
        conn.close()