# pages/4_Fixture_Sales.py

import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from db_utils import (
    ensure_step3_schema,
    get_conn,
    sha1_bytes,
    now_iso,
)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Fixture Sales (Step 3B)", layout="wide")
ensure_step3_schema()

st.title("Step 3B — Sales Aggregation Engine (Fixture-wise Sales + %)")

# Selected store from app.py
store = (st.session_state.get("selected_store") or st.session_state.get("selected_outlet") or "").strip()
if not store:
    st.warning("No store selected. Go to Home (app) page and select store first.")
    st.stop()

st.info(f"✅ Selected store (from Home): **{store}**")
st.caption("Uses Sales Date (DD-MM-YYYY), Quantity as Sold Qty, and Net Total Amt (Rs.) as Sales Amount.")

# Admin mode comes from app.py toggle (key="mode_toggle")
IS_ADMIN = bool(st.session_state.get("mode_toggle", True))


# ----------------------------
# Helpers
# ----------------------------
def norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower())


def pick_col(cols_norm_map: dict, candidates: list[str]):
    for cand in candidates:
        if cand in cols_norm_map:
            return cols_norm_map[cand]
    return None


def parse_any_date_to_ymd(s: str) -> str:
    s = str(s).strip()
    if not s:
        raise ValueError("Sales Date blank")

    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass

    for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass

    raise ValueError(f"Unrecognized date format: '{s}'")


def to_number(x, default=0.0) -> float:
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return float(default)
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return float(default)


def fmt_inr(x: float) -> str:
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return "₹0"


def list_project_data_files():
    files = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(Path(".").glob(ext))
    return sorted([str(f) for f in files])


def auto_pick_sales_file(project_files: list[str]) -> str | None:
    """
    Pick product_report_all (sales file) from project folder.
    Preference: file name contains product_report_all or report_all.
    """
    cand = []
    for f in project_files:
        lf = f.lower()
        if ("product_report_all" in lf) or ("report_all" in lf):
            cand.append(f)

    if not cand:
        return None

    # pick most recently modified
    cand = sorted(cand, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return cand[0]


# ----------------------------
# DB ops
# ----------------------------
def delete_sales_for_store(selected_store: str):
    conn = get_conn()
    try:
        conn.execute("DELETE FROM sales_raw WHERE store = ?;", (selected_store,))
        conn.execute("DELETE FROM sku_sales_daily WHERE store = ?;", (selected_store,))
        conn.execute("DELETE FROM fixture_sales_daily WHERE store = ?;", (selected_store,))
        conn.commit()
    finally:
        conn.close()


def ingest_sales_csv_for_store(file_bytes: bytes, filename: str, selected_store: str, replace_store_sales: bool):
    """
    Ingest only rows for selected_store into sales_raw.
    - replace_store_sales=True => delete existing sales_raw for store first, then ingest.
    - Prevents duplicate ingest by (file_hash + store) ONLY when replace_store_sales=False
    """
    file_hash = sha1_bytes(file_bytes)

    if replace_store_sales:
        delete_sales_for_store(selected_store)

    # Skip if already ingested for this store (only when not replacing)
    if not replace_store_sales:
        conn = get_conn()
        try:
            exists = conn.execute(
                "SELECT 1 FROM sales_raw WHERE file_hash = ? AND store = ? LIMIT 1;",
                (file_hash, selected_store),
            ).fetchone()
            if exists:
                return {"status": "skipped", "reason": "This file already ingested for this store.", "rows": 0}
        finally:
            conn.close()

    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str).fillna("")
    cols_norm_map = {norm_col(c): c for c in df.columns}

    col_date = pick_col(cols_norm_map, ["sales date", "sales_date", "date"])
    col_store = pick_col(cols_norm_map, ["pos outlet", "pos_outlet", "store", "outlet"])
    col_sku = pick_col(cols_norm_map, ["sku", "item sku", "product sku"])
    col_qty = pick_col(cols_norm_map, ["quantity", "qty", "quantity sold", "sold qty", "sold_qty"])
    col_amt = pick_col(cols_norm_map, [
        "net total amt (rs.)", "net total amt(rs.)", "net total amt",
        "net total amount", "net amount", "net_total_amt"
    ])
    col_pname = pick_col(cols_norm_map, ["product name", "product_name", "item name", "item_name", "product"])
    col_invoice = pick_col(cols_norm_map, ["invoice no.", "invoice no", "invoice", "invoice_no"])

    missing = []
    if not col_date: missing.append("Sales Date")
    if not col_store: missing.append("POS Outlet")
    if not col_sku: missing.append("SKU")
    if not col_qty: missing.append("Quantity")
    if not col_amt: missing.append("Net Total Amt (Rs.)")

    if missing:
        return {
            "status": "error",
            "errors": [
                f"Required columns missing (flex match failed): {missing}",
                f"Columns found: {list(df.columns)}"
            ]
        }

    cleaned = []
    errors = []
    skipped = 0
    sku_to_name = {}

    for i, r in df.iterrows():
        try:
            row_store = str(r[col_store]).strip()

            # Skip other stores / blank rows
            if not row_store or row_store != selected_store:
                skipped += 1
                continue

            sku = str(r[col_sku]).strip()
            if not sku:
                skipped += 1
                continue

            # capture product name (if column exists)
            pname = str(r[col_pname]).strip() if col_pname else ""
            if pname:
                sku_to_name[sku] = pname

            raw_date = str(r[col_date]).strip()
            if not raw_date:
                skipped += 1
                continue

            sale_date = parse_any_date_to_ymd(raw_date)
            qty = to_number(r[col_qty], default=0.0)
            amt = to_number(r[col_amt], default=0.0)

            invoice_no = str(r[col_invoice]).strip() if col_invoice else None
            invoice_no = invoice_no or None

            cleaned.append((file_hash, sale_date, row_store, sku, qty, amt, invoice_no, now_iso()))
        except Exception as e:
            skipped += 1
            if len(errors) < 100:
                errors.append(f"Row {i+2}: {e}")

    if not cleaned:
        return {
            "status": "error",
            "errors": [f"No valid rows found for store '{selected_store}'. Check POS Outlet values in CSV."]
        }

    conn = get_conn()
    try:
        conn.executemany(
            """
            INSERT INTO sales_raw (file_hash, sale_date, store, sku, qty, amount, invoice_no, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            cleaned
        )
        conn.commit()

        # ✅ Upsert product names (if available)
        if sku_to_name:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    sku TEXT PRIMARY KEY,
                    product_name TEXT
                );
            """)

            conn.executemany(
                """
                INSERT INTO products (sku, product_name)
                VALUES (?, ?)
                ON CONFLICT(sku) DO UPDATE SET
                    product_name = CASE
                        WHEN excluded.product_name IS NOT NULL AND excluded.product_name <> ''
                        THEN excluded.product_name
                        ELSE products.product_name
                    END;
                """,
                [(k, v) for k, v in sku_to_name.items()]
            )
            conn.commit()

    finally:
        conn.close()

    return {"status": "ok", "rows": len(cleaned), "skipped": skipped, "errors": errors}

def recompute_aggregations(selected_store: str):
    """
    Recompute sku_sales_daily + fixture_sales_daily for selected store only.
    """
    conn = get_conn()
    try:
        conn.execute("DELETE FROM sku_sales_daily WHERE store = ?;", (selected_store,))
        conn.execute("DELETE FROM fixture_sales_daily WHERE store = ?;", (selected_store,))

        # SKU daily aggregation
        conn.execute(
            """
            INSERT INTO sku_sales_daily (store, sale_date, sku, qty, amount)
            SELECT store, sale_date, sku,
                   SUM(qty) AS qty,
                   SUM(amount) AS amount
            FROM sales_raw
            WHERE store = ?
            GROUP BY store, sale_date, sku;
            """,
            (selected_store,)
        )

        # Fixture daily aggregation (join mapping)
        conn.execute(
            """
            INSERT INTO fixture_sales_daily (store, sale_date, fixture_code, qty, amount)
            SELECT s.store, s.sale_date, m.fixture_code,
                   SUM(s.qty) AS qty,
                   SUM(s.amount) AS amount
            FROM sku_sales_daily s
            JOIN fixture_sku_map m
              ON m.store = s.store AND m.sku = s.sku
            WHERE s.store = ?
            GROUP BY s.store, s.sale_date, m.fixture_code;
            """,
            (selected_store,)
        )

        conn.commit()
    finally:
        conn.close()


def get_available_date_range(selected_store: str):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT MIN(sale_date), MAX(sale_date) FROM sku_sales_daily WHERE store = ?;",
            (selected_store,)
        ).fetchone()
        if not row or not row[0]:
            return None, None
        return row[0], row[1]
    finally:
        conn.close()


def fixture_sales_summary(selected_store: str, start_date: str, end_date: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        mapped = pd.read_sql_query(
            """
            SELECT fixture_code,
                   SUM(qty) AS sold_qty,
                   SUM(amount) AS sales_amt
            FROM fixture_sales_daily
            WHERE store = ?
              AND sale_date BETWEEN ? AND ?
            GROUP BY fixture_code
            ORDER BY sales_amt DESC;
            """,
            conn,
            params=(selected_store, start_date, end_date),
        )

        unmapped = pd.read_sql_query(
            """
            SELECT SUM(s.qty) AS sold_qty,
                   SUM(s.amount) AS sales_amt
            FROM sku_sales_daily s
            LEFT JOIN fixture_sku_map m
              ON m.store = s.store AND m.sku = s.sku
            WHERE s.store = ?
              AND s.sale_date BETWEEN ? AND ?
              AND m.sku IS NULL;
            """,
            conn,
            params=(selected_store, start_date, end_date),
        )
    finally:
        conn.close()

    if mapped.empty:
        mapped = pd.DataFrame(columns=["fixture_code", "sold_qty", "sales_amt"])

    unmapped_qty = float(unmapped.iloc[0]["sold_qty"]) if not unmapped.empty and unmapped.iloc[0]["sold_qty"] is not None else 0.0
    unmapped_amt = float(unmapped.iloc[0]["sales_amt"]) if not unmapped.empty and unmapped.iloc[0]["sales_amt"] is not None else 0.0

    total_amt = float(mapped["sales_amt"].sum()) + unmapped_amt
    total_qty = float(mapped["sold_qty"].sum()) + unmapped_qty

    if total_amt > 0:
        mapped["sales_%"] = (mapped["sales_amt"] / total_amt) * 100.0
    else:
        mapped["sales_%"] = 0.0

    out = mapped.copy()

    out = pd.concat([out, pd.DataFrame([{
        "fixture_code": "UNMAPPED",
        "sold_qty": unmapped_qty,
        "sales_amt": unmapped_amt,
        "sales_%": (unmapped_amt / total_amt) * 100.0 if total_amt > 0 else 0.0
    }])], ignore_index=True)

    out = pd.concat([out, pd.DataFrame([{
        "fixture_code": "TOTAL",
        "sold_qty": total_qty,
        "sales_amt": total_amt,
        "sales_%": 100.0 if total_amt > 0 else 0.0
    }])], ignore_index=True)

    out["sold_qty"] = out["sold_qty"].astype(float).round(0).astype(int)
    out["sales_amt"] = out["sales_amt"].astype(float).round(2)
    out["sales_%"] = out["sales_%"].astype(float).round(2)

    return out


def top_unmapped_skus(selected_store: str, start_date: str, end_date: str, top_n: int = 30) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                s.sku,
                COALESCE(p.product_name, '') AS product_name,
                SUM(s.qty) AS sold_qty,
                SUM(s.amount) AS sales_amt
            FROM sku_sales_daily s
            LEFT JOIN fixture_sku_map m
              ON m.store = s.store AND m.sku = s.sku
            LEFT JOIN products p
                ON TRIM(UPPER(p.sku)) = TRIM(UPPER(s.sku))
            WHERE s.store = ?
              AND s.sale_date BETWEEN ? AND ?
              AND m.sku IS NULL
            GROUP BY s.sku, p.product_name
            ORDER BY sales_amt DESC
            LIMIT ?;
            """,
            conn,
            params=(selected_store, start_date, end_date, top_n),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["sold_qty"] = df["sold_qty"].astype(float).round(0).astype(int)
    df["sales_amt"] = df["sales_amt"].astype(float).round(2)
    return df


# ----------------------------
# UI: Ingest (Admin auto OR Upload)
# ----------------------------
st.subheader("1) Load Sales Report (product_report_all.csv)")

replace_store_sales = st.checkbox(
    "Replace existing sales for this store (recommended when loading fresh file)",
    value=False,
    help="If ON: deletes existing sales_raw + aggregations for this store before ingesting."
)

project_files = list_project_data_files()
default_sales_path = auto_pick_sales_file(project_files)

# ADMIN MODE: auto from project folder
if IS_ADMIN:
    if default_sales_path:
        st.success(f"Auto-selected (project): {Path(default_sales_path).name}")
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("📥 Ingest Sales (Auto from Project)", type="primary"):
                b = Path(default_sales_path).read_bytes()
                res = ingest_sales_csv_for_store(b, Path(default_sales_path).name, store, replace_store_sales)
                if res["status"] == "ok":
                    st.success(f"Ingested ✅ Inserted: {res['rows']} | Skipped: {res.get('skipped', 0)}")
                    if res.get("errors"):
                        st.warning("Some rows were skipped (showing up to 100 errors):")
                        st.write(res["errors"])
                elif res["status"] == "skipped":
                    st.info(res["reason"])
                else:
                    st.error("Ingest failed")
                    st.write(res.get("errors", []))
        with colB:
            if st.button("🔁 Recompute Aggregations (SKU Daily + Fixture Daily)"):
                recompute_aggregations(store)
                st.success("Recomputed ✅ Now view fixture sales below.")
    else:
        st.warning("Admin Mode: Could not find sales file in project folder (product_report_all / report_all).")

    # Optional upload even in admin mode
    with st.expander("Or upload a fresh sales CSV manually"):
        sales_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="sales_upload_admin")
        if sales_file and st.button("✅ Ingest Uploaded CSV"):
            res = ingest_sales_csv_for_store(sales_file.getvalue(), sales_file.name, store, replace_store_sales)
            if res["status"] == "ok":
                st.success(f"Ingested ✅ Inserted: {res['rows']} | Skipped: {res.get('skipped', 0)}")
            elif res["status"] == "skipped":
                st.info(res["reason"])
            else:
                st.error("Ingest failed")
                st.write(res.get("errors", []))

# USER MODE: upload required
else:
    sales_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False, key="sales_upload_user")
    c1, c2 = st.columns([1, 1])
    with c1:
        if sales_file and st.button("✅ Ingest Sales CSV", type="primary"):
            res = ingest_sales_csv_for_store(sales_file.getvalue(), sales_file.name, store, replace_store_sales)
            if res["status"] == "ok":
                st.success(f"Ingested ✅ Inserted: {res['rows']} | Skipped: {res.get('skipped', 0)}")
                if res.get("errors"):
                    st.warning("Some rows were skipped (showing up to 100 errors):")
                    st.write(res["errors"])
            elif res["status"] == "skipped":
                st.info(res["reason"])
            else:
                st.error("Ingest failed")
                st.write(res.get("errors", []))
    with c2:
        if st.button("🔁 Recompute Aggregations (SKU Daily + Fixture Daily)"):
            recompute_aggregations(store)
            st.success("Recomputed ✅ Now view fixture sales below.")

st.divider()

# ----------------------------
# UI: View sales
# ----------------------------
st.subheader("2) View Fixture-wise Sales + %")

mn, mx = get_available_date_range(store)
if not mn:
    st.warning("No aggregated sales found for this store yet. Ingest sales and click Recompute Aggregations.")
    st.stop()

d1, d2 = st.columns(2)
with d1:
    start_dt = st.date_input("Start date", value=datetime.strptime(mn, "%Y-%m-%d").date())
with d2:
    end_dt = st.date_input("End date", value=datetime.strptime(mx, "%Y-%m-%d").date())

top_n = st.slider("Show top fixtures (excluding TOTAL/UNMAPPED always shown)", 10, 200, 50, step=10)

if st.button("📊 Show Fixture Sales"):
    start_s = start_dt.strftime("%Y-%m-%d")
    end_s = end_dt.strftime("%Y-%m-%d")

    df = fixture_sales_summary(store, start_s, end_s)

    total_row = df[df["fixture_code"] == "TOTAL"].iloc[0]
    unm_row = df[df["fixture_code"] == "UNMAPPED"].iloc[0]

    total_sales = float(total_row["sales_amt"])
    total_qty = int(total_row["sold_qty"])
    unm_pct = float(unm_row["sales_%"])
    mapped_pct = round(100.0 - unm_pct, 2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sales", fmt_inr(total_sales))
    m2.metric("Total Sold Qty", f"{total_qty:,}")
    m3.metric("Mapped %", f"{mapped_pct:.2f}%")
    m4.metric("Unmapped %", f"{unm_pct:.2f}%")

    core = df[(~df["fixture_code"].isin(["UNMAPPED", "TOTAL"]))].copy()
    core = core.head(top_n)
    tail = df[df["fixture_code"].isin(["UNMAPPED", "TOTAL"])].copy()
    show = pd.concat([core, tail], ignore_index=True)

    show_display = show.rename(columns={
        "fixture_code": "Fixture",
        "sold_qty": "Sold Qty",
        "sales_amt": "Sales (₹)",
        "sales_%": "Sales %"
    }).copy()

    st.markdown("### Fixture-wise Summary")

    st.data_editor(
        show_display,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "Fixture": st.column_config.TextColumn(width="small"),
            "Sold Qty": st.column_config.NumberColumn(format="%d", width="small"),
            "Sales (₹)": st.column_config.NumberColumn(format="₹%.2f", width="small"),
            "Sales %": st.column_config.NumberColumn(format="%.2f", width="small"),
        },
    )

    if unm_pct > 0:
        st.warning(f"UNMAPPED sales is **{unm_pct:.2f}%** — map more SKUs in Fixture Mapping for accurate heatmap.")

    st.divider()

    st.markdown("### Top Unmapped SKUs (by Sales) — map these first")
    unm_df = top_unmapped_skus(store, start_s, end_s, top_n=30)
    if unm_df.empty:
        st.success("No unmapped SKUs 🎉")
    else:
        unm_display = unm_df.rename(columns={
            "product_name": "Product Name",
            "sku": "SKU",
            "sold_qty": "Sold Qty",
            "sales_amt": "Sales (₹)"
        })
        st.data_editor(
            unm_display,
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_config={
                "Product Name": st.column_config.TextColumn(width="large"),
                "SKU": st.column_config.TextColumn(width="medium"),
                "Sold Qty": st.column_config.NumberColumn(format="%d", width="small"),
                "Sales (₹)": st.column_config.NumberColumn(format="₹%.2f", width="small"),
            },
        )
        st.caption("Go to Fixture Mapping → map these SKUs to fixtures → come back here → Recompute.")