import streamlit as st
import pandas as pd
import sqlite3
import os
from pathlib import Path

DB_PATH = "gogigi.db"

REQUIRED_PRODUCT_COLS = ["Product Name", "SKU"]


POS_OUTLETS = [
    "Gogigi - EC",
    "Goodgudi -  Kammanahalli",
    "Goodgudi - J P Nagar",
    "Goodgudi - Koramangala",
    "Goodgudi - Nexus Mall Shantiniketan",
    "Goodgudi Brigade",
    "Goodgudi Commercial Street",
    "Goodgudi HSR Layout",
    "Goodgudi Jayanagar",
    "Goodgudi Malleshwaram",
]

# ---------- helpers ----------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df

def read_upload_to_df(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Upload CSV or Excel.")
    return normalize_cols(df)

def read_project_file_to_df(filepath: str) -> pd.DataFrame:
    fp = Path(filepath)
    if not fp.exists():
        raise ValueError(f"File not found: {filepath}")
    lower = fp.name.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(fp)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(fp)
    else:
        raise ValueError("Unsupported file format in project folder.")
    return normalize_cols(df)

def ensure_cols(df: pd.DataFrame, required_cols: list[str], label: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                sku TEXT PRIMARY KEY,
                product_name TEXT
            )
        """)
        
        conn.commit()

def upsert_products(df_products: pd.DataFrame):
    dfp = df_products[REQUIRED_PRODUCT_COLS].copy()
    dfp["SKU"] = dfp["SKU"].astype(str).str.strip()
    dfp["Product Name"] = dfp["Product Name"].astype(str).str.strip()

    with get_conn() as conn:
        cur = conn.cursor()
        for _, r in dfp.iterrows():
            cur.execute("""
                INSERT INTO products (sku, product_name)
                VALUES (?, ?)
                ON CONFLICT(sku) DO UPDATE SET product_name=excluded.product_name
            """, (r["SKU"], r["Product Name"]))
        conn.commit()


def get_counts():
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM products")
        p = cur.fetchone()[0]

        # Step 4 table (created by db_utils.ensure_step3_schema())
        try:
            cur.execute("SELECT COUNT(*) FROM sales_raw")
            s = cur.fetchone()[0]
        except sqlite3.OperationalError:
            s = 0

    return p, s

def preview_sales_raw_for_outlet(outlet: str, limit=30) -> pd.DataFrame:
    with get_conn() as conn:
        try:
            return pd.read_sql_query(
                """
                SELECT sale_date, store, sku, qty, amount, invoice_no
                FROM sales_raw
                WHERE store = ?
                ORDER BY sale_date DESC, id DESC
                LIMIT ?
                """,
                conn,
                params=(outlet, limit)
            )
        except Exception:
            return pd.DataFrame()


def auto_pick_default_files(project_files: list[str]):
    """
    Auto-picks files from project folder based on name patterns.
    You can adjust patterns anytime.
    """
    product = None
    sales = None

    for f in project_files:
        lf = f.lower()
        if ("product-report" in lf or "product_report" in lf) and ("all" not in lf):
            product = f
        if ("product_report_all" in lf) or ("product_report_all" in lf) or ("report_all" in lf):
            sales = f

    # fallback: if patterns failed, pick first csv/xlsx as product and second as sales
    if product is None and len(project_files) >= 1:
        product = project_files[0]
    if sales is None and len(project_files) >= 2:
        sales = project_files[1]

    return product, sales

def list_project_data_files():
    # scans current folder for csv/xlsx/xls
    files = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(Path(".").glob(ext))
    return sorted([str(f) for f in files])

# ---------- UI ----------
st.set_page_config(page_title="Goodgudi Fixture Sales Dashboard", layout="wide")
init_db()

st.title("Goodgudi Fixture Sales Dashboard")


# --- Larger Toggle CSS ---
st.markdown("""
<style>
/* Make the Toggle bigger (Streamlit new DOM) */
div[data-testid="stToggle"] label {
  font-size: 20px !important;
  font-weight: 700 !important;
}

/* the switch track */
div[data-testid="stToggle"] label span {
  transform: scale(1.8) !important;
  transform-origin: left center !important;
  display: inline-block !important;
}
</style>
""", unsafe_allow_html=True)

header_left, header_right = st.columns([6, 1])
with header_right:
    IS_ADMIN = st.toggle("Admin Mode", value=True, key="mode_toggle")

st.subheader("1) Select Store (POS Outlet)")
selected_outlet = st.session_state.get("selected_outlet", POS_OUTLETS[0])

cols = st.columns(5)
for i, outlet in enumerate(POS_OUTLETS):
    with cols[i % 5]:
        label = f"✅ {outlet}" if outlet == selected_outlet else outlet
        if st.button(label, use_container_width=True, key=f"store_{i}"):
            st.session_state["selected_outlet"] = outlet
            st.session_state["selected_store"] = outlet
            st.rerun()

selected_outlet = st.session_state.get("selected_outlet", POS_OUTLETS[0])
st.success(f"Selected Store: {selected_outlet}")
st.session_state["selected_store"] = selected_outlet   # ✅ add this



st.subheader("2) Load Files (CSV/Excel)")
left, right = st.columns(2)

project_files = list_project_data_files()
default_prod, default_sales = auto_pick_default_files(project_files)


# ---------- PRODUCT ----------
with left:
    st.markdown("### Product Master (Product-Report)")
    df_prod = None

    if IS_ADMIN:
        if default_prod is None:
            st.error("Admin mode: No product file found in project folder.")
        else:
            st.success(f"Auto-selected (project): {Path(default_prod).name}")
            if st.button("Load Product (auto)", key="load_prod_auto"):
                df_prod = read_project_file_to_df(default_prod)
    else:
        prod_file = st.file_uploader(
            "Upload Product Master (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
            key="prod_up"
        )
        if prod_file:
            df_prod = read_upload_to_df(prod_file)

    if df_prod is not None:
        try:
            ensure_cols(df_prod, REQUIRED_PRODUCT_COLS, "Product Master")
            st.info(f"Product file OK. Rows: {len(df_prod)}")
            st.dataframe(df_prod[REQUIRED_PRODUCT_COLS].head(10), width="stretch")

            if st.button("Save Product Master to DB", key="save_prod_db"):
                upsert_products(df_prod)
                p, s = get_counts()
                st.success(f"Saved ✅ | products={p}, sales={s}")
        except Exception as e:
            st.error(str(e))


st.subheader("3) Database Status + Quick Preview")
p_count, s_count = get_counts()
st.write(f"**Products in DB:** {p_count}  |  **Sales rows in DB (Step 4 / sales_raw):** {s_count}")

if s_count > 0:
    st.markdown(f"### Sales Preview (from Step 4 ingestion) for: {selected_outlet}")
    prev = preview_sales_raw_for_outlet(selected_outlet, limit=30)
    if prev.empty:
        st.warning("Sales exist, but none found for this selected store name. Check POS Outlet text match.")
    else:
        st.dataframe(prev, width="stretch")
else:
    st.info("No sales ingested yet. Go to **Fixture Sales (Step 4)** → Upload CSV → Ingest → Recompute.")


st.divider()
st.caption("Dashboard Designed & Developed By RoHIT_Chougule @GoGiGi")