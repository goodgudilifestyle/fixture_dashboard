# pages/5_Heatmap.py

import re
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from db_utils import (
    ensure_step3_schema,
    get_conn,
    get_fixtures_for_store,
    upsert_fixture_sku_rows,
    delete_fixture_sku_rows_for_fixture,
    export_fixture_sku_map,
)

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Step 5 - Heatmap", layout="wide")
ensure_step3_schema()

st.title("Step 5 — Heatmap Overlay (Fixture Sales on Layout)")

store = (st.session_state.get("selected_store") or st.session_state.get("selected_outlet") or "").strip()
if not store:
    st.error("No store selected. Go to Home (app) page and select a store first.")
    st.stop()

st.info(f"✅ Selected store: **{store}**")
st.caption("Heatmap uses: Layout fixtures + SKU→Fixture mapping + aggregated sales (SKU Daily / Fixture Daily).")

# ----------------------------
# Helpers
# ----------------------------
def fmt_inr(x):
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return "₹0"

def parse_skus(txt: str):
    if not txt:
        return []
    raw = re.split(r"[\n,]+", txt)
    out = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
    # unique preserve order
    seen = set()
    final = []
    for s in out:
        if s not in seen:
            final.append(s)
            seen.add(s)
    return final

def recompute_aggregations(selected_store: str):
    conn = get_conn()
    try:
        conn.execute("DELETE FROM sku_sales_daily WHERE store = ?;", (selected_store,))
        conn.execute("DELETE FROM fixture_sales_daily WHERE store = ?;", (selected_store,))

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
            (selected_store,),
        )

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
            (selected_store,),
        )
        conn.commit()
    finally:
        conn.close()

def get_date_range(selected_store: str):
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT MIN(sale_date), MAX(sale_date) FROM sku_sales_daily WHERE store = ?;",
            (selected_store,),
        ).fetchone()
        if not row or not row[0]:
            return None, None
        return row[0], row[1]
    finally:
        conn.close()

def get_fixture_totals(selected_store: str, start_date: str, end_date: str):
    conn = get_conn()
    try:
        fx = pd.read_sql_query(
            """
            SELECT fixture_code,
                   SUM(qty) AS sold_qty,
                   SUM(amount) AS sales_amt
            FROM fixture_sales_daily
            WHERE store = ?
              AND sale_date BETWEEN ? AND ?
            GROUP BY fixture_code;
            """,
            conn,
            params=(selected_store, start_date, end_date),
        )

        total = pd.read_sql_query(
            """
            SELECT SUM(amount) AS total_sales, SUM(qty) AS total_qty
            FROM sku_sales_daily
            WHERE store = ?
              AND sale_date BETWEEN ? AND ?;
            """,
            conn,
            params=(selected_store, start_date, end_date),
        )
    finally:
        conn.close()

    if fx.empty:
        fx = pd.DataFrame(columns=["fixture_code", "sold_qty", "sales_amt"])

    total_sales = float(total.iloc[0]["total_sales"]) if (not total.empty and total.iloc[0]["total_sales"] is not None) else 0.0
    total_qty = float(total.iloc[0]["total_qty"]) if (not total.empty and total.iloc[0]["total_qty"] is not None) else 0.0

    fx["sold_qty"] = fx["sold_qty"].fillna(0).astype(float)
    fx["sales_amt"] = fx["sales_amt"].fillna(0).astype(float)
    fx["sales_%"] = (fx["sales_amt"] / total_sales * 100.0) if total_sales > 0 else 0.0

    return fx, total_sales, total_qty

def get_fixture_product_breakdown(selected_store: str, fixture_code: str, start_date: str, end_date: str):
    """
    Shows BOTH Product Name + SKU.
    If product name is missing in products table, fallback to SKU.
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                m.sku AS sku,
                COALESCE(p.product_name, '') AS product_name,
                SUM(COALESCE(s.qty, 0)) AS sold_qty,
                SUM(COALESCE(s.amount, 0)) AS sales_amt
            FROM fixture_sku_map m
            LEFT JOIN sku_sales_daily s
              ON s.store = m.store
             AND s.sku = m.sku
             AND s.sale_date BETWEEN ? AND ?
            LEFT JOIN products p
              ON p.sku = m.sku
            WHERE m.store = ?
              AND m.fixture_code = ?
            GROUP BY m.sku, p.product_name
            ORDER BY sales_amt DESC;
            """,
            conn,
            params=(start_date, end_date, selected_store, fixture_code),
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["product_name"] = df["product_name"].fillna("").astype(str)
    df.loc[df["product_name"].str.strip().eq(""), "product_name"] = df["sku"]

    df["sold_qty"] = df["sold_qty"].fillna(0).astype(float).round(0).astype(int)
    df["sales_amt"] = df["sales_amt"].fillna(0).astype(float).round(2)
    return df

# ----------------------------
# Load fixtures from layout
# ----------------------------
fixtures = get_fixtures_for_store(store)
if not fixtures:
    st.warning("No fixtures found for this store. Create fixtures in Layout Builder first.")
    st.stop()

mn, mx = get_date_range(store)
if not mn:
    st.warning("No aggregated sales yet. Go to Fixture Sales → ingest sales → recompute.")
    st.stop()

# ----------------------------
# Controls (top)
# ----------------------------
c = st.columns([1.2, 1.2, 1.4, 1.2, 1.4])
with c[0]:
    start_dt = st.date_input("Start date", value=datetime.strptime(mn, "%Y-%m-%d").date())
with c[1]:
    end_dt = st.date_input("End date", value=datetime.strptime(mx, "%Y-%m-%d").date())
with c[2]:
    metric_mode = st.selectbox("Heatmap metric", ["Sales (₹)", "Sold Qty", "Sales %"])
with c[3]:
    zoom = st.slider("Zoom (canvas height)", 0.7, 2.5, 1.2, 0.1)
with c[4]:
    if st.button("🔁 Recompute (after mapping changes)", type="primary", use_container_width=True):
        recompute_aggregations(store)
        st.success("Recomputed ✅")
        st.rerun()

start_s = start_dt.strftime("%Y-%m-%d")
end_s = end_dt.strftime("%Y-%m-%d")

totals_df, total_sales, total_qty = get_fixture_totals(store, start_s, end_s)

fx_df = pd.DataFrame(fixtures)
fx_df = fx_df.merge(totals_df, on="fixture_code", how="left")
fx_df["sold_qty"] = fx_df["sold_qty"].fillna(0.0)
fx_df["sales_amt"] = fx_df["sales_amt"].fillna(0.0)
fx_df["sales_%"] = fx_df["sales_%"].fillna(0.0)

# Choose value for coloring
if metric_mode == "Sales (₹)":
    fx_df["value"] = fx_df["sales_amt"]
elif metric_mode == "Sold Qty":
    fx_df["value"] = fx_df["sold_qty"]
else:
    fx_df["value"] = fx_df["sales_%"]

max_val = float(fx_df["value"].max()) if len(fx_df) else 0.0
max_val = max(max_val, 1e-9)

# Bounds
min_x = float(fx_df["x"].min())
min_y = float(fx_df["y"].min())
max_x = float((fx_df["x"] + fx_df["w"]).max())
max_y = float((fx_df["y"] + fx_df["h"]).max())

pad = 60
min_x -= pad
min_y -= pad
max_x += pad
max_y += pad

# ----------------------------
# FULL WIDTH CANVAS (CRISP RECTANGLES)
# ----------------------------
st.subheader("Heatmap Canvas (Full Width)")

fig = go.Figure()

# Red (low) -> Yellow -> Green (high)
colorscale = "RdYlGn"

centers_x, centers_y, hover_text, custom_data = [], [], [], []

# Background "layout border"
fig.add_shape(
    type="rect",
    x0=min_x,
    y0=min_y,
    x1=max_x,
    y1=max_y,
    line=dict(color="rgba(255,255,255,0.20)", width=4),
    fillcolor="rgba(0,0,0,0)",
    layer="below",
)

for _, r in fx_df.iterrows():
    x0 = float(r["x"])
    y0 = float(r["y"])
    w = float(r["w"])
    h = float(r["h"])
    if w <= 0 or h <= 0:
        continue

    v = float(r["value"])
    fixture_code = str(r["fixture_code"])

    # normalize to 0..1 for colorscale sampling
    t = 0.0 if max_val <= 0 else max(0.0, min(1.0, v / max_val))
    fill = sample_colorscale(colorscale, t)[0]

    # Filled rectangle (CRISP)
    fig.add_shape(
        type="rect",
        x0=x0,
        y0=y0,
        x1=x0 + w,
        y1=y0 + h,
        line=dict(color="rgba(255,255,255,0.55)", width=3),
        fillcolor=fill,
        layer="below",
    )

    # Fixture label ABOVE the block (super clear)
    fig.add_annotation(
        x=x0 + w / 2,
        y=y0 - 12,
        text=f"<b>{fixture_code}</b>",
        showarrow=False,
        font=dict(color="white", size=16),
    )

    ht = (
        f"<span style='font-size:18px'><b>{fixture_code}</b></span><br>"
        f"<span style='font-size:16px'>Sales: {fmt_inr(r['sales_amt'])}<br>"
        f"Qty: {int(r['sold_qty'])}<br>"
        f"Share: {float(r['sales_%']):.2f}%</span>"
    )

    centers_x.append(x0 + w / 2)
    centers_y.append(y0 + h / 2)
    hover_text.append(ht)
    custom_data.append([fixture_code])

# Invisible markers ONLY for hover/click selection (keeps canvas crisp)
fig.add_trace(
    go.Scatter(
        x=centers_x,
        y=centers_y,
        mode="markers",
        marker=dict(size=26, opacity=0),
        hovertext=hover_text,
        hoverinfo="text",
        customdata=custom_data,
        hoverlabel=dict(font=dict(size=20), bgcolor="#0b1220"),
        name="",
    )
)

# Add a visible colorbar using a dummy trace (no blur, only legend)
fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=colorscale,
            cmin=0,
            cmax=max_val,
            color=[0],
            size=0,
            showscale=True,
            colorbar=dict(title=metric_mode),
        ),
        hoverinfo="skip",
        showlegend=False,
    )
)

fig.update_layout(
    height=int(620 * zoom),
    margin=dict(l=8, r=8, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

fig.update_xaxes(range=[min_x, max_x], visible=False)
# invert y to match screen coords
fig.update_yaxes(range=[max_y, min_y], visible=False, scaleanchor="x", scaleratio=1)

event = st.plotly_chart(
    fig,
    use_container_width=True,
    key="heatmap_plot",
    on_select="rerun",
    selection_mode="points",
)

# Store metrics under canvas
m1, m2, m3 = st.columns(3)
m1.metric("Store Total Sales", fmt_inr(total_sales))
m2.metric("Store Total Sold Qty", f"{int(total_qty):,}")

mapped_sales = float(fx_df["sales_amt"].sum())
unmapped_sales = max(total_sales - mapped_sales, 0.0)
unm_pct = (unmapped_sales / total_sales * 100.0) if total_sales > 0 else 0.0
m3.metric("Unmapped % (Sales)", f"{unm_pct:.2f}%")

st.divider()

# ----------------------------
# Selected fixture (click OR dropdown)
# ----------------------------
fixture_list = fx_df["fixture_code"].astype(str).tolist()

clicked_fixture = None
try:
    if event and hasattr(event, "selection") and event.selection and "points" in event.selection:
        pts = event.selection["points"]
        if pts and "customdata" in pts[0]:
            clicked_fixture = pts[0]["customdata"][0]
except Exception:
    clicked_fixture = None

if "selected_fixture" not in st.session_state:
    st.session_state["selected_fixture"] = fixture_list[0]

if clicked_fixture:
    st.session_state["selected_fixture"] = clicked_fixture

sel_fc = st.selectbox(
    "Selected fixture (click a block OR choose here)",
    options=fixture_list,
    index=fixture_list.index(st.session_state["selected_fixture"]) if st.session_state["selected_fixture"] in fixture_list else 0,
    key="fixture_dropdown",
)

# ----------------------------
# Inspector (FULL WIDTH below canvas)
# ----------------------------
st.subheader("Fixture Inspector + Mapping Editor")

row = fx_df[fx_df["fixture_code"].astype(str) == str(sel_fc)].iloc[0]
fc_sales = float(row["sales_amt"])
fc_qty = float(row["sold_qty"])
fc_pct = float(row["sales_%"])

k1, k2, k3 = st.columns(3)
k1.metric("Fixture Sales", fmt_inr(fc_sales))
k2.metric("Fixture Qty", f"{int(fc_qty):,}")
k3.metric("Fixture % Share", f"{fc_pct:.2f}%")

st.markdown("### Products mapped to this fixture (Product Name + SKU + Contribution)")
prod_df = get_fixture_product_breakdown(store, str(sel_fc), start_s, end_s)

if prod_df.empty:
    st.info("No mapped products OR no sales for mapped SKUs in this date range.")
else:
    disp = prod_df.rename(columns={
        "product_name": "Product Name",
        "sku": "SKU",
        "sold_qty": "Sold Qty",
        "sales_amt": "Sales (₹)",
    })
    st.dataframe(disp, use_container_width=True, height=280)

st.divider()

# ----------------------------
# Mapping editor
# ----------------------------
st.markdown("### Edit Mapping for this fixture")

add_txt = st.text_area(
    "Add SKU(s) (one per line or comma-separated)",
    placeholder="Example:\nTAC002129TAS\nSTN001669MBHB",
    height=90,
    key="heat_add_skus",
)

if st.button("➕ Add SKUs to this fixture", use_container_width=True):
    skus = parse_skus(add_txt)
    if not skus:
        st.error("Enter at least 1 SKU.")
    else:
        rows = []
        for sku in skus:
            rows.append({
                "store": store,
                "fixture_code": str(sel_fc),
                "sku": sku.strip(),
                "product_name": None,
                "category": None,
                "priority": None,
            })
        res = upsert_fixture_sku_rows(rows, mode="append")
        st.success(f"Added ✅ Inserted: {res['inserted']} | Skipped duplicates: {res['skipped']}")
        st.session_state["heat_add_skus"] = ""
        st.info("Now click 🔁 Recompute (top) to refresh sales aggregates.")
        st.rerun()

cur_map = export_fixture_sku_map(store)
df_map = pd.DataFrame(cur_map) if cur_map else pd.DataFrame(columns=["fixture_code", "sku"])
df_map = df_map[df_map["fixture_code"].astype(str) == str(sel_fc)] if not df_map.empty else df_map

if df_map.empty:
    st.caption("No SKUs mapped here yet — nothing to remove.")
else:
    rem = st.multiselect(
        "Remove SKU(s) from this fixture",
        options=sorted(df_map["sku"].astype(str).unique().tolist()),
        default=[],
        key="heat_remove_skus",
    )
    if st.button("🗑️ Remove selected SKUs", use_container_width=True):
        if not rem:
            st.error("Select at least 1 SKU.")
        else:
            deleted = delete_fixture_sku_rows_for_fixture(store, str(sel_fc), rem)
            st.success(f"Removed ✅ Rows deleted: {deleted}")
            st.info("Now click 🔁 Recompute (top) to refresh sales aggregates.")
            st.rerun()