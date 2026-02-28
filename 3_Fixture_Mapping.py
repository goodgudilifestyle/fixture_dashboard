import pandas as pd
import re
import streamlit as st

from db_utils import (
    ensure_step3_schema,
    get_fixture_codes_for_store,
    upsert_fixture_sku_rows,
    export_fixture_sku_map,
    get_store_fixture_sku_map,     # ✅ add
    delete_fixture_sku_rows,       # ✅ add
)

st.set_page_config(page_title="Step 3A - Fixture Mapping", layout="wide")
ensure_step3_schema()

st.title("Step 3A — Bulk Mapping Manager (SKU → Fixture)")

# ✅ Store comes ONLY from app.py session state
store = st.session_state.get("selected_store", "").strip()

if not store:
    st.error("No store selected. Go to the Home (app) page and select a store first.")
    st.stop()

st.info(f"✅ Selected store from Step 1: **{store}**")
st.caption("To change store: go to the Home page → select store → come back here.")

# Load fixture codes for this store
try:
    fixture_codes = get_fixture_codes_for_store(store)
except Exception as e:
    st.error(f"Could not load fixtures for store '{store}': {e}")
    st.stop()

if not fixture_codes:
    st.warning("No fixtures found for this store. Create fixtures in Layout Builder first.")
    st.stop()

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

# ----------------------------
# Downloads
# ----------------------------
with c1:
    st.subheader("Download Fixtures (Selected Store)")
    df_fx = pd.DataFrame({"fixture_code": fixture_codes})
    st.download_button(
        "✅ Download Fixtures CSV",
        data=df_fx.to_csv(index=False).encode("utf-8"),
        file_name=f"{store}_fixtures.csv".replace(" ", "_"),
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("This file lists all fixture codes available under the selected store.")

with c2:
    st.subheader("Download Mapping Template")
    # Template: prefill fixture_code rows, sku blank (user duplicates rows for multiple skus)
    df_tpl = pd.DataFrame({"sku": ["" for _ in fixture_codes], "fixture_code": fixture_codes})
    st.download_button(
        "✅ Download Mapping Template CSV",
        data=df_tpl.to_csv(index=False).encode("utf-8"),
        file_name=f"{store}_sku_fixture_template.csv".replace(" ", "_"),
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Fill SKU column. If one fixture has many SKUs, duplicate that fixture_code row.")

with c3:
    st.subheader("Export Current Mapping")
    export_rows = export_fixture_sku_map(store)
    df_exp = pd.DataFrame(export_rows)
    st.download_button(
        "⬇️ Export Mapping CSV (this store)",
        data=df_exp.to_csv(index=False).encode("utf-8"),
        file_name=f"{store}_fixture_sku_map_export.csv".replace(" ", "_"),
        mime="text/csv",
        use_container_width=True,
    )
    st.caption("Exports what is already saved in DB for this selected store.")

st.divider()

# ----------------------------
# Manual Mapping (NEW)
# ----------------------------
st.subheader("Manual Mapping (Quick Add / Remove)")

left, right = st.columns([1.1, 1.4])

with left:
    st.markdown("### Add SKU → Fixture")

    fc_sel = st.selectbox(
        "Select fixture_code",
        options=fixture_codes,
        index=0,
        key="manual_fc"
    )

    sku_text = st.text_area(
        "Enter SKU(s) (one per line or comma-separated)",
        placeholder="Example:\nTAC002129TAS\nSTN001669MBHB\nKID001428MSTP",
        height=120,
        key="manual_skus"
    )

    def parse_skus(txt: str):
        if not txt:
            return []
        # split by comma or newline
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

    add_clicked = st.button("✅ Add Mapping", type="primary", use_container_width=True)

    if add_clicked:
        

        skus = parse_skus(sku_text)

        if not skus:
            st.error("Please enter at least 1 SKU.")
        else:
            rows = []
            for sku in skus:
                rows.append({
                    "store": store,
                    "fixture_code": fc_sel,
                    "sku": sku,
                    "product_name": None,
                    "category": None,
                    "priority": None,
                })

            res = upsert_fixture_sku_rows(rows, mode="append")
            st.success(f"Added ✅ Inserted: {res['inserted']} | Skipped duplicates: {res['skipped']}")
            st.rerun()

with right:
    st.markdown("### Existing Mapping (this store)")

    current = get_store_fixture_sku_map(store)
    if not current:
        st.info("No mappings saved yet for this store.")
    else:
        df_cur = pd.DataFrame(current)

        # Show mapping
        st.dataframe(df_cur, use_container_width=True, height=280)

        # Delete mapping
        st.markdown("### Remove Mapping")
        # Multi select SKUs to delete
        del_skus = st.multiselect(
            "Select SKU(s) to remove",
            options=sorted(df_cur["sku"].astype(str).unique().tolist()),
            default=[],
            key="del_skus"
        )

        if st.button("🗑️ Delete Selected SKU(s)", use_container_width=True):
            if not del_skus:
                st.error("Select at least 1 SKU to delete.")
            else:
                deleted = delete_fixture_sku_rows(store, del_skus)
                st.success(f"Deleted ✅ Rows removed: {deleted}")
                st.rerun()

st.divider()

# ----------------------------
# Upload + Validate + Save
# ----------------------------
st.subheader("Upload Filled Mapping Template (CSV)")

upload_mode = st.radio(
    "Save Mode",
    options=["append", "replace"],
    horizontal=True,
    help="append: keep existing and add new unique rows. replace: delete all mappings for this store and insert fresh."
)

uploaded = st.file_uploader("Upload CSV (columns: sku, fixture_code)", type=["csv"], accept_multiple_files=False)

if uploaded:
    df = pd.read_csv(uploaded, dtype=str).fillna("")
    df.columns = [str(c).strip().lower() for c in df.columns]

    st.write("Preview (first 200 rows)")
    st.dataframe(df.head(200), use_container_width=True)

    errors = []
    valid_rows = []

    # Must have only needed columns (at least these two)
    for col in ["sku", "fixture_code"]:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if not errors:
        # Validate rows
        codes_set = set(fixture_codes)
        seen = set()

        for i, r in df.iterrows():
            sku = str(r.get("sku", "")).strip()
            fc = str(r.get("fixture_code", "")).strip()

            if not sku:
                errors.append(f"Row {i+2}: sku is blank")
                continue
            if not fc:
                errors.append(f"Row {i+2}: fixture_code is blank")
                continue
            if fc not in codes_set:
                errors.append(f"Row {i+2}: fixture_code '{fc}' is not valid for store '{store}'")
                continue

            key = (fc, sku)
            if key in seen:
                errors.append(f"Row {i+2}: duplicate in upload file (fixture_code={fc}, sku={sku})")
                continue
            seen.add(key)

            valid_rows.append({
                "store": store,  # ✅ comes from session
                "fixture_code": fc,
                "sku": sku,
                "product_name": None,
                "category": None,
                "priority": None,
            })

    if errors:
        st.error("Validation errors")
        st.write(errors[:250])
        if len(errors) > 250:
            st.caption(f"Showing first 250 of {len(errors)} errors.")
    else:
        st.success(f"Validation passed ✅ Valid rows: {len(valid_rows)}")

        if st.button("✅ Save Mapping to DB", type="primary"):
            # replace mode should only replace this store
            result = upsert_fixture_sku_rows(valid_rows, mode=upload_mode)
            st.success(
                f"Saved ✅ Inserted: {result['inserted']} | Skipped duplicates: {result['skipped']} | Stores replaced: {result['stores_replaced']}"
            )