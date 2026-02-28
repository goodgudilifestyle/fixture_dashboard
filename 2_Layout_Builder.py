# pages/2_Layout_Builder.py
import streamlit as st
import sqlite3
import pandas as pd
import json
import re
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

DB_PATH = "gogigi.db"

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

# ---------------- DB ----------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_col(cur, table: str, col: str, ddl: str):
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

def init_layout_tables():
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS layouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                store TEXT NOT NULL,
                version_name TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                canvas_w INTEGER NOT NULL DEFAULT 1200,
                canvas_h INTEGER NOT NULL DEFAULT 650,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layout_id INTEGER NOT NULL,
                fixture_code TEXT NOT NULL,
                fixture_type TEXT NOT NULL,
                display_name TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                w REAL NOT NULL,
                h REAL NOT NULL,
                shape_json TEXT
            )
        """)

        # migrations (safe no-op if already exists)
        ensure_col(cur, "layouts", "version_name", "version_name TEXT NOT NULL DEFAULT 'v1'")
        ensure_col(cur, "layouts", "is_active", "is_active INTEGER NOT NULL DEFAULT 1")
        ensure_col(cur, "layouts", "canvas_w", "canvas_w INTEGER NOT NULL DEFAULT 1200")
        ensure_col(cur, "layouts", "canvas_h", "canvas_h INTEGER NOT NULL DEFAULT 650")
        ensure_col(cur, "layouts", "created_at", "created_at TEXT NOT NULL DEFAULT ''")
        ensure_col(cur, "fixtures", "shape_json", "shape_json TEXT")

        conn.commit()

def get_or_create_active_layout(store: str) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM layouts WHERE store=? AND is_active=1 ORDER BY id DESC LIMIT 1",
            (store,),
        )
        r = cur.fetchone()
        if r:
            return int(r[0])

        cur.execute("""
            INSERT INTO layouts (store, version_name, is_active, canvas_w, canvas_h, created_at)
            VALUES (?, 'v1', 1, 1200, 650, ?)
        """, (store, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        return int(cur.lastrowid)

def load_fixtures(layout_id: int) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT fixture_code, fixture_type, display_name, x, y, w, h, shape_json "
            "FROM fixtures WHERE layout_id=?",
            conn,
            params=(layout_id,),
        )

def save_fixtures(layout_id: int, rows: list[dict]):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM fixtures WHERE layout_id=?", (layout_id,))
        for r in rows:
            cur.execute("""
                INSERT INTO fixtures (layout_id, fixture_code, fixture_type, display_name, x, y, w, h, shape_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                layout_id,
                r["fixture_code"],
                r["fixture_type"],
                r["display_name"],
                float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]),
                json.dumps(r.get("shape", {})),
            ))
        conn.commit()

def clear_layout_in_db(layout_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM fixtures WHERE layout_id=?", (layout_id,))
        conn.commit()

# ---------------- helpers ----------------
def split_prefix_num(s: str):
    s = (s or "").strip()
    m = re.match(r"^(.*?)(\d+)$", s)
    if not m:
        return s, None
    return m.group(1), int(m.group(2))

def next_available_code(base_code: str, used: set[str]) -> str:
    prefix, n = split_prefix_num(base_code)
    if n is None:
        prefix, n = base_code, 1
    candidate = f"{prefix}{n}"
    while candidate in used:
        n += 1
        candidate = f"{prefix}{n}"
    return candidate

def increment_like(base: str, i: int):
    prefix, n = split_prefix_num(base)
    if n is None:
        prefix, n = base, 1
    return f"{prefix}{n + i}"

def objects_key(store: str):
    return f"lb_objects::{store}"

def last_canvas_key(store: str):
    return f"lb_last_canvas_json::{store}"

def undo_key(store: str):
    return f"lb_undo_stack::{store}"

def redo_key(store: str):
    return f"lb_redo_stack::{store}"

def get_objects(store: str):
    return st.session_state.get(objects_key(store), [])

def set_objects(store: str, objs: list):
    st.session_state[objects_key(store)] = objs

def set_last_canvas_json(store: str, canvas_json: dict | None):
    st.session_state[last_canvas_key(store)] = canvas_json

def get_last_canvas_json(store: str):
    return st.session_state.get(last_canvas_key(store))

def _deepcopy_objs(objs: list) -> list:
    return json.loads(json.dumps(objs or []))

def push_undo(store: str):
    st.session_state.setdefault(undo_key(store), [])
    st.session_state.setdefault(redo_key(store), [])
    st.session_state[undo_key(store)].append(_deepcopy_objs(get_objects(store)))
    st.session_state[redo_key(store)] = []

def do_undo(store: str):
    ukey = undo_key(store)
    rkey = redo_key(store)
    st.session_state.setdefault(ukey, [])
    st.session_state.setdefault(rkey, [])
    if not st.session_state[ukey]:
        return False
    st.session_state[rkey].append(_deepcopy_objs(get_objects(store)))
    prev = st.session_state[ukey].pop()
    set_objects(store, prev)
    return True

def do_redo(store: str):
    ukey = undo_key(store)
    rkey = redo_key(store)
    st.session_state.setdefault(ukey, [])
    st.session_state.setdefault(rkey, [])
    if not st.session_state[rkey]:
        return False
    st.session_state[ukey].append(_deepcopy_objs(get_objects(store)))
    nxt = st.session_state[rkey].pop()
    set_objects(store, nxt)
    return True

def sync_objects_from_canvas(store: str, canvas_json: dict | None):
    if not canvas_json:
        return

    objs = canvas_json.get("objects", [])

    rect_pos = {}
    for o in objs:
        if o.get("type") == "rect":
            code = (o.get("fixture_code") or "").strip()
            if code:
                rect_pos[code] = (float(o.get("left", 0)), float(o.get("top", 0)))

    for o in objs:
        if o.get("type") == "text":
            code = (o.get("fixture_code") or "").strip()
            if code in rect_pos:
                x, y = rect_pos[code]
                o["left"] = x + 6
                o["top"] = y + 6

    set_objects(store, objs)

def list_existing_codes(layout_id: int) -> set[str]:
    df = load_fixtures(layout_id)
    if df.empty:
        return set()
    return set(df["fixture_code"].astype(str).str.strip().tolist())

# ---------------- Fabric object builders ----------------
def make_fixture_rect_and_label(code: str, ftype: str, name: str, left: float, top: float, w: float, h: float):
    color_map = {
        "W": ("rgba(0,160,255,0.25)", "rgba(0,160,255,1)"),
        "G": ("rgba(0,220,120,0.25)", "rgba(0,220,120,1)"),
        "E": ("rgba(255,180,0,0.25)", "rgba(255,180,0,1)"),
        "T": ("rgba(200,80,255,0.25)", "rgba(200,80,255,1)"),
        "CUSTOM": ("rgba(200,200,200,0.20)", "rgba(200,200,200,1)"),
    }
    fill, stroke = color_map.get(ftype, color_map["CUSTOM"])

    rect = {
        "type": "rect",
        "left": float(left),
        "top": float(top),
        "width": float(w),
        "height": float(h),
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": 2,
        "fixture_code": code,
        "fixture_type": ftype,
        "display_name": name,
    }

    label = {
        "type": "text",
        "text": str(name),
        "left": float(left) + 6,
        "top": float(top) + 6,
        "fontSize": 16,
        "fill": "rgba(255,255,255,0.95)",
        "selectable": False,
        "evented": False,
        "fixture_code": code,
    }

    return rect, label

def fixtures_to_shapes(fixtures_df: pd.DataFrame):
    objs = []
    for _, r in fixtures_df.iterrows():
        sj = r.get("shape_json")
        rect = None
        if sj:
            try:
                rect = json.loads(sj)
            except Exception:
                rect = None

        if not rect or rect.get("type") != "rect":
            rect, label = make_fixture_rect_and_label(
                code=str(r["fixture_code"]),
                ftype=str(r["fixture_type"]),
                name=str(r["display_name"]),
                left=float(r["x"]),
                top=float(r["y"]),
                w=float(r["w"]),
                h=float(r["h"]),
            )
            objs.append(rect); objs.append(label)
        else:
            rect["fixture_code"] = str(r["fixture_code"])
            rect["fixture_type"] = str(r["fixture_type"])
            rect["display_name"] = str(r["display_name"])
            objs.append(rect)

            lbl = {
                "type": "text",
                "text": str(r["display_name"]),
                "left": float(rect.get("left", 0)) + 6,
                "top": float(rect.get("top", 0)) + 6,
                "fontSize": 16,
                "fill": "rgba(255,255,255,0.95)",
                "selectable": False,
                "evented": False,
                "fixture_code": str(r["fixture_code"]),
            }
            objs.append(lbl)

    return objs

# ---------------- Templates ----------------
def used_codes_from_db_and_canvas(store: str, layout_id: int) -> set[str]:
    db_used = list_existing_codes(layout_id)
    canvas_used = {(o.get("fixture_code") or "").strip()
                   for o in get_objects(store)
                   if (o.get("fixture_code") or "").strip()}
    return db_used | canvas_used

def add_template_W(store: str, count: int, start_code: str, start_name: str, layout_id: int, start_left=40, start_top=40):
    used = used_codes_from_db_and_canvas(store, layout_id)
    objs = get_objects(store)

    code_prefix, code_n = split_prefix_num(start_code or "W1")
    if code_n is None:
        code_prefix, code_n = "W", 1

    name_start = (start_name or "WR1").strip()

    for i in range(count):
        desired_code = f"{code_prefix}{code_n + i}"
        code = next_available_code(desired_code, used)
        used.add(code)

        name = increment_like(name_start, i)

        per_row = int(st.session_state.get("fixtures_per_row", 10))
        gap = int(st.session_state.get("fixture_gap", 15))
        fw = float(st.session_state.get("fixture_width", 80))
        fh = float(st.session_state.get("fixture_height", 80))

        row = i // per_row
        col = i % per_row

        left = start_left + col * (fw + gap)
        top  = start_top  + row * (fh + gap)

        r, t = make_fixture_rect_and_label(code, "W", name, left, top, fw, fh)
        objs.append(r); objs.append(t)

    set_objects(store, objs)

def add_template_T(store: str, count: int, start_code: str, start_name: str, layout_id: int, start_left=40, start_top=160):
    used = used_codes_from_db_and_canvas(store, layout_id)
    objs = get_objects(store)

    code_prefix, code_n = split_prefix_num(start_code or "T1")
    if code_n is None:
        code_prefix, code_n = "T", 1

    name_start = (start_name or "T1").strip()  # ✅ default T1

    for i in range(count):
        desired_code = f"{code_prefix}{code_n + i}"
        code = next_available_code(desired_code, used)
        used.add(code)

        name = increment_like(name_start, i)
        left = start_left + i * 210
        top = start_top

        r, t = make_fixture_rect_and_label(code, "T", name, left, top, 180, 80)
        objs.append(r); objs.append(t)

    set_objects(store, objs)

def add_template_G_with_endcaps(
    store: str,
    sets_count: int,
    g_start_num: int,
    e_left_start_num: int,
    e_right_start_num: int,
    layout_id: int,
    rows: int = 2,
    cols: int = 2,
    G_w: float = 90,
    G_h: float = 60,
    E: float = 70,
    gap: float = 8,
    set_gap: float = 60,
    start_left: float = 40,
    start_top: float = 280,
):
    used = used_codes_from_db_and_canvas(store, layout_id)
    objs = get_objects(store)

    block_w = cols * G_w + (cols - 1) * gap
    block_h = rows * G_h + (rows - 1) * gap

    for s in range(sets_count):
        base_left = start_left + s * (E + gap + block_w + gap + E + set_gap)
        base_top = start_top
        e_top = base_top + (block_h - E) / 2

        eL_code = next_available_code(f"E{e_left_start_num}", used); used.add(eL_code)
        r, t = make_fixture_rect_and_label(eL_code, "E", eL_code, base_left, e_top, E, E)
        objs.append(r); objs.append(t)
        e_left_start_num += 1

        g_left = base_left + E + gap
        for rr in range(rows):
            for cc in range(cols):
                g_code = next_available_code(f"G{g_start_num}", used); used.add(g_code); g_start_num += 1
                x = g_left + cc * (G_w + gap)
                y = base_top + rr * (G_h + gap)
                r, t = make_fixture_rect_and_label(g_code, "G", g_code, x, y, G_w, G_h)
                objs.append(r); objs.append(t)

        eR_code = next_available_code(f"E{e_right_start_num}", used); used.add(eR_code)
        eR_left = g_left + block_w + gap
        r, t = make_fixture_rect_and_label(eR_code, "E", eR_code, eR_left, e_top, E, E)
        objs.append(r); objs.append(t)
        e_right_start_num += 1

    set_objects(store, objs)

# ---------------- delete / rotate ----------------
def delete_by_codes(objects: list, codes_to_delete: set[str]):
    return [o for o in objects if ((o.get("fixture_code") or "").strip() not in codes_to_delete)]

def rotate_fixtures_by_code(store: str, codes: set[str], angle: float):
    objs = get_objects(store)
    for o in objs:
        code = (o.get("fixture_code") or "").strip()
        if not code or code not in codes:
            continue
        if o.get("type") == "rect":
            o["angle"] = float(angle)
        if o.get("type") == "text":
            o["angle"] = 0
    set_objects(store, objs)

# ---------------- scaling ----------------
def scale_objects_for_view(objs: list[dict], s: float) -> list[dict]:
    if s == 1.0:
        return objs
    out = []
    for o in objs:
        o2 = dict(o)
        if "left" in o2:   o2["left"] = float(o2["left"]) * s
        if "top" in o2:    o2["top"]  = float(o2["top"]) * s
        if "width" in o2:  o2["width"]  = float(o2["width"]) * s
        if "height" in o2: o2["height"] = float(o2["height"]) * s
        if "fontSize" in o2: o2["fontSize"] = max(10, float(o2["fontSize"]) * s)
        out.append(o2)
    return out

def unscale_canvas_json(canvas_json: dict | None, s: float) -> dict | None:
    if not canvas_json or s == 1.0:
        return canvas_json
    cj = dict(canvas_json)
    objs = cj.get("objects", [])
    out = []
    for o in objs:
        o2 = dict(o)
        if "left" in o2:   o2["left"] = float(o2["left"]) / s
        if "top" in o2:    o2["top"]  = float(o2["top"])  / s
        if "width" in o2:  o2["width"]  = float(o2["width"]) / s
        if "height" in o2: o2["height"] = float(o2["height"]) / s
        if "fontSize" in o2: o2["fontSize"] = float(o2["fontSize"]) / s
        out.append(o2)
    cj["objects"] = out
    return cj

# ---------------- metadata enrichment ----------------
def enrich_rect_meta_from_text(objs: list[dict]):
    texts = [o for o in objs if o.get("type") == "text" and str(o.get("text", "")).strip()]

    def infer_from_label(label: str):
        label = label.strip()
        m = re.match(r"^(WR|TB|W|T|G|E)\s*(\d+)$", label, re.IGNORECASE)
        if not m:
            return None, "CUSTOM", label
        prefix = m.group(1).upper()
        num = int(m.group(2))

        if prefix in ("WR", "W"):
            return f"W{num}", "W", f"WR{num}"
        if prefix in ("TB", "T"):
            return f"T{num}", "T", f"T{num}"   # ✅ normalize to T
        if prefix == "G":
            return f"G{num}", "G", f"G{num}"
        if prefix == "E":
            return f"E{num}", "E", f"E{num}"
        return None, "CUSTOM", label

    for r in [o for o in objs if o.get("type") == "rect"]:
        if r.get("fixture_code") and r.get("fixture_type") and r.get("display_name"):
            continue

        rx = float(r.get("left", 0))
        ry = float(r.get("top", 0))
        rw = float(r.get("width", 0)) * float(r.get("scaleX", 1))
        rh = float(r.get("height", 0)) * float(r.get("scaleY", 1))

        picked = None
        for t in texts:
            tx = float(t.get("left", 0))
            ty = float(t.get("top", 0))
            if rx <= tx <= rx + rw and ry <= ty <= ry + rh:
                picked = str(t.get("text", "")).strip()
                break

        if not picked and texts:
            best_d = None
            for t in texts:
                tx = float(t.get("left", 0))
                ty = float(t.get("top", 0))
                d = (tx - rx) ** 2 + (ty - ry) ** 2
                if best_d is None or d < best_d:
                    best_d = d
                    picked = str(t.get("text", "")).strip()

        if picked:
            code, ftype, name = infer_from_label(picked)
            if not r.get("display_name"):
                r["display_name"] = name
            if not r.get("fixture_type"):
                r["fixture_type"] = ftype
            if not r.get("fixture_code"):
                r["fixture_code"] = code or picked

# ---------------- UI ----------------
st.set_page_config(page_title="Layout Builder", layout="wide")
init_layout_tables()

st.title("Step 2 — Fixture Layout Builder (W / G / E / T)")
st.caption("Add fixtures → move on canvas → Apply → Save to DB. You can keep adding fixtures later too.")

store_default = st.session_state.get("selected_outlet", POS_OUTLETS[0])
store_index = POS_OUTLETS.index(store_default) if store_default in POS_OUTLETS else 0
store = st.selectbox("Select Store", POS_OUTLETS, index=store_index, key="lb_store_select")

layout_id = get_or_create_active_layout(store)

# canvas version key (forces refresh)
cv_key = f"lb_canvas_ver::{store}"
if cv_key not in st.session_state:
    st.session_state[cv_key] = 0

def bump_canvas():
    st.session_state[cv_key] += 1

# Load saved fixtures ONCE per store into session objects
if objects_key(store) not in st.session_state:
    df0 = load_fixtures(layout_id)
    set_objects(store, fixtures_to_shapes(df0) if not df0.empty else [])
    set_last_canvas_json(store, None)
    st.session_state.setdefault(undo_key(store), [])
    st.session_state.setdefault(redo_key(store), [])

left, right = st.columns([3, 2], gap="large")

# ---- LEFT FIRST: Canvas always visible ----
with left:
    st.subheader("Canvas (Build Layout Here)")
    st.caption("Tip: Choose a template on the right → click ➕ Add to Canvas → drag & arrange → Apply → Save.")

    view_zoom = float(st.session_state.get("view_zoom", 1.0))
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        stroke_color="rgba(255,255,255,0.35)",
        background_color="rgba(20,20,20,1)",
        update_streamlit=True,
        height=int(650 * view_zoom),
        width=int(1200 * view_zoom),
        initial_drawing={"version": "4.4.0", "objects": scale_objects_for_view(get_objects(store), view_zoom)},
        drawing_mode="transform",
        display_toolbar=True,
        key=f"canvas_{store}_{st.session_state[cv_key]}",
    )

    if canvas.json_data is not None:
        set_last_canvas_json(store, canvas.json_data)

    apply_changes = st.button("✅ Apply Canvas Changes (Update Positions)", use_container_width=True)
    if apply_changes:
        push_undo(store)
        latest = canvas.json_data or get_last_canvas_json(store)
        latest = unscale_canvas_json(latest, view_zoom)
        sync_objects_from_canvas(store, latest)
        st.success("Applied ✅ Now click Save Layout to DB if final.")

# ---- RIGHT: Controls ----
with right:
    st.subheader("Add Prebuilt Fixtures")

    template = st.selectbox(
        "Choose Template",
        ["W (Wallrack) - Square", "G (Gondola 4 bays + 2 Endcaps)", "T (Table) - Rectangle"],
        key="lb_template",
    )
    count = st.number_input("How many to add?", min_value=1, max_value=50, value=1, step=1, key="lb_add_count")

    st.markdown("---")
    st.subheader("View Controls")
    st.slider("Zoom (fit to view)", 0.4, 1.0, 1.0, 0.05, key="view_zoom")

    u1, u2 = st.columns(2)
    if u1.button("↶ Undo", use_container_width=True):
        if do_undo(store):
            bump_canvas(); st.rerun()
        else:
            st.info("Nothing to undo.")
    if u2.button("↷ Redo", use_container_width=True):
        if do_redo(store):
            bump_canvas(); st.rerun()
        else:
            st.info("Nothing to redo.")

    st.markdown("### Fixture Size")
    st.session_state["fixture_width"]  = st.number_input("Fixture Width",  min_value=20, max_value=500, value=80, step=5)
    st.session_state["fixture_height"] = st.number_input("Fixture Height", min_value=20, max_value=500, value=80, step=5)

    st.markdown("### Layout Settings")
    st.session_state["fixtures_per_row"] = st.number_input("Fixtures Per Row", min_value=1, max_value=20, value=10, step=1)
    st.session_state["fixture_gap"] = st.number_input("Gap Between Fixtures", min_value=0, max_value=100, value=15, step=1)

    st.markdown("### Auto Naming / Numbering")
    if template.startswith("W"):
        w_start_code = st.text_input("Start Fixture Code", value="W1", key="w_start_code")
        w_start_name = st.text_input("Start Display Name", value="WR1", key="w_start_name")
    elif template.startswith("T"):
        t_start_code = st.text_input("Start Fixture Code", value="T1", key="t_start_code")
        t_start_name = st.text_input("Start Display Name", value="T1", key="t_start_name")  # ✅ T1
    else:
        g_start_num = st.number_input("Start G number", min_value=1, value=1, step=1, key="g_start_num")
        e_left_start = st.number_input("Start Left Endcap number", min_value=1, value=1, step=1, key="e_left_start")
        e_right_start = st.number_input("Start Right Endcap number", min_value=1, value=6, step=1, key="e_right_start")
        g_rows = st.number_input("G rows (vertical partitions)", min_value=1, max_value=10, value=2, step=1, key="g_rows")
        g_cols = st.number_input("G columns (usually 2)", min_value=1, max_value=6, value=2, step=1, key="g_cols")
        g_bay_w = st.number_input("G bay width", min_value=20, max_value=400, value=90, step=5, key="g_bay_w")
        g_bay_h = st.number_input("G bay height", min_value=20, max_value=400, value=60, step=5, key="g_bay_h")
        endcap_size = st.number_input("Endcap size", min_value=20, max_value=300, value=70, step=5, key="endcap_size")
        g_gap = st.number_input("Gap inside gondola", min_value=0, max_value=50, value=8, step=1, key="g_gap")
        set_gap = st.number_input("Gap between gondola sets", min_value=0, max_value=200, value=60, step=5, key="g_set_gap")

    if st.button("➕ Add to Canvas", use_container_width=True):
        push_undo(store)
        if template.startswith("W"):
            add_template_W(store, int(count), (w_start_code or "W1").strip(), (w_start_name or "WR1").strip(), layout_id)
        elif template.startswith("T"):
            add_template_T(store, int(count), (t_start_code or "T1").strip(), (t_start_name or "T1").strip(), layout_id)
        else:
            add_template_G_with_endcaps(
                store=store,
                sets_count=int(count),
                g_start_num=int(st.session_state["g_start_num"]),
                e_left_start_num=int(st.session_state["e_left_start"]),
                e_right_start_num=int(st.session_state["e_right_start"]),
                layout_id=layout_id,
                rows=int(st.session_state["g_rows"]),
                cols=int(st.session_state["g_cols"]),
                G_w=float(st.session_state["g_bay_w"]),
                G_h=float(st.session_state["g_bay_h"]),
                E=float(st.session_state["endcap_size"]),
                gap=float(st.session_state["g_gap"]),
                set_gap=float(st.session_state["g_set_gap"]),
            )
        bump_canvas()
        st.rerun()

    st.markdown("---")
    st.subheader("Delete Fixtures")
    objs = get_objects(store)
    codes = sorted(list({(o.get("fixture_code") or "").strip()
                         for o in objs
                         if (o.get("fixture_code") or "").strip()}))
    del_codes = st.multiselect("Select fixture(s) to delete", options=codes, key="del_codes")
    if st.button("❌ Delete Selected", use_container_width=True):
        if del_codes:
            push_undo(store)
            set_objects(store, delete_by_codes(get_objects(store), set(del_codes)))
            bump_canvas()
            st.rerun()
        else:
            st.info("Select fixture(s) first.")

    st.markdown("---")
    st.subheader("Rotate Fixtures (G / E)")
    all_codes = codes
    rot_codes = st.multiselect("Select fixture(s) to rotate", options=all_codes, key="rot_codes")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("0°", use_container_width=True) and rot_codes:
        push_undo(store); rotate_fixtures_by_code(store, set(rot_codes), 0); bump_canvas(); st.rerun()
    if c2.button("90°", use_container_width=True) and rot_codes:
        push_undo(store); rotate_fixtures_by_code(store, set(rot_codes), 90); bump_canvas(); st.rerun()
    if c3.button("180°", use_container_width=True) and rot_codes:
        push_undo(store); rotate_fixtures_by_code(store, set(rot_codes), 180); bump_canvas(); st.rerun()
    if c4.button("270°", use_container_width=True) and rot_codes:
        push_undo(store); rotate_fixtures_by_code(store, set(rot_codes), 270); bump_canvas(); st.rerun()

    st.markdown("---")
    st.subheader("Reset / Save")

    colA, colB, colC = st.columns(3)
    reset_to_saved = colA.button("↩️ Reset DB", use_container_width=True)
    clear_canvas = colB.button("🗑️ New Layout", use_container_width=True)
    save_now = colC.button("💾 Save DB", use_container_width=True)

    if reset_to_saved:
        push_undo(store)
        df_db = load_fixtures(layout_id)
        set_objects(store, fixtures_to_shapes(df_db) if not df_db.empty else [])
        set_last_canvas_json(store, None)
        bump_canvas()
        st.rerun()

    if clear_canvas:
        # ✅ clears DB + canvas + history => numbering restarts from 1
        clear_layout_in_db(layout_id)
        set_objects(store, [])
        set_last_canvas_json(store, None)
        st.session_state[undo_key(store)] = []
        st.session_state[redo_key(store)] = []
        bump_canvas()
        st.success("Cleared ✅ Now add fixtures again and Save DB.")
        st.rerun()

# ---- SAVE LOGIC ----
if save_now:
    view_zoom = float(st.session_state.get("view_zoom", 1.0))
    latest = canvas.json_data or get_last_canvas_json(store)
    latest = unscale_canvas_json(latest, view_zoom)
    sync_objects_from_canvas(store, latest)

    objs = get_objects(store)
    enrich_rect_meta_from_text(objs)
    rects = [o for o in objs if o.get("type") == "rect"]

    if not rects:
        st.warning("Nothing to save.")
    else:
        rows = []
        used_codes = set()

        for i, obj in enumerate(rects, start=1):
            code = (obj.get("fixture_code") or f"TMP{i}").strip()
            name = (obj.get("display_name") or code).strip()
            ftype = (obj.get("fixture_type") or "CUSTOM").strip()

            if code in used_codes:
                code = next_available_code(code, used_codes)
            used_codes.add(code)

            x = float(obj.get("left", 0))
            y = float(obj.get("top", 0))
            w = float(obj.get("width", 50)) * float(obj.get("scaleX", 1))
            h = float(obj.get("height", 50)) * float(obj.get("scaleY", 1))

            rows.append({
                "fixture_code": code,
                "fixture_type": ftype,
                "display_name": name,
                "x": x, "y": y, "w": w, "h": h,
                "shape": obj,
            })

        save_fixtures(layout_id, rows)

        df_db = load_fixtures(layout_id)
        set_objects(store, fixtures_to_shapes(df_db) if not df_db.empty else [])
        set_last_canvas_json(store, None)

        st.success("Saved ✅ Layout persisted to DB.")
        bump_canvas()
        st.rerun()

st.markdown("---")
st.subheader("Current Fixtures (Saved in DB)")
df_show = load_fixtures(layout_id)
if df_show.empty:
    st.info("No fixtures saved yet.")
else:
    st.dataframe(df_show[["fixture_code", "fixture_type", "display_name", "x", "y", "w", "h"]], use_container_width=True)