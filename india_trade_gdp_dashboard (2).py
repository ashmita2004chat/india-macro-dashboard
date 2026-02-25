from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Page config + theme
# ============================================================
st.set_page_config(page_title="India Trade & GDP Analytics", layout="wide", page_icon="📊")

# Plotly defaults
pio.templates.default = "plotly_white"

# --- App-wide styling (soft, clean) ---
st.markdown(
    """
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(255,222,209,0.35), transparent 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(255,210,245,0.25), transparent 55%),
              linear-gradient(180deg, #fffaf6 0%, #ffffff 40%, #ffffff 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.98) 100%);
  border-right: 1px solid rgba(0,0,0,0.06);
}

/* Hero */
.hero {
  padding: 18px 22px;
  border-radius: 18px;
  background: linear-gradient(90deg, rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.95) 60%, rgba(255,240,232,0.75) 100%);
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 12px 28px rgba(0,0,0,0.06);
  margin-bottom: 14px;
}
.hero h1{
  margin: 0;
  font-size: 40px;
  letter-spacing: -0.6px;
}
.hero p{
  margin: 6px 0 0 0;
  color: rgba(0,0,0,0.62);
  font-size: 15px;
}

/* KPI cards */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(180px, 1fr));
  gap: 12px;
  margin: 8px 0 6px 0;
}
.kpi-card {
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 10px 22px rgba(0,0,0,0.05);
}
.kpi-label {
  font-size: 13px;
  color: rgba(0,0,0,0.60);
  margin-bottom: 4px;
}
.kpi-value {
  font-weight: 800;
  letter-spacing: -0.3px;
  line-height: 1.05;
}
.kpi-sub {
  margin-top: 6px;
  font-size: 12px;
  color: rgba(0,0,0,0.55);
}

/* Tables */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.06);
}

/* Tabs pills */
button[data-baseweb="tab"] {
  border-radius: 999px !important;
  padding: 8px 14px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>📊 India Trade &amp; GDP Analytics</h1>
  <p>HS71 trade panel + GDP (WDI format). Interactive trends, ratios, balance, shares &amp; composition.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Utilities
# ============================================================

INDIA_NAMES = {"india", "republic of india"}

def _clean_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c)).strip()

def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _find_local_file(patterns: List[str]) -> Optional[str]:
    """
    Search for files in current working dir (recursively) and /mnt/data (for Colab/sandbox).
    Returns first match (most recent modified if multiple).
    """
    candidates: List[str] = []
    roots = [os.getcwd(), "/mnt/data"]
    for root in roots:
        for pat in patterns:
            candidates += glob.glob(os.path.join(root, "**", pat), recursive=True)
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def _abbr_number(x: float, unit: str = "mn") -> str:
    """
    x is assumed in USD million by default.
    unit: "mn" (million), "bn" (billion), "tn" (trillion)
    """
    if x is None or pd.isna(x):
        return "—"
    if unit == "mn":
        # Always show in millions (M) to match the selected unit.
        # (Do not auto-convert to B, otherwise the label won't sync with 'USD Million'.)
        v = x
        if abs(v) >= 1000:
            return f"{v:,.0f} M"
        return f"{v:,.2f} M"
    if unit == "bn":
        return f"{x/1000:.2f} B"
    if unit == "tn":
        return f"{x/1_000_000:.2f} T"
    return f"{x:.2f}"

def _format_pct(x: float, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{digits}f}%"

def _to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _shorten(text: str, max_len: int = 70) -> str:
    t = str(text).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"

def render_kpis(cards: List[Tuple[str, str, str]]):
    """Render KPI cards (HTML) reliably without markdown turning parts into code blocks."""

    def value_font_px(v: str) -> int:
        l = len(v)
        if l <= 6:
            return 46
        if l <= 8:
            return 40
        if l <= 10:
            return 34
        return 28

    card_bits: List[str] = []
    for title, value, subtitle in cards:
        v = str(value)
        fs = value_font_px(v)
        card_bits.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-title">{title}</div>'
            f'<div class="kpi-value" style="font-size:{fs}px;">{v}</div>'
            f'<div class="kpi-sub">{subtitle}</div>'
            f"</div>"
        )

    cards_html = '<div class="kpi-grid">' + "".join(card_bits) + "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

def apply_labels_line(fig, show_labels: bool, yfmt: str = ".2f", suffix: str = ""):
    """
    For line charts: when show_labels is True, show labels for *all points*.
    """
    if not show_labels:
        fig.update_traces(mode="lines+markers", text=None)
        return fig

    def fmt(v):
        if v is None or pd.isna(v):
            return ""
        try:
            return f"{float(v):{yfmt}}{suffix}"
        except Exception:
            return str(v)

    for tr in fig.data:
        # y values can be list/tuple/np array
        ys = getattr(tr, "y", None)
        if ys is None:
            continue
        tr.text = [fmt(v) for v in ys]
        tr.mode = "lines+markers+text"
        tr.textposition = "top center"
        tr.textfont = dict(size=11)
    return fig

def apply_labels_bar(fig, show_labels: bool, yfmt: str = ".2f", suffix: str = ""):
    if not show_labels:
        fig.update_traces(text=None)
        return fig

    def fmt(v):
        if v is None or pd.isna(v):
            return ""
        try:
            return f"{float(v):{yfmt}}{suffix}"
        except Exception:
            return str(v)

    for tr in fig.data:
        ys = getattr(tr, "y", None)
        if ys is None:
            continue
        tr.text = [fmt(v) for v in ys]
        tr.textposition = "outside"
        tr.textfont = dict(size=11)
    return fig

# ============================================================
# Loaders / parsers
# ============================================================

@st.cache_data(show_spinner=False)
def load_trade_panel(path: str) -> pd.DataFrame:
    """
    Expected (flexible) columns in resulting df:
    - country (str)
    - iso3 (str) optional
    - year (int)
    - exports_hs71_usd_mn (float)
    - imports_hs71_usd_mn (float)
    Optional:
    - gdp_usd_mn
    - total_exports_usd_mn
    - total_imports_usd_mn
    """
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [_clean_col(c) for c in df.columns]
    # Normalize column names heuristically
    colmap: Dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()

        if cl in {"year", "yr"} or "year" in cl:
            colmap[c] = "year"
        elif cl in {"country", "country name"} or "country" in cl:
            colmap[c] = "country"
        elif cl in {"iso3", "iso", "country code", "code"} or ("iso" in cl and "code" in cl) or ("country code" in cl):
            colmap[c] = "iso3"
        elif ("exports" in cl or "export" in cl) and ("hs71" in cl or "g&j" in cl or "gem" in cl) and ("usd" in cl or "value" in cl):
            colmap[c] = "exports_hs71_usd_mn"
        elif ("imports" in cl or "import" in cl) and ("hs71" in cl or "g&j" in cl or "gem" in cl) and ("usd" in cl or "value" in cl):
            colmap[c] = "imports_hs71_usd_mn"
        elif ("total" in cl and "exports" in cl) and ("usd" in cl or "value" in cl):
            colmap[c] = "total_exports_usd_mn"
        elif ("total" in cl and "imports" in cl) and ("usd" in cl or "value" in cl):
            colmap[c] = "total_imports_usd_mn"
        elif ("gdp" in cl) and ("usd" in cl or "value" in cl):
            colmap[c] = "gdp_usd_mn"

    df = df.rename(columns=colmap)

    # Ensure required columns exist
    if "year" not in df.columns:
        raise ValueError("Trade panel: could not find a 'year' column.")
    if "country" not in df.columns:
        # fallback: look for first object column
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if obj_cols:
            df = df.rename(columns={obj_cols[0]: "country"})
        else:
            raise ValueError("Trade panel: could not find a 'country' column.")
    if "exports_hs71_usd_mn" not in df.columns or "imports_hs71_usd_mn" not in df.columns:
        # fallback: try columns containing 'exports'/'imports' without hs71
        for c in df.columns:
            cl = c.lower()
            if "exports_hs71_usd_mn" not in df.columns and ("export" in cl and ("usd" in cl or "value" in cl)):
                df = df.rename(columns={c: "exports_hs71_usd_mn"})
            if "imports_hs71_usd_mn" not in df.columns and ("import" in cl and ("usd" in cl or "value" in cl)):
                df = df.rename(columns={c: "imports_hs71_usd_mn"})
    # types
    df["country"] = df["country"].astype(str).str.strip()
    if "iso3" in df.columns:
        df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    for c in ["exports_hs71_usd_mn", "imports_hs71_usd_mn", "total_exports_usd_mn", "total_imports_usd_mn", "gdp_usd_mn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

@st.cache_data(show_spinner=False)

def load_gdp_wdi(path: str) -> pd.DataFrame:
    """
    Load a WDI-style GDP Excel where years are columns (e.g., 2005 or "2005 [YR2005]"),
    or a long format with a 'year' column.

    Returns a long dataframe with columns:
      - iso3 (may be NaN if not available in the GDP file)
      - country
      - year (int)
      - gdp_usd (float)
    """
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return pd.DataFrame()

    # --- pick the sheet that looks most like WDI (most year-like columns) ---
    best_sheet = None
    best_year_count = -1
    for s in xls.sheet_names:
        try:
            head = pd.read_excel(xls, sheet_name=s, nrows=5)
        except Exception:
            continue
        cols = [str(c).strip() for c in head.columns]
        year_like = 0
        for c in cols:
            m = re.search(r"(19|20)\d{2}", str(c))
            if m:
                y = int(m.group())
                if 1900 <= y <= 2100:
                    year_like += 1
        if year_like > best_year_count:
            best_year_count = year_like
            best_sheet = s

    if best_sheet is None:
        best_sheet = xls.sheet_names[0]

    raw = pd.read_excel(xls, sheet_name=best_sheet)
    raw.columns = [_clean_col(c) for c in raw.columns]

    def _pick_col(needles: list[str]) -> str | None:
        for c in raw.columns:
            cl = str(c).lower()
            if any(k in cl for k in needles):
                return c
        return None

    country_col = _pick_col(["country name", "country", "economy", "name"])
    if country_col is None:
        country_col = raw.columns[0]

    # WDI commonly uses "Country Code" (3-letter) for ISO3
    iso_col = _pick_col(["country code", "iso3", "iso", "code"])

    year_col_existing = _pick_col(["year"])

    # ----------------------------
    # Case A: already long format
    # ----------------------------
    if year_col_existing is not None:
        val_col = None
        for c in raw.columns:
            if c in {country_col, iso_col, year_col_existing}:
                continue
            tmp = pd.to_numeric(raw[c], errors="coerce")
            if tmp.notna().sum() > 0:
                val_col = c
                break
        if val_col is None:
            return pd.DataFrame()

        keep = [country_col, year_col_existing, val_col] + ([iso_col] if iso_col else [])
        df = raw[keep].copy()
        df.rename(
            columns={
                country_col: "country",
                year_col_existing: "year",
                val_col: "gdp_usd",
            },
            inplace=True,
        )
        if iso_col:
            df.rename(columns={iso_col: "iso3"}, inplace=True)
        else:
            df["iso3"] = np.nan

    # ----------------------------
    # Case B: wide format (years as columns)
    # ----------------------------
    else:
        year_map: dict[str, int] = {}
        for c in raw.columns:
            m = re.search(r"(19|20)\d{2}", str(c))
            if m:
                y = int(m.group())
                if 1900 <= y <= 2100:
                    year_map[c] = y

        year_cols = list(year_map.keys())
        if not year_cols:
            return pd.DataFrame()

        id_vars = [country_col] + ([iso_col] if iso_col else [])
        df = raw.melt(
            id_vars=id_vars,
            value_vars=year_cols,
            var_name="_year_col",
            value_name="gdp_usd",
        )
        df["year"] = df["_year_col"].map(year_map)
        df.drop(columns=["_year_col"], inplace=True)
        df.rename(columns={country_col: "country"}, inplace=True)
        if iso_col:
            df.rename(columns={iso_col: "iso3"}, inplace=True)
        else:
            df["iso3"] = np.nan

    # ----------------------------
    # Clean
    # ----------------------------
    df["country"] = df["country"].astype(str).str.strip()
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df.loc[df["iso3"].isin(["NAN", "NONE", "<NA>", ""]), "iso3"] = np.nan

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gdp_usd"] = pd.to_numeric(df["gdp_usd"], errors="coerce")

    df = df.dropna(subset=["year", "gdp_usd"])
    df["year"] = df["year"].astype(int)

    # Normalize GDP to USD million for consistency with trade (trade values are in USD million).
    # If the source looks like raw USD (very large magnitudes), divide by 1e6.
    if "gdp_usd_mn" not in df.columns:
        df["gdp_usd_mn"] = df["gdp_usd"]
    s = pd.to_numeric(df["gdp_usd_mn"], errors="coerce")
    if not s.dropna().empty:
        mx = float(s.max())
        med = float(s.median())
        if mx > 1e9 or med > 1e8:
            df["gdp_usd_mn"] = s / 1e6
        else:
            df["gdp_usd_mn"] = s

    return df

def load_productwise_trade(path: str, kind: str = "exports") -> Tuple[pd.DataFrame, List[int]]:
    """Load product-wise HS71 Excel (wide or long) into a standard long format.

    Expected output columns:
      - product_code (str)
      - product (str)
      - year (int)
      - value (float)      # as in file (typically USD thousand)
      - flow (str)         # 'exports' or 'imports'
    """

    def _detect_header_row(xlsx_path: str, sheet_name: str) -> int:
        # Scan first few rows to find the row that looks like a header.
        raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=8)
        for r in range(min(8, raw.shape[0])):
            row = raw.iloc[r].astype(str).str.strip().str.lower().tolist()
            has_code = any(t in ("code", "hs code", "hs_code", "hs") for t in row)
            has_product = any(("product" in t) or ("label" in t) or ("description" in t) for t in row)
            year_hits = sum(bool(re.search(r"(19|20)\d{2}", t)) for t in row)
            if (has_code and has_product) or year_hits >= 3:
                return r
        return 0

    def _pick_col(cols: List[str], keywords: List[str]) -> Optional[str]:
        for c in cols:
            cl = str(c).strip().lower()
            if any(k in cl for k in keywords):
                return c
        return None

    kind = (kind or "exports").strip().lower()
    flow = "imports" if "imp" in kind else "exports"

    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]

    header_row = _detect_header_row(path, sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop completely empty columns
    df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]

    # Detect columns
    code_col = _pick_col(df.columns.tolist(), ["code", "hs code", "hs_code", "hs"])
    product_col = _pick_col(df.columns.tolist(), ["product", "label", "description", "commodity", "item"])

    # Fallbacks
    if code_col is None:
        code_col = df.columns[0]
    if product_col is None:
        product_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # If it's already long format
    year_col = _pick_col(df.columns.tolist(), ["year", "yr"])
    if year_col is not None and year_col != code_col and year_col != product_col:
        # value column: prefer flow-specific columns first, else most numeric column
        value_candidates = [c for c in df.columns if c not in (code_col, product_col, year_col)]
        if flow == "exports":
            preferred = [c for c in value_candidates if "export" in str(c).lower()]
        else:
            preferred = [c for c in value_candidates if "import" in str(c).lower()]
        cand = preferred + [c for c in value_candidates if c not in preferred]

        value_col = None
        best_n = -1
        for c in cand:
            tmp = pd.to_numeric(df[c], errors="coerce")
            n = int(tmp.notna().sum())
            if n > best_n:
                best_n = n
                value_col = c

        long_df = df[[code_col, product_col, year_col, value_col]].copy()
        long_df.columns = ["product_code", "product", "year", "value"]
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df.dropna(subset=["year", "value"])
        long_df["year"] = long_df["year"].astype(int)
        long_df["product_code"] = long_df["product_code"].astype(str).str.strip().str.lstrip("'")
        long_df["product"] = long_df["product"].astype(str).str.strip()
        long_df["flow"] = flow
        years = sorted(long_df["year"].unique().tolist())
        return long_df, years

    # Otherwise: wide format with years as columns
    year_map: Dict[int, str] = {}
    for c in df.columns:
        if c in (code_col, product_col):
            continue
        cl = str(c).strip().lower()
        # if column name suggests the other flow, deprioritize but still allow if no better match
        m = re.search(r"(19|20)\d{2}", str(c))
        if not m:
            continue
        y = int(m.group(0))
        year_map[y] = c

    if not year_map:
        raise ValueError(f"No year columns found in: {os.path.basename(path)}. Please check the sheet structure.")

    years = sorted(year_map.keys())

    keep_cols = [code_col, product_col] + [year_map[y] for y in years]
    use = df[keep_cols].copy()
    use.columns = ["product_code", "product"] + [str(y) for y in years]

    # Melt
    long_df = use.melt(id_vars=["product_code", "product"], var_name="year", value_name="value")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"])
    long_df["year"] = long_df["year"].astype(int)
    long_df["product_code"] = long_df["product_code"].astype(str).str.strip().str.lstrip("'")
    long_df["product"] = long_df["product"].astype(str).str.strip()
    long_df["flow"] = flow

    return long_df, years
def make_composition_table(long_df: pd.DataFrame, year: int, top_rule: float = 0.75) -> Tuple[pd.DataFrame, int]:
    """
    long_df columns: product_code, product, year, value
    Returns composition table with top N + others; also returns chosen N.
    """
    d = long_df.loc[long_df["year"] == year].copy()
    d = d.dropna(subset=["value"])
    d = d.groupby(["product_code", "product"], as_index=False)["value"].sum()
    d = d.sort_values("value", ascending=False)

    total = d["value"].sum()
    top5_share = d.head(5)["value"].sum() / total if total else 0.0
    top_n = 6 if top5_share < top_rule else 5

    top = d.head(top_n).copy()
    others_val = total - top["value"].sum()

    comp = top.copy()
    if others_val > 0:
        comp = pd.concat(
            [comp, pd.DataFrame([{"product_code": "", "product": "Others", "value": others_val}])],
            ignore_index=True,
        )
    comp["share_pct"] = np.where(total > 0, comp["value"] / total * 100, np.nan)
    return comp, top_n

def pie_figure(comp: pd.DataFrame, title: str, show_labels: bool) -> go.Figure:
    label_col = "product_short"
    comp = comp.copy()
    comp[label_col] = comp["product"].map(lambda x: _shorten(x, 70))
    fig = px.pie(
        comp,
        names=label_col,
        values="value",
        hole=0.55,
        title=title,
    )
    # text labels toggle
    if show_labels:
        fig.update_traces(textinfo="percent", textposition="inside")
    else:
        fig.update_traces(textinfo="none")
    fig.update_layout(
        title_font_size=18,
        legend_title_text="Products",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

# ============================================================
# Sidebar controls
# ============================================================

with st.sidebar:
    st.markdown("## Data")
    show_labels = st.toggle("Show data labels", value=False, help="Turn on to display data labels on charts.")
    st.caption("Auto-detected from the current working directory.")

# ============================================================
# Auto-detect files
# ============================================================

trade_panel_file = _find_local_file([
    "*HS71*WDI*Panel*.xlsx",
    "*HS71*Trade*GDP*Panel*.xlsx",
    "*HS71_plus*Panel*.xlsx",
    "*HS71_plus_TotalTrade_WDI_Panel*.xlsx",
])

gdp_file = _find_local_file([
    "*GDP*Constant*Prices*.xlsx",
    "*GDP*WDI*.xlsx",
    "*GDP*.xlsx",
])

exports_product_file = _find_local_file([
    "Exports 2005-2024 India.xlsx",
    "*Exports*India*.xlsx",
])

imports_product_file = _find_local_file([
    "Imports 2005-2024 India.xlsx",
    "*Imports*India*.xlsx",
])

# ============================================================
# Main tabs
# ============================================================

tabs = st.tabs(["💎🪙 HS71 Trade Panel", "🌍 GDP (WDI format)"])

# ------------------------------------------------------------
# TAB 1: HS71 Trade Panel
# ------------------------------------------------------------
with tabs[0]:
    st.markdown("### HS71 Trade Panel (Exports / Imports / Shares / Trade Balance)")

    if trade_panel_file is None:
        st.error(
            "Trade panel Excel not found. Put the HS71 trade panel Excel in the same folder as this app.\n\n"
            "Expected filename patterns include: `HS71_plus_TotalTrade_WDI_Panel_2005_2024.xlsx`."
        )
        st.stop()

    try:
        trade = load_trade_panel(trade_panel_file)
    except Exception as e:
        st.error(f"Could not load trade panel: {e}")
        st.stop()

    # Merge GDP (optional, used for % of GDP)
    gdp_long = None
    if gdp_file is not None:
        try:
            gdp_long = load_gdp_wdi(gdp_file)
        except Exception:
            gdp_long = None

    if gdp_long is not None and "gdp_usd_mn" not in trade.columns:
        # Merge on iso3+year if possible, else on country+year
        trade_m = trade.copy()
        if "iso3" in trade_m.columns and (trade_m["iso3"].astype(str).str.len() >= 3).any():
            trade_m = trade_m.merge(
                gdp_long[["iso3", "year", "gdp_usd_mn"]],
                on=["iso3", "year"],
                how="left",
            )
        else:
            trade_m = trade_m.merge(
                gdp_long[["country", "year", "gdp_usd_mn"]],
                on=["country", "year"],
                how="left",
            )
        trade = trade_m

    # --- Controls ---
    years_all = sorted(trade["year"].unique().tolist())
    y_min, y_max = min(years_all), max(years_all)

    colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 1.0])
    with colA:
        mode = st.radio("Country selection", ["Auto (India + Top N exporters)", "Manual select"], index=0, key="trade_country_mode")
    with colB:
        yr_range = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max), step=1, key="trade_year_range")
    with colC:
        unit_choice = st.selectbox("Unit for exports/imports", ["USD Million", "USD Billion"], index=1, key="trade_unit")
    with colD:
        top_n = st.number_input("Top N competitors", min_value=1, max_value=20, value=5, step=1, key="trade_top_n")

    yr0, yr1 = int(yr_range[0]), int(yr_range[1])

    # Filter to selected year range first (this fixes disappearing countries when range changes)
    tr = trade.loc[(trade["year"] >= yr0) & (trade["year"] <= yr1)].copy()

    # Determine India rows robustly
    def is_india_row(df: pd.DataFrame) -> pd.Series:
        s1 = df["country"].astype(str).str.strip().str.lower()
        cond = s1.isin(INDIA_NAMES) | (s1 == "india")
        if "iso3" in df.columns:
            cond = cond | (df["iso3"].astype(str).str.upper() == "IND")
        return cond

    # Select countries
    if mode.startswith("Auto"):
        # pick top exporters over the selected range (excluding India)
        tmp = tr.copy()
        tmp["is_india"] = is_india_row(tmp)
        agg = (
            tmp.loc[~tmp["is_india"]]
            .groupby("country", as_index=False)["exports_hs71_usd_mn"]
            .sum()
            .sort_values("exports_hs71_usd_mn", ascending=False)
        )
        pick = agg.head(int(top_n))["country"].tolist()
        selected_countries = ["India"] + pick
    else:
        all_countries = sorted(tr["country"].unique().tolist())
        default = ["India"] if "India" in all_countries else all_countries[:1]
        selected_countries = st.multiselect("Select countries", all_countries, default=default, key="trade_select_countries")

    tr_sel = tr.loc[tr["country"].isin(selected_countries)].copy()

    # Scale for display
    if unit_choice == "USD Billion":
        scale = 1000.0
        unit_suffix = "USD Billion"
        yfmt_num = ".2f"
    else:
        scale = 1.0
        unit_suffix = "USD Million"
        yfmt_num = ".2f"
    kpi_unit = "bn" if unit_choice == "USD Billion" else "mn"


    tr_sel["exports_scaled"] = tr_sel["exports_hs71_usd_mn"] / scale
    tr_sel["imports_scaled"] = tr_sel["imports_hs71_usd_mn"] / scale
    tr_sel["trade_balance_scaled"] = (tr_sel["exports_hs71_usd_mn"] - tr_sel["imports_hs71_usd_mn"]) / scale

    # Ratios
    if "gdp_usd_mn" in tr_sel.columns:
        tr_sel["exports_pct_gdp"] = tr_sel["exports_hs71_usd_mn"] / tr_sel["gdp_usd_mn"] * 100
        tr_sel["imports_pct_gdp"] = tr_sel["imports_hs71_usd_mn"] / tr_sel["gdp_usd_mn"] * 100
    if "total_exports_usd_mn" in tr_sel.columns:
        tr_sel["hs71_share_total_exports_pct"] = tr_sel["exports_hs71_usd_mn"] / tr_sel["total_exports_usd_mn"] * 100
    if "total_imports_usd_mn" in tr_sel.columns:
        tr_sel["hs71_share_total_imports_pct"] = tr_sel["imports_hs71_usd_mn"] / tr_sel["total_imports_usd_mn"] * 100

    # KPI row for India at latest available year within selected range
    india_rows = tr_sel.loc[is_india_row(tr_sel)].copy()
    india_year = None
    if not india_rows.empty:
        india_year = int(india_rows["year"].max())
        india_k = india_rows.loc[india_rows["year"] == india_year].head(1)
        exp_v = float(india_k["exports_hs71_usd_mn"].iloc[0]) if not india_k.empty else np.nan
        imp_v = float(india_k["imports_hs71_usd_mn"].iloc[0]) if not india_k.empty else np.nan
        bal_v = exp_v - imp_v if pd.notna(exp_v) and pd.notna(imp_v) else np.nan
        exp_pct_gdp = float(india_k["exports_pct_gdp"].iloc[0]) if "exports_pct_gdp" in india_k.columns and not india_k.empty else np.nan
        imp_pct_gdp = float(india_k["imports_pct_gdp"].iloc[0]) if "imports_pct_gdp" in india_k.columns and not india_k.empty else np.nan
        hs71_exp_share_total = float(india_k["hs71_share_total_exports_pct"].iloc[0]) if "hs71_share_total_exports_pct" in india_k.columns and not india_k.empty else np.nan
        hs71_imp_share_total = float(india_k["hs71_share_total_imports_pct"].iloc[0]) if "hs71_share_total_imports_pct" in india_k.columns and not india_k.empty else np.nan

        cards = [
            (f"India Exports ({india_year})", _abbr_number(exp_v, kpi_unit), "HS71 exports"),
            (f"India Imports ({india_year})", _abbr_number(imp_v, kpi_unit), "HS71 imports"),
            (f"India Trade Balance ({india_year})", _abbr_number(bal_v, kpi_unit), "Exports − Imports"),
            (f"HS71 Exports as % of GDP ({india_year})", _format_pct(exp_pct_gdp, 3), "HS71 share of GDP"),
            (f"HS71 Imports as % of GDP ({india_year})", _format_pct(imp_pct_gdp, 3), "HS71 share of GDP"),
            (f"HS71 Exports as % of Total Exports ({india_year})", _format_pct(hs71_exp_share_total, 3), "Share in total exports"),
            (f"HS71 Imports as % of Total Imports ({india_year})", _format_pct(hs71_imp_share_total, 3), "Share in total imports"),
        ]
        render_kpis(cards)
    else:
        st.info("India is not present under the current filters. KPIs will appear when India data is available.")

    # Sub-tabs
    t1, t2, t3, t4 = st.tabs(["📉 Exports & Imports", "📈 % of GDP", "⚖️ Balance & Share", "🥧 Composition (India)"])

    # -------------------------
    # Sub-tab 1: Exports & Imports
    # -------------------------
    with t1:
        st.markdown(f"#### HS71 Exports Trend ({unit_suffix})")
        fig_exp = px.line(
            tr_sel.sort_values("year"),
            x="year",
            y="exports_scaled",
            color="country",
            markers=True,
        )
        fig_exp.update_layout(xaxis_title="Year", yaxis_title=f"Exports ({unit_suffix})", legend_title_text="")
        fig_exp.update_xaxes(dtick=1)
        fig_exp = apply_labels_line(fig_exp, show_labels, yfmt=yfmt_num)
        st.plotly_chart(fig_exp, use_container_width=True)

        snap_exp = (
            tr_sel.pivot_table(index="year", columns="country", values="exports_scaled", aggfunc="sum")
            .reset_index()
            .sort_values("year")
        )
        st.dataframe(snap_exp, use_container_width=True, height=320)
        st.download_button("Download snapshot (Exports CSV)", data=_to_csv_download(snap_exp), file_name="hs71_exports_snapshot.csv")

        st.markdown(f"#### HS71 Imports Trend ({unit_suffix})")
        fig_imp = px.line(
            tr_sel.sort_values("year"),
            x="year",
            y="imports_scaled",
            color="country",
            markers=True,
        )
        fig_imp.update_layout(xaxis_title="Year", yaxis_title=f"Imports ({unit_suffix})", legend_title_text="")
        fig_imp.update_xaxes(dtick=1)
        fig_imp = apply_labels_line(fig_imp, show_labels, yfmt=yfmt_num)
        st.plotly_chart(fig_imp, use_container_width=True)

        snap_imp = (
            tr_sel.pivot_table(index="year", columns="country", values="imports_scaled", aggfunc="sum")
            .reset_index()
            .sort_values("year")
        )
        st.dataframe(snap_imp, use_container_width=True, height=320)
        st.download_button("Download snapshot (Imports CSV)", data=_to_csv_download(snap_imp), file_name="hs71_imports_snapshot.csv")

    # -------------------------
    # Sub-tab 2: % of GDP
    # -------------------------
    with t2:
        if "exports_pct_gdp" not in tr_sel.columns or tr_sel["exports_pct_gdp"].notna().sum() == 0:
            st.info("To plot HS71 share of GDP, a GDP series must be available (merged from WDI GDP file).")
        else:
            st.markdown("#### HS71 Exports as % of GDP (HS71 share of GDP)")
            fig = px.line(
                tr_sel.sort_values("year"),
                x="year",
                y="exports_pct_gdp",
                color="country",
                markers=True,
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="HS71 share of GDP (%)", legend_title_text="")
            fig.update_xaxes(dtick=1)
            fig = apply_labels_line(fig, show_labels, yfmt=".2f", suffix="%")
            st.plotly_chart(fig, use_container_width=True)

            snap = (
                tr_sel.pivot_table(index="year", columns="country", values="exports_pct_gdp", aggfunc="mean")
                .reset_index()
                .sort_values("year")
            )
            st.dataframe(snap, use_container_width=True, height=320)
            st.download_button("Download snapshot (Exports % GDP CSV)", data=_to_csv_download(snap), file_name="hs71_exports_pct_gdp_snapshot.csv")

            st.markdown("#### HS71 Imports as % of GDP (HS71 share of GDP)")
            fig = px.line(
                tr_sel.sort_values("year"),
                x="year",
                y="imports_pct_gdp",
                color="country",
                markers=True,
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="HS71 share of GDP (%)", legend_title_text="")
            fig.update_xaxes(dtick=1)
            fig = apply_labels_line(fig, show_labels, yfmt=".2f", suffix="%")
            st.plotly_chart(fig, use_container_width=True)

            snap = (
                tr_sel.pivot_table(index="year", columns="country", values="imports_pct_gdp", aggfunc="mean")
                .reset_index()
                .sort_values("year")
            )
            st.dataframe(snap, use_container_width=True, height=320)
            st.download_button("Download snapshot (Imports % GDP CSV)", data=_to_csv_download(snap), file_name="hs71_imports_pct_gdp_snapshot.csv")

    # -------------------------
    # Sub-tab 3: Balance & Share
    # -------------------------
    with t3:
        st.markdown(f"#### HS71 Trade Balance (Exports − Imports) ({unit_suffix})")
        fig = px.line(
            tr_sel.sort_values("year"),
            x="year",
            y="trade_balance_scaled",
            color="country",
            markers=True,
        )
        fig.update_layout(xaxis_title="Year", yaxis_title=f"Balance ({unit_suffix})", legend_title_text="")
        fig.update_xaxes(dtick=1)
        fig = apply_labels_line(fig, show_labels, yfmt=yfmt_num)
        st.plotly_chart(fig, use_container_width=True)

        snap = (
            tr_sel.pivot_table(index="year", columns="country", values="trade_balance_scaled", aggfunc="sum")
            .reset_index()
            .sort_values("year")
        )
        st.dataframe(snap, use_container_width=True, height=320)
        st.download_button("Download snapshot (Balance CSV)", data=_to_csv_download(snap), file_name="hs71_trade_balance_snapshot.csv")

        # Share of total trade (if totals exist)
        share_cols = []
        if "hs71_share_total_exports_pct" in tr_sel.columns:
            share_cols.append(("HS71 / Total Exports (%)", "hs71_share_total_exports_pct"))
        if "hs71_share_total_imports_pct" in tr_sel.columns:
            share_cols.append(("HS71 / Total Imports (%)", "hs71_share_total_imports_pct"))

        if share_cols:
            st.markdown("#### HS71 Share of Total National Trade")
            plot_long = []
            for label, col in share_cols:
                tmp = tr_sel[["year", "country", col]].rename(columns={col: "value"})
                tmp["series"] = label
                plot_long.append(tmp)
            plot_long = pd.concat(plot_long, ignore_index=True)

            fig = px.line(
                plot_long.sort_values("year"),
                x="year",
                y="value",
                color="country",
                line_dash="series",
                markers=True,
            )
            fig.update_layout(xaxis_title="Year", yaxis_title="Share (%)", legend_title_text="")
            fig.update_xaxes(dtick=1)
            fig = apply_labels_line(fig, show_labels, yfmt=".2f", suffix="%")
            st.plotly_chart(fig, use_container_width=True)

            snap = plot_long.pivot_table(index=["year"], columns=["country", "series"], values="value", aggfunc="mean").reset_index().sort_values("year")
            st.dataframe(snap, use_container_width=True, height=360)
            st.download_button("Download snapshot (Share CSV)", data=_to_csv_download(snap), file_name="hs71_share_total_trade_snapshot.csv")
        else:
            st.info("To plot HS71 share of total trade, include total exports/imports columns in the trade panel (e.g., `total_exports_usd_mn`, `total_imports_usd_mn`).")

    # -------------------------
    # Sub-tab 4: Composition (India)
    # -------------------------

    with t4:
        st.markdown("### HS71 Composition (India) — Product-wise (Top N + Others)")
        st.caption(
            "This section uses the product-wise HS71 export/import Excel files (ITC-style) "
            "to show a donut chart composition for India."
        )

        # exports_product_file / imports_product_file are already resolved local paths
        exp_prod_path = exports_product_file
        imp_prod_path = imports_product_file

        if not exp_prod_path:
            st.warning(f"Could not find exports product-wise file: {exports_product_file}")
        if not imp_prod_path:
            st.warning(f"Could not find imports product-wise file: {imports_product_file}")

        if exp_prod_path and imp_prod_path:
            def _pick_year(target: int, years: List[int]) -> Optional[int]:
                if not years:
                    return None
                ys = sorted(set(int(y) for y in years))
                if target in ys:
                    return target
                older = [y for y in ys if y <= target]
                return max(older) if older else min(ys)

            try:
                exp_long, exp_years = load_productwise_trade(exp_prod_path, kind="exports")
                imp_long, imp_years = load_productwise_trade(imp_prod_path, kind="imports")

                # Align to the selected HS71 range: use India's latest year in range if available, else the range end.
                target_year = int(india_year) if "india_year" in locals() and pd.notna(india_year) else int(yr1)
                y_exp = _pick_year(target_year, exp_years)
                y_imp = _pick_year(target_year, imp_years)

                if y_exp is None or y_imp is None:
                    st.info("Could not detect usable year columns in one of the product-wise files.")
                else:
                    if (y_exp != target_year) or (y_imp != target_year):
                        st.info(f"Using available years — Exports: {y_exp}, Imports: {y_imp}")

                    cexp, cimp = st.tabs(["📤 Exports composition", "📥 Imports composition"])

                    with cexp:
                        comp_exp, top_n_exp = make_composition_table(exp_long, year=y_exp, top_rule=0.75)
                        fig_exp = pie_figure(
                            comp_exp,
                            title=f"HS71 Exports composition ({y_exp}) — Top {top_n_exp} + Others",
                            show_labels=show_labels,
                        )
                        st.plotly_chart(fig_exp, use_container_width=True)

                        # Snapshot table (USD Billion, since file is usually USD thousand)
                        snap = comp_exp.copy()
                        snap["value_usd_bn"] = snap["value"] / 1_000_000
                        snap = snap[["product", "value_usd_bn", "share_pct"]].rename(
                            columns={"product": "Product", "value_usd_bn": "Value (USD Bn)", "share_pct": "Share (%)"}
                        )
                        st.dataframe(snap, use_container_width=True, height=420)
                        st.download_button(
                            "Download snapshot (Exports composition CSV)",
                            data=snap.to_csv(index=False).encode("utf-8"),
                            file_name=f"hs71_exports_composition_{y_exp}.csv",
                            mime="text/csv",
                            key="dl_exports_comp",
                        )

                    with cimp:
                        comp_imp, top_n_imp = make_composition_table(imp_long, year=y_imp, top_rule=0.75)
                        fig_imp = pie_figure(
                            comp_imp,
                            title=f"HS71 Imports composition ({y_imp}) — Top {top_n_imp} + Others",
                            show_labels=show_labels,
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                        snap = comp_imp.copy()
                        snap["value_usd_bn"] = snap["value"] / 1_000_000
                        snap = snap[["product", "value_usd_bn", "share_pct"]].rename(
                            columns={"product": "Product", "value_usd_bn": "Value (USD Bn)", "share_pct": "Share (%)"}
                        )
                        st.dataframe(snap, use_container_width=True, height=420)
                        st.download_button(
                            "Download snapshot (Imports composition CSV)",
                            data=snap.to_csv(index=False).encode("utf-8"),
                            file_name=f"hs71_imports_composition_{y_imp}.csv",
                            mime="text/csv",
                            key="dl_imports_comp",
                        )

            except Exception as e:
                st.error(f"Composition tab error: {e}")

with tabs[1]:
    st.markdown("### GDP Dashboard (WDI-style Excel: Country Name/Code + Year Columns)")

    if gdp_file is None:
        st.error(
            "GDP Excel not found. Put the WDI-style GDP Excel in the same folder as this app.\n\n"
            "Expected filename patterns include: `GDP at Constant Prices (all countries from 2005-2024 from WDI).xlsx`."
        )
        st.stop()

    try:
        gdp = load_gdp_wdi(gdp_file)
    except Exception as e:
        st.error(f"Could not load GDP file: {e}")
        st.stop()

    if gdp is None or gdp.empty or "year" not in gdp.columns:
        st.error(
            "GDP file was found, but it could not be parsed into country-year GDP. "
            "Please ensure the sheet has country + ISO3 (Country Code) and year columns (e.g., 2005 or '2005 [YR2005]')."
        )
        st.stop()

    # Controls
    years = sorted(pd.to_numeric(gdp["year"], errors="coerce").dropna().astype(int).unique().tolist())
    if not years:
        st.error(
            "GDP file parsed, but no year columns were detected. "
            "Ensure the GDP Excel is WDI-style with year columns (e.g., 2005 or '2005 [YR2005]')."
        )
        st.stop()
    y0, y1 = years[0], years[-1]

    # build country label options: prefer "Country Name (ISO3)"
    gdp = gdp.copy()
    gdp["country"] = gdp["country"].astype(str).str.strip()
    gdp["iso3"] = gdp["iso3"].astype(str).str.strip().str.upper()

    # Some WDI sheets include blank iso3; keep but label nicely
    gdp["label"] = np.where(gdp["iso3"].str.len() >= 3, gdp["country"] + " (" + gdp["iso3"] + ")", gdp["country"])

    # default to India + USA if present
    label_opts = sorted(gdp["label"].dropna().unique().tolist())
    defaults = []
    for target in ["India (IND)", "United States (USA)", "United States of America (USA)"]:
        if target in label_opts:
            defaults.append(target)
            break
    if "India (IND)" in label_opts:
        if "India (IND)" not in defaults:
            defaults = ["India (IND)"] + defaults[:1]
    if not defaults and label_opts:
        defaults = label_opts[:2]

    col1, col2, col3 = st.columns([1.6, 1.2, 1.2])
    with col1:
        labels_sel = st.multiselect("Select countries", label_opts, default=defaults, key="gdp_select_countries")
    with col2:
        gdp_unit = st.selectbox("GDP unit", ["USD Trillion", "USD Billion", "USD Million"], index=0)
    with col3:
        yr_gdp = st.slider("Year range", min_value=y0, max_value=y1, value=(y0, y1), step=1, key="gdp_year_range")

    gy0, gy1 = int(yr_gdp[0]), int(yr_gdp[1])

    # Map selected labels back to iso3/country
    sel = gdp.loc[gdp["label"].isin(labels_sel) & (gdp["year"] >= gy0) & (gdp["year"] <= gy1)].copy()

    # Scale
    if gdp_unit == "USD Trillion":
        scale = 1_000_000.0  # million -> trillion
        ytitle = "GDP (USD Trillion)"
        yfmt = ".2f"
    elif gdp_unit == "USD Billion":
        scale = 1000.0
        ytitle = "GDP (USD Billion)"
        yfmt = ".2f"
    else:
        scale = 1.0
        ytitle = "GDP (USD Million)"
        yfmt = ".0f"

    sel["gdp_scaled"] = sel["gdp_usd_mn"] / scale

    if sel.empty:
        st.info("No GDP data under the selected filters.")
        st.stop()

    # GDP Trend line
    st.markdown(f"#### GDP Trend ({gdp_unit})")
    fig = px.line(sel.sort_values("year"), x="year", y="gdp_scaled", color="label", markers=True)
    fig.update_layout(xaxis_title="Year", yaxis_title=ytitle, legend_title_text="")
    # Show all years in selected range
    fig.update_xaxes(dtick=1)

    fig = apply_labels_line(fig, show_labels, yfmt=yfmt)
    st.plotly_chart(fig, use_container_width=True)

    # Optional grouped bars
    st.markdown("#### Snapshot (selected range)")
    snap = sel.pivot_table(index="year", columns="label", values="gdp_scaled", aggfunc="mean").reset_index().sort_values("year")
    st.dataframe(snap, use_container_width=True, height=360)
    st.download_button("Download snapshot (GDP CSV)", data=_to_csv_download(snap), file_name="gdp_snapshot.csv")

    show_bars = st.checkbox("Show grouped bars (can get busy)", value=False)
    if show_bars:
        figb = px.bar(sel.sort_values("year"), x="year", y="gdp_scaled", color="label", barmode="group")
        figb.update_layout(xaxis_title="Year", yaxis_title=ytitle, legend_title_text="")
        figb.update_xaxes(dtick=1)
        figb = apply_labels_bar(figb, show_labels, yfmt=yfmt)
        st.plotly_chart(figb, use_container_width=True)
