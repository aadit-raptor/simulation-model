# dashboard_app.py
# -----------------
# Simulation Model — LBO Analysis Platform
# Run: streamlit run dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import time
import io

from simulation.vectorized_simulation import (
    run_vectorized_simulation_full,
    SimulationParams,
    get_scenario_params,
)
from analytics.risk_metrics import calculate_risk_metrics
from lbo_engine.model import LBOParams, run_lbo
from lbo_engine.capital_structure import build_simple_two_tranche_structure
from lbo_engine.returns import compute_exit_sensitivity, print_sensitivity_table
from pages.settings import init_cfg, get_cfg, build_corr_matrix

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulation Model",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state init (must come before CSS so font_scale is available)
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "page": 1, "mode": "deal", "font_scale": 1.0,
    "lbo_result": None, "mc_result": None,
    # Deal defaults — these are populated from cfg settings after init_cfg()
    "d_ebitda": 100.0, "d_entry_mult": 10.0, "d_exit_mult": 11.0,
    "d_hold": 5, "d_growth": 5.0, "d_gross_margin": 40.0,
    "d_opex": 18.0, "d_tax": 25.0, "d_da": 4.0,
    # WSP-style granular operating assumptions
    "d_rd_pct": 0.0,        # R&D % of revenue (0 = not applicable)
    "d_sga_pct": 18.0,      # SG&A % of revenue (replaces opex when using WSP mode)
    "d_sbc_pct": 2.0,       # Stock-based compensation % of revenue
    "d_ar_days": 45.0,      # Accounts receivable days (for WSP WC calc)
    "d_inv_days": 30.0,     # Inventory days (for WSP WC calc)
    "d_ap_days": 60.0,      # Accounts payable days (for WSP WC calc)
    "d_wsp_mode": False,    # True = use WSP WC drivers; False = use flat NWC %
    "d_dividends": 0.0,     # Annual dividends ($M, for retained earnings roll)
    "d_repurchases": 0.0,   # Annual buybacks ($M, for retained earnings roll)
    "d_other_exp": 0.0,     # Other income/(expense) net ($M, flat)
    "d_capex_abs": 0.0,     # Absolute capex ($M, 0 = use pct instead)
    "d_debt_pct": 60.0, "d_senior_pct": 70.0,
    "d_base_rate": 6.5, "d_mezz_spread": 4.0,
    "d_capex": 4.0, "d_nwc": 1.0, "d_mincash": 0.0,
    "mc_n": 50000, "mc_growth_mean": 5.0, "mc_growth_std": 3.0,
    "mc_exit_mean": 10.0, "mc_exit_std": 1.5,
    "mc_rate_mean": 6.5, "mc_rate_std": 1.5,
    "mc_gm_mean": 40.0, "mc_gm_std": 3.0, "mc_hurdle": 20.0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v
init_cfg()   # load settings defaults

# Apply cfg defaults to deal inputs on first load
# (only if not already overridden by user on page inputs)
_CFG_DEAL_MAP = {
    "d_ebitda": "def_ebitda", "d_entry_mult": "def_entry_mult",
    "d_exit_mult": "def_exit_mult", "d_hold": "def_hold",
    "d_growth": "def_growth", "d_gross_margin": "def_gross_margin",
    "d_opex": "def_opex", "d_tax": "def_tax", "d_da": "def_da",
    "d_debt_pct": "def_debt_pct", "d_senior_pct": "def_senior_pct",
    "d_base_rate": "def_base_rate", "d_mezz_spread": "def_mezz_spread",
    "d_capex": "def_capex", "d_nwc": "def_nwc", "d_mincash": "def_mincash",
}
for _dash_key, _cfg_key in _CFG_DEAL_MAP.items():
    if _dash_key not in st.session_state or st.session_state[_dash_key] == _DEFAULTS.get(_dash_key):
        st.session_state[_dash_key] = get_cfg(_cfg_key)

# ---------------------------------------------------------------------------
# Font scale — compute sizes from session state
# ---------------------------------------------------------------------------
_SCALE = st.session_state.get("font_scale", 1.0)

def _sz(base):
    """Return scaled px size as int."""
    return max(8, int(base * _SCALE))

# ---------------------------------------------------------------------------
# Global CSS — font scale baked in at render time
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: {_sz(15)}px;
}}

/* ── Sidebar shell ─────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: #06060e !important;
    border-right: 1px solid #0f0f20 !important;
    min-width: 220px !important;
    max-width: 220px !important;
    padding: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"],
section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"],
section[data-testid="stSidebar"] nav {{ display: none !important; }}

/* ── All sidebar buttons are the nav items themselves ─── */
/* We remove ALL default Streamlit button chrome and restyle */
section[data-testid="stSidebar"] .stButton > button {{
    width: 100% !important;
    text-align: left !important;
    padding: 7px 10px 7px 14px !important;
    margin: 1px 0 !important;
    border-radius: 4px !important;
    border: none !important;
    background: transparent !important;
    color: #7070a0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: {_sz(10)}px !important;
    letter-spacing: 0.03em !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: #0c0c20 !important;
    color: #a0a0c0 !important;
}}

/* Active nav button — set via data attribute injected below */
section[data-testid="stSidebar"] .stButton > button[data-active="true"] {{
    background: #0c1a2e !important;
    color: #e8e8f8 !important;
    border-left: 2px solid #378add !important;
    padding-left: 12px !important;
}}
/* Green active (new modules) */
section[data-testid="stSidebar"] .stButton > button[data-active-green="true"] {{
    background: #061a10 !important;
    color: #9fe1cb !important;
    border-left: 2px solid #1d9e75 !important;
    padding-left: 12px !important;
}}

/* ── Sidebar group labels ─────────────────────────────── */
.sb-group {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: {_sz(8)}px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 10px 14px 3px;
    margin-top: 2px;
    border-top: 1px solid #10101e;
    color: #3a3a5a;
}}
.sb-group.blue  {{ color: #4a6a90; }}
.sb-group.indigo{{ color: #50509a; }}
.sb-group.green {{ color: #1d7a50; }}

/* ── Font size buttons in sidebar ────────────────────── */
.sb-font-row {{ display: flex; gap: 4px; padding: 6px 8px 10px; }}
.sb-font-row .stButton > button {{
    flex: 1 !important;
    padding: 3px 4px !important;
    text-align: center !important;
    font-size: {_sz(9)}px !important;
    border: 0.5px solid #2a2a42 !important;
    border-radius: 3px !important;
    color: #5a5a72 !important;
    background: #0e0e1c !important;
}}
.sb-font-row .stButton > button:hover {{
    background: #16162a !important;
    color: #a0a0c0 !important;
}}

/* ── Main area ────────────────────────────────────────── */
.main {{ background: #05050c; }}
.block-container {{ padding: 1.2rem 1.8rem 2rem; max-width: 1600px; }}

/* ── Page header ──────────────────────────────────────── */
.page-hdr {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 12px; border-bottom: 1px solid #16162a; margin-bottom: 16px;
}}
.page-hdr-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: {_sz(14)}px; font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; color: #e0e0f0;
}}
.page-hdr-step {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: {_sz(10)}px; color: #44445a; letter-spacing: 0.06em;
}}

/* ── Field labels ─────────────────────────────────────── */
.field-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: {_sz(11)}px; color: #5a5a72; letter-spacing: 0.05em;
    margin-bottom: 2px; display: block;
}}

/* ── Computed chips ───────────────────────────────────── */
.computed {{
    background: #0c2040; border: 0.5px solid #185fa5;
    color: #85b7eb; border-radius: 4px; padding: 5px 10px;
    font-family: 'IBM Plex Mono', monospace; font-size: {_sz(12)}px;
    display: block; width: 100%; margin-bottom: 6px;
}}
.computed.green  {{ background: #061a10; border-color: #1d9e75; color: #5dcaa5; }}
.computed.positive {{
    background: #071a0e; border-color: #0f6e56; color: #9fe1cb;
    font-size: {_sz(15)}px; font-weight: 500;
}}

/* ── SAU rows ─────────────────────────────────────────── */
.sau-row {{
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace; font-size: {_sz(11)}px;
    padding: 4px 0; border-bottom: 0.5px solid #16162a; color: #8888a8;
}}
.sau-row .lbl {{ color: #686884; }}
.sau-row .val {{ color: #c4c4d4; }}
.sau-row.total {{
    color: #e0e0f0; font-weight: 500;
    border-bottom: none; border-top: 0.5px solid #2a2a42;
    margin-top: 2px; padding-top: 6px;
}}
.sau-row.plug .val {{ color: #85b7eb; }}

/* ── Metric cards ─────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: #0e0e1c !important;
    border: 0.5px solid #1e1e30 !important;
    border-radius: 6px !important;
    padding: 10px 12px !important;
}}
[data-testid="metric-container"] label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: {_sz(10)}px !important; letter-spacing: 0.10em;
    text-transform: uppercase; color: #44445a !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: {_sz(22)}px !important; color: #e0e0f0 !important;
}}

/* ── DataFrames ───────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 0.5px solid #1e1e30 !important;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: {_sz(11)}px;
}}

/* ── Tabs ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent; border-bottom: 1px solid #16162a; gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace; font-size: {_sz(10)}px;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #44445a; padding: 7px 16px; background: transparent; border: none;
}}
.stTabs [aria-selected="true"] {{
    color: #e0e0f0 !important;
    border-bottom: 2px solid #6060c0 !important;
}}

/* ── Inputs ───────────────────────────────────────────── */
input[type="number"], input[type="text"] {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: {_sz(13)}px !important;
    background: #0e0e1c !important;
    border: 0.5px solid #2a2a42 !important;
    color: #e0e0f0 !important;
    border-radius: 4px !important;
}}
input[type="number"]:focus, input[type="text"]:focus {{
    border-color: #378add !important;
    box-shadow: 0 0 0 2px rgba(55,138,221,0.15) !important;
}}

/* ── Buttons (main area) ──────────────────────────────── */
.main .stButton > button {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: {_sz(11)}px !important;
    letter-spacing: 0.08em; text-transform: uppercase;
    border-radius: 3px !important;
}}
.main .stButton > button[kind="primary"] {{
    background: #185fa5 !important; border: none !important;
    color: #e6f1fb !important;
}}

/* ── Section headings ─────────────────────────────────── */
h2 {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: {_sz(10)}px !important; letter-spacing: 0.12em;
    color: #44445a !important; text-transform: uppercase;
    border-bottom: 1px solid #16162a; padding-bottom: 5px;
    margin-top: 1.4rem !important;
}}
hr {{ border-color: #16162a !important; }}
</style>
""", unsafe_allow_html=True)

# Matplotlib theme
plt.rcParams.update({
    "figure.facecolor": "#05050c", "axes.facecolor": "#0e0e1c",
    "axes.edgecolor": "#2a2a42", "axes.labelcolor": "#8888a4",
    "text.color": "#c4c4d4", "xtick.color": "#5a5a72", "ytick.color": "#5a5a72",
    "grid.color": "#16162a", "grid.linewidth": 0.5,
    "font.family": "monospace", "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": "#c4c4d4",
    "legend.facecolor": "#0e0e1c", "legend.edgecolor": "#2a2a42", "legend.fontsize": 8,
})

A1="#6060c0"; A2="#40a0c0"; A3="#c06060"; A4="#40c080"; A5="#c0a040"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pf(v):   return f"{v:.1f}%"
def xf(v):   return f"{v:.2f}x"
def mf(v):   return f"${v:,.0f}M"

def step_bar(current):
    """Step progress rendered as styled column cards — no raw text leakage."""
    steps = ["Deal inputs", "Debt & CF", "Returns", "Summary"]
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        n = i + 1
        if n < current:
            bg, bdr, num_c, lbl_c, icon = "#0a1f16","#0f6e56","#9fe1cb","#5dcaa5","✓"
        elif n == current:
            bg, bdr, num_c, lbl_c, icon = "#0a1628","#378add","#b5d4f4","#e0e0f0",str(n)
        else:
            bg, bdr, num_c, lbl_c, icon = "#0e0e1c","#2a2a42","#44445a","#44445a",str(n)
        col.markdown(
            f'<div style="background:{bg};border:1px solid {bdr};border-radius:6px;'
            f'padding:8px 10px;text-align:center;font-family:IBM Plex Mono,monospace">'
            f'<div style="font-size:{_sz(16)}px;font-weight:600;color:{num_c}">{icon}</div>'
            f'<div style="font-size:{_sz(10)}px;color:{lbl_c};letter-spacing:0.06em;'
            f'text-transform:uppercase;margin-top:2px">{label}</div>'
            f'</div>', unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

def sau_row(label, value, cls=""):
    st.markdown(
        f'<div class="sau-row {cls}"><span class="lbl">{label}</span>'
        f'<span class="val">{value}</span></div>',
        unsafe_allow_html=True,
    )

def lbl(text):
    st.markdown(f'<span class="field-label">{text}</span>', unsafe_allow_html=True)

def chip(value, color=""):
    st.markdown(f'<div class="computed {color}">{value}</div>', unsafe_allow_html=True)

def section_hdr(title, color="#85b7eb"):
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{color};border-bottom:1px solid #16162a;padding-bottom:5px;'
        f'margin:16px 0 10px">◈ {title}</div>',
        unsafe_allow_html=True,
    )

def blk_open(border_color, title_color, title):
    """Open a coloured input block div."""
    st.markdown(
        f'<div style="background:#080816;border-left:2px solid {border_color};'
        f'border-radius:6px;padding:10px 14px 4px;margin-bottom:2px">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{title_color};margin-bottom:10px">{title}</div></div>',
        unsafe_allow_html=True,
    )

def _df_to_excel(df):
    """Convert a DataFrame to Excel bytes. Handles empty DataFrames safely."""
    buf = io.BytesIO()
    if df is None or (hasattr(df, "empty") and df.empty):
        # Write a placeholder so openpyxl always has a visible sheet
        df = pd.DataFrame({"Note": ["No data available"]})
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Data", index=True)
        # Ensure the sheet is visible (openpyxl requires at least one)
        w.book.active = 0
    buf.seek(0)
    return buf.getvalue()

def _multi_df_to_excel(sheets):
    """Convert multiple DataFrames to a multi-sheet Excel file."""
    buf = io.BytesIO()
    if not sheets:
        sheets = {"Data": pd.DataFrame({"Note": ["No data"]})}
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            if df is None or (hasattr(df, "empty") and df.empty):
                df = pd.DataFrame({"Note": ["No data"]})
            df.to_excel(w, sheet_name=name[:31], index=True)
        w.book.active = 0
    buf.seek(0)
    return buf.getvalue()

def dl_btn(label, data, filename, key):
    st.download_button(
        label=f"⬇ {label}", data=data, file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )

# ---------------------------------------------------------------------------
# DataFrames helpers
# ---------------------------------------------------------------------------
def wc_from_days(revenue, cogs, ar_days, inv_days, ap_days):
    """
    WSP-style working capital calculation from days outstanding.
    
    AR days  = (AR / Revenue) × 365    → AR  = Revenue × AR_days / 365
    Inv days = (Inv / COGS) × 365      → Inv = COGS × Inv_days / 365
    AP days  = (AP / COGS) × 365       → AP  = COGS × AP_days / 365
    
    Net WC = AR + Inventory − AP  (excludes cash and debt)
    """
    ar  = revenue * ar_days  / 365
    inv = cogs    * inv_days / 365
    ap  = cogs    * ap_days  / 365
    return ar + inv - ap   # net working capital ($M)


def ppe_roll(beginning_ppe, capex, depreciation):
    """PP&E roll-forward: beginning + capex - depreciation = ending."""
    return beginning_ppe + capex - depreciation


def retained_earnings_roll(beginning_re, net_income, dividends, repurchases):
    """
    Retained earnings roll-forward (WSP schedule).
    beginning + net_income − dividends − repurchases = ending
    All values in $M. Dividends and repurchases entered as positive numbers.
    """
    return beginning_re + net_income - abs(dividends) - abs(repurchases)


def pl_dataframe(op):
    yrs = [f"Y{t}" for t in op.years]
    rows = {
        "Revenue ($M)"   : [f"{v:,.0f}" for v in op.revenue],
        "  % growth"     : [f"{v:.1%}"  for v in op.revenue_growth],
        "COGS"           : [f"{v:,.0f}" for v in op.cogs],
        "  COGS %"       : [f"{v:.1%}"  for v in op.cogs_pct],
        "Gross profit"   : [f"{v:,.0f}" for v in op.gross_profit],
        "  Gross margin" : [f"{v:.1%}"  for v in op.gross_margin],
        "OpEx"           : [f"{v:,.0f}" for v in op.opex],
        "EBIT"           : [f"{v:,.0f}" for v in op.ebit],
        "  EBIT margin"  : [f"{v:.1%}"  for v in op.ebit_margin],
        "D&A"            : [f"{v:,.0f}" for v in op.da],
        "EBITDA"         : [f"{v:,.0f}" for v in op.ebitda],
        "  EBITDA margin": [f"{v:.1%}"  for v in op.ebitda_margin],
    }
    if op.net_income:
        rows["Interest exp"] = [f"{v:,.0f}" for v in op.interest_expense]
        rows["EBT"]          = [f"{v:,.0f}" for v in op.ebt]
        rows["Taxes"]        = [f"{v:,.0f}" for v in op.taxes]
        rows["Net income"]   = [f"{v:,.0f}" for v in op.net_income]
        rows["  Net margin"] = [f"{v:.1%}"  for v in op.net_margin]
    # WSP additions — computed from session state if available
    s = st.session_state
    sbc_pct = s.get("d_sbc_pct", 0.0) / 100
    if sbc_pct > 0 and op.revenue:
        sbc_vals = [rev * sbc_pct for rev in op.revenue]
        rows["+ SBC (non-cash)"]   = [f"{v:,.0f}" for v in sbc_vals]
        rows["Adjusted EBITDA"]    = [f"{(e + s_):,.0f}" for e, s_ in zip(op.ebitda, sbc_vals)]
        rows["  Adj EBITDA margin"]= [f"{(e + s_)/r:.1%}" for e, s_, r in zip(op.ebitda, sbc_vals, op.revenue)]
    return pd.DataFrame(rows, index=yrs).T

def fcf_dataframe(cf):
    yrs = [f"Y{t}" for t in cf.years]
    rows = {
        "Net income"     : [f"{v:,.0f}"  for v in cf.net_income],
        "+ D&A"          : [f"{v:,.0f}"  for v in cf.da],
        "- Capex"        : [f"({v:,.0f})" for v in cf.capex],
        "- Change in NWC": [f"({v:,.0f})" for v in cf.delta_nwc],
        "Levered FCF"    : [f"{v:,.0f}"  for v in cf.levered_fcf],
        "Cumulative FCF" : [f"{v:,.0f}"  for v in cf.cumulative_fcf],
    }
    return pd.DataFrame(rows, index=yrs).T

def debt_dataframe(ds):
    yrs = [f"Y{t}" for t in ds.years]
    rows = {
        "Debt — beginning": [f"{v:,.0f}"  for v in ds.total_beginning_debt],
        "Mandatory repay" : [f"({v:,.0f})" for v in ds.total_mandatory_repayment],
        "Cash sweep"      : [f"({v:,.0f})" for v in ds.total_cash_sweep],
        "Debt — ending"   : [f"{v:,.0f}"  for v in ds.total_ending_debt],
        "Cash balance"    : [f"{v:,.0f}"  for v in ds.cash_balance],
        "Interest expense": [f"{v:,.0f}"  for v in ds.total_interest_expense],
    }
    return pd.DataFrame(rows, index=yrs).T

def sens_dataframe(sens):
    ems = sens["exit_multiples"]
    hps = sens["holding_periods"]
    rows = {f"{em:.1f}x": [f"{v*100:.1f}%" for v in sens["table"][i]]
            for i, em in enumerate(ems)}
    return pd.DataFrame(rows, index=[f"{h}yr" for h in hps]).T

# ---------------------------------------------------------------------------
# Run deal model
# ---------------------------------------------------------------------------
def run_deal():
    s = st.session_state
    params = LBOParams(
        entry_ebitda=s.d_ebitda, entry_multiple=s.d_entry_mult,
        exit_multiple=s.d_exit_mult, holding_period=int(s.d_hold),
        debt_pct=s.d_debt_pct/100, senior_pct=s.d_senior_pct/100,
        mezz_spread=s.d_mezz_spread/100, interest_rate=s.d_base_rate/100,
        revenue_growth=s.d_growth/100, gross_margin=s.d_gross_margin/100,
        opex_pct=s.d_opex/100, da_pct=s.d_da/100, tax_rate=s.d_tax/100,
        capex_pct=s.d_capex/100, nwc_pct=s.d_nwc/100,
        minimum_cash=s.d_mincash, n_iterations=3,
    )
    with st.spinner("Running deal model..."):
        result = run_lbo(params)
    st.session_state.lbo_result = result
    return result

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def _get_scenario_params_cfg(scenario: str, base_params: SimulationParams) -> SimulationParams:
    """
    Like get_scenario_params() but reads multipliers from settings (get_cfg).
    Falls back to original hardcoded values if not changed by user.
    """
    import copy
    p = copy.deepcopy(base_params)
    if scenario == "bull":
        p.growth_mean       *= get_cfg("bull_growth_mult")
        p.exit_mean         *= get_cfg("bull_exit_mult")
        p.interest_mean     *= get_cfg("bull_rate_mult")
        p.gross_margin_mean *= get_cfg("bull_margin_mult")
    elif scenario == "base":
        pass
    elif scenario == "recession":
        p.growth_mean        = max(p.growth_mean + get_cfg("rec_growth_adj")/100,
                                   get_cfg("rec_growth_floor")/100)
        p.exit_mean         *= get_cfg("rec_exit_mult")
        p.interest_mean     *= get_cfg("rec_rate_mult")
        p.gross_margin_mean *= get_cfg("rec_margin_mult")
    elif scenario == "stagflation":
        p.growth_mean        = max(p.growth_mean + get_cfg("stag_growth_adj")/100,
                                   get_cfg("stag_growth_floor")/100)
        p.exit_mean         *= get_cfg("stag_exit_mult")
        p.interest_mean     *= get_cfg("stag_rate_mult")
        p.gross_margin_mean *= get_cfg("stag_margin_mult")
    return p


def render_sidebar():
    s = st.session_state
    cur_page = s.get("page", 1)
    cur_mode = s.get("mode", "deal")

    # ── Logo ──────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        f'<div style="padding:16px 14px 4px">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(14)}px;'
        f'font-weight:500;letter-spacing:0.08em;color:#e0e0f0">◈ Simulation Model</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
        f'color:#36365a;margin-top:2px;letter-spacing:0.06em">LBO Platform</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Nav helper — ONE element: a real Streamlit button styled by CSS ───
    # The trick: we inject a JS snippet after each button render that
    # sets a data-active attribute, which CSS then uses for highlighting.
    # This completely avoids the HTML-div-plus-invisible-button layering
    # that caused the click reliability problem.
    def _nav(label, page_id, mode_id, key, green=False):
        active = (cur_page == page_id and cur_mode == mode_id)
        clicked = st.sidebar.button(label, key=key, use_container_width=True)
        if clicked:
            s["page"] = page_id
            s["mode"] = mode_id
            st.rerun()
        # Inject CSS scoped to this button's key to highlight it when active
        if active:
            border_c = "#1d9e75" if green else "#378add"
            bg_c     = "#061a10" if green else "#0c1a2e"
            text_c   = "#9fe1cb" if green else "#e8e8f8"
            st.sidebar.markdown(
                f'<style>'
                f'[data-testid="stSidebar"] [data-testid="{key}"] > button,'
                f'[data-testid="stSidebar"] button[kind="secondary"][data-key="{key}"] {{'
                f'  background: {bg_c} !important;'
                f'  color: {text_c} !important;'
                f'  border-left: 2px solid {border_c} !important;'
                f'  padding-left: 12px !important;'
                f'}}'
                f'</style>',
                unsafe_allow_html=True,
            )

    def _grp(text, cls=""):
        st.sidebar.markdown(
            f'<div class="sb-group {cls}">{text}</div>',
            unsafe_allow_html=True,
        )

    # ── Deal flow ─────────────────────────────────────────────────────────
    _grp("Deal Flow", "blue")
    _nav("1 · Deal inputs",  1, "deal", "nav_p1")
    _nav("2 · Debt & CF",    2, "deal", "nav_p2")
    _nav("3 · Returns",      3, "deal", "nav_p3")
    _nav("4 · Summary",      4, "deal", "nav_p4")

    # ── Simulation ────────────────────────────────────────────────────────
    _grp("Simulation", "indigo")
    _nav("5 · Monte Carlo", 5, "mc", "nav_p5")

    # ── New modules ───────────────────────────────────────────────────────
    _grp("New Modules", "green")
    _nav("6 · Backtesting", 6, "backtest", "nav_p6", green=True)
    _nav("7 · Forecasting", 7, "forecast", "nav_p7", green=True)

    # ── Settings ──────────────────────────────────────────────────────────
    _grp("Settings", "")
    _nav("8 · Settings", 8, "settings", "nav_p8")

    # ── Last run pill ─────────────────────────────────────────────────────
    lbo_result = s.get("lbo_result")
    if lbo_result and getattr(lbo_result, "returns", None):
        st.sidebar.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;'
            f'background:#061a10;border:0.5px solid #0f6e56;border-radius:4px;'
            f'padding:8px 12px;margin:8px 8px 0">'
            f'<div style="color:#3a8060;font-size:{_sz(8)}px;letter-spacing:0.08em;'
            f'text-transform:uppercase">Last run</div>'
            f'<div style="color:#9fe1cb;font-size:{_sz(12)}px;font-weight:500;margin-top:3px">'
            f'IRR {lbo_result.irr*100:.1f}%  ·  {lbo_result.moic:.2f}x</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Font size control ─────────────────────────────────────────────────
    st.sidebar.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(8)}px;'
        f'color:#2e2e4a;letter-spacing:0.12em;text-transform:uppercase;'
        f'padding:10px 14px 4px;margin-top:6px;border-top:1px solid #10101e">Font size</div>',
        unsafe_allow_html=True,
    )
    cur_scale = s.get("font_scale", 1.0)
    fc1, fc2, fc3, fc4 = st.sidebar.columns(4)
    for col, (lbl_fs, sc) in zip(
        [fc1, fc2, fc3, fc4],
        [("A−", 0.85), ("A", 1.0), ("A+", 1.2), ("A++", 1.4)],
    ):
        is_active = abs(cur_scale - sc) < 0.05
        # Highlight active font button
        if is_active:
            col.markdown(
                f'<div style="background:#185fa5;color:#e6f1fb;border-radius:3px;'
                f'text-align:center;font-family:IBM Plex Mono,monospace;'
                f'font-size:{_sz(9)}px;padding:4px 2px">{lbl_fs}</div>',
                unsafe_allow_html=True,
            )
        else:
            if col.button(lbl_fs, key=f"fs_{lbl_fs}", use_container_width=True):
                s["font_scale"] = sc
                st.rerun()

# ---------------------------------------------------------------------------
# PAGE 1 — Deal inputs
# ---------------------------------------------------------------------------
def page_deal_inputs():
    st.markdown(
        '<div class="page-hdr">'
        '<div class="page-hdr-title">Deal inputs & transaction assumptions</div>'
        '<div class="page-hdr-step">Step 1 of 4</div></div>',
        unsafe_allow_html=True,
    )
    step_bar(1)
    s = st.session_state

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        blk_open("#378add", "#85b7eb", "Operating inputs")
        lbl("LTM EBITDA ($M)")
        s.d_ebitda = st.number_input(" ", value=s.d_ebitda, min_value=1.0,
                                      step=5.0, key="fi_ebitda",
                                      label_visibility="collapsed")
        lbl("Revenue growth (%)")
        s.d_growth = st.number_input(" ", value=s.d_growth, step=0.5,
                                      key="fi_growth", label_visibility="collapsed")
        lbl("Gross margin (%)")
        s.d_gross_margin = st.number_input(" ", value=s.d_gross_margin,
                                            min_value=1.0, max_value=99.0, step=1.0,
                                            key="fi_gm", label_visibility="collapsed")
        lbl("OpEx / revenue (%)")
        s.d_opex = st.number_input(" ", value=s.d_opex, min_value=0.0,
                                    max_value=99.0, step=1.0, key="fi_opex",
                                    label_visibility="collapsed")
        lbl("R&D % of revenue (0 if N/A)")
        s.d_rd_pct = st.number_input(" ", value=float(s.get("d_rd_pct", 0.0)),
                                      min_value=0.0, max_value=50.0, step=0.5,
                                      key="fi_rd", label_visibility="collapsed")
        lbl("SBC % of revenue")
        s.d_sbc_pct = st.number_input(" ", value=float(s.get("d_sbc_pct", 2.0)),
                                       min_value=0.0, max_value=20.0, step=0.25,
                                       key="fi_sbc", label_visibility="collapsed")

    with c2:
        blk_open("#1d9e75", "#5dcaa5", "Transaction assumptions")
        lbl("Entry EV / EBITDA (x)")
        s.d_entry_mult = st.number_input(" ", value=s.d_entry_mult, min_value=1.0,
                                          step=0.5, key="fi_emult",
                                          label_visibility="collapsed")
        lbl("Exit EV / EBITDA (x)")
        s.d_exit_mult = st.number_input(" ", value=s.d_exit_mult, min_value=1.0,
                                         step=0.5, key="fi_xmult",
                                         label_visibility="collapsed")
        lbl("Holding period (years)")
        s.d_hold = st.number_input(" ", value=int(s.d_hold), min_value=1,
                                    max_value=15, step=1, key="fi_hold",
                                    label_visibility="collapsed")
        lbl("Tax rate (%)")
        s.d_tax = st.number_input(" ", value=s.d_tax, min_value=0.0,
                                   max_value=60.0, step=1.0, key="fi_tax",
                                   label_visibility="collapsed")

    with c3:
        blk_open("#7f77dd", "#afa9ec", "Financing structure")
        lbl("Senior debt (x EBITDA)")
        senior_x = st.number_input(" ", value=3.4, min_value=0.0, step=0.1,
                                    key="fi_snrx", label_visibility="collapsed")
        lbl("Mezz debt (x EBITDA)")
        mezz_x = st.number_input(" ", value=0.8, min_value=0.0, step=0.1,
                                  key="fi_mzzx", label_visibility="collapsed")
        lbl("Senior interest rate (%)")
        s.d_base_rate = st.number_input(" ", value=s.d_base_rate, min_value=0.0,
                                         step=0.25, key="fi_rate",
                                         label_visibility="collapsed")
        lbl("Mezz spread (%)")
        s.d_mezz_spread = st.number_input(" ", value=s.d_mezz_spread, min_value=0.0,
                                           step=0.25, key="fi_mezz",
                                           label_visibility="collapsed")
        entry_ev = s.d_ebitda * s.d_entry_mult
        total_debt_abs = (senior_x + mezz_x) * s.d_ebitda
        if entry_ev > 0:
            s.d_debt_pct   = min((total_debt_abs / entry_ev) * 100, 99.0)
            s.d_senior_pct = (senior_x / (senior_x + mezz_x)) * 100 \
                             if (senior_x + mezz_x) > 0 else 70.0
        sponsor_eq = max(entry_ev - total_debt_abs, 0)
        lbl("Sponsor equity (plug)")
        chip(mf(sponsor_eq))

    # Sources & Uses
    st.markdown("## Sources & uses of funds")
    entry_ev       = s.d_ebitda * s.d_entry_mult
    total_debt_abs = (senior_x + mezz_x) * s.d_ebitda
    sponsor_eq     = max(entry_ev - total_debt_abs, 0)
    tx_fees        = entry_ev * get_cfg('tx_fee_pct') / 100
    fin_fees       = total_debt_abs * get_cfg('fin_fee_pct') / 100
    total_uses     = entry_ev + tx_fees + fin_fees + get_cfg('other_uses')
    total_sources  = total_debt_abs + sponsor_eq
    check          = total_sources - total_uses

    col_s, col_u = st.columns(2, gap="medium")
    with col_s:
        st.markdown(
            f'<div style="background:#0a0a18;border:0.5px solid #16162a;'
            f'border-radius:8px;padding:12px 14px">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
            f'font-weight:500;letter-spacing:0.10em;text-transform:uppercase;'
            f'color:#ef9f27;margin-bottom:10px">Sources</div>',
            unsafe_allow_html=True,
        )
        sau_row("Senior debt",            mf(senior_x * s.d_ebitda))
        sau_row("Mezz debt",              mf(mezz_x   * s.d_ebitda))
        sau_row("Sponsor equity (plug)",  mf(sponsor_eq), "plug")
        sau_row("Total sources",          mf(total_sources), "total")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_u:
        st.markdown(
            f'<div style="background:#0a0a18;border:0.5px solid #16162a;'
            f'border-radius:8px;padding:12px 14px">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
            f'font-weight:500;letter-spacing:0.10em;text-transform:uppercase;'
            f'color:#5dcaa5;margin-bottom:10px">Uses</div>',
            unsafe_allow_html=True,
        )
        sau_row("Equity purchase price",       mf(entry_ev))
        sau_row(f"Transaction fees ({get_cfg('tx_fee_pct'):.1f}% EV)", mf(tx_fees))
        sau_row(f"Financing fees ({get_cfg('fin_fee_pct'):.1f}% debt)", mf(fin_fees))
        sau_row("Total uses",                  mf(total_uses), "total")
        st.markdown("</div>", unsafe_allow_html=True)

    chk_col = "#40c080" if abs(check) < 1 else "#c06060"
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
        f'color:{chk_col};margin-top:6px">'
        f'S&U check: {mf(check)}  {"✓ balanced" if abs(check)<1 else "⚠ imbalance"}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    _, col_next = st.columns([3, 1])
    with col_next:
        if st.button("Next: Debt & cash flow →", type="primary",
                     use_container_width=True, key="p1_next"):
            st.session_state.page = 2
            st.session_state.mode = "deal"
            st.rerun()

# ---------------------------------------------------------------------------
# PAGE 2 — Debt & Cash Flow
# ---------------------------------------------------------------------------
def page_debt_cashflow():
    st.markdown(
        '<div class="page-hdr">'
        '<div class="page-hdr-title">Debt schedule & cash flow assumptions</div>'
        '<div class="page-hdr-step">Step 2 of 4</div></div>',
        unsafe_allow_html=True,
    )
    step_bar(2)
    s = st.session_state

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        blk_open("#378add", "#85b7eb", "Senior tranche")
        entry_ev   = s.d_ebitda * s.d_entry_mult
        total_debt = entry_ev * s.d_debt_pct / 100
        senior_amt = total_debt * s.d_senior_pct / 100

        lbl("Senior debt / EV (%)")
        s.d_debt_pct = st.number_input(" ", value=float(s.d_debt_pct),
                                        min_value=0.0, max_value=95.0, step=1.0,
                                        key="p2_debtpct", label_visibility="collapsed")
        lbl("Senior / total debt (%)")
        s.d_senior_pct = st.number_input(" ", value=float(s.d_senior_pct),
                                          min_value=1.0, max_value=100.0, step=1.0,
                                          key="p2_snrpct", label_visibility="collapsed")
        lbl("Senior interest rate (%)")
        s.d_base_rate = st.number_input(" ", value=float(s.d_base_rate),
                                         min_value=0.0, step=0.25, key="p2_rate",
                                         label_visibility="collapsed")
        lbl("Senior amount (computed)")
        chip(mf(senior_amt))

    with c2:
        blk_open("#1d9e75", "#5dcaa5", "Mezz / junior tranche")
        mezz_amt  = total_debt * (1 - s.d_senior_pct / 100)
        mezz_rate = s.d_base_rate + s.d_mezz_spread

        lbl("Mezz spread over base (%)")
        s.d_mezz_spread = st.number_input(" ", value=float(s.d_mezz_spread),
                                           min_value=0.0, step=0.25, key="p2_mezz",
                                           label_visibility="collapsed")
        lbl("Mezz amount (computed)"); chip(mf(mezz_amt))
        lbl("Mezz all-in rate (computed)"); chip(f"{mezz_rate:.2f}%")
        lbl("Total debt (computed)"); chip(mf(total_debt))
        lbl("Leverage (computed)"); chip(f"{total_debt/s.d_ebitda:.1f}x EBITDA")

    with c3:
        blk_open("#ba7517", "#ef9f27", "Cash flow assumptions")
        lbl("Capex / revenue (%)")
        s.d_capex = st.number_input(" ", value=float(s.d_capex), min_value=0.0,
                                     step=0.5, key="p2_capex",
                                     label_visibility="collapsed")
        lbl("D&A / revenue (%)")
        s.d_da = st.number_input(" ", value=float(s.d_da), min_value=0.0,
                                  step=0.5, key="p2_da", label_visibility="collapsed")
        lbl("NWC change / revenue (%)")
        s.d_nwc = st.number_input(" ", value=float(s.d_nwc), min_value=0.0,
                                   step=0.25, key="p2_nwc",
                                   label_visibility="collapsed")
        lbl("Minimum cash balance ($M)")
        s.d_mincash = st.number_input(" ", value=float(s.d_mincash), min_value=0.0,
                                       step=5.0, key="p2_mincash",
                                       label_visibility="collapsed")
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(8)}px;'
            f'color:#3a3a5a;text-transform:uppercase;letter-spacing:0.10em;'
            f'margin:10px 0 5px;border-top:0.5px solid #16162a;padding-top:8px">'
            f'WSP-style WC drivers</div>',
            unsafe_allow_html=True,
        )
        lbl("Use AR/AP/Inventory days")
        s.d_wsp_mode = st.checkbox("Replace flat NWC% with days",
                                    value=bool(s.get("d_wsp_mode", False)),
                                    key="p2_wsp_mode")
        if s.get("d_wsp_mode", False):
            lbl("AR days (Revenue×days/365)")
            s.d_ar_days = st.number_input(" ", value=float(s.get("d_ar_days", 45.0)),
                                           min_value=0.0, max_value=365.0, step=1.0,
                                           key="p2_ar", label_visibility="collapsed")
            lbl("Inventory days (COGS×days/365)")
            s.d_inv_days = st.number_input(" ", value=float(s.get("d_inv_days", 30.0)),
                                            min_value=0.0, max_value=365.0, step=1.0,
                                            key="p2_inv", label_visibility="collapsed")
            lbl("AP days (COGS×days/365)")
            s.d_ap_days = st.number_input(" ", value=float(s.get("d_ap_days", 60.0)),
                                           min_value=0.0, max_value=365.0, step=1.0,
                                           key="p2_ap", label_visibility="collapsed")

    st.markdown("---")
    col_run, col_hint = st.columns([1, 2])
    with col_run:
        run_clicked = st.button("▶  Run deal model", type="primary",
                                use_container_width=True, key="p2_run")
    with col_hint:
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
            f'color:#44445a;padding-top:10px">Runs full 3-pass LBO engine. '
            f'Results visible on all pages.</div>',
            unsafe_allow_html=True,
        )

    if run_clicked:
        result = run_deal()
        if result and result.returns:
            st.success(
                f"IRR: {result.irr*100:.1f}%  ·  MOIC: {result.moic:.2f}x  ·  "
                f"Exit equity: {mf(result.exit_equity)}"
            )

    result = st.session_state.lbo_result
    if result and result.debt_schedule:
        st.markdown("## Debt schedule — ending balances")
        ds_df = debt_dataframe(result.debt_schedule)
        st.dataframe(ds_df, use_container_width=True)
        dl_btn("Download debt schedule", _df_to_excel(ds_df),
               "debt_schedule.xlsx", "dl_debt")

        ds = result.debt_schedule
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
        yrs = [f"Y{t}" for t in ds.years]
        tranche_names = list(ds.schedule.keys())
        bottom = np.zeros(len(yrs))
        for i, tn in enumerate(tranche_names):
            endings = [r.ending_balance for r in ds.schedule[tn]]
            axes[0].bar(yrs, endings, bottom=bottom,
                        label=tn, color=[A1,A2,A3,A5][i%4], alpha=0.75, width=0.5)
            bottom += np.array(endings)
        axes[0].set_title("Debt balance by tranche"); axes[0].set_ylabel("$M")
        axes[0].legend(fontsize=7); axes[0].grid(axis="y")

        if result.cash_flow:
            cf = result.cash_flow
            fcf_v = cf.levered_fcf
            bcolors = [A4 if v >= 0 else A3 for v in fcf_v]
            axes[1].bar([f"Y{t}" for t in cf.years], fcf_v,
                        color=bcolors, alpha=0.8, width=0.5)
            axes[1].axhline(0, color="#2a2a42", lw=1)
            axes[1].set_title("Levered FCF"); axes[1].set_ylabel("$M")
            axes[1].grid(axis="y")
            for i, v in enumerate(fcf_v):
                axes[1].text(i, v+(1.5 if v>=0 else -4),
                             f"${v:,.0f}", ha="center", fontsize=7,
                             color=A4 if v>=0 else A3)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    col_back, _, col_next = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Deal inputs", use_container_width=True, key="p2_back"):
            st.session_state.page = 1; st.session_state.mode = "deal"; st.rerun()
    with col_next:
        if st.button("Next: Returns & exit →", type="primary",
                     use_container_width=True, key="p2_next"):
            st.session_state.page = 3; st.session_state.mode = "deal"; st.rerun()

# ---------------------------------------------------------------------------
# PAGE 3 — Returns & Exit
# ---------------------------------------------------------------------------
def page_returns():
    st.markdown(
        '<div class="page-hdr">'
        '<div class="page-hdr-title">Returns & exit assumptions</div>'
        '<div class="page-hdr-step">Step 3 of 4</div></div>',
        unsafe_allow_html=True,
    )
    step_bar(3)
    s = st.session_state
    result = s.lbo_result

    if not result or not result.returns:
        st.info("Run the deal model on Page 2 first to see returns.")
        if st.button("← Go to Debt & CF", key="p3_back_early"):
            st.session_state.page = 2; st.session_state.mode = "deal"; st.rerun()
        return

    r  = result.returns
    br = result.equity_bridge

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        blk_open("#7f77dd", "#afa9ec", "Exit assumptions")
        lbl("Exit year");             chip(str(s.d_hold))
        lbl("Exit EV / EBITDA (x)"); chip(f"{s.d_exit_mult:.1f}x")
        lbl("Exit EBITDA ($M)");      chip(mf(r.exit_ebitda))
        lbl("Net debt at exit ($M)"); chip(mf(r.net_debt_at_exit))

    with c2:
        blk_open("#378add", "#85b7eb", "Exit valuation bridge")
        sau_row("Exit EBITDA ($M)",    mf(r.exit_ebitda))
        sau_row("× Exit multiple",     f"{r.exit_multiple:.1f}x")
        sau_row("= Enterprise value",  mf(r.exit_ev), "plug")
        sau_row("− Net debt at exit",  f"({mf(r.net_debt_at_exit)})")
        sau_row("Equity value at exit",mf(r.net_exit_equity), "total")

    with c3:
        blk_open("#0f6e56", "#5dcaa5", "Sponsor returns")
        lbl("Entry equity");  chip(mf(r.entry_equity))
        lbl("Exit equity");   chip(mf(r.net_exit_equity), "green")
        lbl("MOIC");          chip(xf(r.moic), "positive")
        lbl("IRR");           chip(pf(r.irr),  "positive")

    if br:
        st.markdown("## Equity value bridge")
        col_chart, col_table = st.columns([3, 2], gap="medium")
        with col_chart:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            labels  = ["Entry","EBITDA\ngrowth","Multiple","Deleverage","Exit"]
            heights = [br["entry_equity"], br["ebitda_growth"],
                       br["multiple_expansion"], br["deleveraging"], br["exit_equity"]]
            bottoms = [0, br["entry_equity"],
                       br["entry_equity"]+br["ebitda_growth"],
                       br["entry_equity"]+br["ebitda_growth"]+br["multiple_expansion"], 0]
            for i in range(5):
                ax.bar(i, heights[i], bottom=bottoms[i],
                       color=[A1,A4,A2,A2,A4][i], alpha=0.8, width=0.5,
                       edgecolor="#16162a", linewidth=0.5)
                ax.text(i, bottoms[i]+heights[i]+20, f"${heights[i]:,.0f}",
                        ha="center", fontsize=8, color="#c4c4d4")
            ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("$M"); ax.set_title("Value attribution waterfall")
            ax.grid(axis="y")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with col_table:
            bridge_df = pd.DataFrame({
                "Component": ["Entry equity","EBITDA growth","Multiple expansion",
                               "Deleveraging","Exit equity"],
                "Value ($M)": [f"{br['entry_equity']:,.0f}",
                                f"+{br['ebitda_growth']:,.0f}",
                                f"+{br['multiple_expansion']:,.0f}",
                                f"+{br['deleveraging']:,.0f}",
                                f"{br['exit_equity']:,.0f}"],
                "% of gain":  ["—",
                                f"{br['ebitda_growth_pct']:.1f}%",
                                f"{br['multiple_expansion_pct']:.1f}%",
                                f"{br['deleveraging_pct']:.1f}%","—"],
            })
            st.dataframe(bridge_df.set_index("Component"),
                         use_container_width=True)

    _sens_em_range = f"{get_cfg('sens_em_min'):.1f}x–{get_cfg('sens_em_max'):.1f}x"
    _sens_hp_range = f"{int(get_cfg('sens_hp_min'))}–{int(get_cfg('sens_hp_max'))}yr"
    st.markdown(f"## IRR sensitivity — exit multiple × holding period "
                f"({_sens_em_range} · {_sens_hp_range})")
    if result.exit_sensitivity:
        sens = result.exit_sensitivity
        col_tbl, col_heat = st.columns(2, gap="medium")
        with col_tbl:
            sens_df = sens_dataframe(sens)
            st.dataframe(sens_df, use_container_width=True)
            dl_btn("Download sensitivity", _df_to_excel(sens_df),
                   "irr_sensitivity.xlsx", "dl_sens")
        with col_heat:
            raw = np.array(sens["table"]) * 100
            ems = [f"{v:.1f}x" for v in sens["exit_multiples"]]
            hps = [f"{v}yr"    for v in sens["holding_periods"]]
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(raw, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=50, interpolation="nearest")
            ax.set_xticks(range(len(hps)));  ax.set_xticklabels(hps)
            ax.set_yticks(range(len(ems)));  ax.set_yticklabels(ems)
            ax.set_title("IRR heatmap"); plt.colorbar(im, ax=ax, label="IRR %")
            for i in range(len(ems)):
                for j in range(len(hps)):
                    v = raw[i, j]
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=8, color="black" if 10<v<40 else "white")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")
    col_back, _, col_next = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Debt & CF", use_container_width=True, key="p3_back"):
            st.session_state.page = 2; st.session_state.mode = "deal"; st.rerun()
    with col_next:
        if st.button("Next: Summary →", type="primary",
                     use_container_width=True, key="p3_next"):
            st.session_state.page = 4; st.session_state.mode = "deal"; st.rerun()

# ---------------------------------------------------------------------------
# PAGE 4 — Summary Dashboard
# ---------------------------------------------------------------------------
def page_summary():
    st.markdown(
        '<div class="page-hdr">'
        '<div class="page-hdr-title">Summary dashboard</div>'
        '<div class="page-hdr-step">Step 4 of 4 — complete</div></div>',
        unsafe_allow_html=True,
    )
    step_bar(4)
    result = st.session_state.lbo_result

    if not result or not result.returns:
        st.info("Run the deal model on Page 2 first.")
        if st.button("← Go to Debt & CF", key="p4_back_early"):
            st.session_state.page = 2; st.session_state.mode = "deal"; st.rerun()
        return

    r = result.returns
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("IRR",           pf(r.irr))
    c2.metric("MOIC",          xf(r.moic))
    c3.metric("Entry equity",  mf(r.entry_equity))
    c4.metric("Exit equity",   mf(r.net_exit_equity))
    c5.metric("Value created", mf(r.value_created))

    tabs = st.tabs(["P&L", "Cash flow", "Debt schedule", "Balance sheet (WSP)", "Equity bridge", "Charts"])

    with tabs[0]:
        pl_df = pl_dataframe(result.operating_model)
        st.dataframe(pl_df, use_container_width=True)
        dl_btn("Download P&L", _df_to_excel(pl_df), "pl.xlsx", "dl_pl")

    with tabs[1]:
        cf_df = fcf_dataframe(result.cash_flow)
        st.dataframe(cf_df, use_container_width=True)
        dl_btn("Download cash flow", _df_to_excel(cf_df), "cashflow.xlsx", "dl_cf")

    with tabs[2]:
        ds_df = debt_dataframe(result.debt_schedule)
        st.dataframe(ds_df, use_container_width=True)
        for tn, records in result.debt_schedule.schedule.items():
            with st.expander(f"▸ {tn}"):
                rows = {
                    "Beginning": [f"{r.beginning_balance:,.0f}" for r in records],
                    "Mandatory": [f"({r.mandatory_repayment:,.0f})" for r in records],
                    "Sweep":     [f"({r.cash_sweep:,.0f})" for r in records],
                    "Ending":    [f"{r.ending_balance:,.0f}"  for r in records],
                    "Interest":  [f"{r.interest_expense:,.0f}" for r in records],
                }
                yrs_lbl = [f"Year {r.year}" for r in records]
                st.dataframe(pd.DataFrame(rows, index=yrs_lbl).T,
                             use_container_width=True)
        dl_btn("Download debt schedule", _df_to_excel(ds_df),
               "debt_schedule.xlsx", "dl_ds")

    with tabs[3]:
        # WSP-style Balance Sheet schedules
        section_hdr("PP&E roll-forward (WSP)", "#5dcaa5")
        s_bs = st.session_state
        op_bs = result.operating_model
        if op_bs and op_bs.revenue:
            yrs_bs = [f"Y{t}" for t in op_bs.years]
            n_yrs  = len(yrs_bs)
            # PP&E roll using capex and D&A from model
            da_abs     = [rev * s_bs.get("d_da", 4.0)/100 for rev in op_bs.revenue]
            capex_abs  = [rev * s_bs.get("d_capex", 4.0)/100 for rev in op_bs.revenue]
            entry_ev_bs = s_bs.d_ebitda * s_bs.d_entry_mult
            # Estimate opening PP&E as capex × 3 (rough prior book value)
            opening_ppe = capex_abs[0] * 3 if capex_abs else 0
            ppe_rows = {"Beginning PP&E": [], "+ Capex": [], "− Depreciation (est.)": [], "Ending PP&E": []}
            curr_ppe = opening_ppe
            for i in range(n_yrs):
                ppe_rows["Beginning PP&E"].append(f"${curr_ppe:,.0f}M")
                ppe_rows["+ Capex"].append(f"${capex_abs[i]:,.0f}M")
                dep = da_abs[i]
                ppe_rows["− Depreciation (est.)"].append(f"(${dep:,.0f}M)")
                curr_ppe = ppe_roll(curr_ppe, capex_abs[i], dep)
                ppe_rows["Ending PP&E"].append(f"${curr_ppe:,.0f}M")
            ppe_df = pd.DataFrame(ppe_rows, index=yrs_bs).T
            st.dataframe(ppe_df, use_container_width=True)
            dl_btn("Download PP&E schedule", _df_to_excel(ppe_df), "ppe_schedule.xlsx", "dl_ppe")

            st.markdown("---")
            section_hdr("Working capital schedule (WSP days method)", "#85b7eb")
            if s_bs.get("d_wsp_mode", False):
                cogs_bs   = [rev * (1 - s_bs.d_gross_margin/100) for rev in op_bs.revenue]
                ar_days   = s_bs.get("d_ar_days", 45.0)
                inv_days  = s_bs.get("d_inv_days", 30.0)
                ap_days   = s_bs.get("d_ap_days", 60.0)
                wc_rows   = {
                    "Revenue ($M)":         [f"${v:,.0f}M" for v in op_bs.revenue],
                    "COGS ($M)":            [f"${v:,.0f}M" for v in cogs_bs],
                    "AR (days method)":     [f"${rev * ar_days/365:,.0f}M"  for rev in op_bs.revenue],
                    "Inventory (days)":     [f"${cogs * inv_days/365:,.0f}M" for cogs in cogs_bs],
                    "AP (days method)":     [f"(${cogs * ap_days/365:,.0f}M)" for cogs in cogs_bs],
                    "Net working capital":  [f"${wc_from_days(rev, cogs, ar_days, inv_days, ap_days):,.0f}M"
                                             for rev, cogs in zip(op_bs.revenue, cogs_bs)],
                }
                wc_df = pd.DataFrame(wc_rows, index=yrs_bs).T
                st.dataframe(wc_df, use_container_width=True)
                dl_btn("Download WC schedule", _df_to_excel(wc_df), "wc_schedule.xlsx", "dl_wc")
            else:
                st.info("Enable 'Use AR/AP/Inventory days' on Page 2 to see the WSP working capital schedule.")

            st.markdown("---")
            section_hdr("Adjusted EBITDA bridge (WSP)", "#afa9ec")
            sbc_pct_bs = s_bs.get("d_sbc_pct", 0.0) / 100
            sbc_abs    = [rev * sbc_pct_bs for rev in op_bs.revenue]
            adj_ebitda = [e + s_ for e, s_ in zip(op_bs.ebitda, sbc_abs)]
            adj_rows = {
                "EBITDA ($M)":           [f"${v:,.0f}M" for v in op_bs.ebitda],
                "+ SBC ($M)":            [f"${v:,.0f}M" for v in sbc_abs],
                "Adjusted EBITDA ($M)":  [f"${v:,.0f}M" for v in adj_ebitda],
                "Adj EBITDA margin":     [f"{v/r:.1%}" for v, r in zip(adj_ebitda, op_bs.revenue)],
            }
            adj_df = pd.DataFrame(adj_rows, index=yrs_bs).T
            st.dataframe(adj_df, use_container_width=True)
            dl_btn("Download Adj EBITDA", _df_to_excel(adj_df), "adj_ebitda.xlsx", "dl_adj")
        else:
            st.info("Run the deal model on Page 2 first.")

    with tabs[4]:
        if result.equity_bridge:
            br = result.equity_bridge
            c_l, c_r = st.columns(2, gap="medium")
            with c_l:
                bridge_df = pd.DataFrame({
                    "Component": ["Entry equity","EBITDA growth",
                                   "Multiple expansion","Deleveraging","Exit equity"],
                    "Value ($M)": [f"{br['entry_equity']:,.0f}",
                                    f"+{br['ebitda_growth']:,.0f}",
                                    f"+{br['multiple_expansion']:,.0f}",
                                    f"+{br['deleveraging']:,.0f}",
                                    f"{br['exit_equity']:,.0f}"],
                    "% of gain":  ["—",f"{br['ebitda_growth_pct']:.1f}%",
                                    f"{br['multiple_expansion_pct']:.1f}%",
                                    f"{br['deleveraging_pct']:.1f}%","—"],
                })
                st.dataframe(bridge_df.set_index("Component"),
                             use_container_width=True)
                dl_btn("Download equity bridge",
                       _df_to_excel(bridge_df), "equity_bridge.xlsx", "dl_br")
            with c_r:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                heights = [br["entry_equity"],br["ebitda_growth"],
                           br["multiple_expansion"],br["deleveraging"],br["exit_equity"]]
                bottoms = [0,br["entry_equity"],
                           br["entry_equity"]+br["ebitda_growth"],
                           br["entry_equity"]+br["ebitda_growth"]+br["multiple_expansion"],0]
                for i in range(5):
                    ax.bar(i, heights[i], bottom=bottoms[i],
                           color=[A1,A4,A2,A2,A4][i], alpha=0.8, width=0.5,
                           edgecolor="#16162a", linewidth=0.5)
                ax.set_xticks(range(5))
                ax.set_xticklabels(["Entry","EBITDA\ngrowth","Multiple",
                                     "Deleverage","Exit"], fontsize=8)
                ax.set_ylabel("$M"); ax.grid(axis="y")
                st.pyplot(fig, use_container_width=True); plt.close(fig)

    with tabs[5]:
        op = result.operating_model; cf = result.cash_flow; ds = result.debt_schedule
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        yrs = [f"Y{t}" for t in op.years]
        axes[0,0].bar(yrs, [m*100 for m in op.ebitda_margin],
                      color=A1, alpha=0.7, width=0.5)
        axes[0,0].plot(yrs, [m*100 for m in op.gross_margin],
                       color=A2, lw=1.5, marker="o", ms=4, label="Gross margin")
        axes[0,0].set_title("Margin profile"); axes[0,0].set_ylabel("%")
        axes[0,0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
        axes[0,0].legend(); axes[0,0].grid(axis="y")
        axes[0,1].bar(yrs, op.revenue, color=A2, alpha=0.7, width=0.5)
        axes[0,1].set_title("Revenue ($M)"); axes[0,1].grid(axis="y")
        fcf_v = cf.levered_fcf
        bcolors = [A4 if v>=0 else A3 for v in fcf_v]
        axes[1,0].bar([f"Y{t}" for t in cf.years], fcf_v,
                      color=bcolors, alpha=0.8, width=0.5)
        axes[1,0].axhline(0, color="#2a2a42", lw=1)
        axes[1,0].set_title("Levered FCF ($M)"); axes[1,0].grid(axis="y")
        yrs_d = [f"Y{t}" for t in ds.years]; bottom_d = np.zeros(len(yrs_d))
        for i, tn in enumerate(ds.schedule.keys()):
            endings = [rec.ending_balance for rec in ds.schedule[tn]]
            axes[1,1].bar(yrs_d, endings, bottom=bottom_d,
                          label=tn, color=[A1,A2,A3,A5][i%4], alpha=0.75, width=0.5)
            bottom_d += np.array(endings)
        axes[1,1].set_title("Debt balance ($M)")
        axes[1,1].legend(fontsize=7); axes[1,1].grid(axis="y")
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    sheets = {
        "P&L":           pl_dataframe(result.operating_model),
        "Cash flow":     fcf_dataframe(result.cash_flow),
        "Debt schedule": debt_dataframe(result.debt_schedule),
    }
    dl_btn("Download all tables (Excel)",
           _multi_df_to_excel(sheets), "lbo_summary.xlsx", "dl_summary")

    st.markdown("---")
    col_back, _, col_mc = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Returns", use_container_width=True, key="p4_back"):
            st.session_state.page = 3; st.session_state.mode = "deal"; st.rerun()
    with col_mc:
        if st.button("→ Monte Carlo", type="primary",
                     use_container_width=True, key="p4_mc"):
            st.session_state.page = 5; st.session_state.mode = "mc"; st.rerun()

# ---------------------------------------------------------------------------
# PAGE 5 — Monte Carlo
# ---------------------------------------------------------------------------
def page_monte_carlo():
    st.markdown(
        '<div class="page-hdr">'
        '<div class="page-hdr-title">Monte Carlo simulation</div>'
        '<div class="page-hdr-step">Simulation mode</div></div>',
        unsafe_allow_html=True,
    )
    s = st.session_state
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        blk_open("#378add", "#85b7eb", "Simulation setup")
        lbl("Number of scenarios")
        s.mc_n = st.number_input(" ", value=int(s.mc_n), min_value=1000,
                                  max_value=5_000_000, step=5000,
                                  key="mc_fi_n", label_visibility="collapsed")
        lbl("Entry EBITDA ($M)")
        mc_ebitda = st.number_input(" ", value=float(s.d_ebitda), min_value=1.0,
                                     step=5.0, key="mc_fi_eb",
                                     label_visibility="collapsed")
        lbl("Entry multiple (x)")
        mc_emult = st.number_input(" ", value=float(s.d_entry_mult), min_value=1.0,
                                    step=0.5, key="mc_fi_em",
                                    label_visibility="collapsed")
        lbl("Holding period (yrs)")
        mc_hold = st.number_input(" ", value=int(s.d_hold), min_value=1,
                                   max_value=15, step=1, key="mc_fi_hold",
                                   label_visibility="collapsed")
        lbl("Hurdle rate (%)")
        s.mc_hurdle = st.number_input(" ", value=float(s.mc_hurdle), min_value=0.0,
                                       step=1.0, key="mc_fi_hur",
                                       label_visibility="collapsed")

    with c2:
        blk_open("#7f77dd", "#afa9ec", "Distribution means")
        lbl("Revenue growth mean (%)")
        s.mc_growth_mean = st.number_input(" ", value=float(s.mc_growth_mean),
                                            step=0.5, key="mc_fi_gm",
                                            label_visibility="collapsed")
        lbl("Exit multiple mean (x)")
        s.mc_exit_mean = st.number_input(" ", value=float(s.mc_exit_mean),
                                          min_value=1.0, step=0.5, key="mc_fi_xm",
                                          label_visibility="collapsed")
        lbl("Interest rate mean (%)")
        s.mc_rate_mean = st.number_input(" ", value=float(s.mc_rate_mean),
                                          min_value=0.0, step=0.25, key="mc_fi_rm",
                                          label_visibility="collapsed")
        lbl("Gross margin mean (%)")
        s.mc_gm_mean = st.number_input(" ", value=float(s.mc_gm_mean),
                                        min_value=1.0, max_value=99.0, step=1.0,
                                        key="mc_fi_gmm", label_visibility="collapsed")

    with c3:
        blk_open("#993c1d", "#f0997b", "Distribution volatility")
        lbl("Growth std dev (%)")
        s.mc_growth_std = st.number_input(" ", value=float(s.mc_growth_std),
                                           min_value=0.1, step=0.5, key="mc_fi_gs",
                                           label_visibility="collapsed")
        lbl("Exit multiple std dev (x)")
        s.mc_exit_std = st.number_input(" ", value=float(s.mc_exit_std),
                                         min_value=0.1, step=0.25, key="mc_fi_xs",
                                         label_visibility="collapsed")
        lbl("Interest rate std dev (%)")
        s.mc_rate_std = st.number_input(" ", value=float(s.mc_rate_std),
                                         min_value=0.1, step=0.25, key="mc_fi_rs",
                                         label_visibility="collapsed")
        lbl("Gross margin std dev (%)")
        s.mc_gm_std = st.number_input(" ", value=float(s.mc_gm_std),
                                       min_value=0.1, step=0.5, key="mc_fi_gms",
                                       label_visibility="collapsed")

    st.markdown("---")
    section_hdr("Scenario presets", "#44445a")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5, gap="small")
    scenario_override = None
    with sc1:
        if st.button("RECESSION",   use_container_width=True, key="sc_rec"):
            scenario_override = "recession"
    with sc2:
        if st.button("BASE",        use_container_width=True, key="sc_base"):
            scenario_override = "base"
    with sc3:
        if st.button("BULL",        use_container_width=True, key="sc_bull"):
            scenario_override = "bull"
    with sc4:
        if st.button("STAGFLATION", use_container_width=True, key="sc_stag"):
            scenario_override = "stagflation"
    with sc5:
        run_mc = st.button("▶ RUN SIMULATION", type="primary",
                           use_container_width=True, key="sc_run")

    params = SimulationParams(
        n=int(s.mc_n), entry_ebitda=mc_ebitda, entry_multiple=mc_emult,
        holding_period=int(mc_hold),
        growth_mean=s.mc_growth_mean/100, growth_std=s.mc_growth_std/100,
        exit_mean=s.mc_exit_mean, exit_std=s.mc_exit_std,
        interest_mean=s.mc_rate_mean/100, interest_std=s.mc_rate_std/100,
        gross_margin_mean=s.mc_gm_mean/100, gross_margin_std=s.mc_gm_std/100,
        opex_pct=s.d_opex/100, da_pct=s.d_da/100, tax_rate=s.d_tax/100,
        capex_pct=s.d_capex/100, nwc_pct=s.d_nwc/100,
        debt_pct=s.d_debt_pct/100, senior_pct=s.d_senior_pct/100,
        mezz_spread=s.d_mezz_spread/100,
        n_interest_passes=int(get_cfg('mc_n_passes')),
        corr_matrix=build_corr_matrix(),
    )
    if scenario_override:
        params = _get_scenario_params_cfg(scenario_override, params)
        st.info(f"Scenario preset applied: {scenario_override.upper()}")

    if run_mc:
        with st.spinner(f"Simulating {params.n:,} scenarios..."):
            t0 = time.time()
            sim = run_vectorized_simulation_full(params)
            elapsed = time.time() - t0
        st.session_state.mc_result = sim
        st.success(f"Done — {params.n:,} scenarios in {elapsed:.2f}s "
                   f"({params.n/elapsed/1e6:.2f}M/sec)")

    sim = st.session_state.mc_result
    if not sim:
        return

    df     = sim.df
    irr    = df["IRR"].values
    moic   = df["MOIC"].values
    target = s.mc_hurdle / 100
    metrics = calculate_risk_metrics(df, target)

    section_hdr("Risk metrics")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Mean IRR",        pf(metrics["Mean IRR"] * 100))
    m2.metric("Median IRR",      pf(metrics["Median IRR"] * 100))
    m3.metric("5th pct",         pf(metrics["5% Downside IRR"] * 100))
    m4.metric("95th pct",        pf(metrics["95% Upside IRR"] * 100))
    m5.metric(f"P(>{pf(target * 100)})",pf(metrics["Probability IRR > Target"] * 100))
    m6.metric("Wipeout rate",    pf(sim.wipeout_rate * 100))

    sample = df.sample(min(50_000, len(df)), random_state=42)
    tabs = st.tabs(["Distributions","Scatter plots","Correlations",
                    "Sensitivity","Scenario comparison"])

    with tabs[0]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        axes[0].hist(irr*100, bins=80, color=A1, alpha=0.75,
                     edgecolor="none", density=True)
        axes[0].axvline(np.mean(irr)*100, color=A2, lw=1.5, linestyle="--",
                        label=f"Mean {np.mean(irr):.1%}")
        axes[0].axvline(target*100, color=A3, lw=1.2, linestyle=":",
                        label=f"Hurdle {target:.0%}")
        for pct_v in [5, 95]:
            axes[0].axvline(np.percentile(irr, pct_v)*100, color="#888",
                            lw=1, linestyle=":")
        axes[0].set_xlabel("IRR (%)"); axes[0].set_title("IRR distribution")
        axes[0].legend(fontsize=7); axes[0].grid(axis="y")
        axes[1].hist(moic, bins=80, color=A2, alpha=0.75,
                     edgecolor="none", density=True)
        axes[1].axvline(np.mean(moic), color=A1, lw=1.5, linestyle="--",
                        label=f"Mean {np.mean(moic):.2f}x")
        axes[1].axvline(1.0, color=A3, lw=1.2, linestyle=":",
                        label="1.0x breakeven")
        axes[1].set_xlabel("MOIC (x)"); axes[1].set_title("MOIC distribution")
        axes[1].legend(fontsize=7); axes[1].grid(axis="y")
        sorted_irr = np.sort(irr) * 100
        cdf = np.linspace(0, 1, len(sorted_irr))
        axes[2].plot(sorted_irr, cdf*100, color=A1, lw=1.5)
        axes[2].axvline(target*100, color=A3, lw=1.2, linestyle=":",
                        label=f"Hurdle {target:.0%}")
        p_hit = (irr > target).mean()
        axes[2].set_xlabel("IRR (%)"); axes[2].set_ylabel("Cumulative %")
        axes[2].set_title(f"CDF — P(IRR>{target:.0%}) = {p_hit:.1%}")
        axes[2].legend(fontsize=7); axes[2].grid()
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Download — limit to 10k rows to keep file size manageable
        n_dl = min(10000, len(irr))
        dist_df = pd.DataFrame({
            "IRR (decimal)": irr[:n_dl],
            "IRR (%)":       irr[:n_dl] * 100,
            "MOIC":          moic[:n_dl],
        })
        dl_btn("Download distributions (10k sample)", _df_to_excel(dist_df),
               "mc_distributions.xlsx", "dl_dist")

    with tabs[1]:
        drivers = ["Growth","Exit Multiple","Interest","Gross Margin"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes = axes.flatten()
        for i, col in enumerate(drivers):
            x = sample[col].values; y = sample["IRR"].values * 100
            axes[i].hexbin(x, y, gridsize=50, cmap="Blues",
                           mincnt=1, linewidths=0.1)
            m_fit, b_fit = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            axes[i].plot(xr, m_fit*xr+b_fit, color=A2, lw=1.2, linestyle="--")
            corr = np.corrcoef(x, y)[0, 1]
            axes[i].text(0.04, 0.92, f"r = {corr:.2f}",
                         transform=axes[i].transAxes, fontsize=9, color=A2,
                         bbox=dict(boxstyle="round,pad=0.2",
                                   fc="#0e0e1c", ec="#2a2a42"))
            axes[i].set_xlabel(col); axes[i].set_ylabel("IRR (%)")
            axes[i].set_title(f"IRR vs {col}"); axes[i].grid()
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with tabs[2]:
        cl, cr = st.columns(2, gap="medium")
        with cl:
            section_hdr("Correlation matrix")
            corr_cols = ["IRR","MOIC","Growth","Exit Multiple","Interest","Gross Margin"]
            emp = sample[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(emp, annot=True, fmt=".2f", cmap="RdBu_r",
                        vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                        linecolor="#16162a", annot_kws={"size": 9})
            ax.set_title("Empirical correlations")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with cr:
            section_hdr("Driver tornado (Spearman rho)")
            from scipy.stats import spearmanr
            drv = ["Growth","Exit Multiple","Interest","Gross Margin","EBITDA Shock"]
            rhos = []
            for d in drv:
                if d in sample.columns:
                    rho, _ = spearmanr(sample[d], sample["IRR"])
                    rhos.append((d, rho))
            rhos.sort(key=lambda x: abs(x[1]), reverse=True)
            lbls = [r[0] for r in rhos]; vals = [r[1] for r in rhos]
            bcolors_t = [A4 if v > 0 else A3 for v in vals]
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(lbls, vals, color=bcolors_t, alpha=0.8,
                           height=0.5, edgecolor="#2a2a42", lw=0.5)
            ax.axvline(0, color="#2a2a42", lw=1)
            ax.set_xlabel("Spearman rho with IRR")
            ax.set_title("Driver sensitivity"); ax.grid(axis="x")
            for bar, v in zip(bars, vals):
                ax.text(v+0.01*np.sign(v), bar.get_y()+bar.get_height()/2,
                        f"{v:.2f}", va="center", fontsize=9, color="#c4c4d4")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    with tabs[3]:
        section_hdr("IRR heatmap — growth × exit multiple")
        entry_ev_mc = mc_ebitda * mc_emult
        total_debt_mc = entry_ev_mc * s.d_debt_pct / 100
        eq_in_mc = entry_ev_mc - total_debt_mc
        g_vals = np.linspace(
            max(params.growth_mean - 3*params.growth_std, -0.10),
            params.growth_mean + 3*params.growth_std, 8)
        em_vals = np.linspace(
            max(params.exit_mean - 2*params.exit_std, 2.0),
            params.exit_mean + 2*params.exit_std, 7)
        irr_grid = np.zeros((len(em_vals), len(g_vals)))
        em_base = (params.gross_margin_mean - params.opex_pct + params.da_pct)
        for i, em in enumerate(em_vals):
            for j, g in enumerate(g_vals):
                base_r = mc_ebitda / max(em_base, 0.01)
                rev = base_r * (1+g)**int(mc_hold)
                exit_eq = max(rev * em_base * em - total_debt_mc * 0.70, 0.0)
                irr_grid[i, j] = ((exit_eq/eq_in_mc)**(1/int(mc_hold))-1
                                   if eq_in_mc > 0 else 0)
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(irr_grid*100, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=50, interpolation="nearest")
        ax.set_xticks(range(len(g_vals)))
        ax.set_xticklabels([f"{v:.1%}" for v in g_vals], rotation=30, ha="right")
        ax.set_yticks(range(len(em_vals)))
        ax.set_yticklabels([f"{v:.1f}x" for v in em_vals])
        ax.set_xlabel("Revenue growth"); ax.set_ylabel("Exit multiple")
        ax.set_title("IRR heatmap — growth × exit multiple")
        plt.colorbar(im, ax=ax, label="IRR %")
        for i2 in range(len(em_vals)):
            for j2 in range(len(g_vals)):
                v = irr_grid[i2, j2] * 100
                ax.text(j2, i2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="black" if 10<v<40 else "white")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with tabs[4]:
        section_hdr("All four scenarios")
        scenario_results = {}
        for sc in ["recession","base","bull","stagflation"]:
            sp = _get_scenario_params_cfg(sc, params)
            sr = run_vectorized_simulation_full(sp)
            scenario_results[sc] = sr

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        slabels = ["Recession","Base","Bull","Stagflation"]
        irr_data  = [scenario_results[sc].irr*100
                     for sc in ["recession","base","bull","stagflation"]]
        moic_data = [scenario_results[sc].moic
                     for sc in ["recession","base","bull","stagflation"]]
        bcolors_sc = [A3, A1, A4, A5]
        for ax_i, (data, title, unit) in enumerate([
            (irr_data, "IRR by scenario", "%"),
            (moic_data, "MOIC by scenario", "x"),
        ]):
            bp = axes[ax_i].boxplot(
                data, labels=slabels, patch_artist=True,
                medianprops=dict(color="#ffffff", lw=2),
                whiskerprops=dict(color="#5a5a72"),
                capprops=dict(color="#5a5a72"),
                flierprops=dict(marker=".", color="#5a5a72",
                                markersize=1, alpha=0.3),
            )
            for patch, c in zip(bp["boxes"], bcolors_sc):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            axes[ax_i].set_title(title); axes[ax_i].set_ylabel(unit)
            axes[ax_i].grid(axis="y")
            if ax_i == 0:
                axes[ax_i].axhline(target*100, color=A3, lw=1, linestyle=":")
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        comp = []
        for sc in ["recession","base","bull","stagflation"]:
            sr = scenario_results[sc]
            irr_s = sr.irr
            comp.append({
                "Scenario":        sc.capitalize(),
                "Mean IRR":        pf(float(irr_s.mean()) * 100),
                "Median":          pf(float(np.median(irr_s)) * 100),
                "5th pct":         pf(float(np.percentile(irr_s, 5)) * 100),
                "95th pct":        pf(float(np.percentile(irr_s, 95)) * 100),
                f"P(>{pf(target*100)})":pf(float((irr_s > target).mean()) * 100),
                "Wipeout":         pf(float(sr.wipeout_rate) * 100),
            })
        comp_df = pd.DataFrame(comp).set_index("Scenario")
        st.dataframe(comp_df, use_container_width=True)
        dl_btn("Download scenario comparison",
               _df_to_excel(comp_df), "scenario_comparison.xlsx", "dl_sc")

    # Download full MC sample
    dl_sample = sim.df.sample(min(10000, len(sim.df)), random_state=42)
    dl_btn("Download MC sample (10k scenarios)",
           _df_to_excel(dl_sample), "mc_simulation.xlsx", "dl_mc")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    mode = st.session_state.get("mode", "deal")
    page = st.session_state.get("page", 1)

    if mode == "deal":
        if   page == 1: page_deal_inputs()
        elif page == 2: page_debt_cashflow()
        elif page == 3: page_returns()
        elif page == 4: page_summary()
        else:           page_deal_inputs()
    elif mode == "mc":
        page_monte_carlo()
    elif mode == "backtest":
        import importlib.util, os
        _s = importlib.util.spec_from_file_location(
            "backtesting",
            os.path.join(os.path.dirname(__file__), "pages", "backtesting.py"),
        )
        _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
        _m.render_backtesting()
    elif mode == "forecast":
        import importlib.util, os
        _s = importlib.util.spec_from_file_location(
            "forecasting",
            os.path.join(os.path.dirname(__file__), "pages", "forecasting.py"),
        )
        _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
        _m.render_forecasting()
    elif mode == "settings":
        from pages.settings import render_settings
        render_settings()
    else:
        page_deal_inputs()

if __name__ == "__main__":
    main()