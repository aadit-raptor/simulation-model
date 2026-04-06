# """
# pages/backtesting.py
# --------------------
# Backtesting module: compare our model's predictions against actual
# historical deal results.

# How it works:
#     1. User selects a pre-loaded famous deal OR enters their own deal
#     2. Entry assumptions (what a PE firm would have known at t=0) are input
#     3. Model runs our LBO engine on those entry assumptions
#     4. User enters what ACTUALLY happened year by year
#     5. We compare predicted vs actual on every line
#     6. We show where the model was right, where it was wrong, and why
#     7. We overlay actual IRR on the predicted IRR distribution

# Pre-loaded deals:
#     - Burger King (3G Capital, 2010)
#     - Hilton Hotels (Blackstone, 2007)
#     - Dell (Silver Lake, 2013)
#     - Freescale Semiconductor (Consortium, 2006)
# """

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from simulation.vectorized_simulation import run_vectorized_simulation_full, SimulationParams

# ─── Pre-loaded deal data ─────────────────────────────────────────────────

PRELOADED_DEALS = {
    "Burger King (3G Capital, 2010)": {
        "description": "3G Capital acquired BK in 2010 for $4.0B enterprise value. "
                       "Classic operational turnaround — franchise refranchising, cost cuts, "
                       "and international expansion drove EBITDA nearly doubling over 5 years.",
        "entry": {
            "entry_ebitda": 445.0,
            "entry_multiple": 8.75,
            "exit_multiple": 8.8,
            "holding_period": 5,
            "debt_pct": 61.0,
            "senior_pct": 79.0,
            "base_rate": 6.82,
            "mezz_spread": 3.37,
            "revenue_growth": 2.9,
            "gross_margin": 35.5,
            "opex_pct": 22.2,
            "da_pct": 4.1,
            "tax_rate": 33.8,
            "capex_pct": 7.1,
            "nwc_pct": 1.1,
        },
        "actual_years": [2011, 2012, 2013, 2014, 2015],
        "actual": {
            "revenue":   [2574, 2102, 1884, 1910, 2024],
            "ebitda":    [447,  448,  476,  558,  672],
            "net_income":[89,   104,  131,  191,  273],
            "fcf":       [-16,  18,   54,   113,  191],
            "total_debt":[2644, 2626, 2572, 2459, 2268],
        },
        "actual_exit": {
            "exit_ev": 5914.0,
            "net_debt_at_exit": 2150.0,
            "sponsor_equity_entry": 1560.0,
            "moic": 2.39,
            "irr": 19.0,
        },
        "sector": "QSR / Consumer",
        "geography": "Global",
        "outcome": "SUCCESS",
    },
    "Hilton Hotels (Blackstone, 2007)": {
        "description": "Blackstone acquired Hilton for $26B in 2007 — right before the "
                       "financial crisis. Despite severe distress in 2008-09, Blackstone "
                       "restructured, expanded internationally, and exited via IPO in 2013 "
                       "for a record ~$14B profit.",
        "entry": {
            "entry_ebitda": 1400.0,
            "entry_multiple": 18.5,
            "exit_multiple": 16.0,
            "holding_period": 6,
            "debt_pct": 80.0,
            "senior_pct": 75.0,
            "base_rate": 7.5,
            "mezz_spread": 4.0,
            "revenue_growth": 3.0,
            "gross_margin": 38.0,
            "opex_pct": 24.0,
            "da_pct": 5.5,
            "tax_rate": 32.0,
            "capex_pct": 5.0,
            "nwc_pct": 1.5,
        },
        "actual_years": [2008, 2009, 2010, 2011, 2012, 2013],
        "actual": {
            "revenue":   [8162, 7424, 7683, 8783, 9276, 9738],
            "ebitda":    [1270, 1010, 1190, 1520, 1710, 2010],
            "net_income":[-475, -552, -138,  124,  352,  415],
            "fcf":       [-820, -480, 120,   450,  620,  890],
            "total_debt":[20500,20100,19800,19200,18500,17800],
        },
        "actual_exit": {
            "exit_ev": 32000.0,
            "net_debt_at_exit": 17500.0,
            "sponsor_equity_entry": 5200.0,
            "moic": 2.77,
            "irr": 17.2,
        },
        "sector": "Hospitality / Real Estate",
        "geography": "Global",
        "outcome": "SUCCESS",
    },
    "Dell (Silver Lake, 2013)": {
        "description": "Silver Lake and Michael Dell took Dell private for $24.9B "
                       "to restructure from a PC maker to an enterprise IT solutions provider, "
                       "away from public market short-termism. Returned public via VMware "
                       "tracking stock and then direct listing in 2018.",
        "entry": {
            "entry_ebitda": 3700.0,
            "entry_multiple": 6.7,
            "exit_multiple": 7.2,
            "holding_period": 5,
            "debt_pct": 70.0,
            "senior_pct": 78.0,
            "base_rate": 5.5,
            "mezz_spread": 4.5,
            "revenue_growth": -2.0,
            "gross_margin": 22.0,
            "opex_pct": 15.5,
            "da_pct": 3.5,
            "tax_rate": 28.0,
            "capex_pct": 2.5,
            "nwc_pct": 1.0,
        },
        "actual_years": [2014, 2015, 2016, 2017, 2018],
        "actual": {
            "revenue":   [57400, 54000, 50900, 61642, 78660],
            "ebitda":    [3500,  3200,  3050,  5800,  9200],
            "net_income":[-1800, -1100,  -750,  2200,  3000],
            "fcf":       [1200,  2100,   2800,  4200,  5800],
            "total_debt":[15000, 14500,  14200, 46000, 52000],
        },
        "actual_exit": {
            "exit_ev": 70000.0,
            "net_debt_at_exit": 52000.0,
            "sponsor_equity_entry": 7400.0,
            "moic": 2.43,
            "irr": 19.5,
        },
        "sector": "Technology / Enterprise IT",
        "geography": "Global",
        "outcome": "SUCCESS",
    },
    "Freescale Semiconductor (Consortium, 2006)": {
        "description": "Blackstone, Carlyle, TPG, and Permira acquired Freescale for $17.6B "
                       "in 2006 — one of the largest tech LBOs ever at the time. "
                       "The financial crisis decimated semiconductor demand. "
                       "Freescale filed for bankruptcy in 2009.",
        "entry": {
            "entry_ebitda": 1200.0,
            "entry_multiple": 14.7,
            "exit_multiple": 9.0,
            "holding_period": 5,
            "debt_pct": 85.0,
            "senior_pct": 70.0,
            "base_rate": 7.8,
            "mezz_spread": 5.0,
            "revenue_growth": 4.0,
            "gross_margin": 52.0,
            "opex_pct": 35.0,
            "da_pct": 6.5,
            "tax_rate": 30.0,
            "capex_pct": 6.0,
            "nwc_pct": 2.0,
        },
        "actual_years": [2007, 2008, 2009, 2010, 2011],
        "actual": {
            "revenue":   [5843, 5040, 3500, 4440, 4570],
            "ebitda":    [1050, 680,  120,  650,  720],
            "net_income":[-980, -1850,-2400,-280,  80],
            "fcf":       [-200, -600, -900, 100,  180],
            "total_debt":[16200,16800,18000,7200, 6800],
        },
        "actual_exit": {
            "exit_ev": 6500.0,
            "net_debt_at_exit": 6200.0,
            "sponsor_equity_entry": 2650.0,
            "moic": 0.11,
            "irr": -33.0,
        },
        "sector": "Semiconductors / Technology",
        "geography": "Global",
        "outcome": "DISTRESSED",
    },
    "Custom deal (enter manually)": {
        "description": "",
        "entry": {
            "entry_ebitda": 100.0, "entry_multiple": 10.0,
            "exit_multiple": 10.0, "holding_period": 5,
            "debt_pct": 60.0, "senior_pct": 70.0,
            "base_rate": 6.5, "mezz_spread": 4.0,
            "revenue_growth": 5.0, "gross_margin": 40.0,
            "opex_pct": 18.0, "da_pct": 4.0,
            "tax_rate": 25.0, "capex_pct": 4.0, "nwc_pct": 1.0,
        },
        "actual_years": [1, 2, 3, 4, 5],
        "actual": {
            "revenue":   [0, 0, 0, 0, 0],
            "ebitda":    [0, 0, 0, 0, 0],
            "net_income":[0, 0, 0, 0, 0],
            "fcf":       [0, 0, 0, 0, 0],
            "total_debt":[0, 0, 0, 0, 0],
        },
        "actual_exit": {
            "exit_ev": 0.0, "net_debt_at_exit": 0.0,
            "sponsor_equity_entry": 0.0, "moic": 0.0, "irr": 0.0,
        },
        "sector": "", "geography": "", "outcome": "UNKNOWN",
    },
}

# ─── Colours ─────────────────────────────────────────────────────────────

A_PRED  = "#6060c0"   # predicted — indigo
A_ACT   = "#40c080"   # actual — green
A_MISS  = "#c06060"   # miss — red
A_WARN  = "#c0a040"   # warning — amber
BG      = "#0e0e1c"
BG2     = "#06060c"

plt.rcParams.update({
    "figure.facecolor": BG2, "axes.facecolor": BG,
    "axes.edgecolor": "#2a2a42", "axes.labelcolor": "#8888a4",
    "text.color": "#c4c4d4", "xtick.color": "#5a5a72", "ytick.color": "#5a5a72",
    "grid.color": "#16162a", "grid.linewidth": 0.5,
    "font.family": "monospace", "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": "#c4c4d4",
    "legend.facecolor": BG, "legend.edgecolor": "#2a2a42", "legend.fontsize": 8,
})


# ─── Helpers ─────────────────────────────────────────────────────────────

def _fs():
    """Get current font size multiplier from session state."""
    return st.session_state.get("font_scale", 1.0)

def _label(text):
    sz = int(14 * _fs())
    st.markdown(
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:{sz}px;'
        f'color:#5a5a72;letter-spacing:0.06em">{text}</span>',
        unsafe_allow_html=True,
    )

def _section(text, color="#85b7eb"):
    sz = int(11 * _fs())
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{sz}px;'
        f'font-weight:500;letter-spacing:0.10em;text-transform:uppercase;'
        f'color:{color};border-bottom:1px solid #16162a;padding-bottom:6px;'
        f'margin:14px 0 10px">◈ {text}</div>',
        unsafe_allow_html=True,
    )

def _num(label, value, key, min_val=None, max_val=None, step=1.0):
    sz = int(12 * _fs())
    st.markdown(
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:{sz}px;'
        f'color:#5a5a72">{label}</span>',
        unsafe_allow_html=True,
    )
    kw = {"value": float(value), "step": float(step), "label_visibility": "collapsed"}
    if min_val is not None:
        kw["min_value"] = float(min_val)
    if max_val is not None:
        kw["max_value"] = float(max_val)
    return st.number_input(label, key=key, **kw)

def pct(v):  return f"{v:.1f}%"
def xf(v):   return f"{v:.2f}x"
def mf(v):   return f"${v:,.0f}M"


# ─── Run LBO simulation for predicted distribution ────────────────────────

def _run_prediction_sim(entry, n=30000):
    """Run MC simulation using entry assumptions to get predicted IRR distribution."""
    params = SimulationParams(
        n=n,
        entry_ebitda=entry["entry_ebitda"],
        entry_multiple=entry["entry_multiple"],
        holding_period=entry["holding_period"],
        growth_mean=entry["revenue_growth"] / 100,
        growth_std=0.04,
        exit_mean=entry["exit_multiple"],
        exit_std=1.5,
        interest_mean=entry["base_rate"] / 100,
        interest_std=0.015,
        gross_margin_mean=entry["gross_margin"] / 100,
        gross_margin_std=0.03,
        opex_pct=entry["opex_pct"] / 100,
        da_pct=entry["da_pct"] / 100,
        tax_rate=entry["tax_rate"] / 100,
        capex_pct=entry["capex_pct"] / 100,
        nwc_pct=entry["nwc_pct"] / 100,
        debt_pct=entry["debt_pct"] / 100,
        senior_pct=entry["senior_pct"] / 100,
        mezz_spread=entry["mezz_spread"] / 100,
        n_interest_passes=2,
    )
    sim = run_vectorized_simulation_full(params, seed=42)
    return sim.df["IRR"].values


# ─── Compute predicted EBITDA path ───────────────────────────────────────

def _predicted_ebitda(entry):
    """Compute deterministic predicted EBITDA path using entry assumptions."""
    ebitda_margin = entry["gross_margin"] / 100 - entry["opex_pct"] / 100 + entry["da_pct"] / 100
    ebitda_margin = max(ebitda_margin, 0.01)
    base_rev = entry["entry_ebitda"] / ebitda_margin
    result = []
    rev = base_rev
    for _ in range(entry["holding_period"]):
        rev *= (1 + entry["revenue_growth"] / 100)
        result.append(round(rev * ebitda_margin, 1))
    return result


# ─── Main render function ─────────────────────────────────────────────────

def render_backtesting():
    fs = _fs()

    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{int(15*fs)}px;'
        f'font-weight:500;color:#e0e0f0;letter-spacing:0.08em;'
        f'border-bottom:1px solid #16162a;padding-bottom:12px;margin-bottom:18px">'
        f'BACKTESTING — Predicted vs Actual Deal Results</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{int(10*fs)}px;'
        f'color:#5a5a72;margin-bottom:18px">'
        f'Select a famous historical deal or enter your own. The model runs our LBO engine '
        f'on the entry assumptions (as if you were analysing the deal at close), '
        f'then compares the predicted results against what actually happened.</div>',
        unsafe_allow_html=True,
    )

    # ── Deal selection ────────────────────────────────────────────────────
    _section("Step 1 — Select deal", "#85b7eb")

    deal_name = st.selectbox(
        "Select deal",
        list(PRELOADED_DEALS.keys()),
        label_visibility="collapsed",
        key="bt_deal_select",
    )
    deal = PRELOADED_DEALS[deal_name]

    if deal["description"]:
        outcome_color = {"SUCCESS": "#40c080", "DISTRESSED": "#c06060", "UNKNOWN": "#888888"}.get(
            deal["outcome"], "#888"
        )
        st.markdown(
            f'<div style="background:#0e0e1c;border:0.5px solid #1e1e30;border-left:3px solid '
            f'{outcome_color};border-radius:6px;padding:10px 14px;margin:8px 0;'
            f'font-family:IBM Plex Mono,monospace;font-size:{int(10*fs)}px;color:#8888a4">'
            f'<span style="color:{outcome_color};font-weight:500">{deal["outcome"]}</span>'
            f' &nbsp;·&nbsp; {deal["sector"]} &nbsp;·&nbsp; {deal["geography"]}<br>'
            f'<span style="color:#c4c4d4">{deal["description"]}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Entry assumptions ─────────────────────────────────────────────────
    _section("Step 2 — Entry assumptions (as known at deal close)", "#5dcaa5")
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{int(9*fs)}px;'
        f'color:#44445a;margin-bottom:8px">These are what the PE firm would have modelled '
        f'at time of acquisition. Pre-filled for famous deals — edit if needed.</div>',
        unsafe_allow_html=True,
    )

    e = deal["entry"]
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        _label("Entry EBITDA ($M)")
        entry_ebitda = st.number_input("entry_ebitda", value=e["entry_ebitda"],
            min_value=1.0, step=5.0, label_visibility="collapsed", key="bt_entry_ebitda")
        _label("Entry multiple (x)")
        entry_mult = st.number_input("entry_mult", value=e["entry_multiple"],
            min_value=1.0, step=0.5, label_visibility="collapsed", key="bt_entry_mult")
        _label("Exit multiple (x)")
        exit_mult = st.number_input("exit_mult", value=e["exit_multiple"],
            min_value=1.0, step=0.5, label_visibility="collapsed", key="bt_exit_mult")
        _label("Holding period (yrs)")
        hold = int(st.number_input("hold", value=float(e["holding_period"]),
            min_value=1.0, max_value=15.0, step=1.0, label_visibility="collapsed", key="bt_hold"))
    with c2:
        _label("Debt / EV (%)")
        debt_pct = st.number_input("debt_pct", value=e["debt_pct"],
            min_value=0.0, max_value=99.0, step=1.0, label_visibility="collapsed", key="bt_debt_pct")
        _label("Senior / debt (%)")
        senior_pct = st.number_input("senior_pct", value=e["senior_pct"],
            min_value=1.0, max_value=100.0, step=1.0, label_visibility="collapsed", key="bt_senior_pct")
        _label("Base rate (%)")
        base_rate = st.number_input("base_rate", value=e["base_rate"],
            min_value=0.0, step=0.25, label_visibility="collapsed", key="bt_base_rate")
        _label("Mezz spread (%)")
        mezz_spread = st.number_input("mezz_spread", value=e["mezz_spread"],
            min_value=0.0, step=0.25, label_visibility="collapsed", key="bt_mezz_spread")
    with c3:
        _label("Revenue growth (%)")
        rev_growth = st.number_input("rev_growth", value=e["revenue_growth"],
            step=0.5, label_visibility="collapsed", key="bt_rev_growth")
        _label("Gross margin (%)")
        gross_margin = st.number_input("gross_margin", value=e["gross_margin"],
            min_value=1.0, max_value=99.0, step=1.0, label_visibility="collapsed", key="bt_gross_margin")
        _label("OpEx / revenue (%)")
        opex_pct = st.number_input("opex_pct", value=e["opex_pct"],
            min_value=0.0, step=1.0, label_visibility="collapsed", key="bt_opex_pct")
    with c4:
        _label("D&A / revenue (%)")
        da_pct = st.number_input("da_pct", value=e["da_pct"],
            min_value=0.0, step=0.5, label_visibility="collapsed", key="bt_da_pct")
        _label("Tax rate (%)")
        tax_rate = st.number_input("tax_rate", value=e["tax_rate"],
            min_value=0.0, step=1.0, label_visibility="collapsed", key="bt_tax_rate")
        _label("Capex / revenue (%)")
        capex_pct = st.number_input("capex_pct", value=e["capex_pct"],
            min_value=0.0, step=0.5, label_visibility="collapsed", key="bt_capex_pct")
        _label("NWC / revenue (%)")
        nwc_pct = st.number_input("nwc_pct", value=e["nwc_pct"],
            min_value=0.0, step=0.25, label_visibility="collapsed", key="bt_nwc_pct")

    entry_assumptions = {
        "entry_ebitda": entry_ebitda, "entry_multiple": entry_mult,
        "exit_multiple": exit_mult, "holding_period": hold,
        "debt_pct": debt_pct, "senior_pct": senior_pct,
        "base_rate": base_rate, "mezz_spread": mezz_spread,
        "revenue_growth": rev_growth, "gross_margin": gross_margin,
        "opex_pct": opex_pct, "da_pct": da_pct,
        "tax_rate": tax_rate, "capex_pct": capex_pct, "nwc_pct": nwc_pct,
    }

    # ── Actual results input ──────────────────────────────────────────────
    _section("Step 3 — Actual results (what really happened)", "#ef9f27")
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{int(9*fs)}px;'
        f'color:#44445a;margin-bottom:8px">Enter the actual financial results '
        f'year by year from the deal\'s financials / public disclosures.</div>',
        unsafe_allow_html=True,
    )

    actual_data = deal["actual"]
    years = deal["actual_years"][:hold]
    actual_results = {}

    metrics = ["revenue", "ebitda", "net_income", "fcf", "total_debt"]
    metric_labels = ["Revenue ($M)", "EBITDA ($M)", "Net income ($M)", "Levered FCF ($M)", "Total debt ($M)"]

    for mi, (metric, label) in enumerate(zip(metrics, metric_labels)):
        cols = st.columns(hold + 1, gap="small")
        with cols[0]:
            sz = int(10 * fs)
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:{sz}px;'
                f'color:#8888a4;padding-top:32px">{label}</div>',
                unsafe_allow_html=True,
            )
        vals = []
        for yi in range(hold):
            with cols[yi + 1]:
                default_val = float(actual_data[metric][yi]) if yi < len(actual_data[metric]) else 0.0
                if mi == 0:
                    st.markdown(
                        f'<span style="font-family:IBM Plex Mono,monospace;font-size:{int(9*fs)}px;'
                        f'color:#44445a">{years[yi] if yi < len(years) else f"Yr {yi+1}"}</span>',
                        unsafe_allow_html=True,
                    )
                v = st.number_input(f"{metric}_{yi}", value=default_val,
                    step=1.0, label_visibility="collapsed", key=f"bt_{metric}_{yi}")
                vals.append(v)
        actual_results[metric] = vals

    # Actual exit
    st.markdown("---")
    act_ex = deal["actual_exit"]
    ex_c1, ex_c2, ex_c3, ex_c4, ex_c5 = st.columns(5, gap="small")
    with ex_c1:
        _label("Actual exit EV ($M)")
        act_exit_ev = st.number_input("act_exit_ev", value=act_ex["exit_ev"],
            min_value=0.0, step=50.0, label_visibility="collapsed", key="bt_act_exit_ev")
    with ex_c2:
        _label("Net debt at exit ($M)")
        act_net_debt = st.number_input("act_net_debt", value=act_ex["net_debt_at_exit"],
            min_value=0.0, step=50.0, label_visibility="collapsed", key="bt_act_net_debt")
    with ex_c3:
        _label("Sponsor equity at entry ($M)")
        act_eq_entry = st.number_input("act_eq_entry", value=act_ex["sponsor_equity_entry"],
            min_value=0.0, step=50.0, label_visibility="collapsed", key="bt_act_eq_entry")
    with ex_c4:
        _label("Actual MOIC (x)")
        act_moic = st.number_input("act_moic", value=act_ex["moic"],
            min_value=0.0, step=0.05, label_visibility="collapsed", key="bt_act_moic")
    with ex_c5:
        _label("Actual IRR (%)")
        act_irr = st.number_input("act_irr", value=act_ex["irr"],
            step=0.5, label_visibility="collapsed", key="bt_act_irr")

    # ── Run button ────────────────────────────────────────────────────────
    st.markdown("---")
    run_bt = st.button("▶  RUN BACKTEST", type="primary",
                       use_container_width=False, key="bt_run")

    if not run_bt:
        return

    # ── Compute predicted EBITDA path ─────────────────────────────────────
    with st.spinner("Running simulation..."):
        pred_ebitda = _predicted_ebitda(entry_assumptions)[:hold]
        pred_irr_dist = _run_prediction_sim(entry_assumptions, n=30000)

    entry_ev = entry_ebitda * entry_mult
    pred_exit_ev = pred_ebitda[-1] * exit_mult if pred_ebitda else 0
    pred_net_debt = entry_ev * debt_pct / 100 * 0.75
    pred_equity_entry = entry_ev * (1 - debt_pct / 100)
    pred_exit_equity = max(pred_exit_ev - pred_net_debt, 0)
    pred_moic = pred_exit_equity / pred_equity_entry if pred_equity_entry > 0 else 0
    pred_irr_mean = float(np.mean(pred_irr_dist)) * 100

    # ── Results header ────────────────────────────────────────────────────
    _section("Backtest results", "#40c080")

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    def _metric_card(col, label, pred_val, act_val, fmt=pct, good_if="higher"):
        delta = act_val - pred_val
        good = delta >= 0 if good_if == "higher" else delta <= 0
        col.metric(label,
                   fmt(act_val),
                   f"{'▲' if delta >= 0 else '▼'} {fmt(abs(delta))} vs predicted",
                   delta_color="normal" if good else "inverse")

    r1.metric("Predicted IRR (mean)", pct(pred_irr_mean))
    r2.metric("Actual IRR", pct(act_irr),
              f"{'▲' if act_irr >= pred_irr_mean else '▼'} {pct(abs(act_irr - pred_irr_mean))} vs predicted",
              delta_color="normal" if act_irr >= pred_irr_mean else "inverse")
    r3.metric("Predicted MOIC (base)", xf(pred_moic))
    r4.metric("Actual MOIC", xf(act_moic),
              f"{'▲' if act_moic >= pred_moic else '▼'} {xf(abs(act_moic - pred_moic))} vs predicted",
              delta_color="normal" if act_moic >= pred_moic else "inverse")
    r5.metric("Predicted exit EBITDA", mf(pred_ebitda[-1]) if pred_ebitda else "—")
    r6.metric("Actual exit EBITDA", mf(actual_results["ebitda"][-1]),
              f"{'▲' if actual_results['ebitda'][-1] >= pred_ebitda[-1] else '▼'}"
              f" {mf(abs(actual_results['ebitda'][-1] - pred_ebitda[-1]))}",
              delta_color="normal" if actual_results["ebitda"][-1] >= pred_ebitda[-1] else "inverse")

    # ── Charts ────────────────────────────────────────────────────────────
    tabs = st.tabs(["EBITDA comparison", "IRR distribution", "Full comparison table", "Error attribution"])

    yr_labels = [str(y) for y in years[:hold]]

    # Tab 1: EBITDA predicted vs actual
    with tabs[0]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        x = np.arange(hold)
        w = 0.35
        axes[0].bar(x - w/2, pred_ebitda, width=w, color=A_PRED, alpha=0.8, label="Predicted")
        axes[0].bar(x + w/2, actual_results["ebitda"], width=w, color=A_ACT, alpha=0.8, label="Actual")
        axes[0].set_xticks(x); axes[0].set_xticklabels(yr_labels)
        axes[0].set_ylabel("$M"); axes[0].set_title("EBITDA: Predicted vs Actual")
        axes[0].legend(); axes[0].grid(axis="y")

        # Variance
        variances = [actual_results["ebitda"][i] - pred_ebitda[i] for i in range(hold)]
        bar_colors = [A_ACT if v >= 0 else A_MISS for v in variances]
        axes[1].bar(yr_labels, variances, color=bar_colors, alpha=0.85, width=0.5)
        axes[1].axhline(0, color="#2a2a42", lw=1)
        axes[1].set_ylabel("$M"); axes[1].set_title("EBITDA Variance (Actual - Predicted)")
        axes[1].grid(axis="y")
        for i, v in enumerate(variances):
            axes[1].text(i, v + (2 if v >= 0 else -8), f"${v:+.0f}M",
                         ha="center", fontsize=8, color=A_ACT if v >= 0 else A_MISS)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Tab 2: IRR distribution with actual overlaid
    with tabs[1]:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        irr_pct_arr = pred_irr_dist * 100
        ax.hist(irr_pct_arr, bins=80, color=A_PRED, alpha=0.65,
                edgecolor="none", density=True, label="Predicted distribution")
        ax.axvline(np.mean(irr_pct_arr), color=A_PRED, lw=1.5,
                   linestyle="--", label=f"Predicted mean {np.mean(irr_pct_arr):.1f}%")
        ax.axvline(np.percentile(irr_pct_arr, 5), color="#888", lw=1,
                   linestyle=":", label=f"5th pct {np.percentile(irr_pct_arr,5):.1f}%")
        ax.axvline(np.percentile(irr_pct_arr, 95), color="#888", lw=1,
                   linestyle=":", label=f"95th pct {np.percentile(irr_pct_arr,95):.1f}%")
        # Actual IRR
        ax.axvline(act_irr, color=A_ACT, lw=3,
                   label=f"Actual IRR: {act_irr:.1f}%")
        pct_rank = float(np.mean(irr_pct_arr < act_irr)) * 100
        ax.text(act_irr + 0.5, ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] > 0 else 0.01,
                f"Actual IRR\n{act_irr:.1f}%\n({pct_rank:.0f}th pct)",
                color=A_ACT, fontsize=9, va="top")
        ax.set_xlabel("IRR (%)")
        ax.set_title("Predicted IRR Distribution — Where did the Actual IRR land?")
        ax.legend(); ax.grid(axis="y")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        p_rank = float(np.mean(pred_irr_dist * 100 < act_irr)) * 100
        if p_rank >= 50:
            st.success(f"Actual IRR of {act_irr:.1f}% landed at the **{p_rank:.0f}th percentile** of the predicted distribution — the deal outperformed the median expectation.")
        else:
            st.warning(f"Actual IRR of {act_irr:.1f}% landed at the **{p_rank:.0f}th percentile** of the predicted distribution — the deal underperformed the median expectation.")

    # Tab 3: Full comparison table
    with tabs[2]:
        comparison_rows = []
        for yi in range(hold):
            row = {
                "Year": yr_labels[yi],
                "Revenue (pred)": f"N/A",
                "Revenue (actual)": mf(actual_results["revenue"][yi]),
                "EBITDA (pred)": mf(pred_ebitda[yi]),
                "EBITDA (actual)": mf(actual_results["ebitda"][yi]),
                "EBITDA variance": f"${actual_results['ebitda'][yi] - pred_ebitda[yi]:+.0f}M",
                "FCF (actual)": mf(actual_results["fcf"][yi]),
                "Debt (actual)": mf(actual_results["total_debt"][yi]),
            }
            comparison_rows.append(row)
        df_comp = pd.DataFrame(comparison_rows).set_index("Year")
        st.dataframe(df_comp, use_container_width=True)

        # Summary row
        summary_data = {
            "Metric": ["MOIC", "IRR (%)", "Exit EBITDA ($M)", "Entry equity ($M)", "Exit equity ($M)"],
            "Predicted": [xf(pred_moic), pct(pred_irr_mean),
                          mf(pred_ebitda[-1]), mf(pred_equity_entry), mf(pred_exit_equity)],
            "Actual": [xf(act_moic), pct(act_irr),
                       mf(actual_results["ebitda"][-1]),
                       mf(act_eq_entry),
                       mf(act_exit_ev - act_net_debt)],
            "Variance": [
                xf(act_moic - pred_moic),
                pct(act_irr - pred_irr_mean),
                mf(actual_results["ebitda"][-1] - pred_ebitda[-1]),
                "—", "—",
            ],
        }
        st.markdown("**Returns summary**")
        st.dataframe(pd.DataFrame(summary_data).set_index("Metric"), use_container_width=True)

    # Tab 4: Error attribution
    with tabs[3]:
        ebitda_miss_total = sum(actual_results["ebitda"]) - sum(pred_ebitda)
        avg_ebitda_pred = np.mean(pred_ebitda) if pred_ebitda else 1
        ebitda_miss_pct = ebitda_miss_total / (avg_ebitda_pred * hold) * 100

        actual_ebitda_margin = [
            actual_results["ebitda"][i] / actual_results["revenue"][i] * 100
            if actual_results["revenue"][i] > 0 else 0
            for i in range(hold)
        ]
        pred_ebitda_margin = (gross_margin - opex_pct + da_pct)

        st.markdown("### What caused the gap between predicted and actual?")
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ["EBITDA\ngrowth miss", "Margin\ndifference", "FCF\nconversion", "Debt\npaydown"]
            ebitda_growth_miss = actual_results["ebitda"][-1] - pred_ebitda[-1]
            margin_diff = (np.mean(actual_ebitda_margin) - pred_ebitda_margin) * (sum(actual_results["revenue"]) / hold) / 100
            fcf_diff = sum(actual_results["fcf"]) - sum(pred_ebitda) * 0.3
            debt_diff = actual_results["total_debt"][-1] - entry_ev * debt_pct / 100 * 0.75
            values = [ebitda_growth_miss, margin_diff, fcf_diff * 0.1, -debt_diff * 0.05]
            bar_c = [A_ACT if v >= 0 else A_MISS for v in values]
            ax.barh(categories, values, color=bar_c, alpha=0.8, height=0.5)
            ax.axvline(0, color="#2a2a42", lw=1)
            ax.set_xlabel("$M impact on exit equity (approximate)")
            ax.set_title("Error attribution — approximate drivers")
            ax.grid(axis="x")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_b:
            st.markdown("**Interpretation guide**")
            insight_rows = [
                ("EBITDA growth",
                 f"Actual exit EBITDA: {mf(actual_results['ebitda'][-1])} vs "
                 f"predicted: {mf(pred_ebitda[-1])}. "
                 f"{'Outperformed' if actual_results['ebitda'][-1] >= pred_ebitda[-1] else 'Underperformed'} "
                 f"by {mf(abs(actual_results['ebitda'][-1] - pred_ebitda[-1]))}"),
                ("EBITDA margin",
                 f"Actual avg margin: {np.mean(actual_ebitda_margin):.1f}% vs "
                 f"predicted: {pred_ebitda_margin:.1f}%"),
                ("IRR outcome",
                 f"Actual IRR {act_irr:.1f}% vs predicted mean {pred_irr_mean:.1f}%. "
                 f"Landed at {float(np.mean(pred_irr_dist*100 < act_irr))*100:.0f}th percentile."),
                ("Model accuracy",
                 f"MOIC error: {abs(act_moic - pred_moic):.2f}x "
                 f"({'over-estimated' if pred_moic > act_moic else 'under-estimated'}). "
                 f"For macro shocks (2008 GFC, COVID) the model will always under-estimate downside."),
            ]
            for title, body in insight_rows:
                st.markdown(
                    f'<div style="background:#0e0e1c;border:0.5px solid #1e1e30;'
                    f'border-radius:6px;padding:8px 12px;margin-bottom:6px;'
                    f'font-family:IBM Plex Mono,monospace">'
                    f'<div style="font-size:{int(9*fs)}px;color:#85b7eb;margin-bottom:3px">{title}</div>'
                    f'<div style="font-size:{int(9*fs)}px;color:#8888a4">{body}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )