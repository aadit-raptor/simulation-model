"""
pages/forecasting.py
--------------------
Company Forecasting — two modes:
  1. Financial Modelling  — enter historical data → DCF + traditional projection
                            with Monte Carlo simulation overlay
  2. LBO Simulation       — optional LBO return analysis on same company
Excel download for all outputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import io
from scipy import stats

# ── Colour palette ────────────────────────────────────────────────────────────
BG   = "#05050c"; BG2 = "#0e0e1c"; ACC1 = "#6060c0"; ACC2 = "#40a0c0"
ACC3 = "#c06060"; ACC4 = "#40c080"; ACC5 = "#c0a040"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": BG2,
    "axes.edgecolor": "#2a2a42", "axes.labelcolor": "#8888a4",
    "text.color": "#c4c4d4",  "xtick.color": "#5a5a72", "ytick.color": "#5a5a72",
    "grid.color": "#16162a",  "grid.linewidth": 0.5,
    "font.family": "monospace", "font.size": 10,
    "axes.titlesize": 11,     "axes.titlecolor": "#c4c4d4",
    "legend.facecolor": BG2,  "legend.edgecolor": "#2a2a42", "legend.fontsize": 9,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def _section(title, color="#85b7eb"):
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{color};border-bottom:1px solid #16162a;padding-bottom:6px;'
        f'margin:18px 0 12px">◈ {title}</div>',
        unsafe_allow_html=True,
    )

def _blk(title, color_hex, title_color):
    st.markdown(
        f'<div style="background:#080816;border-left:2px solid {color_hex};'
        f'border-radius:6px;padding:10px 14px 4px;margin-bottom:4px">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:9px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{title_color};margin-bottom:10px">{title}</div></div>',
        unsafe_allow_html=True,
    )

def _lbl(text):
    st.markdown(f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;'
                f'color:#5a5a72;letter-spacing:0.04em;display:block;'
                f'margin-bottom:2px">{text}</span>', unsafe_allow_html=True)

def _to_excel(dfs: dict) -> bytes:
    """Convert dict of {sheet_name: DataFrame} to Excel bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31])
    return buf.getvalue()

def _download_btn(label, data, filename):
    st.download_button(
        label=f"⬇ {label}",
        data=data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ── Monte Carlo engine (standalone, no lbo_engine dependency) ─────────────────
def run_company_simulation(
    base_revenue, growth_mean, growth_std,
    ebitda_margin_mean, ebitda_margin_std,
    capex_pct, da_pct, tax_rate,
    years, n_scenarios=30000,
):
    """Vectorised revenue + EBITDA simulation for a standalone company."""
    rng = np.random.default_rng(42)
    # Correlated draws: growth & margin positively correlated
    L = np.array([[1, 0], [0.5, np.sqrt(1-0.25)]])
    Z = rng.standard_normal((2, n_scenarios))
    corr = L @ Z
    g  = growth_mean/100 + corr[0] * growth_std/100
    em = ebitda_margin_mean/100 + corr[1] * ebitda_margin_std/100
    em = np.clip(em, 0.01, 0.99)

    rev = np.full(n_scenarios, base_revenue)
    rev_paths = [rev.copy()]
    for _ in range(years):
        rev = rev * (1 + g)
        rev_paths.append(rev.copy())

    ebitda = rev * em
    nopat  = ebitda * (1 - tax_rate/100)
    fcf    = nopat + rev * da_pct/100 - rev * capex_pct/100

    return {
        "revenue_final": rev,
        "ebitda_final":  ebitda,
        "fcf_final":     fcf,
        "growth_draws":  g,
        "margin_draws":  em,
        "rev_paths":     np.array(rev_paths),   # (years+1, n_scenarios)
    }

# ── Main render function ──────────────────────────────────────────────────────
def render_forecasting():
    st.markdown(
        '<div class="page-hdr"><div class="page-hdr-title">Company Forecasting</div>'
        '<div class="page-hdr-step">Financial modelling + simulation</div></div>',
        unsafe_allow_html=True,
    )

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Mode",
        ["Financial Modelling & Simulation", "LBO Projection (optional)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if mode == "Financial Modelling & Simulation":
        _render_financial_modelling()
    else:
        _render_lbo_projection()


def _render_financial_modelling():
    """Full financial modelling + simulation mode."""

    # ── STEP 1: Historical data entry ─────────────────────────────────────────
    _section("Step 1 — Historical data", "#85b7eb")
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
        'color:#5a5a72;margin-bottom:12px">'
        'Enter 2–3 years of actual results. All values in $M unless stated.</div>',
        unsafe_allow_html=True,
    )

    n_hist = st.selectbox("Number of historical years", [2, 3], index=1)
    yr_labels = [f"Year -{n_hist - i}" for i in range(n_hist)]

    hist_cols = st.columns(n_hist + 1)
    with hist_cols[0]:
        st.markdown(
            '<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
            'color:#5a5a72;padding-top:28px">Line item</div>',
            unsafe_allow_html=True,
        )
        for item in ["Revenue ($M)", "EBITDA ($M)", "Net income ($M)",
                     "Capex ($M)", "D&A ($M)", "Total debt ($M)"]:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
                f'color:#c4c4d4;padding:6px 0;border-bottom:0.5px solid #16162a">'
                f'{item}</div>',
                unsafe_allow_html=True,
            )

    hist_data = {}
    for j, yr in enumerate(yr_labels):
        with hist_cols[j + 1]:
            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;'
                f'color:#85b7eb;text-align:center;margin-bottom:8px">{yr}</div>',
                unsafe_allow_html=True,
            )
            rev  = st.number_input(" ", value=100.0+j*10, step=5.0, key=f"h_rev_{j}")
            ebit = st.number_input(" ", value=20.0+j*2,  step=1.0, key=f"h_ebit_{j}")
            ni   = st.number_input(" ", value=10.0+j*1,  step=1.0, key=f"h_ni_{j}")
            capx = st.number_input(" ", value=5.0,        step=0.5, key=f"h_capx_{j}")
            da   = st.number_input(" ", value=4.0,        step=0.5, key=f"h_da_{j}")
            debt = st.number_input(" ", value=60.0,       step=5.0, key=f"h_debt_{j}")
            hist_data[yr] = {
                "Revenue": rev, "EBITDA": ebit, "Net income": ni,
                "Capex": capx, "D&A": da, "Total debt": debt,
            }

    # ── Computed historical ratios ─────────────────────────────────────────────
    hist_df = pd.DataFrame(hist_data).T
    hist_df["EBITDA margin"] = hist_df["EBITDA"] / hist_df["Revenue"]
    hist_df["Net margin"]    = hist_df["Net income"] / hist_df["Revenue"]
    hist_df["Capex/Rev"]     = hist_df["Capex"] / hist_df["Revenue"]
    hist_df["D&A/Rev"]       = hist_df["D&A"] / hist_df["Revenue"]
    if len(hist_df) >= 2:
        rev_arr = hist_df["Revenue"].values
        hist_df["Rev growth"] = pd.Series(
            [np.nan] + [(rev_arr[i]/rev_arr[i-1]-1) for i in range(1, len(rev_arr))],
            index=hist_df.index
        )
    else:
        hist_df["Rev growth"] = np.nan

    st.markdown("### Historical metrics (auto-calculated)")
    disp_df = hist_df[["Revenue","EBITDA","EBITDA margin","Net margin",
                        "Rev growth","Capex/Rev","D&A/Rev"]].copy()
    for col in ["EBITDA margin","Net margin","Rev growth","Capex/Rev","D&A/Rev"]:
        disp_df[col] = disp_df[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    for col in ["Revenue","EBITDA"]:
        disp_df[col] = disp_df[col].map(lambda x: f"${x:,.1f}M")
    st.dataframe(disp_df, use_container_width=True)

    # ── STEP 2: Projection assumptions ────────────────────────────────────────
    _section("Step 2 — Projection assumptions", "#5dcaa5")

    # Auto-calibrate from history
    rev_vals = hist_df["Revenue"].values
    if len(rev_vals) >= 2:
        g_hist = [(rev_vals[i]/rev_vals[i-1]-1)*100 for i in range(1, len(rev_vals))]
        g_mean_default = float(np.mean(g_hist))
        g_std_default  = float(np.std(g_hist)) if len(g_hist) > 1 else 3.0
    else:
        g_mean_default, g_std_default = 5.0, 3.0

    em_hist = hist_df["EBITDA margin"].values * 100
    em_mean_default = float(np.mean(em_hist))
    em_std_default  = float(np.std(em_hist)) if len(em_hist) > 1 else 2.0

    capx_pct_def = float(hist_df["Capex/Rev"].mean() * 100)
    da_pct_def   = float(hist_df["D&A/Rev"].mean() * 100)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        _blk("Projection horizon", "#378add", "#85b7eb")
        _lbl("Forecast years"); proj_years = st.number_input(" ", value=5, min_value=1, max_value=10, step=1, key="f_yrs")
        _lbl("Monte Carlo scenarios"); n_sim = st.number_input(" ", value=30000, min_value=1000, max_value=500000, step=5000, key="f_nsim")
        _lbl("Discount rate / WACC (%)"); wacc = st.number_input(" ", value=10.0, step=0.5, key="f_wacc")
        _lbl("Terminal growth rate (%)"); tgr  = st.number_input(" ", value=2.5, step=0.25, key="f_tgr")

    with c2:
        _blk("Growth distribution", "#1d9e75", "#5dcaa5")
        _lbl(f"Growth mean % (hist: {g_mean_default:.1f}%)")
        g_mean = st.number_input(" ", value=round(g_mean_default, 1), step=0.5, key="f_gmean")
        _lbl(f"Growth std dev % (hist: {g_std_default:.1f}%)")
        g_std  = st.number_input(" ", value=max(round(g_std_default, 1), 1.0), min_value=0.1, step=0.5, key="f_gstd")

    with c3:
        _blk("Margin distribution", "#7f77dd", "#afa9ec")
        _lbl(f"EBITDA margin mean % (hist: {em_mean_default:.1f}%)")
        em_mean = st.number_input(" ", value=round(em_mean_default, 1), min_value=1.0, max_value=99.0, step=1.0, key="f_emmean")
        _lbl(f"EBITDA margin std dev % (hist: {em_std_default:.1f}%)")
        em_std  = st.number_input(" ", value=max(round(em_std_default, 1), 1.0), min_value=0.1, step=0.5, key="f_emstd")

    with c4:
        _blk("Operating assumptions", "#ba7517", "#ef9f27")
        _lbl(f"Capex / Revenue % (hist: {capx_pct_def:.1f}%)")
        capx_pct = st.number_input(" ", value=round(capx_pct_def, 1), min_value=0.0, step=0.5, key="f_capx")
        _lbl(f"D&A / Revenue % (hist: {da_pct_def:.1f}%)")
        da_pct   = st.number_input(" ", value=round(da_pct_def, 1), min_value=0.0, step=0.5, key="f_da")
        _lbl("Tax rate (%)"); tax_rate = st.number_input(" ", value=25.0, min_value=0.0, max_value=60.0, step=1.0, key="f_tax")

    # ── Run ────────────────────────────────────────────────────────────────────
    st.markdown("---")
    run_btn = st.button("▶  Run Forecast & Simulation", type="primary", use_container_width=False, key="f_run")

    if not run_btn and "fc_result" not in st.session_state:
        st.info("Fill in historical data and assumptions above, then click Run.")
        return

    if run_btn:
        base_rev = hist_df["Revenue"].iloc[-1]
        with st.spinner(f"Running {n_sim:,} scenarios…"):
            sim = run_company_simulation(
                base_rev, g_mean, g_std, em_mean, em_std,
                capx_pct, da_pct, tax_rate, int(proj_years), int(n_sim)
            )
            # Deterministic base-case projection
            base_rows = []
            r = base_rev
            for yr in range(1, int(proj_years)+1):
                r = r * (1 + g_mean/100)
                ebitda = r * em_mean/100
                da_v   = r * da_pct/100
                capx_v = r * capx_pct/100
                nopat  = ebitda * (1 - tax_rate/100)
                fcf    = nopat + da_v - capx_v
                base_rows.append({
                    "Year": f"Y+{yr}",
                    "Revenue": r,
                    "EBITDA": ebitda,
                    "EBITDA margin": ebitda/r,
                    "D&A": da_v,
                    "Capex": capx_v,
                    "NOPAT": nopat,
                    "FCF": fcf,
                })
            base_proj = pd.DataFrame(base_rows).set_index("Year")

            # DCF terminal value
            last_fcf = base_proj["FCF"].iloc[-1]
            tv = last_fcf * (1 + tgr/100) / (wacc/100 - tgr/100) if wacc > tgr else 0
            pv_fcfs = sum(
                base_proj["FCF"].iloc[i] / (1 + wacc/100)**(i+1)
                for i in range(len(base_proj))
            )
            enterprise_value_dcf = pv_fcfs + tv / (1 + wacc/100)**int(proj_years)

            st.session_state["fc_result"] = {
                "sim": sim, "base_proj": base_proj,
                "ev_dcf": enterprise_value_dcf, "pv_fcfs": pv_fcfs, "tv": tv,
                "params": dict(g_mean=g_mean, g_std=g_std, em_mean=em_mean,
                               em_std=em_std, proj_years=proj_years, wacc=wacc,
                               tgr=tgr, n_sim=n_sim),
                "hist_df": hist_df,
            }

    fc = st.session_state.get("fc_result")
    if not fc:
        return

    sim       = fc["sim"]
    base_proj = fc["base_proj"]
    ev_dcf    = fc["ev_dcf"]
    hist_df_r = fc["hist_df"]
    p         = fc["params"]

    # ── OUTPUTS ───────────────────────────────────────────────────────────────
    _section("Results — Traditional Financial Model", "#85b7eb")

    # Key DCF metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DCF Enterprise Value", f"${ev_dcf:,.0f}M")
    c2.metric("PV of FCFs",           f"${fc['pv_fcfs']:,.0f}M")
    c3.metric("Terminal Value",        f"${fc['tv']:,.0f}M")
    c4.metric("TV as % of EV",         f"{fc['tv']/ev_dcf*100:.0f}%" if ev_dcf > 0 else "—")

    st.markdown("### Base-case projection (deterministic)")
    disp_proj = base_proj.copy()
    for col in ["Revenue","EBITDA","D&A","Capex","NOPAT","FCF"]:
        disp_proj[col] = disp_proj[col].map(lambda x: f"${x:,.1f}M")
    disp_proj["EBITDA margin"] = base_proj["EBITDA margin"].map(lambda x: f"{x:.1%}")
    st.dataframe(disp_proj, use_container_width=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    st.markdown("### Projections with simulation confidence bands")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    proj_years_int = int(p["proj_years"])
    yr_labels_all = [f"Y{i}" for i in range(-len(hist_df_r)+1, proj_years_int+1)]
    hist_rev = list(hist_df_r["Revenue"].values)
    base_rev_proj = list(base_proj["Revenue"].values)

    # Confidence bands from simulation paths
    rev_paths = sim["rev_paths"]  # shape (proj_years+1, n_scenarios)
    p5  = np.percentile(rev_paths, 5,  axis=1)
    p25 = np.percentile(rev_paths, 25, axis=1)
    p50 = np.percentile(rev_paths, 50, axis=1)
    p75 = np.percentile(rev_paths, 75, axis=1)
    p95 = np.percentile(rev_paths, 95, axis=1)
    x_proj = np.arange(proj_years_int + 1)
    x_hist = np.arange(-len(hist_rev)+1, 1)

    # Revenue chart
    axes[0].fill_between(x_proj, p5,  p95, color=ACC1, alpha=0.15, label="5–95%")
    axes[0].fill_between(x_proj, p25, p75, color=ACC1, alpha=0.25, label="25–75%")
    axes[0].plot(x_proj, p50, color=ACC1, lw=2, label="Median")
    axes[0].plot(x_hist, hist_rev, color=ACC2, lw=2, marker="o", ms=5, label="Historical")
    axes[0].set_title("Revenue ($M) — with simulation bands")
    axes[0].set_ylabel("$M"); axes[0].grid(axis="y"); axes[0].legend(fontsize=8)

    # EBITDA margin distribution
    margins = sim["margin_draws"] * 100
    axes[1].hist(margins, bins=60, color=ACC2, alpha=0.75, edgecolor="none", density=True)
    axes[1].axvline(p["em_mean"], color=ACC4, lw=2, linestyle="--",
                    label=f"Mean {p['em_mean']:.1f}%")
    axes[1].axvline(np.percentile(margins, 5),  color="#888", lw=1, linestyle=":")
    axes[1].axvline(np.percentile(margins, 95), color="#888", lw=1, linestyle=":")
    axes[1].set_title("EBITDA margin distribution")
    axes[1].set_xlabel("Margin (%)"); axes[1].grid(axis="y"); axes[1].legend(fontsize=8)

    # FCF distribution
    fcf_vals = sim["fcf_final"]
    axes[2].hist(fcf_vals, bins=60, color=ACC4, alpha=0.75, edgecolor="none", density=True)
    axes[2].axvline(np.mean(fcf_vals),          color=ACC1, lw=2, linestyle="--",
                    label=f"Mean ${np.mean(fcf_vals):,.0f}M")
    axes[2].axvline(0, color=ACC3, lw=1.5, linestyle=":", label="Breakeven")
    axes[2].set_title("Year-N Free Cash Flow distribution")
    axes[2].set_xlabel("FCF ($M)"); axes[2].grid(axis="y"); axes[2].legend(fontsize=8)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    _section("Results — Simulation Analytics", "#afa9ec")

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    # Growth rate distribution
    g_draws = sim["growth_draws"] * 100
    axes2[0].hist(g_draws, bins=60, color=ACC5, alpha=0.75, edgecolor="none", density=True)
    axes2[0].axvline(p["g_mean"], color=ACC4, lw=2, linestyle="--",
                     label=f"Mean {p['g_mean']:.1f}%")
    axes2[0].set_title("Revenue growth distribution"); axes2[0].set_xlabel("Growth (%)")
    axes2[0].grid(axis="y"); axes2[0].legend(fontsize=8)

    # Revenue CDF
    rev_final = sim["revenue_final"]
    sorted_rev = np.sort(rev_final)
    cdf = np.linspace(0, 1, len(sorted_rev))
    axes2[1].plot(sorted_rev, cdf * 100, color=ACC1, lw=2)
    axes2[1].axvline(np.percentile(rev_final, 50), color=ACC2, lw=1.5, linestyle="--",
                     label=f"P50 ${np.percentile(rev_final,50):,.0f}M")
    axes2[1].set_title("Revenue CDF (year N)")
    axes2[1].set_xlabel("Revenue ($M)"); axes2[1].set_ylabel("Cumulative %")
    axes2[1].grid(); axes2[1].legend(fontsize=8)

    # Growth vs FCF scatter
    axes2[2].hexbin(g_draws, fcf_vals, gridsize=40, cmap="Blues",
                    mincnt=1, linewidths=0.1)
    axes2[2].set_title("Revenue growth vs FCF")
    axes2[2].set_xlabel("Growth (%)"); axes2[2].set_ylabel("FCF ($M)")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # ── Summary stats table ────────────────────────────────────────────────────
    st.markdown("### Simulation summary statistics")
    rev_f = sim["revenue_final"]
    ebt_f = sim["ebitda_final"]
    fcf_f = sim["fcf_final"]
    stats_rows = []
    for name, arr in [("Revenue ($M)", rev_f), ("EBITDA ($M)", ebt_f), ("FCF ($M)", fcf_f)]:
        stats_rows.append({
            "Metric": name,
            "Mean":   f"${np.mean(arr):,.1f}M",
            "Median": f"${np.median(arr):,.1f}M",
            "5th pct":f"${np.percentile(arr,5):,.1f}M",
            "25th pct":f"${np.percentile(arr,25):,.1f}M",
            "75th pct":f"${np.percentile(arr,75):,.1f}M",
            "95th pct":f"${np.percentile(arr,95):,.1f}M",
        })
    st.dataframe(pd.DataFrame(stats_rows).set_index("Metric"), use_container_width=True)

    # ── Excel download ─────────────────────────────────────────────────────────
    _section("Download results", "#ef9f27")
    dl_col1, dl_col2 = st.columns(2)

    # Build Excel data
    sim_summary = pd.DataFrame(stats_rows).set_index("Metric")
    sample_sim = pd.DataFrame({
        "Revenue_final": rev_f[:5000],
        "EBITDA_final": ebt_f[:5000],
        "FCF_final": fcf_f[:5000],
        "Growth_draw": g_draws[:5000],
        "Margin_draw": sim["margin_draws"][:5000] * 100,
    })
    hist_out = hist_df_r.reset_index()
    proj_out = base_proj.reset_index()
    for col in ["Revenue","EBITDA","D&A","Capex","NOPAT","FCF"]:
        if col in proj_out: proj_out[col] = proj_out[col].round(2)

    excel_data = _to_excel({
        "Historical data":    hist_out,
        "Base projection":    proj_out,
        "Simulation summary": sim_summary.reset_index(),
        "Simulation sample":  sample_sim,
    })

    with dl_col1:
        _download_btn("Download full results (Excel)", excel_data, "forecast_results.xlsx")
    with dl_col2:
        dcf_df = pd.DataFrame({
            "Item": ["PV of FCFs", "Terminal Value", "Enterprise Value (DCF)",
                     "WACC", "Terminal growth rate"],
            "Value": [f"${fc['pv_fcfs']:,.1f}M", f"${fc['tv']:,.1f}M",
                      f"${ev_dcf:,.1f}M", f"{p['wacc']:.1f}%", f"{p['tgr']:.1f}%"],
        })
        dcf_excel = _to_excel({"DCF Summary": dcf_df})
        _download_btn("Download DCF summary (Excel)", dcf_excel, "dcf_summary.xlsx")


def _render_lbo_projection():
    """Optional LBO projection mode."""
    _section("LBO Projection — optional overlay", "#5dcaa5")
    st.markdown(
        '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#5a5a72;'
        'margin-bottom:16px">Use this mode to model a hypothetical LBO on a company. '
        'Requires entry multiple, debt structure, and an exit assumption.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        _blk("Entry assumptions", "#378add", "#85b7eb")
        _lbl("LTM EBITDA ($M)"); lbo_ebitda = st.number_input(" ", value=50.0, step=5.0, key="lbo_eb")
        _lbl("Entry EV/EBITDA (x)"); lbo_emult = st.number_input(" ", value=10.0, step=0.5, key="lbo_em")
        _lbl("Holding period (yrs)"); lbo_hold = st.number_input(" ", value=5, min_value=1, max_value=15, step=1, key="lbo_hold")
        _lbl("Exit EV/EBITDA (x)"); lbo_xmult = st.number_input(" ", value=11.0, step=0.5, key="lbo_xm")

    with c2:
        _blk("Capital structure", "#7f77dd", "#afa9ec")
        _lbl("Debt / EV (%)"); lbo_debt_pct = st.number_input(" ", value=60.0, step=5.0, key="lbo_dp")
        _lbl("Senior interest rate (%)"); lbo_rate = st.number_input(" ", value=6.5, step=0.25, key="lbo_rate")
        _lbl("Mezz spread (%)"); lbo_mezz = st.number_input(" ", value=4.0, step=0.25, key="lbo_mezz")
        _lbl("Senior / total debt (%)"); lbo_spct = st.number_input(" ", value=70.0, step=5.0, key="lbo_sp")

    with c3:
        _blk("Operating assumptions", "#1d9e75", "#5dcaa5")
        _lbl("Revenue growth mean (%)"); lbo_gmean = st.number_input(" ", value=5.0, step=0.5, key="lbo_gm")
        _lbl("Revenue growth std dev (%)"); lbo_gstd = st.number_input(" ", value=3.0, step=0.5, key="lbo_gs")
        _lbl("EBITDA margin (%)"); lbo_margin = st.number_input(" ", value=20.0, step=1.0, key="lbo_marg")
        _lbl("Scenarios"); lbo_n = st.number_input(" ", value=20000, step=5000, key="lbo_n")

    st.markdown("---")
    if not st.button("▶  Run LBO Simulation", type="primary", key="lbo_run"):
        return

    ev = lbo_ebitda * lbo_emult
    total_debt = ev * lbo_debt_pct / 100
    equity_in  = ev - total_debt
    if equity_in <= 0:
        st.error("Equity is negative — reduce debt %")
        return

    rng = np.random.default_rng(99)
    g   = rng.normal(lbo_gmean/100, lbo_gstd/100, int(lbo_n))
    em  = np.full(int(lbo_n), lbo_margin/100)
    rev = np.full(int(lbo_n), lbo_ebitda / max(lbo_margin/100, 0.01))

    senior_debt = total_debt * lbo_spct / 100
    mezz_debt   = total_debt * (1 - lbo_spct/100)
    senior_bal  = np.full(int(lbo_n), senior_debt)
    mezz_bal    = np.full(int(lbo_n), mezz_debt)
    sr_rate = lbo_rate/100; mz_rate = (lbo_rate + lbo_mezz)/100

    for _ in range(int(lbo_hold)):
        rev = rev * (1 + g)
        ebitda = rev * em
        interest = senior_bal * sr_rate + mezz_bal * mz_rate
        ebt  = ebitda - interest
        tax  = np.maximum(ebt, 0) * 0.25
        ni   = ebt - tax
        fcf  = ni + rev * 0.04 - rev * 0.04   # D&A ≈ Capex for simplicity
        sweep_s = np.minimum(np.maximum(fcf, 0), senior_bal)
        senior_bal = np.maximum(senior_bal - sweep_s, 0)
        rem = np.maximum(fcf - sweep_s, 0)
        sweep_m = np.minimum(rem, mezz_bal)
        mezz_bal = np.maximum(mezz_bal - sweep_m, 0)

    exit_ebitda = rev * em
    exit_ev     = exit_ebitda * lbo_xmult
    net_debt    = senior_bal + mezz_bal
    equity_out  = np.maximum(exit_ev - net_debt, 0)
    moic = equity_out / equity_in
    irr  = np.where(
        equity_in > 0,
        (equity_out / equity_in) ** (1.0 / lbo_hold) - 1,
        -1.0,
    )

    # Results
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean IRR",    f"{np.mean(irr)*100:.1f}%")
    m2.metric("Median IRR",  f"{np.median(irr)*100:.1f}%")
    m3.metric("Mean MOIC",   f"{np.mean(moic):.2f}x")
    m4.metric("P(IRR>20%)",  f"{(irr>0.20).mean():.1%}")
    m5.metric("Wipeout rate",f"{(equity_out==0).mean():.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(irr*100, bins=70, color=ACC1, alpha=0.75, edgecolor="none", density=True)
    axes[0].axvline(np.mean(irr)*100, color=ACC2, lw=2, linestyle="--",
                    label=f"Mean {np.mean(irr):.1%}")
    axes[0].axvline(20, color=ACC3, lw=1.5, linestyle=":", label="20% hurdle")
    axes[0].set_title("IRR distribution"); axes[0].set_xlabel("IRR (%)")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y")

    axes[1].hist(moic, bins=70, color=ACC2, alpha=0.75, edgecolor="none", density=True)
    axes[1].axvline(np.mean(moic), color=ACC1, lw=2, linestyle="--",
                    label=f"Mean {np.mean(moic):.2f}x")
    axes[1].axvline(1.0, color=ACC3, lw=1.5, linestyle=":", label="1x breakeven")
    axes[1].set_title("MOIC distribution"); axes[1].set_xlabel("MOIC (x)")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y")

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Download
    lbo_out = pd.DataFrame({
        "IRR": irr, "MOIC": moic,
        "Exit equity ($M)": equity_out,
        "Growth draw": g,
    })
    excel_lbo = _to_excel({"LBO Simulation": lbo_out.head(5000)})
    _download_btn("Download LBO simulation results (Excel)", excel_lbo, "lbo_simulation.xlsx")