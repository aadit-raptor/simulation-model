"""
pages/settings.py
-----------------
Global model settings — exposes every hardcoded assumption so the user
can override it. All values fall back to the original defaults if not changed.

Sections:
  1. Transaction fees          (dashboard hardcodes)
  2. Deal model defaults       (LBOParams defaults shown on pages 1-4)
  3. Sensitivity table ranges  (returns.py)
  4. Simulation defaults       (SimulationParams)
  5. Correlation matrix        (DEFAULT_CORR — the 5x5 Cholesky matrix)
  6. Scenario preset multipliers
  7. Advanced / convergence

All values are stored in st.session_state under the prefix "cfg_".
Every other page reads from session state via get_cfg() helper.
"""

import streamlit as st
import numpy as np

# ---------------------------------------------------------------------------
# Defaults — exactly the original hardcoded values
# ---------------------------------------------------------------------------

DEFAULTS = {
    # --- Transaction fees (dashboard page 1) ---
    "tx_fee_pct":          2.3,    # % of EV
    "fin_fee_pct":         2.6,    # % of total debt
    "other_uses":          0.0,    # $M flat

    # --- Deal model defaults (pre-fill pages 1-2) ---
    "def_ebitda":          100.0,
    "def_entry_mult":      10.0,
    "def_exit_mult":       11.0,
    "def_hold":            5,
    "def_growth":          5.0,
    "def_gross_margin":    40.0,
    "def_opex":            18.0,
    "def_tax":             25.0,
    "def_da":              4.0,
    "def_debt_pct":        60.0,
    "def_senior_pct":      70.0,
    "def_base_rate":       6.5,
    "def_mezz_spread":     4.0,
    "def_capex":           4.0,
    "def_nwc":             1.0,
    "def_mincash":         0.0,
    "def_senior_amort":    5.0,    # % of original principal per year

    # --- Sensitivity table ranges ---
    "sens_em_min":         6.0,
    "sens_em_max":         12.0,
    "sens_em_steps":       7,
    "sens_hp_min":         3,
    "sens_hp_max":         7,

    # --- Simulation defaults (SimulationParams) ---
    "mc_n":                50000,
    "mc_growth_mean":      5.0,
    "mc_growth_std":       3.0,
    "mc_exit_mean":        10.0,
    "mc_exit_std":         1.5,
    "mc_rate_mean":        6.5,
    "mc_rate_std":         1.5,
    "mc_gm_mean":          40.0,
    "mc_gm_std":           3.0,
    "mc_hurdle":           20.0,
    "mc_n_passes":         2,
    "mc_clip_irr":         True,

    # --- Correlation matrix (10 unique off-diagonal values) ---
    # Variables: [growth, exit_mult, interest, gross_margin, ebitda_shock]
    "corr_g_em":    0.60,   # growth ↔ exit multiple
    "corr_g_ir":   -0.30,   # growth ↔ interest
    "corr_g_gm":    0.40,   # growth ↔ gross margin
    "corr_g_sh":    0.50,   # growth ↔ EBITDA shock
    "corr_em_ir":  -0.50,   # exit multiple ↔ interest
    "corr_em_gm":   0.30,   # exit multiple ↔ gross margin
    "corr_em_sh":   0.30,   # exit multiple ↔ EBITDA shock
    "corr_ir_gm":  -0.20,   # interest ↔ gross margin
    "corr_ir_sh":  -0.20,   # interest ↔ EBITDA shock
    "corr_gm_sh":   0.20,   # gross margin ↔ EBITDA shock

    # --- Scenario preset multipliers ---
    # Bull
    "bull_growth_mult":   1.50,
    "bull_exit_mult":     1.15,
    "bull_rate_mult":     0.85,
    "bull_margin_mult":   1.05,
    # Recession
    "rec_growth_adj":    -6.0,    # pp adjustment (not multiplier)
    "rec_growth_floor":  -10.0,   # pp floor
    "rec_exit_mult":      0.80,
    "rec_rate_mult":      1.20,
    "rec_margin_mult":    0.93,
    # Stagflation
    "stag_growth_adj":   -3.0,
    "stag_growth_floor": -5.0,
    "stag_exit_mult":     0.85,
    "stag_rate_mult":     1.40,
    "stag_margin_mult":   0.90,
}


def init_cfg():
    """Initialise session state with defaults for any key not yet set."""
    for k, v in DEFAULTS.items():
        cfg_key = f"cfg_{k}"
        if cfg_key not in st.session_state:
            st.session_state[cfg_key] = v


def get_cfg(key: str):
    """Read a config value. Falls back to DEFAULTS if not in session state."""
    init_cfg()
    return st.session_state.get(f"cfg_{key}", DEFAULTS.get(key))


def build_corr_matrix() -> np.ndarray:
    """Reconstruct the 5×5 correlation matrix from session state values."""
    g_em  = get_cfg("corr_g_em")
    g_ir  = get_cfg("corr_g_ir")
    g_gm  = get_cfg("corr_g_gm")
    g_sh  = get_cfg("corr_g_sh")
    em_ir = get_cfg("corr_em_ir")
    em_gm = get_cfg("corr_em_gm")
    em_sh = get_cfg("corr_em_sh")
    ir_gm = get_cfg("corr_ir_gm")
    ir_sh = get_cfg("corr_ir_sh")
    gm_sh = get_cfg("corr_gm_sh")

    m = np.array([
        [1.00,  g_em,  g_ir,  g_gm,  g_sh],
        [g_em,  1.00, em_ir, em_gm, em_sh],
        [g_ir, em_ir,  1.00, ir_gm, ir_sh],
        [g_gm, em_gm, ir_gm,  1.00, gm_sh],
        [g_sh, em_sh, ir_sh, gm_sh,  1.00],
    ])
    return m


def is_valid_corr(m: np.ndarray) -> bool:
    """Check that the matrix is positive semi-definite (valid correlation matrix)."""
    try:
        np.linalg.cholesky(m)
        return True
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------------------------
# UI helpers (duplicated lightly to avoid circular imports from dashboard)
# ---------------------------------------------------------------------------

def _sz(base: int) -> int:
    scale = st.session_state.get("font_scale", 1.0)
    return max(8, int(base * scale))


def _section(title: str, color: str = "#85b7eb"):
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(10)}px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{color};border-bottom:1px solid #16162a;padding-bottom:5px;'
        f'margin:20px 0 12px">◈ {title}</div>',
        unsafe_allow_html=True,
    )


def _lbl(text: str):
    st.markdown(
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:{_sz(11)}px;'
        f'color:#5a5a72;display:block;margin-bottom:2px">{text}</span>',
        unsafe_allow_html=True,
    )


def _note(text: str):
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
        f'color:#3a3a5a;font-style:italic;margin-bottom:8px">{text}</div>',
        unsafe_allow_html=True,
    )


def _blk(title: str, border: str, title_color: str):
    st.markdown(
        f'<div style="background:#080816;border-left:2px solid {border};'
        f'border-radius:6px;padding:8px 12px 4px;margin-bottom:6px">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(9)}px;'
        f'font-weight:500;letter-spacing:0.12em;text-transform:uppercase;'
        f'color:{title_color};margin-bottom:8px">{title}</div></div>',
        unsafe_allow_html=True,
    )


def ni(label: str, cfg_key: str, **kwargs):
    """Labelled number input that reads/writes cfg_ session state.
    All numeric kwargs are cast to float to avoid StreamlitMixedNumericTypesError.
    """
    _lbl(label)
    # Cast every numeric kwarg to float so Streamlit doesn't complain
    # about mixing int min_value/step with a float value.
    for k in ("min_value", "max_value", "step"):
        if k in kwargs and kwargs[k] is not None:
            kwargs[k] = float(kwargs[k])
    val = st.number_input(
        " ", value=float(get_cfg(cfg_key)),
        key=f"settings_ni_{cfg_key}",
        label_visibility="collapsed",
        **kwargs,
    )
    st.session_state[f"cfg_{cfg_key}"] = val
    return val


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_settings():
    init_cfg()

    st.markdown(
        f'<div class="page-hdr">'
        f'<div class="page-hdr-title">Model settings</div>'
        f'<div class="page-hdr-step">Global assumptions — all pages</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:{_sz(11)}px;'
        f'color:#5a5a72;margin-bottom:16px">'
        f'Every assumption that was previously hardcoded in the model is configurable here. '
        f'If you leave a value unchanged it uses the original default. '
        f'Changes apply immediately across all pages — no restart needed.</div>',
        unsafe_allow_html=True,
    )

    # Reset button
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("↺  Reset all to defaults", key="cfg_reset"):
            for k in DEFAULTS:
                st.session_state[f"cfg_{k}"] = DEFAULTS[k]
            st.success("All settings reset to defaults.")
            st.rerun()

    # ── 1. TRANSACTION FEES ──────────────────────────────────────────────
    _section("1. Transaction fees", "#85b7eb")
    _note("Applied on Page 1 in the Sources & Uses table. "
          "Transaction fees are typically 1-3% of EV. "
          "Financing fees are typically 2-3% of total debt.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        ni("Transaction fees (% of EV)", "tx_fee_pct", min_value=0.0, max_value=10.0, step=0.1)
    with c2:
        ni("Financing fees (% of total debt)", "fin_fee_pct", min_value=0.0, max_value=10.0, step=0.1)
    with c3:
        ni("Other uses ($M)", "other_uses", min_value=0.0, step=1.0)

    # ── 2. DEAL MODEL DEFAULTS ───────────────────────────────────────────
    _section("2. Deal model defaults", "#5dcaa5")
    _note("These pre-fill Pages 1 and 2 when you open the app. "
          "Changing them here restarts the session with the new defaults.")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        _blk("Operating", "#378add", "#85b7eb")
        ni("LTM EBITDA ($M)",      "def_ebitda",       min_value=1.0,  step=5.0)
        ni("Revenue growth (%)",   "def_growth",                       step=0.5)
        ni("Gross margin (%)",     "def_gross_margin",  min_value=1.0,  step=1.0)
        ni("OpEx / revenue (%)",   "def_opex",          min_value=0.0,  step=1.0)
        ni("D&A / revenue (%)",    "def_da",            min_value=0.0,  step=0.5)
        ni("Tax rate (%)",         "def_tax",           min_value=0.0,  step=1.0)

    with c2:
        _blk("Transaction", "#1d9e75", "#5dcaa5")
        ni("Entry EV/EBITDA (x)",  "def_entry_mult",    min_value=1.0,  step=0.5)
        ni("Exit EV/EBITDA (x)",   "def_exit_mult",     min_value=1.0,  step=0.5)
        ni("Holding period (yrs)", "def_hold",          min_value=1,    step=1)
        ni("Capex / revenue (%)",  "def_capex",         min_value=0.0,  step=0.5)
        ni("NWC / revenue (%)",    "def_nwc",           min_value=0.0,  step=0.25)
        ni("Min cash ($M)",        "def_mincash",       min_value=0.0,  step=5.0)

    with c3:
        _blk("Capital structure", "#7f77dd", "#afa9ec")
        ni("Debt / EV (%)",        "def_debt_pct",      min_value=0.0, max_value=95.0, step=1.0)
        ni("Senior / debt (%)",    "def_senior_pct",    min_value=1.0, max_value=100.0, step=1.0)
        ni("Base rate (%)",        "def_base_rate",     min_value=0.0, step=0.25)
        ni("Mezz spread (%)",      "def_mezz_spread",   min_value=0.0, step=0.25)
        ni("Senior amort (% / yr)","def_senior_amort",  min_value=0.0, max_value=100.0, step=0.5)

    with c4:
        _blk("Sensitivity table", "#ba7517", "#ef9f27")
        _note("Exit multiple range for the IRR heatmap on Page 3.")
        ni("Exit mult min (x)",    "sens_em_min",       min_value=1.0,  step=0.5)
        ni("Exit mult max (x)",    "sens_em_max",       min_value=1.0,  step=0.5)
        _lbl("Number of mult steps")
        steps_val = int(st.number_input(" ", value=int(get_cfg("sens_em_steps")),
            min_value=3, max_value=15, step=1,
            key="settings_ni_sens_em_steps", label_visibility="collapsed"))
        st.session_state["cfg_sens_em_steps"] = steps_val

        _note("Holding period range for the IRR heatmap.")
        ni("Hold period min (yrs)","sens_hp_min",       min_value=1,    step=1)
        ni("Hold period max (yrs)","sens_hp_max",       min_value=1,    step=1)

    # ── 3. CORRELATION MATRIX ─────────────────────────────────────────────
    _section("3. Cholesky correlation matrix", "#afa9ec")
    _note("These 10 values define how the 5 simulation variables move together. "
          "Values must stay between −1.0 and +1.0. "
          "The matrix must remain positive semi-definite — the model will warn you if it is not. "
          "Economic rationale: high growth → higher exit multiple (+0.60); "
          "high rates → lower multiple (−0.50); high growth → tighter rates (−0.30).")

    var_labels = ["Growth", "Exit multiple", "Interest", "Gross margin", "EBITDA shock"]
    corr_keys = [
        [None,         "corr_g_em",  "corr_g_ir",  "corr_g_gm",  "corr_g_sh"],
        ["corr_g_em",  None,         "corr_em_ir", "corr_em_gm", "corr_em_sh"],
        ["corr_g_ir",  "corr_em_ir", None,         "corr_ir_gm", "corr_ir_sh"],
        ["corr_g_gm",  "corr_em_gm", "corr_ir_gm", None,         "corr_gm_sh"],
        ["corr_g_sh",  "corr_em_sh", "corr_ir_sh", "corr_gm_sh", None],
    ]

    # Header row
    hdr_cols = st.columns([2] + [1]*5)
    with hdr_cols[0]:
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;'
                    f'font-size:{_sz(9)}px;color:#44445a;padding-top:28px">Variable</div>',
                    unsafe_allow_html=True)
    for i, lbl in enumerate(var_labels):
        with hdr_cols[i+1]:
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;'
                        f'font-size:{_sz(9)}px;color:#85b7eb;text-align:center;'
                        f'margin-bottom:4px">{lbl}</div>',
                        unsafe_allow_html=True)

    # Matrix rows — only upper triangle is editable (symmetric)
    seen_keys = set()
    for row_i, (row_label, row_keys) in enumerate(zip(var_labels, corr_keys)):
        cols = st.columns([2] + [1]*5)
        with cols[0]:
            st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;'
                        f'font-size:{_sz(10)}px;color:#c4c4d4;padding-top:28px">'
                        f'{row_label}</div>',
                        unsafe_allow_html=True)
        for col_i, key in enumerate(row_keys):
            with cols[col_i + 1]:
                if key is None:
                    # Diagonal
                    st.markdown(
                        f'<div style="background:#0e1e0e;border:0.5px solid #1d9e75;'
                        f'border-radius:4px;padding:5px;text-align:center;'
                        f'font-family:IBM Plex Mono,monospace;font-size:{_sz(11)}px;'
                        f'color:#5dcaa5;margin-top:22px">1.00</div>',
                        unsafe_allow_html=True,
                    )
                elif key in seen_keys:
                    # Lower triangle — show value read-only (mirrors upper)
                    val = get_cfg(key)
                    st.markdown(
                        f'<div style="background:#0a0a18;border:0.5px solid #2a2a42;'
                        f'border-radius:4px;padding:5px;text-align:center;'
                        f'font-family:IBM Plex Mono,monospace;font-size:{_sz(11)}px;'
                        f'color:#5a5a72;margin-top:22px">{val:+.2f}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Upper triangle — editable
                    seen_keys.add(key)
                    _lbl(f"ρ")
                    val = st.number_input(" ",
                        value=float(get_cfg(key)),
                        min_value=-0.99, max_value=0.99, step=0.05,
                        key=f"settings_corr_{key}",
                        label_visibility="collapsed",
                        format="%.2f",
                    )
                    st.session_state[f"cfg_{key}"] = val

    # Validate
    corr_m = build_corr_matrix()
    if is_valid_corr(corr_m):
        st.success("✓ Correlation matrix is valid (positive semi-definite)")
    else:
        st.error("⚠ Correlation matrix is not positive semi-definite — "
                 "the simulation will not run. Reduce the magnitude of some correlations.")

    # ── 4. SCENARIO PRESET MULTIPLIERS ───────────────────────────────────
    _section("4. Scenario preset multipliers", "#ef9f27")
    _note("These control what happens when you press BULL, RECESSION, or STAGFLATION "
          "on the Monte Carlo page. Each preset shifts the distribution means. "
          "Multipliers are applied to the base means (e.g. ×1.50 = 50% higher). "
          "Growth adjustments are additive in percentage points (e.g. −6pp).")

    tab_bull, tab_rec, tab_stag = st.tabs(["Bull", "Recession", "Stagflation"])

    with tab_bull:
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            _lbl("Growth multiplier (×base)")
            ni("", "bull_growth_mult", min_value=0.5, max_value=5.0, step=0.05)
        with c2:
            _lbl("Exit multiple multiplier")
            ni("", "bull_exit_mult", min_value=0.5, max_value=3.0, step=0.05)
        with c3:
            _lbl("Rate multiplier")
            ni("", "bull_rate_mult", min_value=0.5, max_value=2.0, step=0.05)
        with c4:
            _lbl("Gross margin multiplier")
            ni("", "bull_margin_mult", min_value=0.5, max_value=2.0, step=0.05)
        _note("Example: base growth mean = 5%, bull multiplier = 1.50 → bull growth mean = 7.5%")

    with tab_rec:
        c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
        with c1:
            _lbl("Growth adj (pp)")
            ni("", "rec_growth_adj", min_value=-30.0, max_value=0.0, step=0.5)
        with c2:
            _lbl("Growth floor (pp)")
            ni("", "rec_growth_floor", min_value=-50.0, max_value=0.0, step=0.5)
        with c3:
            _lbl("Exit mult multiplier")
            ni("", "rec_exit_mult", min_value=0.1, max_value=1.5, step=0.05)
        with c4:
            _lbl("Rate multiplier")
            ni("", "rec_rate_mult", min_value=0.5, max_value=3.0, step=0.05)
        with c5:
            _lbl("Margin multiplier")
            ni("", "rec_margin_mult", min_value=0.5, max_value=1.5, step=0.01)
        _note("Example: base growth 5%, adj −6pp → recession growth −1%. "
              "Floor prevents unrealistic results (caps at −10%).")

    with tab_stag:
        c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
        with c1:
            _lbl("Growth adj (pp)")
            ni("", "stag_growth_adj", min_value=-30.0, max_value=0.0, step=0.5)
        with c2:
            _lbl("Growth floor (pp)")
            ni("", "stag_growth_floor", min_value=-50.0, max_value=0.0, step=0.5)
        with c3:
            _lbl("Exit mult multiplier")
            ni("", "stag_exit_mult", min_value=0.1, max_value=1.5, step=0.05)
        with c4:
            _lbl("Rate multiplier")
            ni("", "stag_rate_mult", min_value=0.5, max_value=3.0, step=0.05)
        with c5:
            _lbl("Margin multiplier")
            ni("", "stag_margin_mult", min_value=0.5, max_value=1.5, step=0.01)
        _note("Stagflation = mild growth drag + large rate spike + margin squeeze. "
              "The dangerous scenario for leveraged companies.")

    # ── 5. ADVANCED / CONVERGENCE ─────────────────────────────────────────
    _section("5. Advanced / convergence", "#f0997b")
    _note("These control internal model behaviour. "
          "Do not change unless you understand the two-pass convergence system.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        _lbl("Interest convergence passes")
        passes_val = int(st.number_input(" ",
            value=int(get_cfg("mc_n_passes")),
            min_value=1, max_value=5, step=1,
            key="settings_ni_mc_n_passes",
            label_visibility="collapsed",
        ))
        st.session_state["cfg_mc_n_passes"] = passes_val
        _note("2 = standard (fast). 3 = more accurate. "
              "Higher values slow simulation proportionally.")

    with c2:
        _lbl("Clip IRR to range")
        clip = st.checkbox("Clip IRR to [−100%, 500%]",
                           value=bool(get_cfg("mc_clip_irr")),
                           key="settings_clip_irr")
        st.session_state["cfg_mc_clip_irr"] = clip
        _note("Removes numerical artifacts from extreme scenarios. "
              "Recommended: leave ON.")

    with c3:
        _lbl("Default simulation scenarios")
        ni("Default n (MC page)", "mc_n",
           min_value=1000, max_value=5_000_000, step=5000)
        _note("This pre-fills the MC page. "
              "50,000 gives good accuracy in ~0.2s.")

    # ── Summary of current settings ───────────────────────────────────────
    st.markdown("---")
    _section("Current settings summary", "#44445a")

    summary = {
        "Transaction fee": f"{get_cfg('tx_fee_pct'):.1f}% of EV",
        "Financing fee":   f"{get_cfg('fin_fee_pct'):.1f}% of debt",
        "Default EBITDA":  f"${get_cfg('def_ebitda'):.0f}M",
        "Default entry ×": f"{get_cfg('def_entry_mult'):.1f}x",
        "Default hold":    f"{int(get_cfg('def_hold'))} years",
        "Sens EM range":   f"{get_cfg('sens_em_min'):.1f}x – {get_cfg('sens_em_max'):.1f}x",
        "Corr G↔EM":       f"{get_cfg('corr_g_em'):+.2f}",
        "Corr EM↔rate":    f"{get_cfg('corr_em_ir'):+.2f}",
        "Corr G↔rate":     f"{get_cfg('corr_g_ir'):+.2f}",
        "Bull growth ×":   f"{get_cfg('bull_growth_mult'):.2f}",
        "Rec growth adj":  f"{get_cfg('rec_growth_adj'):+.1f}pp",
        "Conv. passes":    str(int(get_cfg("mc_n_passes"))),
    }

    summary_cols = st.columns(4)
    items = list(summary.items())
    for col_i, col in enumerate(summary_cols):
        for k, v in items[col_i::4]:
            col.markdown(
                f'<div style="background:#0e0e1c;border:0.5px solid #1e1e30;'
                f'border-radius:4px;padding:6px 10px;margin-bottom:4px;'
                f'font-family:IBM Plex Mono,monospace">'
                f'<div style="font-size:{_sz(8)}px;color:#44445a;'
                f'text-transform:uppercase;letter-spacing:0.08em">{k}</div>'
                f'<div style="font-size:{_sz(12)}px;color:#e0e0f0">{v}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )