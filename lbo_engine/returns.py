"""
returns.py
----------
Computes sponsor returns from a completed LBO model.

Inputs (all from upstream modules):
    - sponsor_equity at entry         from transaction.py
    - exit_ebitda                     from operating_model.py
    - exit_multiple                   assumption
    - net_debt_at_exit                from debt_model.py
    - levered_fcf[]                   from cashflow_model.py  (optional dividends)
    - holding_period                  years

Core outputs:
    - True IRR    via numpy_financial.irr() on the full cash flow stream
    - MOIC        exit_equity / entry_equity
    - Equity bridge: entry -> EBITDA growth -> multiple expansion -> deleveraging
    - Exit sensitivity table: IRR / MOIC across exit multiples x holding periods

Why numpy_financial.irr() instead of the CAGR approximation:
    Your current model uses (MOIC)^(1/n) - 1. This is the CAGR — it only
    works when ALL cash flows happen at t=0 and t=n. The moment there are
    interim cash flows (dividends, recaps, equity injections), the CAGR
    diverges from the true IRR. numpy_financial.irr() solves for the
    discount rate r such that NPV of the cash flow stream = 0:

        -equity_in + cf1/(1+r) + cf2/(1+r)^2 + ... + exit_proceeds/(1+r)^n = 0

    For most clean LBOs with no interim dividends, CAGR ≈ IRR. But:
        - If you add a dividend recap in year 3, CAGR breaks immediately
        - IRR is what every LP, GP, and investment committee actually quotes
        - CFA Level 2 explicitly tests the difference — use the real formula

Cash flow stream convention:
    t=0: -sponsor_equity            (the equity check at close)
    t=1..n-1: 0                     (no interim distributions, base case)
    t=n: exit_equity_proceeds       (net proceeds to sponsor at exit)

    If the deal includes interim dividends or a recap, those are
    positive cash flows at the relevant year and increase IRR.

Burger King reference:
    Sponsor equity at entry:    $1,560M
    Exit EBITDA (Year 5):         $672M
    Exit multiple:                 8.8x
    Exit EV:                    $5,914M  (672 * 8.8)
    Net debt at exit:           $2,150M
    Sponsor equity at exit:     $3,764M  (BK PDF shows $3,730M)
    MOIC:                          2.4x  (3,730 / 1,560)
    IRR:                          19.0%  (over 5 years)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

try:
    import numpy_financial as npf
    HAS_NPF = True
except ImportError:
    HAS_NPF = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _irr(cashflows: List[float]) -> float:
    """
    Compute IRR from a cash flow stream.

    Uses numpy_financial.irr() if available (preferred).
    Falls back to a Newton-Raphson solver if not installed.
    Falls back to CAGR approximation as last resort with a warning.

    Parameters
    ----------
    cashflows : list[float]
        Cash flows starting at t=0. First element is typically negative
        (equity invested). Last element is the exit proceeds.

    Returns
    -------
    float
        IRR as a decimal (e.g. 0.19 for 19%).
    """
    if HAS_NPF:
        result = npf.irr(cashflows)
        if np.isnan(result):
            return _irr_newton(cashflows)
        return float(result)
    return _irr_newton(cashflows)


def _irr_newton(cashflows: List[float], tol: float = 1e-8, max_iter: int = 1000) -> float:
    """
    Newton-Raphson IRR solver. Used when numpy_financial is not available.

    Solves: sum( cf[t] / (1+r)^t ) = 0  for r.

    Tries multiple starting points to avoid local minima.
    Returns NaN if no solution is found.
    """
    cfs = np.array(cashflows, dtype=float)
    n = len(cfs)
    t = np.arange(n, dtype=float)

    def npv(r):
        return np.sum(cfs / (1 + r) ** t)

    def dnpv(r):
        return np.sum(-t * cfs / (1 + r) ** (t + 1))

    for r0 in [0.10, 0.20, 0.30, 0.05, 0.50]:
        r = r0
        for _ in range(max_iter):
            f = npv(r)
            df = dnpv(r)
            if abs(df) < 1e-15:
                break
            r_new = r - f / df
            if abs(r_new - r) < tol:
                return float(r_new)
            r = r_new

    # Last resort: CAGR approximation
    moic = cashflows[-1] / abs(cashflows[0])
    years = len(cashflows) - 1
    print(
        "WARNING: IRR solver did not converge. "
        "Falling back to CAGR approximation. "
        "Install numpy_financial for accurate IRR: pip install numpy-financial"
    )
    return float(moic ** (1 / years) - 1)


# ---------------------------------------------------------------------------
# Return assumptions
# ---------------------------------------------------------------------------

@dataclass
class ReturnAssumptions:
    """
    Exit assumptions for the returns calculation.

    Parameters
    ----------
    exit_multiple : float
        EV / EBITDA at exit. Can equal entry multiple (same-multiple exit)
        or differ (multiple expansion or compression).
        BK: 8.8x (same as entry).

    holding_period : int
        Years from close to exit. BK: 5 years.

    interim_dividends : list[float]
        Optional cash distributions to sponsor in years 1..n-1 ($M).
        Positive = cash to sponsor. Length must equal holding_period - 1.
        Leave empty for clean exit with no interim distributions.

    entry_equity : float
        Sponsor equity check at close ($M). From transaction.py.

    exit_ebitda : float
        EBITDA in the exit year ($M). From operating_model.exit_ebitda.

    net_debt_at_exit : float
        Total debt minus minimum cash at exit ($M). From debt_model.net_debt_at_exit.

    management_option_pool_pct : float
        Management equity / option pool as % of exit equity value.
        Reduces sponsor proceeds at exit. Default 0 (no dilution modelled).
    """

    exit_multiple: float
    holding_period: int
    entry_equity: float
    exit_ebitda: float
    net_debt_at_exit: float
    interim_dividends: List[float] = field(default_factory=list)
    management_option_pool_pct: float = 0.0


# ---------------------------------------------------------------------------
# Return outputs
# ---------------------------------------------------------------------------

@dataclass
class ReturnsResult:
    """
    Full returns output for a single scenario.

    All dollar values in $M.
    """

    # --- Inputs (echoed for reference) ---
    entry_equity: float
    exit_ebitda: float
    exit_multiple: float
    holding_period: int
    net_debt_at_exit: float

    # --- Exit valuation ---
    exit_ev: float               # exit_ebitda * exit_multiple
    gross_exit_equity: float     # exit_ev - net_debt_at_exit (before mgmt dilution)
    mgmt_dilution: float         # management option pool deduction
    net_exit_equity: float       # proceeds to sponsor after dilution

    # --- Returns ---
    moic: float                  # net_exit_equity / entry_equity
    irr: float                   # true IRR (decimal)
    cash_flow_stream: List[float]  # full stream passed to IRR solver

    # --- Equity bridge components ---
    ebitda_growth_contribution: float   # value created by EBITDA growth
    multiple_expansion_contribution: float  # value from multiple change
    deleveraging_contribution: float    # value from debt paydown
    # Note: bridge components sum to (net_exit_equity - entry_equity)

    @property
    def irr_pct(self) -> float:
        return round(self.irr * 100, 2)

    @property
    def value_created(self) -> float:
        """Total value created ($M) = exit equity - entry equity."""
        return round(self.net_exit_equity - self.entry_equity, 2)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def compute_returns(assumptions: ReturnAssumptions) -> ReturnsResult:
    """
    Compute full sponsor returns from a completed LBO.

    Parameters
    ----------
    assumptions : ReturnAssumptions

    Returns
    -------
    ReturnsResult

    Equity bridge decomposition:
        The bridge answers: where did the return come from?
        Three sources, computed by holding one factor constant at a time:

        1. EBITDA growth contribution:
           What would exit equity be if ONLY EBITDA grew (multiple and
           debt stayed at entry levels)?
           = (exit_ebitda * entry_multiple - net_debt_at_entry) - entry_equity
           But since we don't have entry_multiple here, we approximate:
           ebitda_growth_contrib = (exit_ebitda - entry_ebitda) * exit_multiple

        2. Multiple expansion contribution:
           What was gained purely from the multiple re-rating?
           = exit_ebitda * (exit_multiple - entry_multiple)

        3. Deleveraging contribution:
           What was gained from paying down debt?
           = net_debt_at_entry - net_debt_at_exit

        These three sum to: exit_equity - entry_equity (approximately).
        Small residuals are from interaction effects and rounding.

    Note: entry_ebitda and entry_multiple are not stored in ReturnAssumptions
    because returns.py only needs exit-side inputs. The orchestrator
    (lbo_engine/model.py) computes the bridge using both entry and exit data.
    """

    a = assumptions

    # ------------------------------------------------------------------
    # Step 1: Exit enterprise value
    # ------------------------------------------------------------------
    exit_ev = a.exit_ebitda * a.exit_multiple

    # ------------------------------------------------------------------
    # Step 2: Exit equity value
    # ------------------------------------------------------------------
    gross_exit_equity = exit_ev - a.net_debt_at_exit
    mgmt_dilution = gross_exit_equity * a.management_option_pool_pct
    net_exit_equity = gross_exit_equity - mgmt_dilution

    if net_exit_equity <= 0:
        # Equity is wiped out — return 0 MOIC, -100% IRR
        return ReturnsResult(
            entry_equity=a.entry_equity,
            exit_ebitda=a.exit_ebitda,
            exit_multiple=a.exit_multiple,
            holding_period=a.holding_period,
            net_debt_at_exit=a.net_debt_at_exit,
            exit_ev=round(exit_ev, 2),
            gross_exit_equity=round(gross_exit_equity, 2),
            mgmt_dilution=round(mgmt_dilution, 2),
            net_exit_equity=0.0,
            moic=0.0,
            irr=-1.0,
            cash_flow_stream=[-a.entry_equity] + [0.0] * (a.holding_period - 1) + [0.0],
            ebitda_growth_contribution=0.0,
            multiple_expansion_contribution=0.0,
            deleveraging_contribution=0.0,
        )

    # ------------------------------------------------------------------
    # Step 3: MOIC
    # ------------------------------------------------------------------
    moic = net_exit_equity / a.entry_equity

    # ------------------------------------------------------------------
    # Step 4: Build cash flow stream for IRR
    # t=0:   -entry_equity (cash out)
    # t=1..n-1: interim dividends if any, else 0
    # t=n:   net_exit_equity (cash in)
    # ------------------------------------------------------------------
    n = a.holding_period

    if a.interim_dividends:
        if len(a.interim_dividends) != n - 1:
            raise ValueError(
                f"interim_dividends must have length holding_period - 1 = {n-1}, "
                f"got {len(a.interim_dividends)}."
            )
        interim = list(a.interim_dividends)
    else:
        interim = [0.0] * (n - 1)

    cash_flow_stream = [-a.entry_equity] + interim + [net_exit_equity]

    # ------------------------------------------------------------------
    # Step 5: True IRR
    # ------------------------------------------------------------------
    irr = _irr(cash_flow_stream)

    # ------------------------------------------------------------------
    # Step 6: Equity bridge (simplified 3-factor decomposition)
    # These are approximate — the orchestrator computes the full bridge
    # using entry_ebitda and entry_multiple.
    # Here we store deleveraging as the only exact component.
    # ------------------------------------------------------------------
    # Deleveraging: exact — debt paid down releases equity value
    deleverage_contrib = 0.0   # filled by orchestrator with net_debt_at_entry

    # Placeholder for bridge (orchestrator fills these)
    ebitda_growth_contrib = 0.0
    multiple_expansion_contrib = 0.0

    return ReturnsResult(
        entry_equity=round(a.entry_equity, 2),
        exit_ebitda=round(a.exit_ebitda, 2),
        exit_multiple=a.exit_multiple,
        holding_period=a.holding_period,
        net_debt_at_exit=round(a.net_debt_at_exit, 2),
        exit_ev=round(exit_ev, 2),
        gross_exit_equity=round(gross_exit_equity, 2),
        mgmt_dilution=round(mgmt_dilution, 2),
        net_exit_equity=round(net_exit_equity, 2),
        moic=round(moic, 4),
        irr=round(irr, 6),
        cash_flow_stream=cash_flow_stream,
        ebitda_growth_contribution=round(ebitda_growth_contrib, 2),
        multiple_expansion_contribution=round(multiple_expansion_contrib, 2),
        deleveraging_contribution=round(deleverage_contrib, 2),
    )


# ---------------------------------------------------------------------------
# Equity bridge (full decomposition — needs entry data)
# ---------------------------------------------------------------------------

def compute_equity_bridge(
    entry_equity: float,
    entry_ebitda: float,
    entry_multiple: float,
    net_debt_at_entry: float,
    exit_ebitda: float,
    exit_multiple: float,
    net_debt_at_exit: float,
    management_option_pool_pct: float = 0.0,
) -> dict:
    """
    Full 3-factor equity value bridge.

    Decomposes the gain from entry to exit into:
        1. EBITDA growth     — operational performance
        2. Multiple change   — market re-rating (expansion or compression)
        3. Deleveraging      — debt paydown releasing equity value

    All values in $M.

    Method:
        We hold two factors constant and vary the third to isolate
        each contribution. This is the standard PE attribution framework.

        entry_equity_check  = entry_ev - net_debt_at_entry
                            = (entry_ebitda * entry_multiple) - net_debt_at_entry

        Hypothetical 1: What if ONLY EBITDA grew (multiple and debt unchanged)?
            hypo_ev_1 = exit_ebitda * entry_multiple
            hypo_equity_1 = hypo_ev_1 - net_debt_at_entry
            ebitda_contrib = hypo_equity_1 - entry_equity_check

        Hypothetical 2: Add multiple expansion on top of EBITDA growth
            hypo_ev_2 = exit_ebitda * exit_multiple
            hypo_equity_2 = hypo_ev_2 - net_debt_at_entry
            multiple_contrib = hypo_equity_2 - hypo_equity_1

        Deleveraging: actual exit equity vs hypothetical with same debt
            exit_equity = exit_ev - net_debt_at_exit
            hypo_equity_no_delever = exit_ev - net_debt_at_entry
            delever_contrib = exit_equity - hypo_equity_no_delever
                            = net_debt_at_entry - net_debt_at_exit

        Check: ebitda_contrib + multiple_contrib + delever_contrib
               = exit_equity - entry_equity  (should be exact)

    Returns
    -------
    dict with keys:
        entry_equity, exit_equity, total_gain,
        ebitda_growth, multiple_expansion, deleveraging,
        ebitda_growth_pct, multiple_expansion_pct, deleveraging_pct
    """

    entry_ev = entry_ebitda * entry_multiple
    entry_equity_check = entry_ev - net_debt_at_entry

    exit_ev = exit_ebitda * exit_multiple
    net_exit_eq = exit_ev - net_debt_at_exit
    exit_equity_sponsor = net_exit_eq * (1 - management_option_pool_pct)

    total_gain = exit_equity_sponsor - entry_equity

    # --- Factor 1: EBITDA growth ---
    hypo_ev_1 = exit_ebitda * entry_multiple
    hypo_eq_1 = hypo_ev_1 - net_debt_at_entry
    ebitda_growth = hypo_eq_1 - entry_equity_check

    # --- Factor 2: Multiple expansion ---
    hypo_ev_2 = exit_ebitda * exit_multiple
    hypo_eq_2 = hypo_ev_2 - net_debt_at_entry
    multiple_expansion = hypo_eq_2 - hypo_eq_1

    # --- Factor 3: Deleveraging ---
    deleveraging = net_debt_at_entry - net_debt_at_exit

    check = ebitda_growth + multiple_expansion + deleveraging
    residual = total_gain - check   # should be near 0

    pct = lambda x: round(x / total_gain * 100, 1) if total_gain != 0 else 0.0

    return {
        "entry_equity": round(entry_equity, 2),
        "exit_equity": round(exit_equity_sponsor, 2),
        "total_gain": round(total_gain, 2),
        "ebitda_growth": round(ebitda_growth, 2),
        "multiple_expansion": round(multiple_expansion, 2),
        "deleveraging": round(deleveraging, 2),
        "residual": round(residual, 2),
        "ebitda_growth_pct": pct(ebitda_growth),
        "multiple_expansion_pct": pct(multiple_expansion),
        "deleveraging_pct": pct(deleveraging),
    }


# ---------------------------------------------------------------------------
# Exit sensitivity table
# ---------------------------------------------------------------------------

def compute_exit_sensitivity(
    entry_equity: float,
    exit_ebitda: float,
    net_debt_at_exit: float,
    holding_periods: List[int] = None,
    exit_multiples: List[float] = None,
    metric: str = "irr",
) -> dict:
    """
    IRR or MOIC sensitivity table across exit multiples and holding periods.

    This is the table investment committees look at first.
    Rows = exit multiples. Columns = holding periods.

    Parameters
    ----------
    entry_equity : float
        Sponsor equity check at close ($M).
    exit_ebitda : float
        EBITDA at exit ($M) — same for all cells (operating model fixed).
    net_debt_at_exit : float
        Net debt at exit ($M) — same for all cells (debt model fixed).
    holding_periods : list[int]
        Columns of the table. Default: [3, 4, 5, 6, 7].
    exit_multiples : list[float]
        Rows of the table. Default: [6.0, 7.0, 8.0, 8.8, 9.5, 10.5, 12.0].
    metric : str
        "irr" or "moic".

    Returns
    -------
    dict with:
        "table"           : 2D list [row][col]
        "exit_multiples"  : row labels
        "holding_periods" : column labels
        "metric"          : "irr" or "moic"
    """

    if holding_periods is None:
        holding_periods = [3, 4, 5, 6, 7]
    if exit_multiples is None:
        exit_multiples = [6.0, 7.0, 8.0, 8.8, 9.5, 10.5, 12.0]

    table = []
    for em in exit_multiples:
        row = []
        for hp in holding_periods:
            a = ReturnAssumptions(
                exit_multiple=em,
                holding_period=hp,
                entry_equity=entry_equity,
                exit_ebitda=exit_ebitda,
                net_debt_at_exit=net_debt_at_exit,
            )
            r = compute_returns(a)
            val = r.irr if metric == "irr" else r.moic
            row.append(round(val, 4))
        table.append(row)

    return {
        "table": table,
        "exit_multiples": exit_multiples,
        "holding_periods": holding_periods,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_returns_summary(
    result: ReturnsResult,
    bridge: dict = None,
) -> None:
    """Print formatted returns summary and optional equity bridge."""

    sep = "=" * 55
    dash = "-" * 55

    print(f"\n{sep}")
    print(f"  RETURNS SUMMARY")
    print(sep)
    print(f"  {'Holding period':<32} {result.holding_period} years")
    print(f"  {'Entry equity':<32} ${result.entry_equity:>10,.1f}M")
    print(f"  {'Exit EBITDA':<32} ${result.exit_ebitda:>10,.1f}M")
    print(f"  {'Exit multiple':<32} {result.exit_multiple:>11.1f}x")
    print(f"  {'Exit EV':<32} ${result.exit_ev:>10,.1f}M")
    print(f"  {'Net debt at exit':<32} ${result.net_debt_at_exit:>10,.1f}M")
    print(dash)
    print(f"  {'Exit equity (sponsor)':<32} ${result.net_exit_equity:>10,.1f}M")
    print(dash)
    print(f"  {'MOIC':<32} {result.moic:>11.2f}x")
    print(f"  {'IRR':<32} {result.irr_pct:>10.1f}%")
    print(f"  {'Value created':<32} ${result.value_created:>10,.1f}M")
    print(sep)

    if bridge:
        print(f"\n  EQUITY BRIDGE  (${bridge['entry_equity']:,.0f}M  →  ${bridge['exit_equity']:,.0f}M)")
        print(dash)
        print(f"  {'Entry equity check':<32} ${bridge['entry_equity']:>10,.1f}M")
        print(f"  {'+ EBITDA growth':<32} ${bridge['ebitda_growth']:>10,.1f}M  "
              f"({bridge['ebitda_growth_pct']:>5.1f}%)")
        print(f"  {'+ Multiple expansion':<32} ${bridge['multiple_expansion']:>10,.1f}M  "
              f"({bridge['multiple_expansion_pct']:>5.1f}%)")
        print(f"  {'+ Deleveraging':<32} ${bridge['deleveraging']:>10,.1f}M  "
              f"({bridge['deleveraging_pct']:>5.1f}%)")
        if abs(bridge['residual']) > 0.1:
            print(f"  {'  Residual (rounding)':<32} ${bridge['residual']:>10,.1f}M")
        print(dash)
        print(f"  {'Exit equity (sponsor)':<32} ${bridge['exit_equity']:>10,.1f}M")
        print(sep)


def print_sensitivity_table(sensitivity: dict) -> None:
    """Print IRR or MOIC sensitivity table."""

    metric = sensitivity["metric"].upper()
    hps = sensitivity["holding_periods"]
    ems = sensitivity["exit_multiples"]
    table = sensitivity["table"]

    col_w = 10
    header_w = 12

    print(f"\n  EXIT SENSITIVITY — {metric}")
    print(f"  {'Exit mult':<{header_w}}", end="")
    for hp in hps:
        print(f"  {hp}yr".rjust(col_w), end="")
    print()
    print("  " + "-" * (header_w + len(hps) * (col_w + 2)))

    for i, em in enumerate(ems):
        print(f"  {em:.1f}x".ljust(header_w), end="")
        for j, val in enumerate(table[i]):
            if metric == "IRR":
                formatted = f"{val * 100:.1f}%"
            else:
                formatted = f"{val:.2f}x"
            print(f"  {formatted}".rjust(col_w + 2), end="")
        print()
    print()


# ---------------------------------------------------------------------------
# Quick self-test — Burger King full returns
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- BK reference numbers ---
    bk_assumptions = ReturnAssumptions(
        exit_multiple=8.8,
        holding_period=5,
        entry_equity=1560.0,
        exit_ebitda=672.0,
        net_debt_at_exit=2150.0,
    )

    result = compute_returns(bk_assumptions)
    print_returns_summary(result)

    # --- Equity bridge ---
    bridge = compute_equity_bridge(
        entry_equity=1560.0,
        entry_ebitda=445.0,
        entry_multiple=8.75,
        net_debt_at_entry=2456.0,    # 2644 total debt - 188 cash
        exit_ebitda=672.0,
        exit_multiple=8.8,
        net_debt_at_exit=2150.0,
    )
    print_returns_summary(result, bridge)

    print("VALIDATION vs BK PDF:")
    print(f"  MOIC: {result.moic:.2f}x  (BK PDF: 2.4x)")
    print(f"  IRR:  {result.irr_pct:.1f}%  (BK PDF: 19.0%)")

    # --- CAGR vs true IRR comparison ---
    cagr_approx = (result.moic ** (1 / 5)) - 1
    print(f"\n  True IRR (numpy):      {result.irr_pct:.2f}%")
    print(f"  CAGR approx (old):     {cagr_approx * 100:.2f}%")
    print(f"  Difference:            {(result.irr - cagr_approx) * 100:.4f}%")
    print(f"  (Difference is small here because there are no interim cash flows)")

    # --- Test with interim dividend ---
    print("\n--- With $200M dividend recap in year 3 ---")
    bk_recap = ReturnAssumptions(
        exit_multiple=8.8,
        holding_period=5,
        entry_equity=1560.0,
        exit_ebitda=672.0,
        net_debt_at_exit=2150.0,
        interim_dividends=[0.0, 0.0, 200.0, 0.0],
    )
    recap_result = compute_returns(bk_recap)
    print_returns_summary(recap_result)
    cagr_recap = (recap_result.moic ** (1 / 5)) - 1
    print(f"  True IRR:   {recap_result.irr_pct:.2f}%  (CAGR would give: {cagr_recap*100:.2f}%)")
    print(f"  CAGR error with interim dividend: {(recap_result.irr - cagr_recap)*100:.2f}pp")

    # --- Exit sensitivity ---
    print("\n--- IRR sensitivity ---")
    irr_sens = compute_exit_sensitivity(
        entry_equity=1560.0,
        exit_ebitda=672.0,
        net_debt_at_exit=2150.0,
        metric="irr",
    )
    print_sensitivity_table(irr_sens)

    print("--- MOIC sensitivity ---")
    moic_sens = compute_exit_sensitivity(
        entry_equity=1560.0,
        exit_ebitda=672.0,
        net_debt_at_exit=2150.0,
        metric="moic",
    )
    print_sensitivity_table(moic_sens)

    # --- Equity wiped out scenario ---
    print("--- Distress scenario (equity wipeout) ---")
    distress = ReturnAssumptions(
        exit_multiple=4.0,
        holding_period=5,
        entry_equity=1560.0,
        exit_ebitda=300.0,
        net_debt_at_exit=2150.0,
    )
    distress_result = compute_returns(distress)
    print(f"  Exit EV: ${distress_result.exit_ev:,.0f}M  "
          f"Net debt: ${distress_result.net_debt_at_exit:,.0f}M  "
          f"Exit equity: ${distress_result.net_exit_equity:,.0f}M")
    print(f"  MOIC: {distress_result.moic:.2f}x  IRR: {distress_result.irr_pct:.1f}%  "
          f"(equity wiped out)")