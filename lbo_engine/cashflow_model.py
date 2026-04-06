"""
cashflow_model.py
-----------------
Levered Free Cash Flow bridge between the income statement and debt schedule.

Build order (mirrors BK PDF cash flow section exactly):
    Net Income
    + D&A                       (non-cash add-back, from operating_model)
    - Capital Expenditures      (% of revenue)
    - Change in Net Working Cap (% of revenue, can be positive or negative)
    - Mandatory Debt Repayments (from capital_structure, per tranche)
    = Levered Free Cash Flow    (available for cash sweep or cash build)

Why this file exists as a separate module:
    The LBO cash flow statement is not the same as a standard accounting
    cash flow statement. It is specifically designed to answer one question:
    "How much cash is available to repay debt this year?"

    That answer drives the debt sweep waterfall in debt_model.py.
    Keeping it isolated means you can change Capex or NWC assumptions
    without touching the debt schedule logic.

Burger King reference (FYE 6/30, conservative):
    Year:               2011E   2012E   2013E   2014E   2015E
    Net Income:            89     104     131     191     273
    + D&A:                105      86      77      78      82
    - Capex:              183     149     134     136     144
    - Change in NWC:       27      22      20      20      21
    - Mandatory repay:      0       0       0       0       0
    = Levered FCF:        -16      18      54     113     191

    Key insight: Mandatory debt repayments in the BK model are ZERO
    because the USD term loan's 1% amort ($15.1M/yr) is funded from
    cash on hand, not from the FCF sweep. The FCF sweep is purely
    the excess above the minimum cash floor. We model both.

NWC convention:
    A positive NWC change = cash OUTFLOW (working capital is building up).
    A negative NWC change = cash INFLOW (working capital is releasing).
    BK uses +$27M in year 1 = cash outflow = NWC is increasing.
    In the formula: FCF -= delta_nwc (positive delta = use of cash).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from lbo_engine.operating_model import OperatingModelResult
from lbo_engine.capital_structure import CapitalStructure


# ---------------------------------------------------------------------------
# Assumption inputs
# ---------------------------------------------------------------------------

@dataclass
class CashFlowAssumptions:
    """
    Capital expenditure and working capital assumptions.

    Parameters
    ----------
    holding_period : int
        Number of projection years. Must match OperatingModelResult.

    capex_pct : list[float] or float
        Capex as % of revenue each year.
        BK: [0.071, 0.071, 0.071, 0.071, 0.071] (flat 7.1%)

    nwc_pct : list[float] or float
        Change in net working capital as % of revenue.
        Positive = cash outflow (NWC building).
        Negative = cash inflow (NWC releasing).
        BK: [0.011, 0.011, 0.011, 0.011, 0.011] (flat 1.1%)

    include_mandatory_repayments : bool
        If True, subtract mandatory debt amortization from FCF
        before computing the sweep amount.
        BK model: False (mandatory repayments treated separately).
        Setting True gives a more conservative FCF available for sweep.
    """

    holding_period: int
    capex_pct: List[float] = field(default_factory=list)
    nwc_pct: List[float] = field(default_factory=list)
    include_mandatory_repayments: bool = False

    def __post_init__(self):
        self.capex_pct = self._expand(self.capex_pct, "capex_pct")
        self.nwc_pct   = self._expand(self.nwc_pct,   "nwc_pct")

    def _expand(self, value, name: str) -> List[float]:
        if isinstance(value, (int, float)):
            return [float(value)] * self.holding_period
        if isinstance(value, list):
            if len(value) != self.holding_period:
                raise ValueError(
                    f"'{name}' has {len(value)} elements "
                    f"but holding_period is {self.holding_period}."
                )
            return [float(v) for v in value]
        raise TypeError(
            f"'{name}' must be float or list[float], got {type(value)}."
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class CashFlowResult:
    """
    Year-by-year levered free cash flow output.

    All values in $M. Index 0 = Year 1.

    Fields
    ------
    net_income          From complete_income_statement()
    da                  From operating_model (non-cash add-back)
    capex               Capital expenditures ($M, shown as positive outflow)
    capex_pct           Capex / revenue
    delta_nwc           Change in net working capital ($M, positive = outflow)
    delta_nwc_pct       delta_nwc / revenue
    mandatory_repay     Scheduled debt amortization ($M, all tranches combined)
    levered_fcf         Net Income + D&A - Capex - delta_nwc - mandatory_repay
    cumulative_fcf      Running sum of levered_fcf

    The key output consumed by debt_model.py is levered_fcf.
    """

    holding_period: int
    years: List[int] = field(default_factory=list)

    net_income: List[float] = field(default_factory=list)
    da: List[float] = field(default_factory=list)

    capex: List[float] = field(default_factory=list)
    capex_pct: List[float] = field(default_factory=list)

    delta_nwc: List[float] = field(default_factory=list)
    delta_nwc_pct: List[float] = field(default_factory=list)

    mandatory_repay: List[float] = field(default_factory=list)

    levered_fcf: List[float] = field(default_factory=list)
    cumulative_fcf: List[float] = field(default_factory=list)

    @property
    def total_fcf(self) -> float:
        """Sum of all levered FCF over the holding period ($M)."""
        return round(sum(self.levered_fcf), 2)

    @property
    def avg_annual_fcf(self) -> float:
        """Average annual levered FCF ($M)."""
        return round(self.total_fcf / self.holding_period, 2)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_cashflow_model(
    op_result: OperatingModelResult,
    cf_assumptions: CashFlowAssumptions,
    capital_structure: Optional[CapitalStructure] = None,
) -> CashFlowResult:
    """
    Compute year-by-year levered free cash flow.

    Requires the income statement to be fully complete (net_income
    must be populated — i.e. complete_income_statement() must have
    been called before this function).

    Parameters
    ----------
    op_result : OperatingModelResult
        Fully populated operating model (including net_income).

    cf_assumptions : CashFlowAssumptions
        Capex and NWC assumptions.

    capital_structure : CapitalStructure, optional
        If provided and include_mandatory_repayments is True,
        mandatory amortization is deducted from FCF.
        If None, mandatory_repay defaults to 0 each year.

    Returns
    -------
    CashFlowResult

    Raises
    ------
    ValueError
        If net_income is empty (income statement not completed yet).

    Formula per year:
        capex[t]         = revenue[t] * capex_pct[t]
        delta_nwc[t]     = revenue[t] * nwc_pct[t]
        mandatory[t]     = sum of all tranche mandatory repayments
        levered_fcf[t]   = net_income[t] + da[t]
                           - capex[t]
                           - delta_nwc[t]
                           - mandatory[t]   (if include_mandatory_repayments)
    """

    n = op_result.holding_period

    if not op_result.net_income:
        raise ValueError(
            "net_income is empty. Call complete_income_statement() "
            "before running the cash flow model."
        )

    if len(op_result.net_income) != n:
        raise ValueError(
            f"net_income has {len(op_result.net_income)} elements "
            f"but holding_period is {n}."
        )

    result = CashFlowResult(holding_period=n)
    cumulative = 0.0

    for t in range(n):
        year = t + 1
        revenue = op_result.revenue[t]
        net_income = op_result.net_income[t]
        da = op_result.da[t]

        # --- Capex ---
        capex = revenue * cf_assumptions.capex_pct[t]

        # --- Change in NWC ---
        delta_nwc = revenue * cf_assumptions.nwc_pct[t]

        # --- Mandatory debt repayments ---
        mandatory = 0.0
        if cf_assumptions.include_mandatory_repayments and capital_structure is not None:
            for tranche in capital_structure.tranches:
                mandatory += tranche.mandatory_repayment(year, n)

        # --- Levered FCF ---
        lfcf = net_income + da - capex - delta_nwc - mandatory
        cumulative += lfcf

        # --- Store ---
        result.years.append(year)
        result.net_income.append(round(net_income, 2))
        result.da.append(round(da, 2))
        result.capex.append(round(capex, 2))
        result.capex_pct.append(cf_assumptions.capex_pct[t])
        result.delta_nwc.append(round(delta_nwc, 2))
        result.delta_nwc_pct.append(cf_assumptions.nwc_pct[t])
        result.mandatory_repay.append(round(mandatory, 2))
        result.levered_fcf.append(round(lfcf, 2))
        result.cumulative_fcf.append(round(cumulative, 2))

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_cashflow_model(result: CashFlowResult) -> None:
    """
    Print formatted cash flow bridge to console.
    Mirrors the BK PDF cash flow section layout.
    """

    n = result.holding_period
    yr_labels = [f"Year {t}" for t in result.years]
    sep = "-" * (30 + n * 10)

    def row(label, values, sign=""):
        line = f"  {label:<28}"
        for v in values:
            if sign == "-":
                line += f"  ({abs(v):>7,.0f})"
            else:
                line += f"  {v:>8,.0f}"
        print(line)

    print(f"\n{'=' * (30 + n * 10)}")
    print(f"  LEVERED FREE CASH FLOW BRIDGE")
    print(f"{'=' * (30 + n * 10)}")

    header = f"  {'':28}"
    for yr in yr_labels:
        header += f"  {yr:>8}"
    print(header)
    print(sep)

    row("Net Income",               result.net_income)
    row("+ D&A",                    result.da)
    row("- Capex",                  result.capex,       sign="-")
    row("- Change in NWC",          result.delta_nwc,   sign="-")

    if any(v != 0 for v in result.mandatory_repay):
        row("- Mandatory repayments",  result.mandatory_repay, sign="-")

    print(sep)
    row("Levered FCF",              result.levered_fcf)
    row("Cumulative FCF",           result.cumulative_fcf)

    print(f"{'=' * (30 + n * 10)}")
    print(f"  Total FCF over period:  ${result.total_fcf:,.1f}M")
    print(f"  Avg annual FCF:         ${result.avg_annual_fcf:,.1f}M\n")


# ---------------------------------------------------------------------------
# Factory: Burger King cash flow assumptions
# ---------------------------------------------------------------------------

def build_bk_cashflow_assumptions() -> CashFlowAssumptions:
    """
    BK PDF cash flow assumptions (conservative scenario).

    Capex: flat 7.1% of revenue throughout.
    NWC:   flat 1.1% of revenue throughout.
    Mandatory repayments: NOT deducted from FCF in BK model
    (they are handled separately in the debt schedule).
    """
    return CashFlowAssumptions(
        holding_period=5,
        capex_pct=0.071,
        nwc_pct=0.011,
        include_mandatory_repayments=False,
    )


def build_generic_cashflow_assumptions(
    holding_period: int = 5,
    capex_pct: float = 0.04,
    nwc_pct: float = 0.01,
    include_mandatory_repayments: bool = False,
) -> CashFlowAssumptions:
    """
    Generic flat-assumption cash flow model.
    Used by simulation engine for Monte Carlo scenarios.
    """
    return CashFlowAssumptions(
        holding_period=holding_period,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        include_mandatory_repayments=include_mandatory_repayments,
    )


# ---------------------------------------------------------------------------
# Quick self-test — Burger King conservative scenario
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from lbo_engine.operating_model import (
        build_bk_conservative,
        run_operating_model,
        complete_income_statement,
    )

    # --- Build full operating model ---
    bk_op_assumptions = build_bk_conservative()
    op_result = run_operating_model(bk_op_assumptions)

    bk_interest = [208.0, 208.0, 205.0, 199.0, 189.0]
    op_result = complete_income_statement(
        op_result,
        interest_expense=bk_interest,
        tax_rate=bk_op_assumptions.tax_rate,
        minimum_cash=118.0,
        interest_income_rate=0.005,
    )

    # --- Run cash flow model ---
    cf_assumptions = build_bk_cashflow_assumptions()
    cf_result = run_cashflow_model(op_result, cf_assumptions)

    print_cashflow_model(cf_result)

    # --- Validate against BK PDF ---
    bk_fcf = [-16, 18, 54, 113, 191]

    print("VALIDATION vs BK PDF (conservative):")
    print(f"\n  {'Year':<6} {'FCF (model)':>12} {'FCF (BK PDF)':>14} {'Delta':>8}")
    print("  " + "-" * 44)
    for i in range(5):
        delta = cf_result.levered_fcf[i] - bk_fcf[i]
        flag = "  <-- check" if abs(delta) > 2 else ""
        print(
            f"  {i+1:<6} "
            f"{cf_result.levered_fcf[i]:>11,.0f}  "
            f"{bk_fcf[i]:>13,}  "
            f"{delta:>7,.1f}"
            f"{flag}"
        )

    print(f"\n  Total FCF: ${cf_result.total_fcf:,.0f}M")

    # --- Test with mandatory repayments included ---
    print("\n--- With mandatory repayments deducted ---")
    from lbo_engine.capital_structure import build_bk_capital_structure
    cs = build_bk_capital_structure()

    cf_with_mandatory = CashFlowAssumptions(
        holding_period=5,
        capex_pct=0.071,
        nwc_pct=0.011,
        include_mandatory_repayments=True,
    )
    cf_result_2 = run_cashflow_model(op_result, cf_with_mandatory, cs)
    print_cashflow_model(cf_result_2)

    print(f"  Mandatory repayments Year 1: ${cf_result_2.mandatory_repay[0]:,.1f}M")
    print(f"  FCF after mandatory Year 1:  ${cf_result_2.levered_fcf[0]:,.1f}M")

    # --- Test generic assumptions ---
    print("\n--- Generic assumptions (for simulation) ---")
    from lbo_engine.operating_model import build_generic_assumptions

    generic_op = build_generic_assumptions(
        holding_period=5,
        base_revenue=1000.0,
        revenue_growth=0.05,
        gross_margin=0.40,
        opex_pct=0.18,
        da_pct=0.04,
        tax_rate=0.25,
    )
    generic_op_result = run_operating_model(generic_op)
    generic_op_result = complete_income_statement(
        generic_op_result,
        interest_expense=[50.0] * 5,
        tax_rate=[0.25] * 5,
        minimum_cash=50.0,
    )

    generic_cf = build_generic_cashflow_assumptions(
        holding_period=5,
        capex_pct=0.04,
        nwc_pct=0.01,
    )
    generic_cf_result = run_cashflow_model(generic_op_result, generic_cf)
    print_cashflow_model(generic_cf_result)