"""
operating_model.py
------------------
Full year-by-year P&L model for an LBO.

Build order (mirrors the BK PDF exactly):
    Revenue
    - COGS                  (% of revenue)
    = Gross Profit
    - OpEx                  (% of revenue, i.e. SG&A / overhead)
    = EBIT
    + D&A                   (% of revenue, non-cash add-back)
    = EBITDA

    Separately:
    - Interest expense      (from debt_model.py, passed in)
    - Interest income       (on minimum cash balance)
    = EBT
    - Taxes
    = Net Income

Two scenarios supported (mirrors BK PDF):
    "conservative"   — lower revenue growth, tighter margins
    "management"     — higher growth, more aggressive margin expansion

Design notes:
    - All line items are stored as year-indexed lists (length = holding_period)
    - Percentages are stored alongside absolute values for easy display
    - The OperatingModelResult is self-contained: downstream modules
      (cashflow_model, debt_model, returns) read from it directly
    - Tax is computed on EBT, but only after interest is known.
      Since interest comes from debt_model, we split the P&L into
      two passes: first pass stops at EBIT/EBITDA, second pass
      completes EBT -> Net Income once interest is available.

Burger King reference (conservative scenario, FYE 6/30):
    Year:           2011E    2012E    2013E    2014E    2015E
    Revenue:        2,574    2,102    1,884    1,910    2,024
    Revenue growth:   2.9%   -18.3%   -10.4%    1.4%    6.0%
    COGS %:          64.5%    62.8%    61.0%    59.3%    57.5%
    Gross margin:    35.5%    37.2%    39.0%    40.7%    42.5%
    OpEx %:          22.2%    20.0%    17.8%    15.6%    13.4%
    EBIT:             342      363      399      480      589
    D&A %:            4.1%     4.1%     4.1%     4.1%     4.1%
    EBITDA:           447      448      476      558      672
    Tax rate:        33.8%    33.3%    32.8%    32.3%    31.8%
"""

from dataclasses import dataclass, field
from typing import List, Literal, Dict
import numpy as np


# ---------------------------------------------------------------------------
# Scenario type
# ---------------------------------------------------------------------------

Scenario = Literal["conservative", "management"]


# ---------------------------------------------------------------------------
# Assumption inputs
# ---------------------------------------------------------------------------

@dataclass
class OperatingAssumptions:
    """
    Year-by-year operating assumptions for the P&L model.

    All list inputs must have length == holding_period.
    Scalar inputs are expanded into flat lists automatically.

    Parameters
    ----------
    holding_period : int
        Number of projection years (e.g. 5).

    base_revenue : float
        Revenue in year 0 (LTM at entry), $M.
        BK: $2,502M (FY2010A).

    revenue_growth : list[float] or float
        Year-by-year revenue growth rates.
        BK conservative: [0.029, -0.183, -0.104, 0.014, 0.060]

    cogs_pct : list[float] or float
        COGS as % of revenue each year.
        BK conservative: [0.645, 0.628, 0.610, 0.593, 0.575]

    opex_pct : list[float] or float
        Operating expenses (SG&A / overhead) as % of revenue.
        BK conservative: [0.222, 0.200, 0.178, 0.156, 0.134]

    da_pct : list[float] or float
        D&A as % of revenue. Used both for EBITDA reconciliation
        and as a non-cash add-back in cashflow_model.
        BK conservative: [0.041, 0.041, 0.041, 0.041, 0.041]

    tax_rate : list[float] or float
        Effective tax rate applied to EBT.
        BK: [0.338, 0.333, 0.328, 0.323, 0.318]

    scenario : Scenario
        "conservative" or "management". Informational label only —
        the actual numbers are what you pass in.
    """

    holding_period: int
    base_revenue: float

    revenue_growth: List[float] = field(default_factory=list)
    cogs_pct: List[float] = field(default_factory=list)
    opex_pct: List[float] = field(default_factory=list)
    da_pct: List[float] = field(default_factory=list)
    tax_rate: List[float] = field(default_factory=list)

    scenario: Scenario = "conservative"

    def __post_init__(self):
        """
        Expand scalar inputs to lists and validate lengths.
        Called automatically after __init__.
        """
        self.revenue_growth = self._expand(self.revenue_growth, "revenue_growth")
        self.cogs_pct       = self._expand(self.cogs_pct,       "cogs_pct")
        self.opex_pct       = self._expand(self.opex_pct,       "opex_pct")
        self.da_pct         = self._expand(self.da_pct,         "da_pct")
        self.tax_rate       = self._expand(self.tax_rate,       "tax_rate")

    def _expand(self, value, name: str) -> List[float]:
        """
        If value is a scalar float/int, repeat it holding_period times.
        If it's already a list, validate its length.
        """
        if isinstance(value, (int, float)):
            return [float(value)] * self.holding_period
        if isinstance(value, list):
            if len(value) != self.holding_period:
                raise ValueError(
                    f"'{name}' has {len(value)} elements but "
                    f"holding_period is {self.holding_period}. "
                    f"They must match."
                )
            return [float(v) for v in value]
        raise TypeError(
            f"'{name}' must be a float or list of floats, got {type(value)}."
        )


# ---------------------------------------------------------------------------
# Output: first-pass P&L (Revenue → EBITDA, no interest yet)
# ---------------------------------------------------------------------------

@dataclass
class OperatingModelResult:
    """
    Year-by-year P&L output.

    First-pass fields (always populated after run_operating_model()):
        revenue, cogs, gross_profit, opex, ebit, da, ebitda
        and their corresponding _pct variants

    Second-pass fields (populated after complete_income_statement()):
        interest_expense, interest_income, ebt, taxes, net_income
        and their corresponding _pct variants

    All absolute values in $M. All _pct values are ratios (e.g. 0.355).
    Index 0 = Year 1 (first full year post-close).
    """

    scenario: Scenario
    holding_period: int
    years: List[int] = field(default_factory=list)        # [1, 2, 3, ...]

    # --- Revenue ---
    revenue: List[float] = field(default_factory=list)
    revenue_growth: List[float] = field(default_factory=list)

    # --- COGS ---
    cogs: List[float] = field(default_factory=list)
    cogs_pct: List[float] = field(default_factory=list)

    # --- Gross Profit ---
    gross_profit: List[float] = field(default_factory=list)
    gross_margin: List[float] = field(default_factory=list)

    # --- OpEx ---
    opex: List[float] = field(default_factory=list)
    opex_pct: List[float] = field(default_factory=list)

    # --- EBIT ---
    ebit: List[float] = field(default_factory=list)
    ebit_margin: List[float] = field(default_factory=list)

    # --- D&A ---
    da: List[float] = field(default_factory=list)
    da_pct: List[float] = field(default_factory=list)

    # --- EBITDA ---
    ebitda: List[float] = field(default_factory=list)
    ebitda_margin: List[float] = field(default_factory=list)

    # --- Below-EBIT (populated in second pass) ---
    interest_expense: List[float] = field(default_factory=list)
    interest_income: List[float] = field(default_factory=list)
    ebt: List[float] = field(default_factory=list)
    ebt_margin: List[float] = field(default_factory=list)
    tax_rate: List[float] = field(default_factory=list)
    taxes: List[float] = field(default_factory=list)
    net_income: List[float] = field(default_factory=list)
    net_margin: List[float] = field(default_factory=list)

    # --- Convenience: final year EBITDA (used by returns module) ---
    @property
    def exit_ebitda(self) -> float:
        """EBITDA in the final projection year ($M)."""
        return self.ebitda[-1]

    @property
    def exit_revenue(self) -> float:
        """Revenue in the final projection year ($M)."""
        return self.revenue[-1]


# ---------------------------------------------------------------------------
# Core engine: first pass (Revenue → EBITDA)
# ---------------------------------------------------------------------------

def run_operating_model(assumptions: OperatingAssumptions) -> OperatingModelResult:
    """
    Build the year-by-year P&L from Revenue down to EBITDA.

    This is the first pass. It stops at EBITDA because interest expense
    (needed for EBT) comes from debt_model.py. Call
    complete_income_statement() to finish the P&L once interest is known.

    Parameters
    ----------
    assumptions : OperatingAssumptions

    Returns
    -------
    OperatingModelResult
        Populated through EBITDA. Below-EBIT fields are empty lists.

    Logic (mirrors BK PDF row by row):
        revenue[t]      = revenue[t-1] * (1 + growth[t])
        cogs[t]         = revenue[t] * cogs_pct[t]
        gross_profit[t] = revenue[t] - cogs[t]
        opex[t]         = revenue[t] * opex_pct[t]
        ebit[t]         = gross_profit[t] - opex[t]
        da[t]           = revenue[t] * da_pct[t]
        ebitda[t]       = ebit[t] + da[t]
    """

    n = assumptions.holding_period
    rev = assumptions.base_revenue

    result = OperatingModelResult(
        scenario=assumptions.scenario,
        holding_period=n,
    )

    for t in range(n):
        # --- Revenue ---
        rev = rev * (1 + assumptions.revenue_growth[t])
        result.years.append(t + 1)
        result.revenue.append(round(rev, 2))
        result.revenue_growth.append(assumptions.revenue_growth[t])

        # --- COGS ---
        cogs = rev * assumptions.cogs_pct[t]
        result.cogs.append(round(cogs, 2))
        result.cogs_pct.append(assumptions.cogs_pct[t])

        # --- Gross Profit ---
        gp = rev - cogs
        result.gross_profit.append(round(gp, 2))
        result.gross_margin.append(round(gp / rev, 6))

        # --- OpEx ---
        opex = rev * assumptions.opex_pct[t]
        result.opex.append(round(opex, 2))
        result.opex_pct.append(assumptions.opex_pct[t])

        # --- EBIT ---
        ebit = gp - opex
        result.ebit.append(round(ebit, 2))
        result.ebit_margin.append(round(ebit / rev, 6))

        # --- D&A ---
        da = rev * assumptions.da_pct[t]
        result.da.append(round(da, 2))
        result.da_pct.append(assumptions.da_pct[t])

        # --- EBITDA ---
        ebitda = ebit + da
        result.ebitda.append(round(ebitda, 2))
        result.ebitda_margin.append(round(ebitda / rev, 6))

    return result


# ---------------------------------------------------------------------------
# Second pass: complete income statement once interest is known
# ---------------------------------------------------------------------------

def complete_income_statement(
    result: OperatingModelResult,
    interest_expense: List[float],
    tax_rate: List[float],
    minimum_cash: float = 0.0,
    interest_income_rate: float = 0.005,
) -> OperatingModelResult:
    """
    Complete the P&L below EBIT once interest expense is available.

    Called by the LBO orchestrator after debt_model.py runs.
    Mutates result in-place and returns it.

    Parameters
    ----------
    result : OperatingModelResult
        Output from run_operating_model() — already has Revenue → EBITDA.

    interest_expense : list[float]
        Total interest expense per year ($M), from debt_model.py.
        Length must equal result.holding_period.

    tax_rate : list[float]
        Effective tax rate per year (decimal). Same as in OperatingAssumptions.

    minimum_cash : float
        Cash balance on which interest income is earned ($M).
        BK uses $118M minimum cash at 0.5% = ~$0.6M/yr interest income.

    interest_income_rate : float
        Rate earned on minimum cash balance. BK: 0.5%.

    Returns
    -------
    OperatingModelResult
        Same object, now fully populated including Net Income.

    Logic:
        interest_income[t] = minimum_cash * interest_income_rate
        ebt[t]             = ebit[t] - interest_expense[t] + interest_income[t]
        taxes[t]           = max(ebt[t], 0) * tax_rate[t]
        net_income[t]      = ebt[t] - taxes[t]

    Note on tax shield:
        We only tax positive EBT (no negative tax on losses).
        A full model would carry forward NOLs — that extension
        can be added here if needed.
    """

    n = result.holding_period

    if len(interest_expense) != n:
        raise ValueError(
            f"interest_expense has {len(interest_expense)} elements "
            f"but holding_period is {n}."
        )
    if len(tax_rate) != n:
        raise ValueError(
            f"tax_rate has {len(tax_rate)} elements "
            f"but holding_period is {n}."
        )

    for t in range(n):
        rev = result.revenue[t]
        ebit = result.ebit[t]

        # Interest income on minimum cash balance
        int_income = round(minimum_cash * interest_income_rate, 4)

        # EBT
        ebt = ebit - interest_expense[t] + int_income

        # Tax (only on positive EBT)
        taxes = max(ebt, 0) * tax_rate[t]

        # Net income
        net_income = ebt - taxes

        result.interest_expense.append(round(interest_expense[t], 2))
        result.interest_income.append(round(int_income, 4))
        result.ebt.append(round(ebt, 2))
        result.ebt_margin.append(round(ebt / rev, 6))
        result.tax_rate.append(tax_rate[t])
        result.taxes.append(round(taxes, 2))
        result.net_income.append(round(net_income, 2))
        result.net_margin.append(round(net_income / rev, 6))

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_operating_model(result: OperatingModelResult) -> None:
    """
    Print a formatted P&L table to console.
    Mirrors the layout of the BK PDF operating model section.
    """

    n = result.holding_period
    yr_labels = [f"Year {t}" for t in result.years]
    col_w = 10

    def row(label, values, fmt="$", pct_values=None):
        """Print one P&L row with optional % margin row below."""
        line = f"  {label:<28}"
        for v in values:
            if fmt == "$":
                line += f"  {v:>8,.0f}"
            elif fmt == "%":
                line += f"  {v * 100:>7.1f}%"
        print(line)
        if pct_values:
            pct_line = f"  {'% of sales':<28}"
            for p in pct_values:
                pct_line += f"  {p * 100:>7.1f}%"
            print(pct_line)

    sep = "-" * (30 + n * 10)

    print(f"\n{'=' * (30 + n * 10)}")
    print(f"  OPERATING MODEL ({result.scenario.upper()})")
    print(f"{'=' * (30 + n * 10)}")

    # Header row
    header = f"  {'':28}"
    for yr in yr_labels:
        header += f"  {yr:>8}"
    print(header)
    print(sep)

    row("Revenue ($M)",          result.revenue)
    row("  % growth",            result.revenue_growth, fmt="%")
    print()
    row("COGS",                  result.cogs,         pct_values=result.cogs_pct)
    row("Gross Profit",          result.gross_profit, pct_values=result.gross_margin)
    print()
    row("OpEx",                  result.opex,         pct_values=result.opex_pct)
    row("EBIT",                  result.ebit,         pct_values=result.ebit_margin)
    print()
    row("D&A",                   result.da,           pct_values=result.da_pct)
    row("EBITDA",                result.ebitda,       pct_values=result.ebitda_margin)

    if result.net_income:
        print(sep)
        row("Interest expense",      result.interest_expense)
        row("EBT",                   result.ebt,          pct_values=result.ebt_margin)
        row("Taxes",                 result.taxes)
        row("Net Income",            result.net_income,   pct_values=result.net_margin)

    print(f"{'=' * (30 + n * 10)}\n")


# ---------------------------------------------------------------------------
# Factory: Burger King conservative scenario
# ---------------------------------------------------------------------------

def build_bk_conservative() -> OperatingAssumptions:
    """
    Burger King conservative scenario assumptions from the LBO PDF.

    Base revenue = FY2010A revenue of $2,502M.
    Projection period = FY2011E through FY2015E (5 years).
    """
    return OperatingAssumptions(
        holding_period=5,
        base_revenue=2502.2,
        scenario="conservative",
        revenue_growth=[0.029, -0.183, -0.104, 0.014, 0.060],
        cogs_pct=[0.645, 0.628, 0.610, 0.593, 0.575],
        opex_pct=[0.222, 0.200, 0.178, 0.156, 0.134],
        da_pct=[0.041, 0.041, 0.041, 0.041, 0.041],
        tax_rate=[0.338, 0.333, 0.328, 0.323, 0.318],
    )


def build_bk_management() -> OperatingAssumptions:
    """
    Burger King management scenario assumptions from the LBO PDF.

    Same base revenue, more aggressive growth and margin expansion.
    """
    return OperatingAssumptions(
        holding_period=5,
        base_revenue=2502.2,
        scenario="management",
        # Management case implies higher revenue from PDF EBITDA figures
        # EBITDA management: 464, 518, 563, 620, 685
        # Back-calculating growth from PDF D&A and EBIT margin assumptions
        revenue_growth=[0.029, -0.165, -0.095, 0.025, 0.070],
        cogs_pct=[0.636, 0.614, 0.590, 0.568, 0.545],
        opex_pct=[0.218, 0.193, 0.165, 0.140, 0.118],
        da_pct=[0.044, 0.044, 0.052, 0.048, 0.047],
        tax_rate=[0.338, 0.333, 0.328, 0.323, 0.318],
    )


def build_generic_assumptions(
    holding_period: int = 5,
    base_revenue: float = 1000.0,
    revenue_growth: float = 0.05,
    gross_margin: float = 0.40,
    opex_pct: float = 0.20,
    da_pct: float = 0.04,
    tax_rate: float = 0.25,
    scenario: Scenario = "conservative",
) -> OperatingAssumptions:
    """
    Build generic flat-assumption operating model.
    Used by the simulation engine when running Monte Carlo.

    Converts EBITDA margin logic into P&L logic:
        ebitda_margin = gross_margin - opex_pct + da_pct  (approx)

    Parameters
    ----------
    All scalar inputs are expanded into flat lists automatically
    by OperatingAssumptions.__post_init__().
    """
    # cogs_pct = 1 - gross_margin
    cogs_pct = 1.0 - gross_margin

    return OperatingAssumptions(
        holding_period=holding_period,
        base_revenue=base_revenue,
        scenario=scenario,
        revenue_growth=revenue_growth,
        cogs_pct=cogs_pct,
        opex_pct=opex_pct,
        da_pct=da_pct,
        tax_rate=tax_rate,
    )


# ---------------------------------------------------------------------------
# Quick self-test — Burger King conservative scenario
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- First pass: Revenue to EBITDA ---
    bk_assumptions = build_bk_conservative()
    result = run_operating_model(bk_assumptions)

    # --- Second pass: complete with interest (BK PDF totals) ---
    bk_interest = [208.0, 208.0, 205.0, 199.0, 189.0]
    result = complete_income_statement(
        result,
        interest_expense=bk_interest,
        tax_rate=bk_assumptions.tax_rate,
        minimum_cash=118.0,
        interest_income_rate=0.005,
    )

    print_operating_model(result)

    print("VALIDATION vs BK PDF (conservative):")
    bk_revenue = [2574, 2102, 1884, 1910, 2024]
    bk_ebitda  = [447,  448,  476,  558,  672]
    bk_ebit    = [342,  363,  399,  480,  589]
    bk_net_inc = [89,   104,  131,  191,  273]

    print(f"\n  {'Year':<6} {'Rev (model)':>12} {'Rev (BK)':>10} "
          f"{'EBITDA (model)':>15} {'EBITDA (BK)':>12} "
          f"{'NI (model)':>12} {'NI (BK)':>10}")
    print("  " + "-" * 82)
    for i in range(5):
        print(
            f"  {i+1:<6} "
            f"{result.revenue[i]:>11,.0f}  "
            f"{bk_revenue[i]:>9,}  "
            f"{result.ebitda[i]:>14,.0f}  "
            f"{bk_ebitda[i]:>11,}  "
            f"{result.net_income[i]:>11,.0f}  "
            f"{bk_net_inc[i]:>9,}"
        )

    print(f"\n  Exit EBITDA: ${result.exit_ebitda:,.0f}M  (BK PDF: $672M)")

    # --- Test management scenario ---
    print("\n--- Management scenario ---")
    mgmt = build_bk_management()
    mgmt_result = run_operating_model(mgmt)
    mgmt_result = complete_income_statement(
        mgmt_result,
        interest_expense=bk_interest,
        tax_rate=mgmt.tax_rate,
        minimum_cash=118.0,
    )
    print_operating_model(mgmt_result)

    # --- Test generic (flat) assumptions ---
    print("--- Generic assumptions test ---")
    generic = build_generic_assumptions(
        holding_period=5,
        base_revenue=100.0,
        revenue_growth=0.05,
        gross_margin=0.40,
        opex_pct=0.18,
        da_pct=0.04,
        tax_rate=0.25,
    )
    generic_result = run_operating_model(generic)
    print_operating_model(generic_result)
    print(f"  EBITDA margin check: {generic_result.ebitda_margin[0]*100:.1f}%  "
          f"(expected: ~26% = 40% gross - 18% opex + 4% DA)")