"""
debt_model.py
-------------
Year-by-year debt schedule for every tranche in the capital structure.

For each tranche, each year:
    Beginning balance
    - Mandatory amortization      (from Tranche.mandatory_repayment())
    - Cash sweep                  (excess FCF, applied in priority order)
    = Ending balance
    Interest expense              (rate * beginning balance)

The cash sweep waterfall:
    1. Compute available cash = levered_fcf + beginning_cash - minimum_cash
    2. If available > 0, distribute to sweep-eligible tranches in
       sweep_priority order (lowest number first)
    3. Each tranche absorbs as much as possible (capped at its balance)
    4. Remaining unswept cash builds on the balance sheet

Why interest on beginning balance:
    Standard LBO convention. Interest is charged on the balance at the
    START of the year. This matches the BK PDF exactly:
        USD term loan Year 1: $1,510M * 6.82% = $103M  (PDF: $103M)
        EUR term loan Year 1:   $334M * 7.11% =  $24M  (PDF: $24M)
        Senior Notes  Year 1:   $800M * 10.19% =  $82M  (PDF: $82M)
        Total interest Year 1:                   $208M  (PDF: $208M)

The outputs of this module feed two places:
    1. operating_model.complete_income_statement()
       needs interest_expense[] per year
    2. returns.py
       needs net_debt_at_exit = sum of ending balances in final year

Burger King reference (debt schedule, conservative):
    Year:                   2011E   2012E   2013E   2014E   2015E
    Total debt begin:       2,644   2,644   2,626   2,572   2,459
    Mandatory repay:            0       0       0       0       0
    Cash sweep (USD TL):        0      18      54     113     191
    Total debt end:         2,644   2,626   2,572   2,459   2,268
    Total interest:           208     208     205     199     189

    Note: The BK model shows $0 mandatory repayment in year 1 because
    the $15M annual amort on the USD term loan is funded from excess
    balance sheet cash, not modelled as a separate FCF line.
    We make this explicit via the include_mandatory_repayments flag
    in cashflow_model.py.
"""

from dataclasses import dataclass, field
from typing import List, Dict
from lbo_engine.capital_structure import CapitalStructure, Tranche
from lbo_engine.cashflow_model import CashFlowResult


# ---------------------------------------------------------------------------
# Per-tranche year record
# ---------------------------------------------------------------------------

@dataclass
class TrancheYearRecord:
    """
    Single year snapshot for one tranche.

    All values in $M.
    """
    year: int
    tranche_name: str
    beginning_balance: float
    mandatory_repayment: float
    cash_sweep: float
    ending_balance: float
    interest_expense: float
    interest_rate: float


# ---------------------------------------------------------------------------
# Full debt schedule output
# ---------------------------------------------------------------------------

@dataclass
class DebtScheduleResult:
    """
    Complete debt schedule for all tranches over the holding period.

    Structure:
        schedule[tranche_name] = list of TrancheYearRecord (one per year)
        interest_expense[year_index] = total interest across all tranches
        net_debt_at_exit = sum of all ending balances in final year

    The interest_expense list is what gets passed to
    complete_income_statement() in operating_model.py.
    """

    holding_period: int
    years: List[int] = field(default_factory=list)

    # Per-tranche schedule: {tranche_name: [TrancheYearRecord, ...]}
    schedule: Dict[str, List[TrancheYearRecord]] = field(default_factory=dict)

    # Aggregates per year (lists, index 0 = Year 1)
    total_beginning_debt: List[float] = field(default_factory=list)
    total_mandatory_repayment: List[float] = field(default_factory=list)
    total_cash_sweep: List[float] = field(default_factory=list)
    total_ending_debt: List[float] = field(default_factory=list)
    total_interest_expense: List[float] = field(default_factory=list)
    cash_balance: List[float] = field(default_factory=list)   # balance sheet cash each year
    available_for_sweep: List[float] = field(default_factory=list)

    @property
    def net_debt_at_exit(self) -> float:
        """
        Net debt at exit = total ending debt in final year - final cash balance.
        Fed directly into returns.py.

        BK reference: $2,268M debt - $118M cash = $2,150M net debt.
        """
        return round(self.total_ending_debt[-1] - self.cash_balance[-1], 2)

    @property
    def interest_expense(self) -> List[float]:
        """
        Alias for total_interest_expense.
        Passed to complete_income_statement() in operating_model.py.
        """
        return self.total_interest_expense

    def ending_balance(self, tranche_name: str) -> List[float]:
        """Ending balances for a specific tranche over all years."""
        return [r.ending_balance for r in self.schedule[tranche_name]]

    def total_debt_repaid(self) -> float:
        """Total principal repaid over the holding period ($M)."""
        return round(
            sum(self.total_mandatory_repayment) + sum(self.total_cash_sweep), 2
        )


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def run_debt_model(
    capital_structure: CapitalStructure,
    cf_result: CashFlowResult,
    minimum_cash: float = 0.0,
    opening_cash: float = 0.0,
    pure_sweep_mode: bool = False,
) -> DebtScheduleResult:
    """
    Run the full year-by-year debt schedule and cash sweep waterfall.

    Parameters
    ----------
    capital_structure : CapitalStructure
        All tranches with their rates, amort schedules, and sweep flags.

    cf_result : CashFlowResult
        Output from cashflow_model.py. The levered_fcf list drives the sweep.

    minimum_cash : float
        Minimum cash balance maintained on the balance sheet ($M).
        BK: $118M. Cash above this floor is swept to debt.

    opening_cash : float
        Cash balance at the START of year 1 ($M).
        In BK: $118M (minimum cash retained post-close).
        Excess cash above minimum was used as a source of funds at close.

    pure_sweep_mode : bool
        If True, mandatory amortization is NOT deducted from FCF before
        computing the available sweep. Instead, all FCF above minimum
        cash is swept directly (BK PDF convention).

        Background: The BK deal pre-funded mandatory amortization from
        the excess cash at close ($188M total - $118M minimum = $70M).
        So the FCF sweep is purely the excess over the minimum cash floor,
        with no deduction for scheduled amort. This exactly replicates
        the BK PDF debt schedule (sweep: 0, 18, 54, 113, 191).

        If False (default, standard convention): mandatory amortization
        is deducted from FCF first, then excess cash sweeps debt. This
        is more conservative and correct for deals without pre-funded amort.

    Returns
    -------
    DebtScheduleResult

    Algorithm per year t:
        Step 1: Record beginning balances (prior year ending balances)
        Step 2: Compute mandatory amortization per tranche
        Step 3: Compute interest on beginning balance per tranche
        Step 4: Compute cash available for sweep
        Step 5: Distribute sweep to eligible tranches in priority order
        Step 6: Compute ending balances
        Step 7: Compute ending cash balance
    """

    n = cf_result.holding_period
    tranches = capital_structure.tranches
    sweep_tranches = capital_structure.sweep_eligible_tranches  # sorted by priority

    # Initialise running balances
    balances = {t.name: t.amount for t in tranches}
    cash = opening_cash

    result = DebtScheduleResult(holding_period=n)

    # Initialise schedule dict
    for t in tranches:
        result.schedule[t.name] = []

    for yr_idx in range(n):
        year = yr_idx + 1
        result.years.append(year)

        # ------------------------------------------------------------------
        # Step 1: Beginning balances (snapshot at start of year)
        # ------------------------------------------------------------------
        beg_balances = {t.name: balances[t.name] for t in tranches}
        total_beg = sum(beg_balances.values())

        # ------------------------------------------------------------------
        # Step 2: Mandatory amortization per tranche
        # ------------------------------------------------------------------
        mandatory = {}
        for t in tranches:
            # Cannot repay more than current balance
            sched = t.mandatory_repayment(year, n)
            mandatory[t.name] = min(sched, beg_balances[t.name])

        total_mandatory = sum(mandatory.values())

        # ------------------------------------------------------------------
        # Step 3: Interest on beginning balance
        # ------------------------------------------------------------------
        interest = {}
        for t in tranches:
            interest[t.name] = t.annual_interest(beg_balances[t.name])

        total_interest = sum(interest.values())

        # ------------------------------------------------------------------
        # Step 4: Cash available for sweep
        #
        # Standard mode (pure_sweep_mode=False):
        #   cash_in_hand = cash + levered_fcf
        #   after_mandatory = cash_in_hand - mandatory_repayments
        #   available_sweep = max(after_mandatory - minimum_cash, 0)
        #
        # BK / pure sweep mode (pure_sweep_mode=True):
        #   Mandatory amortization is pre-funded from balance sheet cash.
        #   available_sweep = max(cash + levered_fcf - minimum_cash, 0)
        #   This exactly replicates the BK PDF schedule.
        #
        # In both cases we cannot sweep below the minimum cash floor.
        # ------------------------------------------------------------------
        levered_fcf = cf_result.levered_fcf[yr_idx]
        cash_in_hand = cash + levered_fcf

        if pure_sweep_mode:
            # All FCF above minimum cash floor goes to sweep
            # Mandatory amortization funded separately (balance sheet cash)
            available_sweep = max(cash_in_hand - minimum_cash, 0.0)
        else:
            # Mandatory amortization deducted from FCF first
            cash_after_mandatory = cash_in_hand - total_mandatory
            available_sweep = max(cash_after_mandatory - minimum_cash, 0.0)

        # ------------------------------------------------------------------
        # Step 5: Cash sweep waterfall
        # Distribute to sweep-eligible tranches in priority order.
        # Each tranche is paid down as much as possible.
        # ------------------------------------------------------------------
        sweep = {t.name: 0.0 for t in tranches}
        remaining_sweep = available_sweep

        for t in sweep_tranches:
            if remaining_sweep <= 0:
                break
            # Balance after mandatory amortization
            balance_post_mandatory = beg_balances[t.name] - mandatory[t.name]
            # Sweep amount is capped at remaining balance
            swept = min(remaining_sweep, balance_post_mandatory)
            sweep[t.name] = round(max(swept, 0.0), 4)
            remaining_sweep -= sweep[t.name]

        total_sweep = sum(sweep.values())

        # ------------------------------------------------------------------
        # Step 6: Ending balances
        # ------------------------------------------------------------------
        end_balances = {}
        for t in tranches:
            end_bal = beg_balances[t.name] - mandatory[t.name] - sweep[t.name]
            end_balances[t.name] = round(max(end_bal, 0.0), 4)

        total_end = sum(end_balances.values())

        # ------------------------------------------------------------------
        # Step 7: Ending cash balance
        # Cash = minimum_cash + any unswept excess
        # In BK, cash stays at minimum_cash = $118M throughout.
        # ------------------------------------------------------------------
        unswept = remaining_sweep   # any cash that couldn't be used (debt fully repaid)
        ending_cash = minimum_cash + unswept
        cash = ending_cash          # carry forward to next year

        # ------------------------------------------------------------------
        # Step 8: Store per-tranche records
        # ------------------------------------------------------------------
        for t in tranches:
            record = TrancheYearRecord(
                year=year,
                tranche_name=t.name,
                beginning_balance=round(beg_balances[t.name], 2),
                mandatory_repayment=round(mandatory[t.name], 2),
                cash_sweep=round(sweep[t.name], 2),
                ending_balance=round(end_balances[t.name], 2),
                interest_expense=round(interest[t.name], 2),
                interest_rate=t.interest_rate,
            )
            result.schedule[t.name].append(record)

        # ------------------------------------------------------------------
        # Step 9: Store aggregate row
        # ------------------------------------------------------------------
        result.total_beginning_debt.append(round(total_beg, 2))
        result.total_mandatory_repayment.append(round(total_mandatory, 2))
        result.total_cash_sweep.append(round(total_sweep, 2))
        result.total_ending_debt.append(round(total_end, 2))
        result.total_interest_expense.append(round(total_interest, 2))
        result.cash_balance.append(round(ending_cash, 2))
        result.available_for_sweep.append(round(available_sweep, 2))

        # Update running balances for next year
        balances = end_balances

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_debt_schedule(result: DebtScheduleResult) -> None:
    """
    Print per-tranche debt schedule and aggregate summary.
    Mirrors the BK PDF debt schedule section layout.
    """

    n = result.holding_period
    yr_labels = [f"Year {t}" for t in result.years]
    sep = "-" * (32 + n * 10)
    wide_sep = "=" * (32 + n * 10)

    def row(label, values, indent=0):
        pad = "  " + " " * indent
        line = f"{pad}{label:<{30 - indent}}"
        for v in values:
            line += f"  {v:>8,.0f}"
        print(line)

    # --- Header ---
    print(f"\n{wide_sep}")
    print(f"  DEBT SCHEDULE")
    print(wide_sep)

    header = f"  {'':30}"
    for yr in yr_labels:
        header += f"  {yr:>8}"
    print(header)

    # --- Per-tranche detail ---
    for tranche_name, records in result.schedule.items():
        print(f"\n  {tranche_name}")
        print(sep)
        row("Beginning balance",
            [r.beginning_balance for r in records], indent=2)
        row("Mandatory repayment",
            [r.mandatory_repayment for r in records], indent=2)
        row("Cash sweep",
            [r.cash_sweep for r in records], indent=2)
        row("Ending balance",
            [r.ending_balance for r in records], indent=2)
        row("Interest expense",
            [r.interest_expense for r in records], indent=2)
        rate = records[0].interest_rate
        print(f"    Interest rate: {rate * 100:.2f}%")

    # --- Aggregate summary ---
    print(f"\n  {'AGGREGATE SUMMARY':30}")
    print(wide_sep)
    row("Total debt — beginning",  result.total_beginning_debt)
    row("Total mandatory repay",   result.total_mandatory_repayment)
    row("Total cash sweep",        result.total_cash_sweep)
    row("Total debt — ending",     result.total_ending_debt)
    print(sep)
    row("Cash balance",            result.cash_balance)
    row("Available for sweep",     result.available_for_sweep)
    print(sep)
    row("Total interest expense",  result.total_interest_expense)

    print(wide_sep)
    print(f"  Net debt at exit:      ${result.net_debt_at_exit:>10,.1f}M")
    print(f"  Total debt repaid:     ${result.total_debt_repaid():>10,.1f}M")
    print(f"  Debt / entry debt:     "
          f"{result.total_ending_debt[-1] / result.total_beginning_debt[0] * 100:.1f}%"
          f"  remaining at exit\n")


# ---------------------------------------------------------------------------
# Factory: BK debt model setup
# ---------------------------------------------------------------------------

def build_bk_debt_model_inputs():
    """
    Return the capital structure and minimum cash needed to run
    the BK debt model. Convenience wrapper for the self-test.
    """
    from lbo_engine.capital_structure import build_bk_capital_structure
    cs = build_bk_capital_structure()
    minimum_cash = 118.0
    opening_cash = 118.0   # minimum cash retained post-close
    return cs, minimum_cash, opening_cash


# ---------------------------------------------------------------------------
# Quick self-test — Burger King conservative scenario end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from lbo_engine.operating_model import (
        build_bk_conservative,
        run_operating_model,
        complete_income_statement,
    )
    from lbo_engine.cashflow_model import (
        build_bk_cashflow_assumptions,
        run_cashflow_model,
    )

    # --- Step 1: Operating model (first pass) ---
    bk_op = build_bk_conservative()
    op_result = run_operating_model(bk_op)

    # --- Step 2: Debt model with placeholder interest for first pass ---
    # We use BK PDF interest values to bootstrap.
    # In the full orchestrator (lbo_engine/model.py), this will be
    # solved iteratively: operating model -> debt model -> complete P&L.
    placeholder_interest = [208.0, 208.0, 205.0, 199.0, 189.0]
    op_result = complete_income_statement(
        op_result,
        interest_expense=placeholder_interest,
        tax_rate=bk_op.tax_rate,
        minimum_cash=118.0,
        interest_income_rate=0.005,
    )

    # --- Step 3: Cash flow model ---
    cf_assumptions = build_bk_cashflow_assumptions()
    cf_result = run_cashflow_model(op_result, cf_assumptions)

    # --- Step 4: Debt schedule ---
    cs, min_cash, open_cash = build_bk_debt_model_inputs()

    # BK PDF convention: mandatory amort pre-funded from balance sheet cash
    debt_result = run_debt_model(
        cs, cf_result, min_cash, open_cash, pure_sweep_mode=True
    )

    print_debt_schedule(debt_result)

    # --- Validate against BK PDF ---
    bk_total_interest = [208, 208, 205, 199, 189]
    bk_total_debt_end = [2644, 2626, 2572, 2459, 2268]
    bk_usd_sweep      = [0, 18, 54, 113, 191]

    print("VALIDATION vs BK PDF (pure_sweep_mode=True):")
    print()
    print(f"  {'Year':<6} "
          f"{'Int (model)':>12} {'Int (BK)':>10} "
          f"{'Debt end (model)':>17} {'Debt end (BK)':>14} "
          f"{'Sweep (model)':>14} {'Sweep (BK)':>11}")
    print("  " + "-" * 88)

    usd_tl_name = "USD Secured Term Loan"
    for i in range(5):
        model_sweep = debt_result.schedule[usd_tl_name][i].cash_sweep
        print(
            f"  {i+1:<6} "
            f"{debt_result.total_interest_expense[i]:>11,.0f}  "
            f"{bk_total_interest[i]:>9,}  "
            f"{debt_result.total_ending_debt[i]:>16,.0f}  "
            f"{bk_total_debt_end[i]:>13,}  "
            f"{model_sweep:>13,.0f}  "
            f"{bk_usd_sweep[i]:>10,}"
        )

    print(f"\n  Net debt at exit: ${debt_result.net_debt_at_exit:,.0f}M  "
          f"(BK PDF: $2,150M)")
    print(f"  Cash balance Y5:  ${debt_result.cash_balance[-1]:,.0f}M  "
          f"(BK PDF: $118M)")

    # --- Standard mode comparison (more conservative, for generic deals) ---
    print("\n--- Standard mode (pure_sweep_mode=False) for comparison ---")
    debt_result_std = run_debt_model(
        cs, cf_result, min_cash, open_cash, pure_sweep_mode=False
    )
    print(f"  Net debt at exit (standard): ${debt_result_std.net_debt_at_exit:,.0f}M")
    print(f"  Net debt at exit (BK mode):  ${debt_result.net_debt_at_exit:,.0f}M")
    print(f"  Difference: ${debt_result_std.net_debt_at_exit - debt_result.net_debt_at_exit:,.0f}M "
          f"(standard is more conservative — less sweep, more debt remaining)")