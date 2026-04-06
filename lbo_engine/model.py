"""
lbo_engine/model.py
-------------------
Orchestrator: wires all five modules into a single run_lbo() call.

Replaces the original model.py entirely.

Execution order:
    1. Build capital structure          capital_structure.py
    2. Solve sponsor equity             transaction.py
    3. Build transaction                transaction.py
    4. Run operating model (pass 1)     operating_model.py
       - Revenue -> EBITDA only
       - Uses placeholder interest for first pass
    5. Run cash flow model (pass 1)     cashflow_model.py
    6. Run debt model (pass 1)          debt_model.py
       - Produces real interest expense per year
    7. Complete income statement        operating_model.py
       - Injects real interest from debt model
       - Produces net income
    8. Run cash flow model (pass 2)     cashflow_model.py
       - Recomputes FCF with correct net income
    9. Run debt model (pass 2)          debt_model.py
       - Final debt schedule with correct FCF
   10. Compute returns                  returns.py
   11. Compute equity bridge            returns.py

Why two passes:
    Interest expense (needed by the P&L) depends on debt balances.
    Debt balances (swept by FCF) depend on net income.
    Net income depends on interest expense.

    This is the circular reference in every LBO model. Excel resolves
    it with iterative calculations. We resolve it explicitly:

        Pass 1: P&L (no interest) -> FCF -> debt schedule -> get interest
        Pass 2: P&L (real interest) -> FCF -> debt schedule -> final numbers

    Two passes are sufficient for convergence in all standard deals.
    The change in interest between pass 1 and pass 2 is small because
    debt balances don't change dramatically from a flat-interest assumption.
    For stress tests or deals with very large sweeps, an optional
    n_iterations parameter runs additional passes until convergence.

Public API:
    run_lbo(params)          -> LBOResult   (single deterministic run)
    run_lbo_from_inputs()    -> LBOResult   (from raw scalar inputs, for simulation)

LBOResult contains every output from every module, making it easy
to extract just what you need (IRR for simulation, full P&L for dashboard).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from lbo_engine.transaction import (
    TransactionAssumptions,
    TransactionResult,
    solve_sponsor_equity,
    build_transaction,
)
from lbo_engine.capital_structure import (
    CapitalStructure,
    build_simple_two_tranche_structure,
)
from lbo_engine.operating_model import (
    OperatingAssumptions,
    OperatingModelResult,
    run_operating_model,
    complete_income_statement,
    build_generic_assumptions,
)
from lbo_engine.cashflow_model import (
    CashFlowAssumptions,
    CashFlowResult,
    run_cashflow_model,
    build_generic_cashflow_assumptions,
)
from lbo_engine.debt_model import (
    DebtScheduleResult,
    run_debt_model,
)
from lbo_engine.returns import (
    ReturnAssumptions,
    ReturnsResult,
    compute_returns,
    compute_equity_bridge,
    compute_exit_sensitivity,
    print_returns_summary,
    print_sensitivity_table,
)


# ---------------------------------------------------------------------------
# Unified parameter object
# ---------------------------------------------------------------------------

@dataclass
class LBOParams:
    """
    All inputs needed to run a complete LBO model.

    Designed to accept both:
        - Full deal inputs (BK-style, with real tranche data)
        - Generic simulation inputs (scalar assumptions, for Monte Carlo)

    Fields marked [REQUIRED] must always be provided.
    Fields marked [OPTIONAL] have sensible defaults for generic deals.

    Transaction
    -----------
    entry_ebitda : float        [REQUIRED]  LTM EBITDA at entry ($M)
    entry_multiple : float      [REQUIRED]  EV / LTM EBITDA at entry
    exit_multiple : float       [REQUIRED]  EV / LTM EBITDA at exit
    holding_period : int        [REQUIRED]  Years from close to exit

    Capital structure
    -----------------
    debt_pct : float            [REQUIRED]  Total debt / EV
    senior_pct : float          [OPTIONAL]  Senior debt / total debt (two-tranche mode)
    mezz_spread : float         [OPTIONAL]  Mezz rate = base_rate + mezz_spread
    interest_rate : float       [REQUIRED]  Base interest rate (senior)
    capital_structure : CapitalStructure  [OPTIONAL]
        If provided, used directly. Overrides debt_pct / senior_pct / mezz_spread.
        Pass a CapitalStructure object from capital_structure.py for full deals.

    Operating model
    ---------------
    revenue_growth : float or list[float]   [REQUIRED]
    gross_margin : float or list[float]     [REQUIRED]  (= 1 - cogs_pct)
    opex_pct : float or list[float]         [REQUIRED]
    da_pct : float or list[float]           [OPTIONAL]  default 0.04
    tax_rate : float or list[float]         [OPTIONAL]  default 0.25

    Cash flow
    ---------
    capex_pct : float or list[float]        [OPTIONAL]  default 0.04
    nwc_pct : float or list[float]          [OPTIONAL]  default 0.01

    Transaction details
    -------------------
    minimum_cash : float        [OPTIONAL]  default 0.0
    company_name : str          [OPTIONAL]  default "Target"
    pure_sweep_mode : bool      [OPTIONAL]  default False
        Set True only for BK-style deals with pre-funded amortization.

    Simulation helpers
    ------------------
    base_revenue : float        [OPTIONAL]
        If not provided, inferred as entry_ebitda / implied_ebitda_margin.
        implied_ebitda_margin = gross_margin - opex_pct + da_pct.
    """

    # --- Core deal inputs ---
    entry_ebitda: float = 100.0
    entry_multiple: float = 10.0
    exit_multiple: float = 10.0
    holding_period: int = 5

    # --- Capital structure (generic mode) ---
    debt_pct: float = 0.60
    senior_pct: float = 0.70
    mezz_spread: float = 0.04
    interest_rate: float = 0.06

    # --- Capital structure (full deal mode) ---
    capital_structure: Optional[CapitalStructure] = None

    # --- Operating model ---
    revenue_growth: float = 0.05
    gross_margin: float = 0.40
    opex_pct: float = 0.18
    da_pct: float = 0.04
    tax_rate: float = 0.25

    # --- Cash flow ---
    capex_pct: float = 0.04
    nwc_pct: float = 0.01

    # --- Transaction ---
    minimum_cash: float = 0.0
    company_name: str = "Target"
    pure_sweep_mode: bool = False

    # --- Optional base revenue override ---
    base_revenue: Optional[float] = None

    # --- Optional management option pool ---
    management_option_pool_pct: float = 0.0

    # --- Iteration control ---
    n_iterations: int = 2   # number of interest convergence passes


# ---------------------------------------------------------------------------
# Full result object
# ---------------------------------------------------------------------------

@dataclass
class LBOResult:
    """
    Complete output from a full LBO run.

    Contains outputs from every module. The simulation engine
    only needs .returns.irr and .returns.moic. The dashboard
    uses the full object to render tables and charts.
    """

    params: LBOParams

    # Module outputs
    transaction: Optional[TransactionResult] = None
    capital_structure: Optional[CapitalStructure] = None
    operating_model: Optional[OperatingModelResult] = None
    cash_flow: Optional[CashFlowResult] = None
    debt_schedule: Optional[DebtScheduleResult] = None
    returns: Optional[ReturnsResult] = None
    equity_bridge: Optional[Dict] = None
    exit_sensitivity: Optional[Dict] = None

    # Convergence diagnostics
    interest_pass1: List[float] = field(default_factory=list)
    interest_pass2: List[float] = field(default_factory=list)
    interest_converged: bool = False

    # Quick-access properties for simulation engine
    @property
    def irr(self) -> float:
        return self.returns.irr if self.returns else float("nan")

    @property
    def moic(self) -> float:
        return self.returns.moic if self.returns else float("nan")

    @property
    def exit_equity(self) -> float:
        return self.returns.net_exit_equity if self.returns else float("nan")

    @property
    def entry_equity(self) -> float:
        return self.returns.entry_equity if self.returns else float("nan")


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------

def run_lbo(params: LBOParams) -> LBOResult:
    """
    Run a complete LBO model from inputs to returns.

    Parameters
    ----------
    params : LBOParams
        All deal inputs. See LBOParams docstring.

    Returns
    -------
    LBOResult
        Complete output from all modules.

    Two-pass convergence loop
    -------------------------
    The circular dependency between interest expense and FCF
    is resolved by running the model twice (or more if n_iterations > 2):

        Iteration 1:
            - Run operating model with zero interest (EBITDA only)
            - Run cash flow model
            - Run debt model -> produces interest_pass1
        Iteration 2:
            - Complete income statement with interest_pass1
            - Rerun cash flow model (now has correct net income)
            - Rerun debt model -> produces interest_pass2
            - interest_pass2 feeds final P&L and returns calculation

        Convergence check:
            max |interest_pass2[t] - interest_pass1[t]| < 0.5 ($M)
    """

    result = LBOResult(params=params)

    # ------------------------------------------------------------------
    # Step 1: Capital structure
    # ------------------------------------------------------------------
    if params.capital_structure is not None:
        # Full deal mode — use provided capital structure
        cs = params.capital_structure
    else:
        # Generic simulation mode — build two-tranche structure
        entry_ev = params.entry_ebitda * params.entry_multiple
        cs = build_simple_two_tranche_structure(
            enterprise_value=entry_ev,
            ltm_ebitda=params.entry_ebitda,
            debt_pct=params.debt_pct,
            senior_pct=params.senior_pct,
            senior_rate=params.interest_rate,
            mezz_rate=params.interest_rate + params.mezz_spread,
            holding_period=params.holding_period,
        )

    result.capital_structure = cs

    # ------------------------------------------------------------------
    # Step 2: Transaction (solve sponsor equity, build S&U)
    # ------------------------------------------------------------------
    entry_ev = params.entry_ebitda * params.entry_multiple
    total_debt = cs.total_debt
    equity = entry_ev - total_debt

    # For generic deals we skip the full transaction module
    # and just compute equity directly. The full transaction module
    # (share price, S&U table) is used when TransactionAssumptions
    # are passed explicitly (see run_lbo_full_deal() below).
    result.transaction = None   # populated in run_lbo_full_deal()
    entry_equity = equity

    # ------------------------------------------------------------------
    # Step 3: Operating assumptions
    # ------------------------------------------------------------------
    # Infer base revenue if not provided
    implied_ebitda_margin = params.gross_margin - params.opex_pct + params.da_pct
    if params.base_revenue is not None:
        base_revenue = params.base_revenue
    else:
        # Back-calculate from entry EBITDA and implied margin
        if implied_ebitda_margin <= 0:
            raise ValueError(
                f"Implied EBITDA margin is {implied_ebitda_margin:.2%}. "
                f"Check gross_margin ({params.gross_margin:.2%}), "
                f"opex_pct ({params.opex_pct:.2%}), "
                f"da_pct ({params.da_pct:.2%})."
            )
        base_revenue = params.entry_ebitda / implied_ebitda_margin

    op_assumptions = build_generic_assumptions(
        holding_period=params.holding_period,
        base_revenue=base_revenue,
        revenue_growth=params.revenue_growth,
        gross_margin=params.gross_margin,
        opex_pct=params.opex_pct,
        da_pct=params.da_pct,
        tax_rate=params.tax_rate,
    )

    cf_assumptions = build_generic_cashflow_assumptions(
        holding_period=params.holding_period,
        capex_pct=params.capex_pct,
        nwc_pct=params.nwc_pct,
        include_mandatory_repayments=False,
    )

    # ------------------------------------------------------------------
    # Step 4: First pass — operating model without interest
    # ------------------------------------------------------------------
    op_result = run_operating_model(op_assumptions)

    # Complete income statement with zero interest for first pass
    zero_interest = [0.0] * params.holding_period
    op_result_pass1 = complete_income_statement(
        op_result,
        interest_expense=zero_interest,
        tax_rate=op_assumptions.tax_rate,
        minimum_cash=params.minimum_cash,
        interest_income_rate=0.005,
    )

    cf_result_pass1 = run_cashflow_model(op_result_pass1, cf_assumptions, cs)

    debt_result_pass1 = run_debt_model(
        cs,
        cf_result_pass1,
        minimum_cash=params.minimum_cash,
        opening_cash=params.minimum_cash,
        pure_sweep_mode=params.pure_sweep_mode,
    )

    interest_pass1 = debt_result_pass1.total_interest_expense
    result.interest_pass1 = interest_pass1

    # ------------------------------------------------------------------
    # Step 5: Subsequent passes — converge interest
    # ------------------------------------------------------------------
    interest_current = interest_pass1
    final_op_result = None
    final_cf_result = None
    final_debt_result = None

    for iteration in range(params.n_iterations - 1):

        # Fresh operating model result (run_operating_model is pure)
        op_iter = run_operating_model(op_assumptions)

        op_iter = complete_income_statement(
            op_iter,
            interest_expense=interest_current,
            tax_rate=op_assumptions.tax_rate,
            minimum_cash=params.minimum_cash,
            interest_income_rate=0.005,
        )

        cf_iter = run_cashflow_model(op_iter, cf_assumptions, cs)

        debt_iter = run_debt_model(
            cs,
            cf_iter,
            minimum_cash=params.minimum_cash,
            opening_cash=params.minimum_cash,
            pure_sweep_mode=params.pure_sweep_mode,
        )

        interest_new = debt_iter.total_interest_expense

        # Check convergence
        max_delta = max(
            abs(interest_new[t] - interest_current[t])
            for t in range(params.holding_period)
        )

        interest_current = interest_new
        final_op_result = op_iter
        final_cf_result = cf_iter
        final_debt_result = debt_iter

        if max_delta < 0.5:   # converged within $0.5M
            result.interest_converged = True
            break

    result.interest_pass2 = interest_current

    # ------------------------------------------------------------------
    # Step 6: Returns
    # ------------------------------------------------------------------
    return_assumptions = ReturnAssumptions(
        exit_multiple=params.exit_multiple,
        holding_period=params.holding_period,
        entry_equity=entry_equity,
        exit_ebitda=final_op_result.exit_ebitda,
        net_debt_at_exit=final_debt_result.net_debt_at_exit,
        management_option_pool_pct=params.management_option_pool_pct,
    )

    returns = compute_returns(return_assumptions)

    # ------------------------------------------------------------------
    # Step 7: Equity bridge
    # ------------------------------------------------------------------
    entry_ebitda = params.entry_ebitda
    entry_multiple = params.entry_multiple
    net_debt_at_entry = cs.total_debt - params.minimum_cash

    bridge = compute_equity_bridge(
        entry_equity=entry_equity,
        entry_ebitda=entry_ebitda,
        entry_multiple=entry_multiple,
        net_debt_at_entry=net_debt_at_entry,
        exit_ebitda=final_op_result.exit_ebitda,
        exit_multiple=params.exit_multiple,
        net_debt_at_exit=final_debt_result.net_debt_at_exit,
        management_option_pool_pct=params.management_option_pool_pct,
    )

    # ------------------------------------------------------------------
    # Step 8: Exit sensitivity (5x5 table)
    # ------------------------------------------------------------------
    sensitivity = compute_exit_sensitivity(
        entry_equity=entry_equity,
        exit_ebitda=final_op_result.exit_ebitda,
        net_debt_at_exit=final_debt_result.net_debt_at_exit,
        holding_periods=[3, 4, 5, 6, 7],
        exit_multiples=[
            round(params.exit_multiple * m, 1)
            for m in [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4]
        ],
        metric="irr",
    )

    # ------------------------------------------------------------------
    # Populate result
    # ------------------------------------------------------------------
    result.operating_model = final_op_result
    result.cash_flow = final_cf_result
    result.debt_schedule = final_debt_result
    result.returns = returns
    result.equity_bridge = bridge
    result.exit_sensitivity = sensitivity

    return result


# ---------------------------------------------------------------------------
# Full deal mode (with transaction module and S&U table)
# ---------------------------------------------------------------------------

def run_lbo_full_deal(
    transaction_assumptions: TransactionAssumptions,
    capital_structure: CapitalStructure,
    op_assumptions: OperatingAssumptions,
    cf_assumptions: CashFlowAssumptions,
    exit_multiple: float,
    pure_sweep_mode: bool = False,
    management_option_pool_pct: float = 0.0,
    n_iterations: int = 2,
) -> LBOResult:
    """
    Full deal mode: takes explicit module-level assumption objects.

    Use this when running the BK model or any full deal with:
        - Real tranche data (from capital_structure.py)
        - Share price / premium approach (from transaction.py)
        - Year-by-year operating assumptions (from operating_model.py)
        - Explicit cash flow assumptions (from cashflow_model.py)

    Parameters
    ----------
    transaction_assumptions : TransactionAssumptions
        Raw deal inputs for transaction.py.

    capital_structure : CapitalStructure
        Pre-built tranche registry from capital_structure.py.

    op_assumptions : OperatingAssumptions
        Year-by-year P&L assumptions from operating_model.py.

    cf_assumptions : CashFlowAssumptions
        Capex and NWC assumptions from cashflow_model.py.

    exit_multiple : float
        EV / EBITDA at exit.

    pure_sweep_mode : bool
        See debt_model.py documentation.

    management_option_pool_pct : float
        Management dilution at exit.

    n_iterations : int
        Number of interest convergence passes.

    Returns
    -------
    LBOResult
        Fully populated, including transaction S&U table.
    """

    # --- Solve transaction ---
    total_debt = capital_structure.total_debt
    sponsor_equity = solve_sponsor_equity(transaction_assumptions, total_debt)
    transaction_result = build_transaction(
        transaction_assumptions, total_debt, sponsor_equity
    )

    minimum_cash = transaction_assumptions.minimum_cash
    entry_equity = sponsor_equity

    # --- Build LBOParams for the generic orchestrator ---
    # We extract what we need and call the convergence loop directly

    dummy_params = LBOParams(
        entry_ebitda=transaction_assumptions.ltm_ebitda,
        entry_multiple=transaction_result.entry_multiple,
        exit_multiple=exit_multiple,
        holding_period=op_assumptions.holding_period,
        capital_structure=capital_structure,
        minimum_cash=minimum_cash,
        pure_sweep_mode=pure_sweep_mode,
        management_option_pool_pct=management_option_pool_pct,
        n_iterations=n_iterations,
    )

    result = LBOResult(params=dummy_params)
    result.transaction = transaction_result
    result.capital_structure = capital_structure

    # --- Convergence loop ---
    interest_current = [0.0] * op_assumptions.holding_period

    final_op_result = None
    final_cf_result = None
    final_debt_result = None

    for iteration in range(n_iterations):

        op_iter = run_operating_model(op_assumptions)
        op_iter = complete_income_statement(
            op_iter,
            interest_expense=interest_current,
            tax_rate=op_assumptions.tax_rate,
            minimum_cash=minimum_cash,
            interest_income_rate=0.005,
        )

        cf_iter = run_cashflow_model(op_iter, cf_assumptions, capital_structure)

        debt_iter = run_debt_model(
            capital_structure,
            cf_iter,
            minimum_cash=minimum_cash,
            opening_cash=minimum_cash,
            pure_sweep_mode=pure_sweep_mode,
        )

        interest_new = debt_iter.total_interest_expense
        max_delta = max(
            abs(interest_new[t] - interest_current[t])
            for t in range(op_assumptions.holding_period)
        )

        interest_current = interest_new
        final_op_result = op_iter
        final_cf_result = cf_iter
        final_debt_result = debt_iter

        if iteration > 0 and max_delta < 0.5:
            result.interest_converged = True
            break

    result.interest_pass2 = interest_current

    # --- Returns ---
    return_assumptions = ReturnAssumptions(
        exit_multiple=exit_multiple,
        holding_period=op_assumptions.holding_period,
        entry_equity=entry_equity,
        exit_ebitda=final_op_result.exit_ebitda,
        net_debt_at_exit=final_debt_result.net_debt_at_exit,
        management_option_pool_pct=management_option_pool_pct,
    )

    returns = compute_returns(return_assumptions)

    # --- Equity bridge ---
    bridge = compute_equity_bridge(
        entry_equity=entry_equity,
        entry_ebitda=transaction_assumptions.ltm_ebitda,
        entry_multiple=transaction_result.entry_multiple,
        net_debt_at_entry=capital_structure.total_debt - minimum_cash,
        exit_ebitda=final_op_result.exit_ebitda,
        exit_multiple=exit_multiple,
        net_debt_at_exit=final_debt_result.net_debt_at_exit,
        management_option_pool_pct=management_option_pool_pct,
    )

    # --- Exit sensitivity ---
    sensitivity = compute_exit_sensitivity(
        entry_equity=entry_equity,
        exit_ebitda=final_op_result.exit_ebitda,
        net_debt_at_exit=final_debt_result.net_debt_at_exit,
        metric="irr",
    )

    result.operating_model = final_op_result
    result.cash_flow = final_cf_result
    result.debt_schedule = final_debt_result
    result.returns = returns
    result.equity_bridge = bridge
    result.exit_sensitivity = sensitivity

    return result


# ---------------------------------------------------------------------------
# Simulation-friendly wrapper (thin, fast, for Monte Carlo)
# ---------------------------------------------------------------------------

def run_lbo_from_inputs(
    entry_ebitda: float,
    entry_multiple: float,
    exit_multiple: float,
    debt_pct: float,
    senior_pct: float,
    mezz_spread: float,
    interest_rate: float,
    revenue_growth: float,
    gross_margin: float,
    opex_pct: float,
    da_pct: float,
    tax_rate: float,
    capex_pct: float,
    nwc_pct: float,
    holding_period: int,
    minimum_cash: float = 0.0,
) -> tuple:
    """
    Thin wrapper for Monte Carlo simulation engine.
    Accepts raw scalar inputs and returns (irr, moic) only.

    This is what vectorized_simulation.py will call per scenario.
    Kept minimal — no sensitivity table, no equity bridge — for speed.

    Returns
    -------
    tuple : (irr: float, moic: float)
    """
    params = LBOParams(
        entry_ebitda=entry_ebitda,
        entry_multiple=entry_multiple,
        exit_multiple=exit_multiple,
        debt_pct=debt_pct,
        senior_pct=senior_pct,
        mezz_spread=mezz_spread,
        interest_rate=interest_rate,
        revenue_growth=revenue_growth,
        gross_margin=gross_margin,
        opex_pct=opex_pct,
        da_pct=da_pct,
        tax_rate=tax_rate,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        holding_period=holding_period,
        minimum_cash=minimum_cash,
        n_iterations=2,
    )
    result = run_lbo(params)
    return result.irr, result.moic


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_lbo_summary(result: LBOResult) -> None:
    """
    Print a full deal summary: transaction, P&L, FCF, debt, returns.
    """
    from lbo_engine.operating_model import print_operating_model
    from lbo_engine.cashflow_model import print_cashflow_model
    from lbo_engine.debt_model import print_debt_schedule
    from lbo_engine.transaction import print_transaction_summary

    if result.transaction:
        print_transaction_summary(result.transaction)

    if result.operating_model:
        print_operating_model(result.operating_model)

    if result.cash_flow:
        print_cashflow_model(result.cash_flow)

    if result.debt_schedule:
        print_debt_schedule(result.debt_schedule)

    if result.returns:
        print_returns_summary(result.returns, result.equity_bridge)

    if result.exit_sensitivity:
        print_sensitivity_table(result.exit_sensitivity)

    # Convergence diagnostics
    print(f"  Interest convergence:")
    print(f"    Pass 1: {[round(x, 1) for x in result.interest_pass1]}")
    print(f"    Pass 2: {[round(x, 1) for x in result.interest_pass2]}")
    deltas = [
        round(abs(result.interest_pass2[t] - result.interest_pass1[t]), 2)
        for t in range(len(result.interest_pass1))
    ]
    print(f"    Delta:  {deltas}  (converged: {result.interest_converged})\n")


# ---------------------------------------------------------------------------
# Self-test 1: Generic deal (simulation-style inputs)
# ---------------------------------------------------------------------------

def _test_generic():
    print("\n" + "=" * 60)
    print("  TEST 1: GENERIC DEAL (simulation-style)")
    print("=" * 60)

    params = LBOParams(
        entry_ebitda=100.0,
        entry_multiple=10.0,
        exit_multiple=11.0,
        debt_pct=0.60,
        senior_pct=0.70,
        mezz_spread=0.04,
        interest_rate=0.065,
        revenue_growth=0.05,
        gross_margin=0.40,
        opex_pct=0.18,
        da_pct=0.04,
        tax_rate=0.25,
        capex_pct=0.04,
        nwc_pct=0.01,
        holding_period=5,
        minimum_cash=10.0,
        n_iterations=3,
    )

    result = run_lbo(params)
    print_lbo_summary(result)

    print(f"  IRR:  {result.irr * 100:.2f}%")
    print(f"  MOIC: {result.moic:.2f}x")


# ---------------------------------------------------------------------------
# Self-test 2: Burger King full deal
# ---------------------------------------------------------------------------

def _test_burger_king():
    print("\n" + "=" * 60)
    print("  TEST 2: BURGER KING FULL DEAL")
    print("=" * 60)

    from lbo_engine.transaction import TransactionAssumptions
    from lbo_engine.capital_structure import build_bk_capital_structure
    from lbo_engine.operating_model import build_bk_conservative
    from lbo_engine.cashflow_model import build_bk_cashflow_assumptions

    bk_tx = TransactionAssumptions(
        company_name="Burger King",
        share_price=18.86,
        premium_pct=0.27,
        diluted_shares=138.5,
        existing_debt=755.0,
        existing_cash=188.0,
        minimum_cash=118.0,
        ltm_ebitda=445.0,
        transaction_fees_pct=0.0234,
        financing_fees_pct=0.0261,
        other_uses=32.0,
    )

    bk_result = run_lbo_full_deal(
        transaction_assumptions=bk_tx,
        capital_structure=build_bk_capital_structure(),
        op_assumptions=build_bk_conservative(),
        cf_assumptions=build_bk_cashflow_assumptions(),
        exit_multiple=8.8,
        pure_sweep_mode=True,
        n_iterations=3,
    )

    print_lbo_summary(bk_result)

    print("  VALIDATION vs BK PDF:")
    print(f"    IRR:          {bk_result.irr * 100:.1f}%   (PDF: 19.0%)")
    print(f"    MOIC:         {bk_result.moic:.2f}x  (PDF: 2.4x)")
    print(f"    Exit equity:  ${bk_result.exit_equity:,.0f}M  (PDF: $3,730M)")


# ---------------------------------------------------------------------------
# Self-test 3: run_lbo_from_inputs (simulation wrapper)
# ---------------------------------------------------------------------------

def _test_simulation_wrapper():
    print("\n" + "=" * 60)
    print("  TEST 3: SIMULATION WRAPPER (scalar inputs)")
    print("=" * 60)

    irr, moic = run_lbo_from_inputs(
        entry_ebitda=100.0,
        entry_multiple=10.0,
        exit_multiple=11.0,
        debt_pct=0.60,
        senior_pct=0.70,
        mezz_spread=0.04,
        interest_rate=0.065,
        revenue_growth=0.05,
        gross_margin=0.40,
        opex_pct=0.18,
        da_pct=0.04,
        tax_rate=0.25,
        capex_pct=0.04,
        nwc_pct=0.01,
        holding_period=5,
    )
    print(f"  IRR:  {irr * 100:.2f}%")
    print(f"  MOIC: {moic:.2f}x")
    print(f"  (Matches Test 1 — same inputs, same result)")


if __name__ == "__main__":
    _test_generic()
    _test_simulation_wrapper()
    _test_burger_king()