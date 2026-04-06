"""
capital_structure.py
--------------------
Generic N-tranche debt capital structure for an LBO.

Design philosophy:
    Each debt tranche is a self-contained dataclass with its own rate,
    maturity, amortization schedule, and fees. The CapitalStructure
    container holds a list of tranches and exposes aggregate views.

    This design means adding a PIK tranche, a revolver, or a second
    lien is just appending one more Tranche object — no other file changes.

Burger King reference tranches:
    1. USD Secured Term Loan   $1,510M  6.82%  7yr  1% annual amort
    2. EUR Secured Term Loan     $334M  7.11%  6yr  ~0.4% annual amort
    3. Senior Notes              $800M 10.19%  8yr  bullet (no amort)

Amortization types supported:
    "bullet"      — no principal payments until maturity
    "amortizing"  — fixed annual % of original principal
    "custom"      — pass in a year-by-year schedule list

Fees:
    Upfront financing fees are tracked per tranche. They reduce the
    net proceeds to the borrower (OID effect) but for simplicity we
    treat them as a use of funds in Sources & Uses and amortise them
    as a non-cash charge in the operating model if needed.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Amortization type
# ---------------------------------------------------------------------------

AmortizationType = Literal["bullet", "amortizing", "custom"]


# ---------------------------------------------------------------------------
# Tranche dataclass
# ---------------------------------------------------------------------------

@dataclass
class Tranche:
    """
    A single debt tranche in the capital structure.

    Parameters
    ----------
    name : str
        Descriptive label, e.g. "USD Term Loan A", "Senior Notes".

    amount : float
        Original principal ($M).

    interest_rate : float
        Annual interest rate as a decimal, e.g. 0.0682 for 6.82%.
        For floating rate debt this is the all-in rate at close.
        You can override per-year rates in the debt_model.

    maturity_years : int
        Number of years until final maturity.

    amort_type : AmortizationType
        "bullet"      — full principal at maturity.
        "amortizing"  — fixed % of original principal per year.
        "custom"      — year-by-year schedule provided in amort_schedule.

    amort_pct : float
        Annual amortization as % of original principal.
        Only used when amort_type == "amortizing".
        e.g. 0.01 means 1% of original principal per year.

    amort_schedule : list[float]
        Explicit list of principal repayments per year ($M).
        Required when amort_type == "custom". Length must equal holding_period.

    fee_pct : float
        Upfront financing fee as % of principal.
        e.g. 0.027 for 2.7%.

    is_cash_sweep : bool
        Whether excess FCF is applied to this tranche after mandatory
        amortization. In BK, only the USD term loan was swept.
        Senior Notes were not swept.

    sweep_priority : int
        Lower number = higher sweep priority. The debt_model.py will
        sort tranches by this value when distributing excess FCF.
        Typical order: secured term loans first, then mezz, then notes.

    currency : str
        For informational use. All values stored in USD equivalent.
    """

    name: str
    amount: float                             # $M original principal
    interest_rate: float                      # Annual rate, decimal
    maturity_years: int                       # Years to maturity
    amort_type: AmortizationType = "bullet"
    amort_pct: float = 0.0                    # Annual amort % of original principal
    amort_schedule: List[float] = field(default_factory=list)  # custom schedule
    fee_pct: float = 0.0                      # Upfront fee %
    is_cash_sweep: bool = False               # Eligible for excess FCF sweep
    sweep_priority: int = 99                  # Lower = swept first
    currency: str = "USD"

    # -------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------

    @property
    def fee_amount(self) -> float:
        """Upfront fee in $M."""
        return round(self.amount * self.fee_pct, 4)

    @property
    def net_proceeds(self) -> float:
        """Proceeds after upfront fee ($M)."""
        return round(self.amount - self.fee_amount, 4)

    def mandatory_repayment(self, year: int, holding_period: int) -> float:
        """
        Mandatory principal repayment in a given year ($M).

        Parameters
        ----------
        year : int
            1-indexed projection year (1 = first full year post-close).
        holding_period : int
            Total holding period in years. Used to detect final year
            bullet for amortizing tranches.

        Returns
        -------
        float
            Mandatory repayment amount ($M) for that year.
        """
        if self.amort_type == "bullet":
            # Pay full outstanding balance only at maturity
            if year == self.maturity_years:
                return self.amount   # Simplified: full original amount
            return 0.0

        elif self.amort_type == "amortizing":
            annual = self.amount * self.amort_pct
            # In the final year of the holding period, there may be
            # a balloon payment. The debt_model handles the actual
            # remaining balance — this returns the scheduled annual amount.
            return round(annual, 4)

        elif self.amort_type == "custom":
            if not self.amort_schedule:
                raise ValueError(
                    f"Tranche '{self.name}' has amort_type='custom' "
                    f"but no amort_schedule was provided."
                )
            if year > len(self.amort_schedule):
                return 0.0
            return self.amort_schedule[year - 1]

        else:
            raise ValueError(
                f"Unknown amort_type '{self.amort_type}' on tranche '{self.name}'."
            )

    def annual_interest(self, beginning_balance: float) -> float:
        """
        Interest expense for a year given the beginning-of-year balance.

        In the BK model, interest is calculated on beginning balance.
        This is standard for term loans. PIK instruments would add to
        principal — that extension can be added here if needed.

        Parameters
        ----------
        beginning_balance : float
            Outstanding principal at start of the year ($M).

        Returns
        -------
        float
            Interest expense ($M).
        """
        return round(beginning_balance * self.interest_rate, 4)


# ---------------------------------------------------------------------------
# Capital structure container
# ---------------------------------------------------------------------------

@dataclass
class CapitalStructure:
    """
    Container for all debt tranches in the deal.

    This object is constructed once at the transaction level and passed
    to the debt_model, which then runs the year-by-year schedule.

    Parameters
    ----------
    tranches : list[Tranche]
        All debt tranches, in any order. The debt_model.py will sort
        by sweep_priority when distributing excess FCF.

    ltm_ebitda : float
        LTM EBITDA at entry. Used to compute leverage multiples.
    """

    tranches: List[Tranche]
    ltm_ebitda: float = 445.0

    # -------------------------------------------------------------------
    # Aggregate views
    # -------------------------------------------------------------------

    @property
    def total_debt(self) -> float:
        """Total par value of all tranches ($M)."""
        return round(sum(t.amount for t in self.tranches), 2)

    @property
    def total_fees(self) -> float:
        """Total upfront financing fees across all tranches ($M)."""
        return round(sum(t.fee_amount for t in self.tranches), 2)

    @property
    def total_leverage_multiple(self) -> float:
        """Total debt / LTM EBITDA."""
        return round(self.total_debt / self.ltm_ebitda, 2)

    @property
    def blended_interest_rate(self) -> float:
        """
        Weighted average interest rate across all tranches.
        Weighted by tranche amount (par value).
        """
        if self.total_debt == 0:
            return 0.0
        weighted = sum(t.amount * t.interest_rate for t in self.tranches)
        return round(weighted / self.total_debt, 6)

    @property
    def sweep_eligible_tranches(self) -> List[Tranche]:
        """
        Tranches eligible for excess FCF sweep, sorted by priority.
        Priority 1 gets swept first (most senior).
        """
        eligible = [t for t in self.tranches if t.is_cash_sweep]
        return sorted(eligible, key=lambda t: t.sweep_priority)

    # -------------------------------------------------------------------
    # Per-tranche summary
    # -------------------------------------------------------------------

    def tranche_summary(self) -> List[dict]:
        """
        Returns a list of dicts for easy DataFrame construction.
        Each dict represents one tranche row.
        """
        rows = []
        for t in self.tranches:
            rows.append({
                "Tranche": t.name,
                "Amount ($M)": t.amount,
                "x EBITDA": round(t.amount / self.ltm_ebitda, 2),
                "Rate (%)": round(t.interest_rate * 100, 2),
                "Fee (%)": round(t.fee_pct * 100, 2),
                "Fee ($M)": t.fee_amount,
                "Maturity (yrs)": t.maturity_years,
                "Amort type": t.amort_type,
                "Annual amort (%)": round(t.amort_pct * 100, 2),
                "Cash sweep": t.is_cash_sweep,
                "Currency": t.currency,
            })
        rows.append({
            "Tranche": "TOTAL",
            "Amount ($M)": self.total_debt,
            "x EBITDA": self.total_leverage_multiple,
            "Rate (%)": round(self.blended_interest_rate * 100, 2),
            "Fee (%)": "",
            "Fee ($M)": self.total_fees,
            "Maturity (yrs)": "",
            "Amort type": "",
            "Annual amort (%)": "",
            "Cash sweep": "",
            "Currency": "",
        })
        return rows

    def print_summary(self) -> None:
        """Print formatted capital structure table to console."""
        col_w = 22

        print(f"\n{'=' * 80}")
        print(f"  CAPITAL STRUCTURE")
        print(f"{'=' * 80}")
        print(
            f"  {'Tranche':<28} {'Amount':>8} {'xEBITDA':>8} "
            f"{'Rate':>7} {'Fee%':>6} {'Fee$':>7} {'Maturity':>9}"
        )
        print(f"  {'-' * 76}")

        for t in self.tranches:
            print(
                f"  {t.name:<28} "
                f"${t.amount:>7,.0f}M "
                f"{t.amount / self.ltm_ebitda:>6.1f}x "
                f"{t.interest_rate * 100:>6.2f}% "
                f"{t.fee_pct * 100:>5.1f}% "
                f"${t.fee_amount:>5.1f}M "
                f"{t.maturity_years:>6}yr"
            )

        print(f"  {'-' * 76}")
        print(
            f"  {'TOTAL':<28} "
            f"${self.total_debt:>7,.0f}M "
            f"{self.total_leverage_multiple:>6.1f}x "
            f"{self.blended_interest_rate * 100:>6.2f}% "
            f"{'':>6} "
            f"${self.total_fees:>5.1f}M"
        )
        print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Factory functions — pre-built deal structures
# ---------------------------------------------------------------------------

def build_bk_capital_structure() -> CapitalStructure:
    """
    Reconstruct the exact Burger King LBO capital structure from the PDF.

    Tranche details from the model:
        USD Term Loan:  $1,510M  6.82%  6yr  2.7% fee  1% annual amort  sweep
        EUR Term Loan:    $334M  7.11%  6yr  2.5% fee  0.4% annual amort  no sweep
        Senior Notes:     $800M 10.19%  8yr  2.5% fee  bullet             no sweep

    Notes:
        - USD term loan is the primary sweep vehicle in the BK model.
          In the PDF the paydown goes: $0, $18M, $54M, $113M, $191M
          which matches excess FCF after minimum cash floor.
        - EUR term loan and Senior Notes have no FCF sweep in the BK model.
        - Amort rates are approximated from the PDF debt schedule.
    """
    usd_term_loan = Tranche(
        name="USD Secured Term Loan",
        amount=1510.0,
        interest_rate=0.0682,
        maturity_years=6,
        amort_type="amortizing",
        amort_pct=0.01,            # 1% annual = $15.1M/yr mandatory
        fee_pct=0.027,
        is_cash_sweep=True,
        sweep_priority=1,          # Swept first — most senior
        currency="USD",
    )

    eur_term_loan = Tranche(
        name="EUR Secured Term Loan",
        amount=334.0,
        interest_rate=0.0711,
        maturity_years=6,
        amort_type="amortizing",
        amort_pct=0.004,           # ~0.4% per year
        fee_pct=0.025,
        is_cash_sweep=False,
        sweep_priority=2,
        currency="EUR",
    )

    senior_notes = Tranche(
        name="Senior Notes",
        amount=800.0,
        interest_rate=0.1019,
        maturity_years=8,
        amort_type="bullet",
        amort_pct=0.0,
        fee_pct=0.025,
        is_cash_sweep=False,
        sweep_priority=3,
        currency="USD",
    )

    return CapitalStructure(
        tranches=[usd_term_loan, eur_term_loan, senior_notes],
        ltm_ebitda=445.0,
    )


def build_simple_two_tranche_structure(
    enterprise_value: float,
    ltm_ebitda: float,
    debt_pct: float = 0.60,
    senior_pct: float = 0.70,
    senior_rate: float = 0.06,
    mezz_rate: float = 0.10,
    holding_period: int = 5,
) -> CapitalStructure:
    """
    Build a simple two-tranche structure (senior + mezz) for generic deals.

    This is what your current model.py approximates.
    Use this when you don't have a full tranche breakdown.

    Parameters
    ----------
    enterprise_value : float
        Entry EV ($M).
    ltm_ebitda : float
        LTM EBITDA at entry ($M).
    debt_pct : float
        Total debt as % of EV.
    senior_pct : float
        Senior debt as % of total debt.
    senior_rate : float
        Senior interest rate.
    mezz_rate : float
        Mezz interest rate.
    holding_period : int
        Holding period in years.

    Returns
    -------
    CapitalStructure
    """
    total_debt = enterprise_value * debt_pct
    senior_amount = total_debt * senior_pct
    mezz_amount = total_debt * (1 - senior_pct)

    senior = Tranche(
        name="Senior Term Loan",
        amount=round(senior_amount, 2),
        interest_rate=senior_rate,
        maturity_years=holding_period,
        amort_type="amortizing",
        amort_pct=0.05,            # 5% annual amortization
        fee_pct=0.02,
        is_cash_sweep=True,
        sweep_priority=1,
    )

    mezz = Tranche(
        name="Mezzanine",
        amount=round(mezz_amount, 2),
        interest_rate=mezz_rate,
        maturity_years=holding_period,
        amort_type="bullet",
        amort_pct=0.0,
        fee_pct=0.03,
        is_cash_sweep=True,
        sweep_priority=2,
    )

    return CapitalStructure(
        tranches=[senior, mezz],
        ltm_ebitda=ltm_ebitda,
    )


# ---------------------------------------------------------------------------
# Quick self-test — Burger King capital structure
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cs = build_bk_capital_structure()
    cs.print_summary()

    print("Sweep eligible tranches (in priority order):")
    for t in cs.sweep_eligible_tranches:
        print(f"  [{t.sweep_priority}] {t.name}  ${t.amount:,.0f}M")

    print(f"\nBlended rate: {cs.blended_interest_rate * 100:.2f}%")
    print(f"Total leverage: {cs.total_leverage_multiple:.1f}x EBITDA")

    # Test mandatory repayment
    print("\nMandatory repayments (USD Term Loan, years 1-5):")
    for yr in range(1, 6):
        print(f"  Year {yr}: ${cs.tranches[0].mandatory_repayment(yr, 5):,.1f}M")