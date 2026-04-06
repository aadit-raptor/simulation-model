"""
transaction.py
--------------
Handles the entry-side of an LBO transaction.

Responsibilities:
    1. Compute entry enterprise value from either:
       - Share price + premium approach (public deal, e.g. Burger King)
       - Direct EBITDA multiple approach (private deal)
    2. Build the Sources & Uses table, which MUST balance.
    3. Expose a clean TransactionResult dataclass consumed by all
       downstream modules (capital_structure, operating_model, returns).

Burger King reference:
    Share price:      $18.86
    Premium:          27%
    Offer price:      $24.01
    Diluted shares:   139M
    Equity value:     $3,325M
    + Net debt:       $567M  ($755M debt - $188M cash)
    = EV:             $3,892M
    Entry multiple:   8.75x  ($3,892M / $445M LTM EBITDA)
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TransactionAssumptions:
    """
    Raw inputs for the transaction module.

    Two entry modes are supported:

    Mode 1 — Public deal (share price approach):
        Provide: share_price, premium_pct, diluted_shares
        Leave:   direct_equity_value = None

    Mode 2 — Private deal (direct equity value or EV approach):
        Provide: direct_equity_value
        Leave:   share_price = None, premium_pct = None, diluted_shares = None

    In both modes you MUST provide:
        ltm_ebitda, existing_debt, existing_cash, minimum_cash,
        sponsor_equity, transaction_fees_pct, financing_fees_pct
    """

    # --- Company identifier ---
    company_name: str = "Target Company"

    # --- Public deal inputs (Mode 1) ---
    share_price: Optional[float] = None          # Latest closing price ($)
    premium_pct: Optional[float] = None          # Acquisition premium, e.g. 0.27 for 27%
    diluted_shares: Optional[float] = None       # Diluted shares outstanding (millions)

    # --- Private deal input (Mode 2) ---
    direct_equity_value: Optional[float] = None  # Equity value ($M) if not using share approach

    # --- Balance sheet at entry ---
    existing_debt: float = 0.0                   # Debt being refinanced ($M)
    existing_cash: float = 0.0                   # Total cash on balance sheet ($M)
    minimum_cash: float = 0.0                    # Cash retained in business post-close ($M)

    # --- LTM financials ---
    ltm_ebitda: float = 445.0                    # Last twelve months EBITDA ($M)

    # --- Fee assumptions ---
    transaction_fees_pct: float = 0.02           # % of EV (advisory, legal, etc.)
    financing_fees_pct: float = 0.02             # % of total debt raised (amortised via OID)

    # --- Other uses ---
    other_uses: float = 0.0                      # Any additional uses of funds ($M)


@dataclass
class SourcesAndUses:
    """
    The Sources & Uses table.
    Sources == Uses by construction (enforced in TransactionResult).

    All values in $M.
    """
    # Sources
    total_debt_raised: float = 0.0
    cash_on_hand_used: float = 0.0     # Excess cash used to fund the deal
    sponsor_equity: float = 0.0

    # Uses
    equity_purchase_price: float = 0.0
    debt_refinanced: float = 0.0
    transaction_fees: float = 0.0
    financing_fees: float = 0.0
    other_uses: float = 0.0

    @property
    def total_sources(self) -> float:
        return self.total_debt_raised + self.cash_on_hand_used + self.sponsor_equity

    @property
    def total_uses(self) -> float:
        return (
            self.equity_purchase_price
            + self.debt_refinanced
            + self.transaction_fees
            + self.financing_fees
            + self.other_uses
        )

    @property
    def check(self) -> float:
        """Sources minus Uses. Must be ~0. Any nonzero value indicates a modelling error."""
        return round(self.total_sources - self.total_uses, 4)


@dataclass
class TransactionResult:
    """
    Fully computed transaction output.
    This object is passed as-is to every downstream module.

    Key values:
        enterprise_value    — entry EV ($M)
        entry_multiple      — EV / LTM EBITDA (x)
        sponsor_equity      — PE firm's equity check ($M)
        total_debt          — total new debt raised ($M)
        net_debt_at_entry   — total_debt - minimum_cash ($M)
        sources_and_uses    — the balanced S&U table
    """
    company_name: str
    offer_price_per_share: float         # $0 if private deal
    equity_value: float                  # $M
    enterprise_value: float              # $M
    entry_multiple: float                # EV / LTM EBITDA
    ltm_ebitda: float                    # $M
    existing_debt: float                 # $M
    existing_cash: float                 # $M
    minimum_cash: float                  # $M retained post-close
    total_debt: float                    # New debt raised ($M)
    sponsor_equity: float                # PE equity check ($M)
    net_debt_at_entry: float             # total_debt - minimum_cash
    sources_and_uses: SourcesAndUses


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def build_transaction(
    assumptions: TransactionAssumptions,
    total_debt_raised: float,            # Computed by capital_structure.py and passed back in
    sponsor_equity: float                # Residual after debt and fees
) -> TransactionResult:
    """
    Construct the full transaction object given deal assumptions,
    total debt raised (from the capital structure), and sponsor equity.

    Parameters
    ----------
    assumptions : TransactionAssumptions
        Raw deal inputs.
    total_debt_raised : float
        Sum of all debt tranche amounts ($M). Comes from CapitalStructure.
    sponsor_equity : float
        Sponsor equity check ($M). Comes from solve_sponsor_equity().

    Returns
    -------
    TransactionResult
        Fully populated transaction with balanced S&U table.

    Raises
    ------
    ValueError
        If neither a share-price approach nor a direct equity value is provided.
    ValueError
        If the Sources & Uses table does not balance within $0.01M.
    """

    # ------------------------------------------------------------------
    # Step 1: Compute equity value
    # ------------------------------------------------------------------
    if assumptions.share_price is not None and assumptions.diluted_shares is not None:
        # Mode 1: Public deal — price × shares
        if assumptions.premium_pct is None:
            raise ValueError(
                "premium_pct is required when using the share-price approach."
            )
        offer_price = assumptions.share_price * (1 + assumptions.premium_pct)
        equity_value = offer_price * assumptions.diluted_shares   # $M
    elif assumptions.direct_equity_value is not None:
        # Mode 2: Private deal — equity value given directly
        offer_price = 0.0
        equity_value = assumptions.direct_equity_value
    else:
        raise ValueError(
            "Provide either (share_price, premium_pct, diluted_shares) "
            "or direct_equity_value."
        )

    # ------------------------------------------------------------------
    # Step 2: Enterprise value
    # EV = Equity Value + Net Debt at entry
    # Net debt at entry = existing debt - existing cash
    # ------------------------------------------------------------------
    net_existing_debt = assumptions.existing_debt - assumptions.existing_cash
    enterprise_value = equity_value + net_existing_debt
    entry_multiple = enterprise_value / assumptions.ltm_ebitda

    # ------------------------------------------------------------------
    # Step 3: Compute fees (absolute $M)
    # ------------------------------------------------------------------
    transaction_fees = enterprise_value * assumptions.transaction_fees_pct
    financing_fees = total_debt_raised * assumptions.financing_fees_pct

    # ------------------------------------------------------------------
    # Step 4: Cash on hand available to fund deal
    # The BK model used $69M of excess cash (total $188M - $118M minimum)
    # as a source of funds.
    # ------------------------------------------------------------------
    cash_available = max(assumptions.existing_cash - assumptions.minimum_cash, 0.0)

    # ------------------------------------------------------------------
    # Step 5: Build Sources & Uses
    # ------------------------------------------------------------------
    sau = SourcesAndUses(
        # Sources
        total_debt_raised=total_debt_raised,
        cash_on_hand_used=cash_available,
        sponsor_equity=sponsor_equity,
        # Uses
        equity_purchase_price=equity_value,
        debt_refinanced=assumptions.existing_debt,
        transaction_fees=transaction_fees,
        financing_fees=financing_fees,
        other_uses=assumptions.other_uses,
    )

    # ------------------------------------------------------------------
    # Step 6: Validate balance
    # Sources must equal Uses within $0.01M rounding tolerance
    # ------------------------------------------------------------------
    if abs(sau.check) > 0.01:
        raise ValueError(
            f"Sources & Uses do not balance: delta = ${sau.check:.4f}M. "
            f"Total Sources = ${sau.total_sources:.2f}M, "
            f"Total Uses = ${sau.total_uses:.2f}M. "
            f"Check sponsor_equity calculation."
        )

    # ------------------------------------------------------------------
    # Step 7: Net debt at entry (for returns module)
    # ------------------------------------------------------------------
    net_debt_at_entry = total_debt_raised - assumptions.minimum_cash

    return TransactionResult(
        company_name=assumptions.company_name,
        offer_price_per_share=round(offer_price, 4),
        equity_value=round(equity_value, 2),
        enterprise_value=round(enterprise_value, 2),
        entry_multiple=round(entry_multiple, 2),
        ltm_ebitda=assumptions.ltm_ebitda,
        existing_debt=assumptions.existing_debt,
        existing_cash=assumptions.existing_cash,
        minimum_cash=assumptions.minimum_cash,
        total_debt=total_debt_raised,
        sponsor_equity=sponsor_equity,
        net_debt_at_entry=round(net_debt_at_entry, 2),
        sources_and_uses=sau,
    )


def solve_sponsor_equity(
    assumptions: TransactionAssumptions,
    total_debt_raised: float,
) -> float:
    """
    Solve for the sponsor equity check that makes Sources == Uses.

    This mirrors how an actual LBO model works: debt tranches are set
    first, then sponsor equity is the residual that plugs the gap.

    Formula:
        Total Uses = equity_purchase_price + debt_refinanced
                     + transaction_fees + financing_fees + other_uses

        Total Sources = total_debt_raised + cash_on_hand_used + sponsor_equity

        sponsor_equity = Total Uses - total_debt_raised - cash_on_hand_used

    Parameters
    ----------
    assumptions : TransactionAssumptions
    total_debt_raised : float
        Sum of all tranche par amounts.

    Returns
    -------
    float
        Sponsor equity check in $M.
    """

    # Recompute equity value (same logic as build_transaction)
    if assumptions.share_price is not None and assumptions.diluted_shares is not None:
        offer_price = assumptions.share_price * (1 + assumptions.premium_pct)
        equity_value = offer_price * assumptions.diluted_shares
    else:
        equity_value = assumptions.direct_equity_value

    net_existing_debt = assumptions.existing_debt - assumptions.existing_cash
    enterprise_value = equity_value + net_existing_debt

    transaction_fees = enterprise_value * assumptions.transaction_fees_pct
    financing_fees = total_debt_raised * assumptions.financing_fees_pct
    cash_available = max(assumptions.existing_cash - assumptions.minimum_cash, 0.0)

    total_uses = (
        equity_value
        + assumptions.existing_debt
        + transaction_fees
        + financing_fees
        + assumptions.other_uses
    )

    sponsor_equity = total_uses - total_debt_raised - cash_available

    if sponsor_equity < 0:
        raise ValueError(
            f"Sponsor equity is negative (${sponsor_equity:.2f}M). "
            f"Debt raised (${total_debt_raised:.2f}M) exceeds total uses. "
            f"Reduce total debt or increase uses."
        )

    return round(sponsor_equity, 2)


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def print_transaction_summary(result: TransactionResult) -> None:
    """
    Print a formatted transaction summary to console.
    Mirrors the layout of a real IB deal summary page.
    """
    sau = result.sources_and_uses
    sep = "-" * 55

    print(f"\n{'=' * 55}")
    print(f"  TRANSACTION SUMMARY — {result.company_name.upper()}")
    print(f"{'=' * 55}")

    print(f"\n  ENTRY VALUATION")
    print(sep)
    if result.offer_price_per_share > 0:
        print(f"  Offer price per share       ${result.offer_price_per_share:>10.2f}")
    print(f"  Equity value                ${result.equity_value:>10,.1f}M")
    print(f"  + Existing debt             ${result.existing_debt:>10,.1f}M")
    print(f"  - Existing cash             ${result.existing_cash:>10,.1f}M")
    print(f"  Enterprise value            ${result.enterprise_value:>10,.1f}M")
    print(f"  LTM EBITDA                  ${result.ltm_ebitda:>10,.1f}M")
    print(f"  Entry multiple              {result.entry_multiple:>11.1f}x")

    print(f"\n  SOURCES & USES")
    print(sep)
    print(f"  {'SOURCES':<30} {'USES':<20}")
    print(f"  {sep}")
    print(f"  New debt raised  ${sau.total_debt_raised:>7,.1f}M   "
          f"Equity purchase  ${sau.equity_purchase_price:>7,.1f}M")
    print(f"  Cash on hand     ${sau.cash_on_hand_used:>7,.1f}M   "
          f"Debt refinanced  ${sau.debt_refinanced:>7,.1f}M")
    print(f"  Sponsor equity   ${sau.sponsor_equity:>7,.1f}M   "
          f"Transaction fees ${sau.transaction_fees:>7,.1f}M")
    print(f"  {'':>26}   Financing fees   ${sau.financing_fees:>7,.1f}M")
    if sau.other_uses > 0:
        print(f"  {'':>26}   Other            ${sau.other_uses:>7,.1f}M")
    print(sep)
    print(f"  Total            ${sau.total_sources:>7,.1f}M   "
          f"Total            ${sau.total_uses:>7,.1f}M")
    print(f"\n  Check (must be 0): ${sau.check:.4f}M")
    print(f"{'=' * 55}\n")


# ---------------------------------------------------------------------------
# Quick self-test — mirrors the Burger King deal exactly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from lbo_engine.capital_structure import build_bk_capital_structure

    bk_assumptions = TransactionAssumptions(
        company_name="Burger King",
        share_price=18.86,
        premium_pct=0.27,
        diluted_shares=138.5,       # ~139M shares → equity value ~$3,325M
        existing_debt=755.0,
        existing_cash=188.0,
        minimum_cash=118.0,
        ltm_ebitda=445.0,
        transaction_fees_pct=0.0234,   # $91.2M / $3,893M EV
        financing_fees_pct=0.0261,     # $69M / $2,644M debt
        other_uses=32.0,
    )

    # Capital structure total is known from the BK deal
    total_debt = 2644.0

    sponsor_equity = solve_sponsor_equity(bk_assumptions, total_debt)
    result = build_transaction(bk_assumptions, total_debt, sponsor_equity)
    print_transaction_summary(result)
    