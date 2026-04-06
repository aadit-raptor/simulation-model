from lbo_engine.model import run_lbo

result = run_lbo(
    entry_ebitda=100,
    entry_multiple=10,
    debt_pct=0.6,
    interest_rate=0.06,
    revenue_growth=0.05,
    ebitda_margin=0.25,
    capex_pct=0.04,
    exit_multiple=11,
)

print(result)