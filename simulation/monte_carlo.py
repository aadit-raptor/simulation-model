import numpy as np
import pandas as pd
from lbo_engine.model import run_lbo


def run_simulation(
    n,
    growth_mean, growth_std,
    exit_mean, exit_std,
    interest_mean, interest_std,
    debt_pct,
    senior_pct,
    mezz_spread,
    tax_rate,
    ebitda_margin,
    capex_pct,
    years
):

    results = []

    corr = np.array([
        [1, 0.6, -0.3],
        [0.6, 1, -0.5],
        [-0.3, -0.5, 1]
    ])

    L = np.linalg.cholesky(corr)

    for _ in range(n):

        z = np.random.normal(size=3)
        correlated = L @ z

        growth = growth_mean + correlated[0] * growth_std
        exit_multiple = exit_mean + correlated[1] * exit_std
        interest = interest_mean + correlated[2] * interest_std

        res = run_lbo(
            entry_ebitda=100,
            entry_multiple=10,
            debt_pct=debt_pct,
            senior_pct=senior_pct,
            mezz_spread=mezz_spread,
            interest_rate=interest,
            tax_rate=tax_rate,
            revenue_growth=growth,
            ebitda_margin=ebitda_margin,
            capex_pct=capex_pct,
            exit_multiple=exit_multiple,
            years=years
        )

        results.append({
            "IRR": res["IRR"],
            "MOIC": res["MOIC"],
            "Growth": growth,
            "Exit Multiple": exit_multiple,
            "Interest": interest
        })

    return pd.DataFrame(results)