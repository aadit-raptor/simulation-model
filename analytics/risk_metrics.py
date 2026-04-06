import pandas as pd


def calculate_risk_metrics(df, target_irr=0.20):

    metrics = {}

    metrics["Mean IRR"] = df["IRR"].mean()
    metrics["Median IRR"] = df["IRR"].median()

    metrics["Probability IRR > Target"] = (df["IRR"] > target_irr).mean()

    metrics["Probability IRR < 10%"] = (df["IRR"] < 0.10).mean()

    metrics["5% Downside IRR"] = df["IRR"].quantile(0.05)

    metrics["95% Upside IRR"] = df["IRR"].quantile(0.95)

    metrics["Worst Case IRR"] = df["IRR"].min()

    metrics["Best Case IRR"] = df["IRR"].max()

    return metrics