from simulation.monte_carlo import run_simulation
from analytics.risk_metrics import calculate_risk_metrics

df = run_simulation(10000)

metrics = calculate_risk_metrics(df)

print("\nRISK ANALYTICS\n")

for k, v in metrics.items():
    print(k, ":", round(v,4))