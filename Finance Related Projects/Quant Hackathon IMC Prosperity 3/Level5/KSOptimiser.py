import subprocess
import re
import pandas as pd
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import os

# Storage for all runs
results = []

def run_strategy(**params):
    print("\n🔍 Running strategy with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    # Enforce condition: high > mid > low > neg for PB1, PB2
    # for prefix in ["pb1", "pb2"]:
    #     high = params[f"{prefix}high"]
    #     mid = params[f"{prefix}mid"]
    #     low = params[f"{prefix}low"]
    #     neg = params[f"{prefix}neg"]
    #     if not (high >= mid >= low >= neg):
    #         print(
    #             f"❌ Constraint violated: {prefix}_high > {prefix}_mid > {prefix}_low > {prefix}_neg not satisfied."
    #         )
    #         return -1e9  # penalize

    # Modify the trading script
    with open("Level5\KelpSquinkFINAL3.py", "r") as file:
        code = file.read()

    for key, val in params.items():
        old_line = re.search(rf"{key}\s*=\s*[-\d.]+", code)
        if old_line:
            print(f"    Replacing in script: {old_line.group(0)} → {key} = {val:.4f}")
        code = re.sub(rf"{key}\s*=\s*[-\d.]+", f"{key} = {val:.4f}", code)

    with open("Level5\KelpSquinkFINAL3.py", "w") as file:
        file.write(code)

    print("🚀 Executing simulation...")
    result = subprocess.run(
        ["prosperity3bt", "Level5\KelpSquinkFINAL3.py",'4',"5",'--merge-pnl',"--no-out"],
        capture_output=True,
        text=True,
    )
    output = result.stdout

    # Extract profit
    matches = re.findall(r"Total profit:\s*(-?[\d,]+)", output)
    if not matches:
        profit = -1e9
        print("❌ No profit found. Penalizing with -1e9.")
    else:
        profit_str = matches[-1].replace(",", "")
        try:
            profit = int(profit_str)
            print(f"✅ Last Total profit found: {profit}")
        except ValueError:
            profit = -1e9
            print("❌ Profit conversion failed. Penalizing with -1e9.")

    run_result = {**params, "profit": profit}
    results.append(run_result)
    return profit

# Define parameter space
pbounds = {
    "Kz": (0.01, 4),
    "Sz": (0.01, 4),
    "Jz": (0.01, 4),
    "Dz": (0.01, 4),
    "Cz": (0.01, 4),
}

print("🧠 Initializing Bayesian Optimizer...")
optimizer = BayesianOptimization(
    f=run_strategy,
    pbounds=pbounds,
    random_state=72,
)

init_points = 200
n_iter = 100

print("\n⚙️ Running initial random points...")
optimizer.maximize(init_points=init_points, n_iter=0)

print("\n⚙️ Running optimization iterations...")
for _ in tqdm(range(n_iter), desc="🚀 Bayesian Optimization"):
    optimizer.maximize(init_points=0, n_iter=1)
    print(f"📈 Best so far: {optimizer.max['target']:.2f}")


# Final results
print("\n🏁 Optimization complete. Sorting top 20 strategies by profit...")
top_10 = sorted(results, key=lambda x: x["profit"], reverse=True)[:20]
df = pd.DataFrame(top_10)

print("\n📊 Top 10 parameter combinations:\n")
print(df.to_string(index=False))

df.to_csv("top_10_strategies.csv", index=False)
print("\n💾 Results saved to EN8_strategies.csv")
