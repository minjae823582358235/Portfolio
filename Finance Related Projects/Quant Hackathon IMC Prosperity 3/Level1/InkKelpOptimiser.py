import subprocess
import re
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Storage for all runs
results = []


def run_strategy(**params):
    print("\n🔍 Running strategy with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    for prefix in ["S", "K"]:
        high = params[f"{prefix}_high"]
        mid = params[f"{prefix}_mid"]
        low = params[f"{prefix}_low"]
        neg = params[f"{prefix}_neg"]
        if not (high >= mid >= low >= neg):
            print(
                f"❌ Constraint violated: {prefix}_high > {prefix}_mid > {prefix}_low > {prefix}_neg not satisfied."
            )
            return -1e9  # penalize
    # Modify the P2P8.py script with current parameters
    with open("Level1\ClassicPairsTrading.py", "r") as file:
        code = file.read()

    for key, val in params.items():
        old_line = re.search(rf"{key}\s*=\s*[-\d.]+", code)
        if old_line:
            print(f"    Replacing in script: {old_line.group(0)} → {key} = {val:.4f}")
        code = re.sub(rf"{key}\s*=\s*[-\d.]+", f"{key} = {val:.4f}", code)

    with open("Level1\ClassicPairsTrading.py", "w") as file:
        file.write(code)

    print("🚀 Executing simulation...")
    result = subprocess.run(
        ["prosperity3bt", "Level1\ClassicPairsTrading.py", "2"],
        capture_output=True,
        text=True,
    )
    output = result.stdout

    # Extract all Total profit lines and use the last one
    matches = re.findall(r"Total profit:\s*(-?[\d,]+)", output)
    if not matches:
        profit = -1e9
        print("❌ No profit found. Penalizing with -1e9.")
    else:
        profit_str = matches[-1].replace(",", "")
        profit = int(profit_str)

    print(f"📈 Profit returned: {profit}")

    run_result = {**params, "profit": profit}
    results.append(run_result)
    return profit


# Define parameter search space
pbounds = {
    "lookbackwindow": (5, 500),
    "gamma": (0.05, 10),
    "sigma": (1, 3),
    "k": (1, 30),
    "max_order_AS_size": (1, 500),
    "buffer": (0, 50),
    "zscorethreshold": (0, 2.5),
    "holdFactor": (0.1, 40),
    "K_high": (0.1, 40),
    "K_mid": (0.1, 40),
    "K_low": (-10, 10),
    "K_neg": (-40, 11),
    "S_high": (0.1, 40),
    "S_mid": (0.1, 40),
    "S_low": (-10, 10),
    "S_neg": (-40, 11),
}

print("🧠 Initializing Bayesian Optimizer...")
optimizer = BayesianOptimization(
    f=run_strategy,
    pbounds=pbounds,
    random_state=42,
)

logger = JSONLogger(path="./bayes_log.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

print("\n⚙️ Starting optimization process...")
optimizer.maximize(
    init_points=200,
    n_iter=20,
)

# Display final top results
print("\n🏁 Optimization complete. Sorting top 10 strategies by profit...")
top_10 = sorted(results, key=lambda x: x["profit"], reverse=True)[:10]
df = pd.DataFrame(top_10)

print("\n📊 Top 10 parameter combinations:\n")
print(df.to_string(index=False))

# Save to CSV
df.to_csv("top_10_strategies.csv", index=False)
print("\n💾 Results saved to top_10_strategies.csv")
