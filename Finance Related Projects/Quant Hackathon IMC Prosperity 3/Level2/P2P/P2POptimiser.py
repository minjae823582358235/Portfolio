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

    # Enforce condition: high > mid > low > neg for PB1, PB2, DJ
    for prefix in ["PB1", "PB2", "DJ"]:
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
    with open("Level2/P2P/P2P10.py", "r") as file:
        code = file.read()

    for key, val in params.items():
        old_line = re.search(rf"{key}\s*=\s*[-\d.]+", code)
        if old_line:
            print(f"    Replacing in script: {old_line.group(0)} → {key} = {val:.4f}")
        code = re.sub(rf"{key}\s*=\s*[-\d.]+", f"{key} = {val:.4f}", code)

    with open("Level2/P2P/P2P10.py", "w") as file:
        file.write(code)

    print("🚀 Executing simulation...")
    result = subprocess.run(
        ["prosperity3bt", "Level2/P2P/P2P10.py", "2"],
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
        print(f"✅ Last Total profit found: {profit}")

    run_result = {**params, "profit": profit}
    results.append(run_result)
    return profit


# Define parameter search space
pbounds = {
    "zfactor1": (0.05, 2.5),
    "zfactor2": (0.05, 2.5),
    "PB1_high": (1, 40),
    "PB1_mid": (1, 40),
    "PB1_low": (-10, 10),
    "PB1_neg": (-40, 11),
    "PB2_high": (1, 40),
    "PB2_mid": (1, 40),
    "PB2_low": (-10, 10),
    "PB2_neg": (-40, 10),
    "DJ_high": (1, 40),
    "DJ_mid": (0, 20),
    "DJ_low": (-10, 10),
    "DJ_neg": (-40, 10),
    "hold_factor": (0.01, 20.0),
    "pb1ZscoreThreshold": (0.01, 5.0),
    "pb2ZscoreThreshold": (0.01, 5.0),
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
    init_points=500,
    n_iter=100,
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
