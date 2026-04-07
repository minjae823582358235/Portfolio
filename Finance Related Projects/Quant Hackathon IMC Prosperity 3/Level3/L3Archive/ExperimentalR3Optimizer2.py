import subprocess
import re
import pandas as pd
from bayes_opt import BayesianOptimization
from tqdm import tqdm
import os
import multiprocessing
# Storage for all runs
results = []
def run_strategy_wrapper(params):
    return run_strategy(**params)

def run_strategy(**params):
    print("\n🔍 Running strategy with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    # Enforce condition: high > mid > low > neg for PB1, PB2
    for prefix in ["pb1", "pb2"]:
        high = params[f"{prefix}high"]
        mid = params[f"{prefix}mid"]
        low = params[f"{prefix}low"]
        neg = params[f"{prefix}neg"]
        if not (high >= mid >= low >= neg):
            print(
                f"❌ Constraint violated: {prefix}_high > {prefix}_mid > {prefix}_low > {prefix}_neg not satisfied."
            )
            return -1e9  # penalize

    # Modify the trading script
    with open("Level3\Round2CommoditiesTests.py", "r") as file:
        code = file.read()

    for key, val in params.items():
        old_line = re.search(rf"{key}\s*=\s*[-\d.]+", code)
        if old_line:
            print(f"    Replacing in script: {old_line.group(0)} → {key} = {val:.4f}")
        code = re.sub(rf"{key}\s*=\s*[-\d.]+", f"{key} = {val:.4f}", code)

    with open("Level3\Round2CommoditiesTests.py", "w") as file:
        file.write(code)

    print("🚀 Executing simulation...")
    result = subprocess.run(
        ["prosperity3bt", "Level3\Round2CommoditiesTests.py", "3",'--no-out'],
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
    "s1Zscore": (0.05, 2.5),
    "s2Zscore": (0.05, 2.5),
    "pb1high": (1, 40),
    "pb1mid": (1, 40),
    "pb1low": (-10, 10),
    "pb1neg": (-40, 11),
    "pb2high": (1, 40),
    "pb2mid": (1, 40),
    "pb2low": (-10, 10),
    "pb2neg": (-40, 10),
    "kelptw": (1, 20),
    "kelpminvolume": (0, 50),
    "kelpspreadedge": (1.0,5.0),
    'squidz':(0,3),
    'squidhistorylength':(1,1000),
    'squidrsi':(0,50),
    'squidoverbought':(0,100),
    'squidoversold':(0,100),
    'squidrsitradesize':(0, 50),
    'croissanthistorylength':(1,1000),
    'croissantzthreshold':(0.05, 2.5),
    'pb1Hold' : (0,50),
    'pb2Hold' : (0,50),
    'diff_threshold_b1_b2':(30,300)
}

print("🧠 Initializing Bayesian Optimizer...")
optimizer = BayesianOptimization(
    f=run_strategy,
    pbounds=pbounds,
    random_state=42,
)

init_points = 500
n_iter = 100

print("\n⚙️ Running initial random points (parallelized)...")

# Generate random points from the optimizer space
init_params = [optimizer._space.array_to_params(optimizer._space.random_sample()) for _ in range(init_points)]


# Run in parallel using 4 cores
with multiprocessing.Pool(processes=4) as pool:
    results_parallel = pool.map(run_strategy_wrapper, init_params)

# Register results with the optimizer
for param_dict, profit in zip(init_params, results_parallel):
    optimizer.register(params=param_dict, target=profit)


print("\n⚙️ Running optimization iterations...")
for _ in tqdm(range(n_iter), desc="🚀 Bayesian Optimization"):
    optimizer.maximize(init_points=0, n_iter=1)
    print(f"📈 Best so far: {optimizer.max['target']:.2f}")


# Final results
print("\n🏁 Optimization complete. Sorting top 10 strategies by profit...")
top_10 = sorted(results, key=lambda x: x["profit"], reverse=True)[:10]
df = pd.DataFrame(top_10)

print("\n📊 Top 10 parameter combinations:\n")
print(df.to_string(index=False))

df.to_csv("top_10_strategies.csv", index=False)
print("\n💾 Results saved to top_10_strategies.csv")
