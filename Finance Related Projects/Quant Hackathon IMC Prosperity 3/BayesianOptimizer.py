import optuna
import subprocess
import re
import os
from collections import defaultdict
import numpy as np

# ---------------- Define objective function ----------------
def compute_sortino(returns):
    """
    Compute the Sortino ratio given target rf rate (0 as default)
    """
    returns = np.array(returns)
    # Calculate the average return over all periods
    avg_return = np.mean(returns)
    
    # Returns that are below the target (downside)
    downside_returns = returns[returns < 0]  # consider changing this 0 threshold since its usuall rf rate
    
    # Downside deviation of the negative returns - if no negative returns, set to a small number (avoid division by zero)
    if len(downside_returns) == 0:
        downside_deviation = 1e-6
    else:
        downside_deviation = np.sqrt(np.mean((downside_returns - 0) ** 2))
    
    # Compute Sortino ratio.
    sortino_ratio = (avg_return - 0) / downside_deviation
    return sortino_ratio
# -----------------------------------------------------------



# ---------------- Optimizer function ----------------
n_trials = 100
def objective(trial):
    """
    Optimizes parameters for the backtester - set parameters, file to run, and days to test
    """
    # Define the parameters to optimize - CHECK IF FLOAT OR INT
    param_name1 = trial.suggest_float("param_name1", 0, 10)
    param_name2 = trial.suggest_int("param_name2", 0, 10)

    # Set the environment variables for the backtester
    env = os.environ.copy()
    env["major"] = str(major)
    env["minor"] = str(minor)

    # Compile command - CHECK CODE AND DAYS BEING TESTED ON
    cmd = ["prosperity3bt", "FILENAME.py", "DAY"]

    # Run the backtester with the current set of parameters.
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = result.stdout

    # Extract profits from each asset for each day
    pattern = r"^(CROISSANTS|DJEMBES|JAMS|KELP|PICNIC_BASKET1|PICNIC_BASKET2|RAINFOREST_RESIN|SQUID_INK):\s*([-0-9,]+)"
    matches = re.findall(pattern, output, flags=re.MULTILINE)

    # Store the profits in a dictionary
    asset_returns = defaultdict(list)
    for asset, value in matches:
        asset_returns[asset].append(int(value.replace(",", "")))
    
    # Compute the Sortino ratio for each asset
    asset_sortino = {}
    for asset, returns in asset_returns.items():
        asset_sortino[asset] = compute_sortino(returns)
    
    # Combine the individual Sortino ratios into a composite average - THIS CAN BE IMPROVED BUT HOW
    overall_sortino = np.mean(list(asset_sortino.values()))
    
    return overall_sortino

# Create and run the study to maximise RESIN profit
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

print("\n-------- Optimization Complete --------")


completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:n_trials//10]  # Indexing top 10% of trials - in case repeated values etc.

print(f"\n-------- Top {n_trials} Trials: --------")
for trial in top_trials:
    print(f"Trial {trial.number}:")
    print(f"Params: {trial.params}")
    print(f"Averaged Sortino: {trial.value}")