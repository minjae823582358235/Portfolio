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
    downside_returns = returns[
        returns < 0
    ]  # consider changing this 0 threshold since its usuall rf rate

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
n_trials = 1000


def objective(trial):
    """
    Optimizes parameters for the backtester - set parameters, file to run, and days to test
    """

    # ---- Parameter suggestions ----
    # Example group: PB1
    PB1_high = trial.suggest_float("PB1_high", 2, 10)
    PB1_mid = trial.suggest_float("PB1_mid", 1, 5)
    PB1_low = trial.suggest_float("PB1_low", -3, 2)
    PB1_neg = trial.suggest_float("PB1_neg", -4, 1)

    PB2_high = trial.suggest_float("PB2_high", 2, 10)
    PB2_mid = trial.suggest_float("PB2_mid", 1, 5)
    PB2_low = trial.suggest_float("PB2_low", -3, 2)
    PB2_neg = trial.suggest_float("PB2_neg", -4, 1)

    DJ_high = trial.suggest_float("DJ_high", 1, 4)
    DJ_mid = trial.suggest_float("DJ_mid", 0, 2)
    DJ_low = trial.suggest_float("DJ_low", -1, 1)
    DJ_neg = trial.suggest_float("DJ_neg", -2, 1)

    # Other parameters
    zfactor1 = trial.suggest_float("zfactor1", 0.05, 2.5)
    zfactor2 = trial.suggest_float("zfactor2", 0.05, 2.5)
    hold_factor = trial.suggest_float("hold_factor", 0.01, 5.0)

    # ---- Constraint checking ----
    for prefix in ["PB1", "PB2", "DJ"]:
        high = eval(f"{prefix}_high")
        mid = eval(f"{prefix}_mid")
        low = eval(f"{prefix}_low")
        neg = eval(f"{prefix}_neg")
        if not (high > mid > low > neg):
            return -1e9  # Constraint violated → penalize

    # ---- Set environment vars (if used in script) ----
    env = os.environ.copy()
    env["PB1_high"] = str(PB1_high)
    env["PB1_mid"] = str(PB1_mid)
    env["PB1_low"] = str(PB1_low)
    env["PB1_neg"] = str(PB1_neg)

    env["PB2_high"] = str(PB2_high)
    env["PB2_mid"] = str(PB2_mid)
    env["PB2_low"] = str(PB2_low)
    env["PB2_neg"] = str(PB2_neg)

    env["DJ_high"] = str(DJ_high)
    env["DJ_mid"] = str(DJ_mid)
    env["DJ_low"] = str(DJ_low)
    env["DJ_neg"] = str(DJ_neg)

    env["zfactor1"] = str(zfactor1)
    env["zfactor2"] = str(zfactor2)
    env["hold_factor"] = str(hold_factor)

    # ---- Run simulation ----
    cmd = ["prosperity3bt", "P2P\ArchiveP2P\RaysAllStrat", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = result.stdout

    # ---- Extract profits ----
    pattern = r"^(CROISSANTS|DJEMBES|JAMS|KELP|PICNIC_BASKET1|PICNIC_BASKET2|RAINFOREST_RESIN|SQUID_INK):\s*([-0-9,]+)"
    matches = re.findall(pattern, output, flags=re.MULTILINE)

    asset_returns = defaultdict(list)
    for asset, value in matches:
        asset_returns[asset].append(int(value.replace(",", "")))

    # ---- Compute average Sortino ratio ----
    asset_sortino = {
        asset: compute_sortino(returns) for asset, returns in asset_returns.items()
    }
    overall_sortino = np.mean(list(asset_sortino.values()))

    return overall_sortino


# Create and run the study to maximise RESIN profit
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

print("\n-------- Optimization Complete --------")


completed_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[
    : n_trials // 10
]  # Indexing top 10% of trials - in case repeated values etc.

print(f"\n-------- Top {n_trials} Trials: --------")
for trial in top_trials:
    print(f"Trial {trial.number}:")
    print(f"Params: {trial.params}")
    print(f"Averaged Sortino: {trial.value}")
