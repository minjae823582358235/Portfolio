import optuna
import subprocess
import re
import os


def objective(trial):
    # Define the parameters to optimize - CHECK IF FLOAT OR INT
    bullmajor = trial.suggest_int("bullmajor", -20, 20)
    bullminor = trial.suggest_int("bullminor", -20, 20)
    bearmajor = trial.suggest_int("bearmajor", -20, 20)
    bearminor = trial.suggest_int("bearminor", -20, 20)
    # Set the environment variables for the backtester
    env = os.environ.copy()
    env["bullmajor"] = str(bullmajor)
    env["bullminor"] = str(bullminor)
    env["bearmajor"] = str(bearmajor)
    env["bearminor"] = str(bearminor)
    

    # Compile command - CHECK CODE AND DAYS BEING TESTED ON
    cmd = ["prosperity3bt", "JamesInkOnly.py", "2", "--no-out"]

    # Run the backtester with the current set of parameters.
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = result.stdout

    print(f"Testing with bullmajor={bullmajor}, bullminor={bullminor}, bearmajor={bearmajor}, bearminor={}")
    
    # Find total profits
    profit_strings = re.findall(r"Total profit:\s*([-0-9,]+)", output)
    # Convert the extracted strings into integers
    profits = [int(profit_str.replace(",", "")) for profit_str in profit_strings]
    print(profits)
    # Calculate the average profit
    average_profit = profits[-1]/3  # 3 for 3 days of testing - CHECK IF ALL ROUNDS ARE 3 DAYS

    return average_profit

# Create and run the study to maximise RESIN profit
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500)

print("\n-------- Optimization Complete --------")


completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]

print("\n-------- Top 10 Trials: --------")
for trial in top_trials:
    print(f"Trial {trial.number}: Params: {trial.params}, Value: {trial.value}")

#                /Level2/James/Ink
