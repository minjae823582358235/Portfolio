import pandas as pd
import re

# Read the top 10 strategies from the CSV
df = pd.read_csv("top_10_strategies.csv")

# The base code path (you'll modify this path to the actual file name if needed)
base_code_file = "Level4\ExtremelyNaive8.py"

# Read the base code once
with open(base_code_file, "r") as file:
    base_code = file.read()

# Loop through the top 10 strategies and generate new scripts
for i, row in df.iterrows():
    # Clone the base code for each script
    modified_code = base_code

    # Replace parameters in the base code with values from the row
    for param, value in row.items():
        if param != 'profit':  # Ignore the 'profit' column (not a parameter)
            # Use regex to find and replace the parameter in the base code
            modified_code = re.sub(rf"{param}\s*=\s*[-\d.]+", f"{param} = {value:.4f}", modified_code)

    # Write the modified code to a new Python file
    output_filename = f"Round3BS_optimized_{i+1}.py"
    with open(output_filename, "w") as output_file:
        output_file.write(modified_code)

    print(f"Generated script: {output_filename}")
