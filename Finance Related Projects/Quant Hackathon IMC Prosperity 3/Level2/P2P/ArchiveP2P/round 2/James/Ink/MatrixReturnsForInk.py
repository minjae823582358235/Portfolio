import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os


# File paths
data_dir = "/Users/jameszhao/Documents/IMC3Round1"


day2_file = os.path.join(data_dir, "day2.csv")
day1_file = os.path.join(data_dir, "day1.csv")
day0_file = os.path.join(data_dir, "day0.csv")

# Load the provided CSVs (adjust paths as needed)
day2PriceDF = pd.read_csv(day2_file, sep=";")
day1PriceDF = pd.read_csv(day1_file, sep=";")
day0PriceDF = pd.read_csv(day0_file, sep=";")

# Store them in a list
DFArr = [day2PriceDF, day1PriceDF, day0PriceDF]
dictofDF = {}
timestamplimit = 999900
increment = 100
commodities = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]

# Loop over each dataframe (except the last one)
for i in range(len(DFArr)):
    df = DFArr[i]
    for product in commodities:
        print(f"Processing: {product}")

        # Filter for product
        productDF = df[df["product"] == product]

        # Offset the timestamp column directly, don't use df[df["timestamp"]]
        productDF = productDF.copy()  # Avoid SettingWithCopyWarning
        productDF["timestamp"] += i * 1_000_000
        if i == 0:
            dictofDF[product] = productDF[
                [
                    "timestamp",
                    "mid_price",
                    "bid_price_1",
                    "bid_price_2",
                    "bid_price_3",
                    "ask_price_1",
                    "ask_price_2",
                    "ask_price_3",
                    "ask_volume_1",
                    "ask_volume_2",
                    "ask_volume_3",
                    "bid_volume_1",
                    "bid_volume_2",
                    "bid_volume_3",
                ]
            ].copy()
        if i != 0:
            dictofDF[product] = pd.concat(
                [
                    dictofDF[product],
                    productDF[
                        [
                            "timestamp",
                            "mid_price",
                            "bid_price_1",
                            "bid_price_2",
                            "bid_price_3",
                            "ask_price_1",
                            "ask_price_2",
                            "ask_price_3",
                            "ask_volume_1",
                            "ask_volume_2",
                            "ask_volume_3",
                            "bid_volume_1",
                            "bid_volume_2",
                            "bid_volume_3",
                        ]
                    ],
                ],
                ignore_index=True,
            )
        # Print the result


kelp = dictofDF["KELP"]
ink = dictofDF["SQUID_INK"]
resin = dictofDF["RAINFOREST_RESIN"]
kelpmid_price = kelp["mid_price"]
inkmid_price = ink["mid_price"]
resinmid_price = resin["mid_price"]
kelpReturns = []
inkReturns = []
resinReturns = []


def returns(i, array):
    return (array[i] - array[i - 1]) / array[i - 1]


for i in range(len(kelpmid_price)):
    if i == 0:
        continue
    kelpReturns.append(returns(i, kelpmid_price))
    inkReturns.append(returns(i, inkmid_price))
    resinReturns.append(returns(i, resinmid_price))
kelplines = [
    0.000494,
    -0.000247,
    0.0,
    -0.000494,
    0.000247,
    0.000741,
    -0.000741,
    0.000989,
    -0.000988,
    -0.00123,
    0.00148,
    -0.00148,
    0.00124,
]
inklines = [  ## atol plus minus 0.006
    -4.959e-03,
    -4.706e-03,
    -4.520e-03,
    -4.263e-03,
    -3.991e-03,
    -3.776e-03,
    -3.246e-03,
    -3.014e-03,
    -2.744e-03,
    -2.512e-03,
    -2.239e-03,
    -2.008e-03,
    -1.748e-03,
    -1.511e-03,
    -1.258e-03,
    -1.013e-03,
    -7.590e-04,
    -5.100e-04,
    -2.540e-04,
    0.000e00,
    3.000e-06,
    9.000e-06,
    2.540e-04,
    5.090e-04,
    7.600e-04,
    1.014e-03,
    1.262e-03,
    1.504e-03,
    1.736e-03,
    2.021e-03,
    2.243e-03,
    2.513e-03,
    2.754e-03,
    3.012e-03,
    3.251e-03,
    3.536e-03,
    3.743e-03,
    4.014e-03,
    4.229e-03,
    4.528e-03,
    4.685e-03,
    4.913e-03,
]

kelplines = np.sort(kelplines)
inklines = np.round(inklines, decimals=6)


def find_nearest_index(array, value, mode):
    if mode == "KELP":
        tol = 0.0015  # TODO slightly skewed
    if mode == "INK":
        tol = 0.0058
    if abs(value) > tol:
        if value > tol:
            return "PlusAnomaly"
        else:
            return "MinusAnomaly"
    return (np.abs(array - value)).argmin()


def find_nearest_value(array, value, mode):
    if mode == "KELP":
        tol = 0.0015
    if mode == "INK":
        tol = 0.006
    if abs(value) > tol:
        return value
    return array[(np.abs(array - value)).argmin()]


# Define the dimensions of the matrix
rows = len(inklines) + 2
cols = len(kelplines) + 2

# Create a matrix filled with empty arrays
pMatrix = np.empty((rows, cols), dtype=object)

# Initialize each element with an empty array
for i in range(rows):
    for j in range(cols):
        pMatrix[i, j] = []

i = 0
for kelpreturn, inkreturn in zip(kelpReturns, inkReturns):
    KIndex, IIndex = (
        find_nearest_index(kelplines, kelpreturn, mode="KELP"),
        find_nearest_index(inklines, inkreturn, mode="INK"),
    )
    if i == len(inkReturns) - 1:
        continue
    if KIndex == "MinusAnomaly":
        if IIndex == "MinusAnomaly":
            pMatrix[0][0].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
        elif IIndex == "PlusAnomaly":
            pMatrix[-1][0].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
        else:
            pMatrix[int(IIndex) + 1][0].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
    elif KIndex == "PlusAnomaly":
        if IIndex == "MinusAnomaly":
            pMatrix[0][-1].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
        elif IIndex == "PlusAnomaly":
            pMatrix[-1][-1].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
        else:
            pMatrix[int(IIndex) + 1][-1].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
    else:
        if IIndex == "MinusAnomaly":
            pMatrix[0][int(KIndex) + 1].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
        elif IIndex == "PlusAnomaly":
            pMatrix[-1][int(KIndex) + 1].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
        else:
            pMatrix[int(IIndex) + 1][int(KIndex) + 1].append(
                find_nearest_value(inklines, inkReturns[i + 1], mode="INK")
            )
    i += 1
# Calculate the mean of the arrays in each cell of pMatrix
# TOTAL MATRIX
# # Adjust layout and show the plot
# fig, axes = plt.subplots(
#     rows, cols, figsize=(15, 15)
# )  # Adjust the figure size as needed

# # Iterate through each cell in pMatrix
# for i in range(rows):
#     for j in range(cols):
#         ax = axes[i, j]  # Get the subplot for the current cell
#         cell_data = pMatrix[i, j]  # Get the data in the current cell

#         # Check if the cell contains data
#         if len(cell_data) > 0:
#             # Plot a histogram of the data in the cell with bins equal to the size of inklines
#             ax.hist(cell_data, bins=len(inklines), color="blue", alpha=0.7)
#         else:
#             # If the cell is empty, fill the background with red and display a message
#             ax.set_facecolor("red")
#             ax.text(
#                 0.5, 0.5, "Empty", ha="center", va="center", fontsize=6, color="white"
#             )

#         # Annotate each subplot with its cell index

#         # Remove x and y ticks for better visualization
#         ax.set_xticks([])
#         ax.set_yticks([])

# # Add global axis labels
# fig.text(0.5, 0.04, "Kelp Line", ha="center", fontsize=12)  # X-axis label
# fig.text(
#     0.04, 0.5, "Ink Line", va="center", rotation="vertical", fontsize=12
# )  # Y-axis label

# # Adjust layout to prevent overlap
# plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Leave space for axis labels

# # Show the plot
# plt.show()

# CUSTOM CELL
# Define the cell coordinates (row and column) for the histogram
# row, col = 13, 7  # Example: row 5, column 3

# # Get the data for the specified cell
# cell_data = pMatrix[row, col]

# # Check if the cell contains data
# if len(cell_data) > 0:
#     # Plot the histogram for the specified cell
#     plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
#     plt.hist(cell_data, bins=len(inklines), color="blue", alpha=0.7)
#     plt.title(f"Histogram for Cell ({row}, {col})")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
# else:
#     # If the cell is empty, display a message
#     print(f"Cell ({row}, {col}) is empty.")

# # Show the plot
# plt.show()


# MODE MODE
# from scipy.stats import mode

# # Create a matrix to store the mode values
# mode_matrix = np.empty((rows, cols), dtype=object)

# # Calculate the mode for each cell in pMatrix
# for i in range(rows):
#     for j in range(cols):
#         cell_data = pMatrix[i, j]
#         if len(cell_data) > 0:
#             # Calculate the mode of the cell data
#             mode_result = mode(
#                 cell_data, nan_policy="omit"
#             )  # Handle NaN values if present
#             # Check if mode_result.mode is a scalar or array
#             if np.isscalar(mode_result.mode):
#                 mode_matrix[i, j] = mode_result.mode  # Store the scalar mode value
#             else:
#                 mode_matrix[i, j] = mode_result.mode[0]  # Store the first mode value
#         else:
#             mode_matrix[i, j] = np.nan  # Assign NaN for empty cells

# # Adjust layout and show the plot
# fig, axes = plt.subplots(
#     rows, cols, figsize=(15, 15)
# )  # Adjust the figure size as needed

# # Iterate through each cell in pMatrix
# for i in range(rows):
#     for j in range(cols):
#         ax = axes[i, j]  # Get the subplot for the current cell
#         cell_data = pMatrix[i, j]  # Get the data in the current cell

#         # Check if the cell contains data
#         if len(cell_data) > 0:
#             # Plot a histogram of the data in the cell with bins equal to the size of inklines
#             ax.hist(cell_data, bins=len(inklines), color="blue", alpha=0.7)
#             # Annotate the mode value with 6 decimal places
#             ax.text(
#                 0.5,
#                 0.9,
#                 f"Mode: {mode_matrix[i, j]:.6f}",
#                 ha="center",
#                 va="center",
#                 transform=ax.transAxes,
#                 fontsize=6,
#                 color="black",
#             )
#         else:
#             # If the cell is empty, fill the background with red and display a message
#             ax.set_facecolor("red")
#             ax.text(
#                 0.5, 0.5, "Empty", ha="center", va="center", fontsize=6, color="white"
#             )

#         # Remove x and y ticks for better visualization
#         ax.set_xticks([])
#         ax.set_yticks([])

# # Add global axis labels
# fig.text(0.5, 0.04, "Kelp Line", ha="center", fontsize=12)  # X-axis label
# fig.text(
#     0.04, 0.5, "Ink Line", va="center", rotation="vertical", fontsize=12
# )  # Y-axis label

# # Adjust layout to prevent overlap
# plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Leave space for axis labels

# # Show the plot
# plt.show()


# MEAN MATRIX
# Create a matrix to store the mean values
mean_matrix = np.empty((rows, cols))  # Create a matrix to store the mean values

# Calculate the mean for each cell in pMatrix
for i in range(rows):
    for j in range(cols):
        cell_data = pMatrix[i, j]
        if len(cell_data) > 0:
            mean_matrix[i, j] = np.mean(cell_data)  # Calculate the mean
        else:
            mean_matrix[i, j] = np.nan  # Assign NaN for empty cells

# Adjust layout and show the plot
fig, axes = plt.subplots(
    rows, cols, figsize=(15, 15)
)  # Adjust the figure size as needed

# Iterate through each cell in pMatrix
for i in range(rows):
    for j in range(cols):
        ax = axes[i, j]  # Get the subplot for the current cell
        cell_data = pMatrix[i, j]  # Get the data in the current cell

        # Check if the cell contains data
        if len(cell_data) > 0:
            if mean_matrix[i, j] > 0:
                colour = "g"
            if mean_matrix[i, j] == 0:
                colour = "black"
            if mean_matrix[i, j] < 0:
                colour = "r"
            # Plot a histogram of the data in the cell with bins equal to the size of inklines
            ax.hist(cell_data, bins=len(inklines), color=colour, alpha=0.7)
            # Annotate the mean value with 6 decimal places
            ax.text(
                0.5,
                0.9,
                f"Mean: {mean_matrix[i, j]:.6f}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=6,
                color="black",
            )
        else:
            # If the cell is empty, fill the background with red and display a message
            ax.set_facecolor("red")
            ax.text(
                0.5, 0.5, "Empty", ha="center", va="center", fontsize=6, color="white"
            )

        # Remove x and y ticks for better visualization
        ax.set_xticks([])
        ax.set_yticks([])

# Add global axis labels
fig.text(0.5, 0.04, "Kelp Line", ha="center", fontsize=12)  # X-axis label
fig.text(
    0.04, 0.5, "Ink Line", va="center", rotation="vertical", fontsize=12
)  # Y-axis label

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Leave space for axis labels
mean_matrix = np.round(mean_matrix, decimals=6)
mean_matrix = [list(row) for row in mean_matrix.tolist()]
print(mean_matrix)
# Show the plot
plt.show()


