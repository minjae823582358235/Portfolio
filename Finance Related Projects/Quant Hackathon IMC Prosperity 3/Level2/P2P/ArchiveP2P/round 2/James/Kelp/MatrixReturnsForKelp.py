import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Mockup returns for simplicity, normally these would come from actual price series
kelpReturns = np.random.normal(0, 0.001, 100)
inkReturns = np.random.normal(0, 0.002, 100)

kelplines = np.sort([
    0.000494, -0.000247, 0.0, -0.000494, 0.000247, 0.000741,
    -0.000741, 0.000989, -0.000988, -0.00123, 0.00148, -0.00148, 0.00124
])
inklines = np.round([
    -0.004959, -0.004706, -0.004520, -0.004263, -0.003991, -0.003776,
    -0.003246, -0.003014, -0.002744, -0.002512, -0.002239, -0.002008,
    -0.001748, -0.001511, -0.001258, -0.001013, -0.000759, -0.000510,
    -0.000254, 0.000, 0.000003, 0.000009, 0.000254, 0.000509, 0.000760,
    0.001014, 0.001262, 0.001504, 0.001736, 0.002021, 0.002243, 0.002513,
    0.002754, 0.003012, 0.003251, 0.003536, 0.003743, 0.004014, 0.004229,
    0.004528, 0.004685, 0.004913
], decimals=6)

def find_nearest_index(array, value, mode):
    tol = 0.0015 if mode == "KELP" else 0.0058
    if abs(value) > tol:
        return "PlusAnomaly" if value > tol else "MinusAnomaly"
    return (np.abs(array - value)).argmin()

def find_nearest_value(array, value, mode):
    tol = 0.0015 if mode == "KELP" else 0.006
    if abs(value) > tol:
        return value
    return array[(np.abs(array - value)).argmin()]

rows = len(inklines) + 2
cols = len(kelplines) + 2
pMatrix = np.empty((rows, cols), dtype=object)
for i in range(rows):
    for j in range(cols):
        pMatrix[i, j] = []

# Fill the matrix with kelpReturns[t+1] conditioned on kelpReturns[t] and inkReturns[t]
for i in range(len(kelpReturns) - 1):
    KIndex = find_nearest_index(kelplines, kelpReturns[i], "KELP")
    IIndex = find_nearest_index(inklines, inkReturns[i], "INK")
    kelpNext = find_nearest_value(kelplines, kelpReturns[i + 1], "KELP")

    if KIndex == "MinusAnomaly":
        if IIndex == "MinusAnomaly":
            pMatrix[0][0].append(kelpNext)
        elif IIndex == "PlusAnomaly":
            pMatrix[-1][0].append(kelpNext)
        else:
            pMatrix[int(IIndex) + 1][0].append(kelpNext)
    elif KIndex == "PlusAnomaly":
        if IIndex == "MinusAnomaly":
            pMatrix[0][-1].append(kelpNext)
        elif IIndex == "PlusAnomaly":
            pMatrix[-1][-1].append(kelpNext)
        else:
            pMatrix[int(IIndex) + 1][-1].append(kelpNext)
    else:
        if IIndex == "MinusAnomaly":
            pMatrix[0][int(KIndex) + 1].append(kelpNext)
        elif IIndex == "PlusAnomaly":
            pMatrix[-1][int(KIndex) + 1].append(kelpNext)
        else:
            pMatrix[int(IIndex) + 1][int(KIndex) + 1].append(kelpNext)

# Compute mean matrix
mean_matrix = np.empty((rows, cols))
for i in range(rows):
    for j in range(cols):
        cell_data = pMatrix[i, j]
        mean_matrix[i, j] = np.mean(cell_data) if len(cell_data) > 0 else np.nan

mean_matrix = np.round(mean_matrix, decimals=6)
mean_matrix_list = [list(row) for row in mean_matrix.tolist()]
mean_matrix_list[:3]  # Show a sample for brevity


# Plotting the matrix of histograms with means for KELP predictions
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

for i in range(rows):
    for j in range(cols):
        ax = axes[i, j]
        cell_data = pMatrix[i, j]

        if len(cell_data) > 0:
            mean_val = mean_matrix[i, j]
            if mean_val > 0:
                color = "g"
            elif mean_val < 0:
                color = "r"
            else:
                color = "black"

            ax.hist(cell_data, bins=len(kelplines), color=color, alpha=0.7)
            ax.text(
                0.5,
                0.9,
                f"Mean: {mean_val:.6f}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=6,
                color="black",
            )
        else:
            ax.set_facecolor("red")
            ax.text(
                0.5, 0.5, "Empty", ha="center", va="center", fontsize=6, color="white"
            )

        ax.set_xticks([])
        ax.set_yticks([])

# Axis labels
fig.text(0.5, 0.04, "Kelp Line (t)", ha="center", fontsize=12)
fig.text(0.04, 0.5, "Ink Line (t)", va="center", rotation="vertical", fontsize=12)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()
