import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# Read the CSVs
day2PriceDF = pd.read_csv("round 1/prices_round_1_day_-2.csv", sep=";")
day1PriceDF = pd.read_csv("round 1/prices_round_1_day_-1.csv", sep=";")
day0PriceDF = pd.read_csv("round 1/prices_round_1_day_0.csv", sep=";")

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


def moment(df):
    df.fillna(0)
    return (
        df["bid_volume_1"] * df["bid_price_1"]
        + df["bid_volume_2"] * df["bid_price_2"]
        + df["bid_volume_3"] * df["bid_price_3"]
    ) / (df["bid_volume_1"] + df["bid_volume_2"] + df["bid_volume_3"])


kelp = dictofDF["KELP"]
ink = dictofDF["SQUID_INK"]
resin = dictofDF["RAINFOREST_RESIN"]
kelpmid_price = kelp["mid_price"]
kelpmoment = moment(kelp)
inkmid_price = ink["mid_price"]
resinmid_price = resin["mid_price"]
inkmoment = moment(ink)
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
inklines = [
    0.0,
    9.3452022939531e-06,
    3.0331964847770748e-06,
    -0.0007589588386768175,
    -0.0002540117117792504,
    0.0007601333923670026,
    0.0010136092972173707,
    -0.0012580974550694332,
    -0.001510586080391952,
    0.00025402486733725133,
    -0.000509630837167474,
    0.001735757474718193,
    0.0005093741937142857,
    -0.0010134658387485316,
    -0.0017479306886413846,
    0.0012618002061133378,
    0.0020206536328920776,
    0.001504117739616198,
    -0.002743573323169716,
    0.002243422436623028,
    -0.0022394157917402053,
    -0.002008354734870741,
    -0.0030142930432078365,
    0.006032927743470007,
    0.002513466645070161,
    0.0030119266027009007,
    -0.0042634977921961895,
    -0.0032459997065433788,
    0.0046848179900707155,
    -0.00452006286159514,
    0.003250948479182156,
    0.002753654166342292,
    -0.0039908919162827804,
    -0.005450941526263627,
    0.003535547283344697,
    -0.002512020249513628,
    -0.003775889823086874,
    -0.004705736509268734,
    -0.006802755340643274,
    0.006947771921418304,
    -0.004959033870489804,
    0.010861694424330196,
    0.0040142111285407425,
    -0.006202726243801828,
    0.008626887131560028,
    0.008553100498930863,
    0.003742859681903285,
    -0.00876361913784936,
    -0.006468799270700232,
    0.004228722216898299,
    0.006421452128067481,
    -0.00595393241098862,
    0.004528268880737295,
    0.005456702253855279,
    0.008905852417302799,
    0.007326932794340576,
    -0.006944444444444444,
    0.011292016806722689,
    0.009392121054004696,
    0.0051773233238415735,
    -0.005155968032998196,
    0.011734028683181226,
    -0.01004895645452203,
    0.00491336953710887,
]

kelplines = np.sort(kelplines)
inklines = np.round(inklines, decimals=6)


def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()


def find_nearest_value(array, value):
    return array[(np.abs(array - value)).argmin()]


# Define the dimensions of the matrix
rows = len(inklines)
cols = len(kelplines)

# Create a matrix filled with empty arrays
pMatrix = np.empty((rows, cols), dtype=object)

# Initialize each element with an empty array
for i in range(rows):
    for j in range(cols):
        pMatrix[i, j] = []

i = 0
for kelpreturn, inkreturn in zip(kelpReturns, inkReturns):
    KIndex, IIndex = (
        find_nearest_index(kelplines, kelpreturn),
        find_nearest_index(inklines, inkreturn),
    )
    if i == len(inkReturns) - 1:
        continue
    pMatrix[IIndex][KIndex].append(find_nearest_value(inklines, inkReturns[i + 1]))
    i += 1
# Adjust layout and show the plot
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
            # Plot a histogram of the data in the cell with bins equal to the size of inklines
            ax.hist(cell_data, bins=len(inklines), color="blue", alpha=0.7)
        else:
            # If the cell is empty, fill the background with red and display a message
            ax.set_facecolor("red")
            ax.text(
                0.5, 0.5, "Empty", ha="center", va="center", fontsize=6, color="white"
            )

        # Annotate each subplot with its cell index

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

# Show the plot
plt.show()
