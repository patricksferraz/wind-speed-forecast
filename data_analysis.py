import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def show_all(dataframe, index):
    nans = np.where(np.isnan(dataframe.values))
    print(
        "\nColumn {}\nMax {}\nMédia {}\nMediana {}\nMínimo {}\nNan {}".format(
            index,
            dataframe.max(),
            dataframe.mean(),
            dataframe.median(),
            dataframe.min(),
            np.where(np.isnan(dataframe.values))
            # dataframe.mode(),
        )
    )
    return nans


dataframe = pd.read_csv(
    "confidential/train150.txt", delimiter="\t", header=None
)
dataset = dataframe.values

print("[INFO] Dataframe info")
dataframe.info()

print("[INFO] Columns info")
nans_4 = show_all(dataframe[4], 4)
nans_5 = show_all(dataframe[5], 5)
nans_6 = show_all(dataframe[6], 6)
nans_7 = show_all(dataframe[7], 7)
nans_8 = show_all(dataframe[8], 8)

# Get all where column 0 is 15 value
# subset = dataframe[dataframe[0] == 15]
# values = subset[6].values
# index_nan = np.where(np.isnan(values))
# values = np.delete(values, index_nan[0])

# print(np.average(values))
# print(np.average([24.33333333, 24.08333333]))

plt.style.use("ggplot")
_, axs = plt.subplots(5, 1)
axs[0].plot(range(0, len(dataset[:, 3])), dataset[:, 4])
axs[0].set_title("velo")
axs[1].plot(range(0, len(dataset[:, 3])), dataset[:, 5])
axs[1].set_title("dir")
axs[2].plot(range(0, len(dataset[:, 3])), dataset[:, 6])
axs[2].set_title("temp")
axs[3].plot(range(0, len(dataset[:, 3])), dataset[:, 7])
axs[3].set_title("umi")
axs[4].plot(range(0, len(dataset[:, 3])), dataset[:, 8])
axs[4].set_title("press")
plt.legend()
plt.savefig("out/plot.png")
# plt.show()
