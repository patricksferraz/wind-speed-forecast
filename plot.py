import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv(
    "confidential/train150.txt", delimiter="\t", header=None
)
dataset = dataframe.values

# dataset = np.loadtxt(args["dataset"], dtype="float", delimiter="\t")
# Dia, Mês, Ano, Hora, Velocidade, Direção,
# Temperatura, Umidade, Pressão

# X = dataset[:-1]
# Y = dataset[:, 4][1:]

plt.style.use("ggplot")
# plt.figure()
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
plt.show()
