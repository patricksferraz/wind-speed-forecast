# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import argparse

matplotlib.use("Agg")

np.random.seed(8)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument(
    "-m", "--model", required=True, help="path to output trained model"
)
ap.add_argument(
    "-p", "--plot", required=True, help="path to output accuracy/loss plot"
)
args = vars(ap.parse_args())

dataframe = pd.read_csv(args["dataset"], delimiter="\t", header=None)
dataset = dataframe.values

# dataset = np.loadtxt(args["dataset"], dtype="float", delimiter="\t")

X = dataset[:-1]
Y = dataset[:, 4][1:]

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(X_train, X_test, Y_train, Y_test) = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scalert = preprocessing.MinMaxScaler()
Y_train = scalert.fix_transform(Y_train.reshape(-1, 1))
Y_test = scalert.transform(Y_test.reshape(-1, 1))

print(X_train)
print(Y_train)

# define the 9-9-6-1 architecture using tf.keras
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(9, input_shape=(9,), activation="tanh"))
model.add(tf.keras.layers.Dense(9, activation="tanh"))
model.add(tf.keras.layers.Dense(6, activation="tanh"))
model.add(tf.keras.layers.Dense(1, activation="linear"))

# initialize our initial learning rate and # of epochs to train for
# INIT_LR = 0.01
EPOCHS = 16

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
# opt = tf.keras.optimizers.SGD(lr=INIT_LR)
model.compile(loss="mse", optimizer="sgd", metrics=["mae"])

# train the neural network
H = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=EPOCHS,
    batch_size=1,
)

# evaluate the network
print("[INFO] evaluating network...")
# predictions = model.predict(X_test, batch_size=1)
# print(classification_report(Y_test, predictions, target_names=["velocidade"]))

# evaluate the model
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
