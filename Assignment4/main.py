import sys, math
from functools import partial
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
import sklearn as skl, sklearn.datasets as skds, tensorflow as tf
from sklearn.datasets import load_sample_images
import seaborn as sns

# import MNIST fashion dataset
mnist = tf.keras.datasets.fashion_mnist.load_data()

# split data into training and testing sets
(X_train_full, y_train_full), (X_test, y_test) = mnist

# transform data values to range from 0-1 (normalization)
X_train_full = np.expand_dims(X_train_full, axis=-1).astype(np.float32) / 255
X_test = np.expand_dims(X_test.astype(np.float32), axis=-1) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# basic CNN creation, alter initial parameters
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

# create the model
model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),

    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam",
              metrics=["accuracy"])

from sklearn import metrics

# fit model to data and evaluate
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# generate confusion matrix
p_labels = np.argmax(model.predict(X_test), axis=1)
cm = tf.math.confusion_matrix(y_test, y_pred)
print(cm)

# display confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# print metrics
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

