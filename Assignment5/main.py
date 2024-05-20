import sys, math
from functools import partial
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
import sklearn as skl, sklearn.datasets as skds, tensorflow as tf
from sklearn.datasets import load_sample_images
import seaborn as sns

# import CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10.load_data()

# split data into training and testing sets
(X_train_full, y_train_full), (X_test, y_test) = cifar10

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
    ,DefaultConv2D(filters=64, kernel_size=3, input_shape=[32, 32, 3]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256)

    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=2, mode='min',
                                         verbose=1, restore_best_weights=True)

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam",
              metrics=["accuracy"])

from sklearn import metrics

# fit model to data and evaluate
cnn_hist = model.fit(X_train, y_train, epochs=50,
                     validation_data=(X_valid, y_valid), callbacks=[es])
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

from sklearn import ensemble

# fit model to data and evaluate
model = ensemble.RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5)
X_train_temp = np.reshape(X_train, (X_train.shape[0],-1))
X_test_temp = np.reshape(X_test, (X_test.shape[0],-1))
rf = model.fit(X_train_temp, y_train.ravel())
y_pred = model.predict(X_test_temp)
y_pred = np.argmax(y_pred, axis=0)

from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, X_test_temp, y_test)

# generate confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# display confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# print metrics
report = classification_report(y_test, y_pred)
print(report)
