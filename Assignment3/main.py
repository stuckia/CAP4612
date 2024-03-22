import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl
import sklearn as skl, sklearn.datasets as skds

# import data, split data and target
dataset = skds.fetch_openml('mnist_784', as_frame=False, parser='auto')
X,y = dataset.data, dataset.target

# split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# train random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=12)
rf.fit(X_train, y_train)

# train KNN
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=10)
kn.fit(X_train, y_train)

# calculate cross validation and confusion matrices
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
rf_y_train_pred = cross_val_predict(rf, X_train, y_train)
rf_cm = confusion_matrix(y_train, rf_y_train_pred)
kn_y_train_pred = cross_val_predict(kn, X_train, y_train)
kn_cm = confusion_matrix(y_train, kn_y_train_pred)

# print confusion matrices
print(rf_cm)
print(kn_cm)

# print metrics for each model
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
print('rf weighted:')
print('precision=' + str(precision_score(y_train,rf_y_train_pred,average='weighted')))
print('recall=' + str(recall_score(y_train,rf_y_train_pred,average='weighted')))
print('accuracy=' + str(accuracy_score(y_train,rf_y_train_pred)))
print('f1=' + str(f1_score(y_train,rf_y_train_pred,average='weighted')))
print('kn weighted:')
print('precision=' + str(precision_score(y_train,kn_y_train_pred,average='weighted')))
print('recall=' + str(recall_score(y_train,kn_y_train_pred,average='weighted')))
print('accuracy=' + str(accuracy_score(y_train,kn_y_train_pred)))
print('f1=' + str(f1_score(y_train,kn_y_train_pred,average='weighted')))

# scale inputs for presentation
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train_stdscaled = std_scaler.fit_transform(X_train.astype("float64"))

# visualize the confusion matrices
from sklearn.metrics import ConfusionMatrixDisplay
rf_y_train_mpred = cross_val_predict(rf, X_train_stdscaled, y_train)
ConfusionMatrixDisplay.from_predictions(y_train, rf_y_train_mpred, normalize='true', values_format='.0%')
kn_y_train_mpred = cross_val_predict(kn, X_train_stdscaled, y_train)
ConfusionMatrixDisplay.from_predictions(y_train, kn_y_train_mpred, normalize='true', values_format='.0%')
