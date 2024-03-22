#########################
# Data importation
#########################

import pandas as pd

# read in dataset
HouseDF = pd.read_csv('1553768847-housing.csv')

# ensure no empty values in dataframe
HouseDF = HouseDF.fillna(0)

# print top 4 rows to ensure proper read-in
print(HouseDF.head())

#########################
# Data pre-processing
#########################

# set X, y
X=HouseDF.iloc[:,:-1]
y=HouseDF.iloc[:,-1]

# fix data so that ocean_proximity becomes a float as well
X = X.apply(pd.to_numeric, errors='coerce')
Y = y.apply(pd.to_numeric, errors='coerce')

# ensure no empty data
X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

#########################
# Model selection and training
#########################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# split data into test set and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=101)

# linear regression model, no hyper-parameters
lm = LinearRegression()
lm.fit(X_train,y_train)

# ridge regression, no hyper-parameters
rm = Ridge()
rm.fit(X_train, y_train)

# random forest regression, no hyper-parameters
rf = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
rf.fit(X_train, y_train)

#########################
# Model evaluation
#########################

import numpy as np
from sklearn import metrics

# store linear model predictions
predict1 = lm.predict(X_test)

# store ridge model predictions
predict2 = rm.predict(X_test)

# store random forest model predictions
predict3 = rf.predict(X_test)

# print analysis metrics
print('Linear Regression Analysis:')
print('   MAE:', metrics.mean_absolute_error(y_test, predict1))
print('   MSE:', metrics.mean_squared_error(y_test, predict1))
print('   RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict1)))

print('Ridge Regression Analysis:')
print('   MAE:', metrics.mean_absolute_error(y_test, predict2))
print('   MSE:', metrics.mean_squared_error(y_test, predict2))
print('   RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict2)))

print('Random Forest Regression Analysis:')
print('   MAE:', metrics.mean_absolute_error(y_test, predict3))
print('   MSE:', metrics.mean_squared_error(y_test, predict3))
print('   RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict3)))

#########################
# Model visual evaluation
#########################

from matplotlib import pyplot as plt

# create scatter plot for linear regression model
plt.figure(figsize=(10,10))
plt.scatter(y_test, predict1, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(predict1), max(y_test))
p2 = min(min(predict1), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# create scatter plot for ridge regression model
plt.figure(figsize=(10,10))
plt.scatter(y_test, predict2, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(predict2), max(y_test))
p2 = min(min(predict2), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# create scatter plot for random forest regression model
plt.figure(figsize=(10,10))
plt.scatter(y_test, predict3, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(predict3), max(y_test))
p2 = min(min(predict3), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

#########################
# Hyper-parameter with random forest
#########################

from sklearn.model_selection import GridSearchCV

# use n_estimators field to parse for optimal number
rf_params = {
    'n_estimators':[25,50,75,100,125,150,175,200]
}

# iterate through rf_params to find optimal n_estimators value
rf2 = RandomForestRegressor()
rf_optimize = GridSearchCV(rf2, rf_params)
rf_optimize.fit(X_train, y_train)

#print optimal value
print("Optimal parameters:", rf_optimize.best_params_)

# display scatter plot
rf2 = rf_optimize.best_estimator_
predict4 = rf2.predict(X_test)
plt.figure(figsize=(10,10))
plt.scatter(y_test, predict4, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(predict4), max(y_test))
p2 = min(min(predict4), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# calculate fields for comparison
print('Random Forest Hyper-parameter Regression Analysis:')
print('   MAE:', metrics.mean_absolute_error(y_test, predict4))
print('   MSE:', metrics.mean_squared_error(y_test, predict4))
print('   RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict4)))