# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:47:14 2025

@author: Ibrahim
"""

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


# DATA PREPARATION

boston = pd.read_csv('boston.csv')
X = boston.drop(columns=['medv']) #Independent columns
y = boston['medv'] # Dependent column - Median value of house

'''
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax ate per
 $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homms in $1000's
'''

boston.head()
y[1:10] # response

# Splitting dataset as test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training and evaluation

# New GB Regressor object
gradient_regressor = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)

# n_estimators: Number of weak learners to train iteratively
# learning rate: It contributes to the weights of weak learners. It uses 1 as a default value

# Train Gradient Boost Regressor
model = gradient_regressor.fit(X_train, y_train)

# Predicting the response for test dataset
y_pred = model.predict(X_test)

r2_score(y_pred, y_test)

import matplotlib.pyplot as plt
# %matplotlib inline

# Plot feature importance
feature_importance = model.feature_importances_ # Which one is more important for this model

# Making importance relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# Tunning the Hypermeters
from sklearn.model_selection import GridSearchCV
LR = {'learning_rate':[0.15, 0.1, 0.1, 0.05], 'n_estimators':[100, 150, 200, 250]}

tuning = GridSearchCV(estimator = GradientBoostingRegressor(), param_grid = LR, scoring='r2')
tuning.fit(X_train, y_train)
tuning.best_params_, tuning.best_score_