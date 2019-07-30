# Import the following packages pandas, numpy and sklearn.
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the data-set and preprocessing
df = pd.read_csv('winequality-red.csv')


X = df.iloc[:, :-1].values
y = df.iloc[:, 11].values

# Data-set is divided into test data and train data based on test_size variable.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




##############################################################################


#  MULTIPLE LINEAR REGRESSION  (Algorithm number - 1)

# Fit the training data into multiple linrear regression.

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

y_predicted = linear_regressor.predict(X_test)

# CALCULATING THE ROOT MEAN SQUARED ERROR AND R SQUARED VALUES
rmse_multiple_regression = np.sqrt(mean_squared_error(y_predicted, y_test))
r_2_score_multiple_regression = r2_score(y_predicted, y_test)
##############################################################################

#############################################################################
# POLYNOMIAL REGRESSION (Algorithm number - 2)
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
y_predicted_poly = lin_reg_2.predict(poly_reg.fit_transform(X_test))

# CALCULATING THE ROOT MEAN SQUARED ERROR AND R SQUARED VALUES
rmse_polynomial_regression = np.sqrt(mean_squared_error(y_predicted_poly, y_test))
r_2_score_polynomial_regression = r2_score(y_predicted_poly, y_test)
##############################################################################

