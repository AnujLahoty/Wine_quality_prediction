# Import the following packages pandas, numpy and sklearn.
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

###############################################################################
# LOGISTIC REGRESSION (Algorithm number - 3)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(multi_class="multinomial",solver="lbfgs", random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


###############################################################################
# SUPPORT VECTOR REGRESSION (Algorithm number - 4)
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0, decision_function_shape='ovo')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))

###############################################################################
# DECISION TREE CLASSIFICATION (Algorithm number - 5)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
                            DecisionTreeClassifier(), 
                            n_estimators = 100,
                            bootstrap = True,
                            n_jobs = -1,
                            oob_score = True
                            )

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
