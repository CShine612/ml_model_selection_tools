import pandas as pd
import random

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

DATA = "Regression_Data.csv"
RANDOM = random.randint(1, 100)

dataframe = pd.read_csv(DATA)
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM)

model_fits = {}

#Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

linear_regression_y_pred = linear_regressor.predict(X_test)

model_fits["Linear Regression"] = r2_score(y_test, linear_regression_y_pred)

#Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y_train)

poly_y_pred = poly_regressor.predict(poly_reg.transform((X_test)))

model_fits["Polynomial Regression"] = r2_score(y_test, poly_y_pred)

#Decision Tree Regression
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

decision_tree_y_pred = dt_regressor.predict(X_test)

model_fits["Decision Tree Regression"] = r2_score(y_test, decision_tree_y_pred)

#Random Forest Regression
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)

rf_y_pred = rf_regressor.predict(X_test)

model_fits["Random Forest Regression"] = r2_score(y_test, rf_y_pred)

#Support Vector Regression
y_reshaped = y.reshape(len(y), 1)
X_train, X_test, y_train, y_test = train_test_split(X, y_reshaped, test_size=0.2, random_state=RANDOM)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

svr_regressor = SVR(kernel="rbf")
svr_regressor.fit(X_train, y_train)

y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_X.transform(X_test)).reshape(-1,1))

model_fits["Support Vector Regression"] = r2_score(y_test, y_pred)

model_fits = dict(sorted(model_fits.items(), key=lambda item: item[1], reverse=True))

i = 1
for fit in model_fits.items():
    print(f"{i} - {fit[0]} - {'{:.4f}'.format(fit[1])}")
    i += 1
