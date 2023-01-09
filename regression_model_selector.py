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
N_TESTS = 2

dataframe = pd.read_csv(DATA)
X = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values


# Linear Regression
def linear_regression_accuracy(X, y, random_state=random.randint(1, 100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)

    linear_regression_y_pred = linear_regressor.predict(X_test)

    return r2_score(y_test, linear_regression_y_pred)


# Polynomial Regression
def polynomial_regression_accuracy(X, y, random_state=random.randint(1, 100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    poly_regressor = LinearRegression()
    poly_regressor.fit(X_poly, y_train)

    poly_y_pred = poly_regressor.predict(poly_reg.transform((X_test)))

    return r2_score(y_test, poly_y_pred)


# Decision Tree Regression
def decision_tree_regression_accuracy(X, y, random_state=random.randint(1, 100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(X_train, y_train)

    decision_tree_y_pred = dt_regressor.predict(X_test)

    return r2_score(y_test, decision_tree_y_pred)


# Random Forest Regression
def random_forest_regression_accuracy(X, y, random_state=random.randint(1, 100)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)

    rf_y_pred = rf_regressor.predict(X_test)

    return r2_score(y_test, rf_y_pred)


# Support Vector Regression
def support_vector_accuracy(X, y, random_state=random.randint(1, 100)):
    y_reshaped = y.reshape(len(y), 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_reshaped, test_size=0.2, random_state=random_state)

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)

    svr_regressor = SVR(kernel="rbf")
    svr_regressor.fit(X_train, y_train)

    y_pred = sc_y.inverse_transform(svr_regressor.predict(sc_X.transform(X_test)).reshape(-1, 1))

    return r2_score(y_test, y_pred)


def find_best_fit(X, y, n_tests=1):
    model_fits = {"Linear Regression": [],
                  "Polynomial Regression": [],
                  "Decision Tree Regression": [],
                  "Random Forest Regression": [],
                  "Support Vector Regression": []}
    for _ in range(n_tests):
        random = random.randint(1, 100)
        model_fits["Linear Regression"].append(linear_regression_accuracy(X, y, random))
        model_fits["Polynomial Regression"].append(polynomial_regression_accuracy(X, y, random))
        model_fits["Decision Tree Regression"].append(decision_tree_regression_accuracy(X, y, random))
        model_fits["Random Forest Regression"].append(random_forest_regression_accuracy(X, y, random))
        model_fits["Support Vector Regression"].append(support_vector_accuracy(X, y, random))

    model_fits = dict(sorted(model_fits.items(), key=lambda item: sum(item[1]) / len(item[1]), reverse=True))

    i = 1
    fit_info = []
    for fit in model_fits.items():
        fit_info.append(f"{i} - {fit[0]} - {'{:.4f}'.format(sum(fit[1]) / len(fit[1]))}")
        i += 1

    return fit_info


if __name__ == "__main__":
    for fit in find_best_fit(X, y, n_tests=N_TESTS):
        print(fit)
