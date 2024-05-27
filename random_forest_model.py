from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def get_data():


    dataset_train = pd.read_csv('train.csv')
    dataset_train = dataset_train.to_numpy()

    y_train = dataset_train[:, -1] # Last column is target column
    X_train = dataset_train[:, :-1] # other columns are feature


    dataset_test = pd.read_csv('test.csv')
    dataset_test = dataset_test.to_numpy()

    y_test = dataset_test[:, -1] # Last column is target column
    X_test = dataset_test[:, :-1] # other columns are feature

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = get_data()


def polynomial_features(X_train, X_test, degree):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    return X_train_poly, X_test_poly

degree = 2
X_train_poly, X_test_poly = polynomial_features(X_train, X_test, degree= degree)


n_estimators = 200
model = RandomForestRegressor(n_estimators= n_estimators, random_state=42, verbose= 1)


model.fit(X_train_poly, y_train)


y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)


train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print metrics and hyperparameters
print(f'Poly Degree : {degree} - N Estimators : {n_estimators}')
print(f'Final Train MSE: {train_mse}')
print(f'Final Test MSE: {test_mse}')

print(f'Final Train MAE: {train_mae}')
print(f'Final Test MAE: {test_mae}')

print(f'Final Train R2: {train_r2}')
print(f'Final Test R2: {test_r2}')
