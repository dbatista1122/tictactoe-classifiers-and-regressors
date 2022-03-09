import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load(file):

    game = np.loadtxt(file)
    x = game[:, :9]
    y = game[:, 9:]

    return x, y


def theta(x_train, y_train):
    x_transpose = x_train.T

    X_XT = x_transpose.dot(x_train)

    X_XT_inverse = np.linalg.inv(X_XT)

    X_XT_inverse_XT = X_XT_inverse.dot(x_transpose)

    theta = X_XT_inverse_XT.dot(y_train)

    return theta


def mse_calc(prediction, y_test):

    total_data = len(prediction)
    error = (np.sum((prediction - y_test)**2))/total_data
    return error


def rsq(prediction, y_test):

    total_data = len(y_test)

    y_avg = np.sum(y_test)/total_data

    tot_err = np.sum((y_test-y_avg)**2)

    res_err = np.sum((y_test-prediction)**2)

    r2 = 1 - (res_err / tot_err)

    return r2


def linear_regression(file):
    x, y = load(file)

    avg_error = 0
    avg_r = 0
    avg_accuracy = 0
    for i in range(9):

        y_new = y[:, i:i+1]

        x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size=0.2, random_state=42)

        t = theta(x_train, y_train)

        predictions = x_test.dot(t)

        error = mse_calc(predictions, y_test)
        avg_error += error

        r = rsq(predictions, y_test)
        avg_r += r

    avg_accuracy /= 9
    avg_error /= 9
    avg_r /= 9

    print("=" * 40, file[9:], "=" * 40, "\n")
    print("Linear Regression Evaluation: \n")
    print("\n")
    print("MSE: ", avg_error)
    print("R Squared: ", avg_r)

