import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


def load(file_name):
    game = np.loadtxt(file_name)
    x = game[:, :9]
    y = game[:, 9:]
    return x, y


def con_mat(y_test, predictions):
    confusion = confusion_matrix(y_test, predictions, normalize='true')

    return confusion


def knn_regressor(file):
    x, y = load(file)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, shuffle=True, test_size=0.2)

    model = neighbors.KNeighborsRegressor(1, weights='distance')

    model.fit(x_train, y_train)
    model.predict(x_test)

    acc = model.score(x_test, y_test)

    print("=" * 40, file[9:], "=" * 40, "\n")
    print("K-Nearest Neighbors Regression Accuracy", acc, "\n")
    print("\n")

    return model
