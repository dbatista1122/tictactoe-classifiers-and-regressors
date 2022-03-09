import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
np.set_printoptions(suppress=True)


def load(file):
    game = np.loadtxt(file)
    x = game[:, :9]
    y = game[:, 9:]
    return x, y


def con_mat(y_test, predictions):
    confusion = confusion_matrix(y_test, predictions, normalize='true')

    return confusion


def mlp_regressor(file):
    x, y = load(file)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    skf = StratifiedKFold(n_splits=10,
                          random_state=42,
                          shuffle=True)

    model = MLPRegressor(solver='adam',
                         alpha=1e-6,
                         max_iter=500,
                         hidden_layer_sizes=(300, 300),
                         random_state=42,
                         activation='relu')

    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)

    print("=" * 40, file[9:], "=" * 40, "\n")
    print("MLP Regression Accuracy: ", acc, "\n")
    print("\n")

