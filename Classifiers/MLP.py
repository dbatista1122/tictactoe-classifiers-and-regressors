import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


def classifier_mlp(file):
    x, y = load(file)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    skf = StratifiedKFold(n_splits=10,
                          random_state=42,
                          shuffle=True)
    model = MLPClassifier(max_iter=300,
                          hidden_layer_sizes=(256, 128),
                          random_state=3)

    for train_indices, test_indices in skf.split(x_train, y_train):
        model.fit(x[train_indices], np.ravel(y[train_indices], order='C'))

    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions, normalize='true')

    print("=" * 40, file[9:], "=" * 40, "\n")
    print("MLP Classification Accuracy: ", acc, "\n")
    print("Confusion Matrix:\n")
    print(confusion)
    print("\n")

