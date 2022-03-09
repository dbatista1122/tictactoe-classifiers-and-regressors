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


def classifier_knn(file):

    x, y = load(file)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5, test_size=0.20, shuffle=True)

    model = neighbors.KNeighborsClassifier(1, metric='euclidean')
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_indices, test_indices in skf.split(x_train, y_train):
        model.fit(x_train[train_indices], y_train[train_indices])

    predictions = model.predict(x_test)
    acc = model.score(x_test, y_test)
    confusion = con_mat(y_test, predictions)

    print("=" * 40, file[9:], "=" * 40, "\n")
    print("K-Nearest Neighbors Classification Accuracy", acc, "\n")
    print("Confusion Matrix:\n")
    print(confusion)
    print("\n")

    return model
