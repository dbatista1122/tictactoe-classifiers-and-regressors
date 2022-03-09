""" Importing required libraries """
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    import warnings


def load(file_name):
    game = np.loadtxt(file_name)
    x = game[:, :9]
    y = game[:, 9:]
    return x, y


def con_mat(y_test, predictions):
    confusion = confusion_matrix(y_test, predictions, normalize='true')

    return confusion


def final(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.20,
                                                        shuffle=True)

    svm = SVC(kernel='linear', gamma='scale')
    svm.fit(x_train, y_train)

    accuracy = svm.score(x_test, y_test)

    predictions = svm.predict(x_test)
    confusion_mtrx = con_mat(y_test, predictions)

    return accuracy, confusion_mtrx, svm


def single_label(x, y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.20, shuffle=True)

    svm = SVC(kernel='rbf', gamma='scale', probability=True, decision_function_shape='ovo',
              random_state=3)
    svm.fit(x_train, y_train)

    accuracy = svm.score(x_test, y_test)

    predictions = svm.predict(x_test)
    confusion = con_mat(y_test, predictions)

    return accuracy, confusion, svm


def classifier_svm(file):
    x, y = load(file)

    if 'final' in file:
        accuracy, confusion, svm = final(x, y)
    elif 'single' in file:
        accuracy, confusion, svm = single_label(x, y)
    else:
        return 0

    print("=" * 40, file[9:], "=" * 40, "\n")
    print("SVM Classification Accuracy: ", accuracy, "\n")
    print("Confusion Matrix:\n")
    print(confusion)
    print("\n")

    return svm
