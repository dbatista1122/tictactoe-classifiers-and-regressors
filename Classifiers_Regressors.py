import warnings
from Classifiers.SVM import classifier_svm
from Classifiers.KNN import classifier_knn
from Classifiers.MLP import classifier_mlp
from Regressors.LinearRegression import linear_regression
from Regressors.KNNRegressor import knn_regressor
from Regressors.MLPRegressor import mlp_regressor


def main():
    warnings.filterwarnings('ignore')
    print("\n")
    print(" " * 42, "Classifiers", " " * 42, "\n")

    print("*" * 98, "\n")
    print(" " * 42, "Linear  SVM", " " * 42, "\n")
    classifier_svm('Datasets/tictac_final.txt')
    classifier_svm('Datasets/tictac_single.txt')
    print("*" * 98, "\n")

    print(" " * 39, "K-Nearest Neighbors", " " * 38, "\n")
    classifier_knn('Datasets/tictac_final.txt')
    classifier_knn('Datasets/tictac_single.txt')

    print("*" * 98, "\n")
    print(" " * 42, "Multilayer Perceptron", " " * 42, "\n")

    classifier_mlp('Datasets/tictac_final.txt')
    classifier_mlp('Datasets/tictac_single.txt')

    print("\n")
    print(" " * 42, "Regressors", " " * 42, "\n")

    print("*" * 98, "\n")
    print(" " * 40, "Linear Regression", " " * 42, "\n")
    linear_regression('Datasets/tictac_multi.txt')

    print("*" * 98, "\n")
    print(" " * 32, "K-Nearest Neighbors Regression", " " * 42, "\n")
    knn_regressor('Datasets/tictac_multi.txt')

    print("*" * 98, "\n")
    print(" " * 32, "MultiLayer Perceptron Regression", " " * 42, "\n")
    mlp_regressor('Datasets/tictac_multi.txt')


if __name__ == "__main__":
    main()
