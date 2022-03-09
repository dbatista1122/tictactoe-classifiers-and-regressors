import pickle
from Classifiers.SVM import classifier_svm
from Classifiers.KNN import classifier_knn
from Classifiers.MLP import classifier_mlp

model = open('trained_model.mdl','wb')

pickle.dump(classifier_knn('Datasets/tictac_single.txt'), model)

