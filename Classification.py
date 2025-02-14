import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib as mpl

mnist = fetch_openml('mnist_784', version=1)

X ,y = mnist["data"], mnist["target"]
X = np.array(X)
y = np.array(y)

some_digit = X[0]

# Binary Classifier
y_train = y[1:60001]
X_train = X[1:60001]

y_train_5 = [Y == '5' for Y in y_train]

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

some_digit = some_digit.reshape(1, -1)
print(sgd_clf.predict(some_digit))

# Cross-validation
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))

# Performance measures
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never5_clf = Never5Classifier()
print(cross_val_score(never5_clf, X_train, y_train_5, cv=3, scoring='accuracy'))

# Confusion Matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(y_train_pred)

from sklearn.metrics import confusion_matrix

def conf_mat(y_true, y_pred):
    if len(y_true) != len(y_pred):
        pass
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and y_true == True:
            TP += 1
        elif y_true[i] == y_pred[i] and y_true == False:
            TN += 1
        elif y_true[i] != y_pred[i] and y_true == True:
            FN += 1
        elif y_true[i] != y_pred[i] and y_true == False:
            FP += 1

    return np.array([[TN, FP], [FN, TP]])

print(confusion_matrix(y_train_5, y_train_pred))
assert confusion_matrix(y_train_5, y_train_pred).all() == conf_mat(y_train_5, y_train_pred).all()
y_train_perfect_pred = y_train_5
print(confusion_matrix(y_train_5, y_train_perfect_pred))

# Precision
def precision(confusion_matrix):
    return confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])

# Recall
def recall(confusion_matrix):
    return confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])

# F1
def F1(confusion_matrix):
    return confusion_matrix[1,1] / (confusion_matrix[1,1] + (confusion_matrix[0,1] + confusion_matrix[1,0]) / 2)

print(type(confusion_matrix(y_train_5, y_train_pred)))
print(precision(confusion_matrix(y_train_5, y_train_pred)))
print(recall(confusion_matrix(y_train_5, y_train_pred)))

