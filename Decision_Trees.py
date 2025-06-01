from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # Petal length & petal width
y = iris.target

tree_classifier = DecisionTreeClassifier(max_depth=2)
tree_classifier.fit(X, y)

print(tree_classifier.predict_proba([[5, 1.5]])) # Predicting the probability of belonging to each class
print(tree_classifier.predict([[5, 1.5]])) # The predicted class of the data for prediction. 1 means the class whose index is 1.

tree_classifier = DecisionTreeClassifier(max_depth=3,
                                         min_samples_split=4, # Limiting the minimum number of samples a node must have before splitting,
                                         min_samples_leaf=4) # and the minimum number of samples a node must have before becoming a leaf.

tree_classifier.fit(X, y)
print(tree_classifier.predict_proba([[5, 1.5]]))
print(tree_classifier.predict([[5, 1.5]]))