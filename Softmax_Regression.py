'''

Softmax Regression:

When given an instance x, it first computes a score sk(x) for each class k, then estimates
the probability of each class by applying the softmax function to the scores.

pk_hat = theta(s(x))k) = e^(sk(x)) / (Sum from j=1 to K)(e^(sj(x)))
y_hat = argmax(theta(s(x))k)

'''


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
X = iris["data"][:, (2, 3)] # Petal length & petal width
y = iris["target"]

softmax.fit(X, y)

print(softmax.predict([[5, 2]]))

print(softmax.predict_proba([[5, 2]]))

X_min, X_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
XX, yy = np.meshgrid(
    np.linspace(X_min, X_max, 1000),
        np.linspace(y_min, y_max, 1000))

X_new = np.c_[XX.ravel(), yy.ravel()]
y_hat = softmax.predict(X_new)
zz = y_hat.reshape(XX.shape)


plt.contourf(XX, yy, zz, cmap=plt.cm.rainbow, alpha=0.3)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

plt.legend(*scatter.legend_elements(), loc='lower right', title='Classes')
plt.title("Softmax Regression and Decision Boundary")
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()
