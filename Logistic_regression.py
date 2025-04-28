from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

# 1 feature
X = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(int)

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_pred = log_reg.predict_proba(X_new)
plt.plot(X_new, y_pred[:, 1], 'g-', label='Iris virginica')
plt.plot(X_new, y_pred[:, 0], 'b--', label='NOT Iris virginica')
plt.legend()
plt.show()


print(log_reg.predict([[1.7], [1.5]]))


# 2 features
X = iris['data'][:, 2:]
y = (iris['target'] == 2).astype(int)

log_reg = LogisticRegression()
log_reg.fit(X, y)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', edgecolor='k')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000),np.linspace(y_min, y_max, 1000))

Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], cmap='gray', linestyles='solid')
plt.show()