'''

Ridge, Lasso, Elastic Net

'''

from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.rand(m, 1) / 1.5


ridge = Ridge(alpha=1, solver='cholesky', random_state=42)
ridge.fit(X, y)
print(ridge.predict([[1.5]]))

lasso = Lasso(alpha=1, random_state=42)
lasso.fit(X, y)
print(lasso.predict([[1.5]]))

elastic_net = ElasticNet(alpha=1, random_state=42)
elastic_net.fit(X, y)
print(elastic_net.predict([[1.5]]))

