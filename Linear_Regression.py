from sklearn.datasets import load_diabetes
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# Take diabetes data as example
diabetes = load_diabetes()

# Set X and Y
X = diabetes.data
y = diabetes.target
column = X.shape[1]
print(X)
# We then know that the L.R. model is: y_hat = theta.T * x
# MSE(X, h_theta) = 1/m * (Sum for i in 1 to m) 1/m * (theta.T * x_i - y_i) ^ 2
# theta_hat = (X.T * X).inverse * X.T * y
# (∂-theta) MSE(theta) = 2 * X.T * X * theta - 2 * X.T * y. We need to know when this partial derivative is zero,
# Thus we need theta.

X = np.c_[np.ones((X.shape[0], 1)), X] # We need x0 = 1 in each instance
theta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print("theta hat = {}".format(theta_hat))
print(X[0])


# Then we can predict data using theta_hat

X_new = np.array([[random.uniform(-0.1, 0.1) for _ in range(column)], [random.uniform(-0.1, 0.1) for _ in range(column)], [random.uniform(-0.1, 0.1) for _ in range(column)]])
X_new = np.c_[np.ones((X_new.shape[0], 1)), X_new]
y_new = X_new.dot(theta_hat)


lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Here should be "assert True".
assert np.allclose(lin_reg.predict(X_new), y_new)

# Batch gradient descent

eta = 0.1
n_iterations = 1000
m = 100
theta_hat = [random.uniform(min(theta_hat), max(theta_hat))] * X.shape[1]

for iteration in range(n_iterations):
    grad = 2/m * X.T.dot(X.dot(theta_hat) - y)
    theta_hat = theta_hat - eta * grad

print(theta_hat)

# Stochastic GD
n_epochs = 100
t0, t1 = 5, 100

def learning_schedule(t):
    return t0 / (t + t1)

theta_hat = [random.uniform(min(theta_hat), max(theta_hat))] * X.shape[1]

for epoch in range(n_epochs):
    for i in range(m):
        random_i = np.random.randint(m)
        xi = X[random_i:random_i+1]
        yi = y[random_i:random_i+1]
        grad = 2 * xi.T.dot(xi.dot(theta_hat) - yi)
        eta = learning_schedule(epoch * m + i)
        theta_hat = theta_hat - eta * grad

print(theta_hat)
