from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
# Example of voting classifiers

log = LogisticRegression()
RF = RandomForestClassifier()
SVM = SVC(probability=True)

iris = load_iris()
X = iris.data[:, 2:] # Petal length & petal width
y = iris.target

X_train, y_train = X[:len(X) * 3 // 4 + 1], y[:len(y) * 3 // 4 + 1]
X_test, y_test = X[len(X) * 3 // 4 + 1:], y[len(y) * 3 // 4 + 1:]

# Hard voting
voting = VotingClassifier(
    estimators=[('lr', log), ('rf', RF), ('svc', SVM)],
    voting='hard')
print('--------Hard Voting--------')
for model in (log, RF, SVM, voting):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.__class__.__name__, accuracy_score(y_test, y_pred))


# Soft voting
voting = VotingClassifier(
    estimators=[('lr', log), ('rf', RF), ('svc', SVM)],
    voting='soft')
print('--------Soft Voting--------')
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(model.__class__.__name__, accuracy_score(y_test, y_pred))


# Bagging and Pasting
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1
)

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print('--------Bagging--------')
print(bagging.__class__.__name__, accuracy_score(y_test, y_pred))

# RF
RF = RandomForestClassifier(n_estimators=500,
                            max_leaf_nodes=16,
                            n_jobs=-1)

RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
print('--------Random Forest--------')
print(RF.__class__.__name__, accuracy_score(y_test, y_pred))

# Feature importance
iris = load_iris()
RF = RandomForestClassifier(n_estimators=500, n_jobs=-1)
RF.fit(iris['data'], iris['target'])
for name, score in zip(iris['feature_names'], RF.feature_importances_):
    print(name, score)

# Boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X_train = data.data[:len(data.data) * 3 // 4 + 1]
y_train = data.target[:len(data.target) * 3 // 4 + 1]
X_values = data.data[len(data.data) * 3 // 4 + 1:]
y_values = data.target[len(data.target) * 3 // 4 + 1:]
GB = GradientBoostingRegressor(max_depth=2, warm_start=True)
min_val_error = float('inf')
E = 0
for n_estimators in range(1, 120):
    GB.n_estimators = n_estimators
    GB.fit(X, y)
    y_pred = GB.predict(X_values[0])
    val_error = mean_squared_error(y_values[0], y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        E = 0
    else:
        E += 1
        if E == 5:
            break # Early Stopping