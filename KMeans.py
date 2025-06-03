from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

k = 3
iris = load_iris()
X = iris.data
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

print(kmeans.cluster_centers_)

print(y_pred)

# Silhouette score: finding the optimal number of clusters
from sklearn.metrics import silhouette_score

print(silhouette_score(X, kmeans.labels_))

Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
silhouettes = []
for k in Ks:
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X)
    silhouettes.append(silhouette_score(X, kmeans.labels_))

plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.plot(Ks, silhouettes)
plt.show()
