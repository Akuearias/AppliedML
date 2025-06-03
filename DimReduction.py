from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Function for visualization of data after Dimension Reduction.
def plot_2D_data(X_2D, y=None, title='2D Projection', cmap='viridis', figsize=(8, 6), legend=True):
    plt.figure(figsize=figsize)

    if y is None:
        plt.scatter(X_2D[:, 0], X_2D[:, 1], alpha=0.7)
    else:
        y = np.array(y)
        classes = np.unique(y)
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(classes)))
        for cls, color in zip(classes, colors):
            idx = y == cls
            plt.scatter(X_2D[idx, 0], X_2D[idx, 1], label=f"Class {cls}", color=color, alpha=0.7)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    if y is not None and legend:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# PCA
print('--------PCA--------')
pca = PCA(n_components=2)
iris = load_iris()
X = iris.data
y = iris.target
X_2D = pca.fit_transform(X)
print(X_2D)
print(pca.explained_variance_ratio_)
print(pca.inverse_transform(X_2D))

# Kernel PCA
print('--------KernelPCA--------')
Linear_PCA = KernelPCA(n_components=2, kernel='linear') # Linear kernel
X_2D = Linear_PCA.fit_transform(X)
plot_2D_data(X_2D, y, title='Linear PCA', cmap='viridis')

RBF_PCA = KernelPCA(n_components=2, kernel='rbf') # RBF kernel
X_2D = RBF_PCA.fit_transform(X)
plot_2D_data(X_2D, y, title='RBF PCA', cmap='viridis')

Sigmoid_PCA = KernelPCA(n_components=2, kernel='sigmoid') # Sigmoid kernel
X_2D = Sigmoid_PCA.fit_transform(X)
plot_2D_data(X_2D, y, title='Sigmoid PCA', cmap='viridis')


# LLE
from sklearn.manifold import LocallyLinearEmbedding
LLE = LocallyLinearEmbedding(n_components=2, n_neighbors=5)
X_2D = LLE.fit_transform(X)
plot_2D_data(X_2D, y, title='LLE Embedding', cmap='viridis')

