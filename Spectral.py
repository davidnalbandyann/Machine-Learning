import numpy as np
from sklearn.cluster import KMeans
class spectral_clustering:
    def __init__(self, n_clusters=2, alpha=1, k_eigen=20):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.k_eigen = k_eigen

    def fit(self, X):
        n_samples = X.shape[0]
        laplacian = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                w = np.exp(-(np.linalg.norm(X[i] - X[j])**2) / (self.alpha**2))
                laplacian[i][j] = w
        for i in range(n_samples):
            d = np.sum(laplacian[i])
            laplacian[i][i] = -d
        eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        sort_id = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = np.take(eigenvalues, sort_id)[:self.k_eigen], np.take(eigenvectors, sort_id)[:, :self.k_eigen]
        self.transformed_data_ = eigenvectors
        return self

    def predict(self, X):
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(self.transformed_data_)
        return km.labels_
