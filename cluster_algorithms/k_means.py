from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from cluster_algorithms.cross_cor_eqns import get_cross_cor
from cluster_algorithms.similarity_matrix import distance_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from cluster_algorithms.normalize import normalize_X
import time
import numpy as np

def kmeans(pixel_seds, sed, *args, norm_method=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)

    if norm_method=='_pca': # Probably don't use since we lose information
        l2_norms = np.sqrt(np.sum(X**2, axis=1))
        for i in range(len(X)):
            X[i] = X[i]/l2_norms[i]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X)
        pca = PCA(n_components=4) 
        X = pca.fit_transform(scaled_data) # Dimension of X now reduced to n_components. Shape (121, n_components)
        print(pca.explained_variance_ratio_)
    
    range_n_clusters = [2, 4, 6, 8, 10, 12, 14]
    range_n_clusters = [4]

    for n_clusters in range_n_clusters:
        kmeans_out = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(X) # requires (samples, features). In this case, each pixel is sample and each image is a feature
        silhouette_avg = silhouette_score(X, kmeans_out.labels_)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
    cluster_values = kmeans_out.labels_ + 1
    
    return cluster_values