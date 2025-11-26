
from sklearn.cluster import SpectralClustering
import numpy as np
from helper_functions.normalize import normalize_X
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt


def spectral_cluster(pixel_seds, sed, *args, norm_method=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)

    min_samples = 5
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

    sc = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', random_state=0)
    clustering_labels = sc.fit_predict(X)

    cluster_values = clustering_labels + 1

    
    return cluster_values