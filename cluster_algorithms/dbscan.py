from sklearn.cluster import DBSCAN
import numpy as np
from helper_functions.normalize import normalize_X
from helper_functions.distances import get_distance_metric
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt


def dbscan_clustering(pixel_seds, sed, waves, *args, norm_method='', distances=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)

    X, metric, _ = get_distance_metric(X, distances, waves)

    min_samples = 5
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    # plt.plot(distances)
    # plt.show()

    clustering = DBSCAN(eps=2, min_samples=min_samples, metric=metric).fit(X)

    cluster_values = clustering.labels_ + 1

    
    return cluster_values