from sklearn.cluster import DBSCAN
import numpy as np
from cluster_algorithms.normalize import normalize_X
from cluster_algorithms.similarity_matrix import distance_matrix, find_sim_matrix
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt


def dbscan_clustering(pixel_seds, sed, waves, *args, norm_method='', distances=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)


    if distances=='_weightednorm':
        X = distance_matrix(X, waves)
        metric = 'precomputed'
    elif distances == '_crosscor':
        X = find_sim_matrix(X)
        X = 1-X
        metric = 'precomputed'
    else:
        metric = 'euclidean'


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