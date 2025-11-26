from sklearn.cluster import AgglomerativeClustering
import numpy as np
from cluster_algorithms.normalize import normalize_X
from cluster_algorithms.similarity_matrix import distance_matrix, find_sim_matrix
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def agglomerative_cluster(pixel_seds, sed, waves, *args, norm_method='', distances=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)

    if distances=='_weightednorm':
        X = distance_matrix(X, waves)
        metric = 'precomputed'
        linkage = 'complete'
    elif distances == '_crosscor':
        X = find_sim_matrix(X)
        X = 1-X
        metric = 'precomputed'
        linkage = 'complete'
    else:
        metric = 'euclidean'
        linkage = 'ward'

    agg = AgglomerativeClustering(n_clusters=4, metric=metric, linkage=linkage)


    clustering_labels = agg.fit_predict(X)


    cluster_values = clustering_labels + 1

    
    return cluster_values