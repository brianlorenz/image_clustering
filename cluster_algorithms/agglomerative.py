from sklearn.cluster import AgglomerativeClustering
import numpy as np
from helper_functions.normalize import normalize_X
from helper_functions.distances import get_distance_metric

def agglomerative_cluster(pixel_seds, sed, waves, *args, norm_method='', distances=''):
    X = pixel_seds.T

    X = normalize_X(X, norm_method=norm_method, sed=sed)

    X, metric, linkage = get_distance_metric(X, distances, waves)

    agg = AgglomerativeClustering(n_clusters=4, metric=metric, linkage=linkage)

    clustering_labels = agg.fit_predict(X)

    cluster_values = clustering_labels + 1
    
    return cluster_values