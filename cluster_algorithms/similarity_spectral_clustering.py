import numpy as np
from sklearn.cluster import SpectralClustering
from helper_functions.similarity_matrix import find_sim_matrix

def spectral_cluster_cross_cor(pixel_seds, *args):
    X = pixel_seds.T

    sim_matrix = find_sim_matrix(X)

    clustering_aff = SpectralClustering(n_clusters=4, assign_labels="discretize", random_state=0, affinity='precomputed').fit(sim_matrix)

    cluster_values = clustering_aff.labels_ + 1
    
    return cluster_values

