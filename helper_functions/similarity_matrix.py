import time
import numpy as np
from itertools import combinations_with_replacement
from helper_functions.cross_cor_eqns import get_cross_cor
import matplotlib.pyplot as plt


def weighted_euclidean(y1, y2, x):
    # y1, y2: shape (n,) — normalized curves
    # x: shape (n,) — non-uniform x-grid, shared across all samples
    
    diffs = y1 - y2
    # Compute trapezoidal weights (approximate integral over non-uniform grid)
    dx = np.diff(x)
    weights = np.zeros_like(y1)
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2  # middle points
    weights[0] = dx[0] / 2
    weights[-1] = dx[-1] / 2
    
    # Weighted L2 norm (like integrating squared difference over x)
    return np.sqrt(np.sum(weights * diffs**2))

def euclidean(y1, y2, x):
    distance = np.linalg.norm(y1 - y2)
    return distance

def distance_matrix(X, waves):
    """Distance matrix using np.linalg.norm"""
    n_pixels = len(X)

    def setup_distance_matrix(distance_function, X, waves):
        t0 = time.time()
        distance_matrix = np.zeros((n_pixels, n_pixels))
        
        for i, j in combinations_with_replacement(range(n_pixels), 2):
            distance = distance_function(X[i], X[j], waves)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
        t1 = time.time()
        print(f'Computed similarity matrix in {t1-t0} seconds')
        return distance_matrix
    
    weighted_matrix = setup_distance_matrix(weighted_euclidean, X, waves)
    euclidean_matrix =  setup_distance_matrix(euclidean, X, waves)    
    

    return weighted_matrix



def find_sim_matrix(X):
    """Similarity matrix using cross cor function"""
    n_pixels = len(X)

    sim_matrix = np.zeros((n_pixels, n_pixels))

    t0 = time.time()
    for i, j in combinations_with_replacement(range(n_pixels), 2):
        _, b12 = get_cross_cor(X[i], X[j])
        sim_matrix[i, j] = 1-b12 # Need to do 1-b12 since we want identical items to have a score of 1
        sim_matrix[j, i] = 1-b12
    t1 = time.time()
    print(f'Computed similarity matrix in {t1-t0} seconds')

    # eigenvals, eignvectors = np.linalg.eig(sim_matrix)
    # x_axis = np.arange(1, len(eigenvals)+1, 1)
    # dx = 1
    # derivative = np.gradient(eigenvals, dx)
    # plt.plot(x_axis, eigenvals, ls='-', marker='o', color='black')
    # plt.plot(x_axis, derivative, ls='-', marker='o', color='orange')
    # plt.xlim(0, 15)
    # plt.show()

    return sim_matrix
