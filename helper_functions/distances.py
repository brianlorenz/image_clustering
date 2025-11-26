from helper_functions.similarity_matrix import distance_matrix, find_sim_matrix

def get_distance_metric(X, distances, waves):
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
    
    return X, metric