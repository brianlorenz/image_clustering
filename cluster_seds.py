from data_paths import pixel_sed_save_loc, read_saved_pixels, read_sed, check_and_make_dir, get_cluster_save_path, read_uncover_filters, get_wavelength
import numpy as np
from cluster_algorithms.k_means import kmeans
from cluster_algorithms.similarity_spectral_clustering import spectral_cluster_cross_cor
from cluster_algorithms.gaussian_mixture import gaussian_mixture_model
from cluster_algorithms.dbscan import dbscan_clustering
from cluster_algorithms.agglomerative import agglomerative_cluster
from cluster_algorithms.spectral import spectral_cluster
from helper_functions.helpers import delete_bad_values

### Note - curerntly each cluster method is a separate .py file.
### The calls for each of them all seem similar enough that you could probably consolidate them into a single file
### My plan with this project was to explore a bunch of methods, and then cleanly implement the one that seemed most successful 
def define_cluster_methods():
    """All current options for clustering can be found here"""
    cluster_dict = {
        'test': cluster_method_test, # Just a test function, not actual clustering
        'kmeans': kmeans,
        'spectral_cross_cor': spectral_cluster_cross_cor,
        'gaussian_mixture': gaussian_mixture_model,
        'dbscan': dbscan_clustering, 
        'agglomerative': agglomerative_cluster,
        'spectral': spectral_cluster,
    }
    return cluster_dict


def cluster_pixels(id_dr3_list, cluster_method='test', norm_method='', distances=''):
    """Performs the acutal clustering
    
    Parameters:
    id_dr3_list (list): List of ids to cluster on. Currently set up for id_dr3s fromt he UNCOVER/MegaScience Catalogs
    cluster_method (str): Name of the clustering method. See "define_cluster_methods" in cluster_seds.py for options
    norm_method (str): Can either be blank (for no normalization), '_sed', or '_L2'. See normalize.py
    distances (str): Can either be blank for default metric, '_weightednorm', or '_crosscor'. See distances.py. Not currently implemeted for all algorithms
    """
    save_path = get_cluster_save_path(cluster_method, norm_method=norm_method, distances=distances)
    check_and_make_dir(save_path)
    for id_dr3 in id_dr3_list:
        save_location = save_path + f'{id_dr3}_clustered.npz'
        
        print(f'Reading cutouts for id_dr3 = {id_dr3}')
        pixel_data = read_saved_pixels(id_dr3) # Reasd outputs from prepare_images()
        pixel_seds = pixel_data['pixel_seds']
        image_cutouts = pixel_data['image_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
        bad_image_idxs = pixel_data['bad_image_idxs']
        masked_indicies = pixel_data['masked_indicies']
        filter_names = pixel_data['filter_names']
        sed_data = read_sed(id_dr3)
        sed = sed_data['sed']
        err_sed = sed_data['err_sed']
    

        # Remove the images flagged as bad if there are any
        pixel_seds = delete_bad_values(pixel_seds, bad_image_idxs)
        image_cutouts = delete_bad_values(image_cutouts, bad_image_idxs)
        sed = delete_bad_values(sed, bad_image_idxs)
        err_sed = delete_bad_values(err_sed, bad_image_idxs)
        filter_names = delete_bad_values(filter_names, bad_image_idxs)
        
        filt_dict = read_uncover_filters()
        waves = [get_wavelength(filt_dict, 'f_'+name) for name in filter_names] 

        # Now need to cluster on the pixel_seds
        cluster_dict = define_cluster_methods()
        ### HERE is the call to do the actual clustering. As stated at the top of the sript, there is likely a way to do this more cleanly
        # There are lots of parameters to tune in each of the possible methods
        cluster_values = cluster_dict[cluster_method](pixel_seds, sed, waves, norm_method=norm_method, distances=distances)
        # cluster_values is an array where each unique integer is a label for the cluster that the pixel belongs to
        # For example, it may look like: cluster_values = (1, 3, 3, 1, 2, 1, 2). This would be three unique clusters, and the pixels at index 0, 3, 5 would be part of the same cluster

        # Reconstruct the image but with clusteres given their categorical varaibles - useful for visualization
        clustered_image = setup_cluster_image(cluster_values, masked_indicies, image_cutouts[0].shape)
        # The value at the location of each pixel is its cluster label

        # Save the outputs
        np.savez(save_location, cluster_values=cluster_values, clustered_image=clustered_image)        


def cluster_method_test(pixel_seds, *args):
    """This is a test method - it simply clusters the first half of the array as part of cluster 1, and the second half as cluster 2"""
    cluster_values = np.ones(pixel_seds.shape[1])
    cluster_values[:round(len(cluster_values)/2)] = 2
    return cluster_values

def setup_cluster_image(cluster_values, masked_indicies, image_shape):
    """Reconstruces the image where the value at each pixel is its cluster ID number
    
    May look something like this:
    0 0 1 1 0
    0 1 1 2 0
    0 1 2 2 0
    0 3 3 2 0
    3 3 0 0 0 

    Where each number corresponds to a unique clutsre of pixels, and the 0s are ignored/sky
    """
    cluster_image = np.zeros(image_shape)
    row_idx, col_idx = masked_indicies
    cluster_image[row_idx, col_idx] = cluster_values
    return cluster_image


if __name__ == '__main__':
    cluster_pixels([46339, 44283, 30804], cluster_method='spectral_cross_cor', norm_method='', distances='')