from read_images import prepare_images
from vizualize_outputs import plot_cutout_overview, plot_cluster_summary
from cluster_seds import cluster_pixels
from data_paths import image_save_dir, check_and_make_dir
import os
import shutil
import faulthandler; faulthandler.enable()
from make_composite_sed import get_composite_sed


def main(id_dr3_list, snr_thresh=3, cluster_method='test', norm_method='', distances=''):
    """Runs the full pipeline, from reading in the images to clusters with composite SEDs
    
    Parameters:
    id_dr3_list (list): List of ids to cluster on. Currently set up for id_dr3s fromt he UNCOVER/MegaScience Catalogs
    snr_thresh (float): Minimum signal-to-noise ratio needed to include a pixel. Currently checks against the median SNR from all images
    cluster_method (str): Name of the clustering method. See "define_cluster_methods" in cluster_seds.py for options
    norm_method (str): Can either be blank (for no normalization), '_sed', or '_L2'. See normalize.py
    distances (str): Can either be blank for default metric, '_weightednorm', or '_crosscor'. See distances.py. Not currently implemeted for all algorithms
    """
    prepare_images(id_dr3_list, snr_thresh=snr_thresh) # reads images and sets up the pixels for clustering
    plot_cutout_overview(id_dr3_list) # makes an overview plot - not necessary for the pipeline
    cluster_pixels(id_dr3_list, cluster_method=cluster_method, norm_method=norm_method, distances=distances) # Performs the clustering
    plot_cluster_summary(id_dr3_list, cluster_method=cluster_method, norm_method=norm_method, distances=distances) # Plots the output of clustering
    get_composite_sed(id_dr3_list, cluster_method=cluster_method, norm_method=norm_method, distances=distances) # Makes composite SEDs from the clusters


def overview_single_galaxy(id_dr3):
    """Grabs all attempted methods for a single galaxy and copies them into one folder - easier to compare to each other and see what's working"""
    cluster_methods = os.listdir(image_save_dir+'clustered/')
    cluster_methods = [cluster_method for cluster_method in cluster_methods if '.' not in cluster_method]
    single_gal_folder = image_save_dir+'single_galaxies/'
    for cluster_method in cluster_methods:
        source_location = f'{image_save_dir}clustered/{cluster_method}/{id_dr3}_clustered.pdf'
        check_and_make_dir(f'{single_gal_folder}{id_dr3}')
        if os.path.exists(source_location):
            shutil.copy(source_location, single_gal_folder+f'{id_dr3}/{cluster_method}.pdf')


# Examples of lots of clustering methods that I was trying
if __name__ == '__main__':
    main([46339, 44283, 30804], snr_thresh=3, cluster_method='kmeans', norm_method='') # Simplist kmeans clustering, will just end up clustering by brightness since there is no normalization
    main([46339, 44283, 30804], snr_thresh=3, cluster_method='kmeans', norm_method='_L2') # kmeans with normalization - this should be a reasonable attempt at clustering, but still lots to improve
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='kmeans', norm_method='_sed')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='kmeans', norm_method='_pca')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='gaussian_mixture', norm_method='_L2')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='gaussian_mixture', norm_method='_sed')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='gaussian_mixture', norm_method='')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='dbscan', norm_method='_L2')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='dbscan', norm_method='_L2', distances='_weightednorm')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='dbscan', norm_method='_L2', distances='_crosscor')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='agglomerative', norm_method='_L2', distances='_crosscor')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='agglomerative', norm_method='_L2')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='spectral', norm_method='_L2')


    # overview_single_galaxy(30804)