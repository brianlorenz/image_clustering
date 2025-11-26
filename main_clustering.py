from read_images import prepare_images
from vizualize_outputs import plot_cutout_overview, plot_cluster_summary
from cluster_seds import cluster_pixels
from data_paths import image_save_dir, check_and_make_dir
import os
import shutil
import faulthandler; faulthandler.enable()
from make_composite_sed import get_composite_sed


def main(id_dr3_list, snr_thresh=3, cluster_method='test', norm_method='', distances=''):
    # prepare_images(id_dr3_list, snr_thresh=snr_thresh)
    # plot_cutout_overview(id_dr3_list)
    # cluster_pixels(id_dr3_list, cluster_method=cluster_method, norm_method=norm_method, distances=distances)
    # plot_cluster_summary(id_dr3_list, cluster_method=cluster_method, norm_method=norm_method, distances=distances)
    get_composite_sed(id_dr3_list, cluster_method=cluster_method, norm_method=norm_method, distances=distances)


def overview_single_galaxy(id_dr3):
    cluster_methods = os.listdir(image_save_dir+'clustered/')
    cluster_methods = [cluster_method for cluster_method in cluster_methods if '.' not in cluster_method]
    single_gal_folder = image_save_dir+'single_galaxies/'
    for cluster_method in cluster_methods:
        source_location = f'{image_save_dir}clustered/{cluster_method}/{id_dr3}_clustered.pdf'
        check_and_make_dir(f'{single_gal_folder}{id_dr3}')
        if os.path.exists(source_location):
            shutil.copy(source_location, single_gal_folder+f'{id_dr3}/{cluster_method}.pdf')


if __name__ == '__main__':
    main([46339, 44283, 30804], snr_thresh=3, cluster_method='kmeans', norm_method='')
    # main([46339, 44283, 30804], snr_thresh=3, cluster_method='kmeans', norm_method='_L2')
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