from data_paths import pixel_sed_save_loc, read_saved_pixels, image_save_dir, get_cluster_save_path, check_and_make_dir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import time


def plot_cluster_summary(id_dr3_list, cluster_method, norm_method='', distances='distances'):
    """Makes a plot that shoulds the clusters made by cluster_seds.py"""
    image_filter = 'f444w'
    for id_dr3 in id_dr3_list:
        pixel_data = read_saved_pixels(id_dr3)
        cluster_savepath = get_cluster_save_path(cluster_method, norm_method=norm_method, distances=distances, id_dr3=id_dr3)
        clustered_data = np.load(cluster_savepath)
        image_cutouts = pixel_data['image_cutouts'] 
        filter_names = pixel_data['filter_names']
        mask = pixel_data['mask']

        clustered_image = clustered_data['clustered_image']

        fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
        ax_image = axarr[0]
        ax_overlay = axarr[1]
        ax_clustered = axarr[2]

        for ax in axarr:
            ax.axis('off')

        # Setup for plotting
        image_filter_idx = np.argmax(filter_names==image_filter)
        unique_values, counts = np.unique(clustered_image, return_counts=True)
        cmap_cluster, cmap_cluster_black = generate_cluster_cmap(len(unique_values)-1)
        masked_clustered = np.ma.masked_where(clustered_image == 0, clustered_image)
        
        # plotting
        ax_image.imshow(image_cutouts[image_filter_idx], cmap='gray', origin='lower')
        ax_clustered.imshow(clustered_image, cmap=cmap_cluster_black, origin='lower')
        ax_overlay.imshow(image_cutouts[image_filter_idx], cmap='gray', origin='lower')
        ax_overlay.imshow(masked_clustered, cmap=cmap_cluster, origin='lower', alpha=0.5)

        add_textbox(ax_image, image_filter.upper(), color='white')

        check_and_make_dir(image_save_dir+f'clustered/{cluster_method}{norm_method}{distances}/')
        fig.savefig(image_save_dir+f'clustered/{cluster_method}{norm_method}{distances}/{id_dr3}_clustered.pdf', bbox_inches='tight')

        
        
        


def plot_cutout_overview(id_dr3_list):
    """Overview plot of which pixels were selected for clustering and showing the galaxy in all filters"""
    for id_dr3 in id_dr3_list:
        pixel_data = read_saved_pixels(id_dr3) # Reads the outputs from prepare_images()
        image_cutouts = pixel_data['image_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
        noise_cutouts = pixel_data['noise_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
        boolean_segmap = pixel_data['boolean_segmap'] # shape of (cutout_y_size, cutout_x_size)
        mask = pixel_data['mask']
        obj_segmap = pixel_data['obj_segmap'] # shape of (cutout_y_size, cutout_x_size)
        filter_names = pixel_data['filter_names'] # shape of (n_images,)
        snr_thresh = pixel_data['snr_thresh']
        bad_image_idxs = pixel_data['bad_image_idxs']

        # Number of plots
        n_rows = 3
        n_cols = 10

        fig, axarr = plt.subplots(n_rows, n_cols, figsize=(4*n_cols,4*n_rows))
                
        # Show the segmap in last spot
        ax_segmap = axarr[n_rows-1, n_cols-1]
        unique_segmap_ids, indexed_segmap = np.unique(obj_segmap, return_inverse=True)
        cmap = create_cmap_segmap(id_dr3, unique_segmap_ids)
        ax_segmap.imshow(indexed_segmap, cmap=cmap, origin='lower')
        add_textbox(ax_segmap, 'Segmap')

        # Show an SNR map in second to last spot
        ax_snr = axarr[n_rows-1, n_cols-2]
        median_snr = np.median(image_cutouts/noise_cutouts, axis=0)
        snr_masked = median_snr > snr_thresh # SNR thresh defined above
        cmap_snr = ListedColormap(['black', 'cornflowerblue'])
        ax_snr.imshow(snr_masked, cmap=cmap_snr, origin='lower')
        add_textbox(ax_snr, f'SNR > {snr_thresh}')

        # Show all images
        for i in range(n_rows):
            for j in range(n_cols):
                array_index = i*n_cols+j

                text_color = 'white'
                if array_index in bad_image_idxs:
                    text_color = 'red'

                ax = axarr[i, j]
                ax.axis('off')
                if array_index >= len(filter_names):
                    continue
                  
                add_textbox(ax, filter_names[array_index].upper(), color=text_color)
                image_data = image_cutouts[array_index, :, :]
                ax.imshow(image_data, cmap='gray', origin='lower')

                # ax.imshow(mask, cmap='Greys', origin='lower', alpha=0.5) # shows the full mask, but covers the image
                ax.contour(mask, levels=[0.5], colors='cornflowerblue', linewidths=2) # contours the mask
        
        plt.tight_layout()
        fig.savefig(image_save_dir+f'overviews/{id_dr3}_overview.pdf')
    pass


def create_cmap_segmap(id_dr3, unique_segmap_ids):
    """Making a colormap for the segmap where our target object stands out, and the background is alwyas black"""
    id_to_index = {gal_id: i for i, gal_id in enumerate(unique_segmap_ids)} # Dict that goes from id to its index in the color list
    base_cmap = plt.cm.Reds
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    base_cmap = truncate_colormap(base_cmap, minval=0.0, maxval=0.8)
    other_gal_ids = [gal_id for gal_id in unique_segmap_ids if gal_id != 0 and gal_id != id_dr3]
    n_colors = len(other_gal_ids)
    color_list = [None] * len(unique_segmap_ids)
    color_list[0] = (0, 0, 0, 1) # black
    color_list[id_to_index[id_dr3]] = 'cornflowerblue'
    color_samples = [base_cmap(i / (n_colors - 1)) for i in range(n_colors)]
    for i in range(len(other_gal_ids)):
        gal_id = other_gal_ids[i]
        color_list[id_to_index[gal_id]] = color_samples[i]
    cmap = ListedColormap(color_list)
    return cmap

def generate_cluster_cmap(n_colors):
    cmap = plt.get_cmap('rainbow', n_colors)
    new_colors = np.vstack(([0, 0, 0, 1], cmap(np.arange(n_colors)))) 
    cmap_with_black = ListedColormap(new_colors)
    return cmap, cmap_with_black


def add_textbox(ax, text, color='white'):
    text_height = 0.905
    text_start = 0.035
    ax.text(text_start, text_height, text, fontsize=24, transform=ax.transAxes, color='black', bbox=dict(facecolor=color, alpha=0.8, boxstyle='round'))

# Example usage
if __name__ == '__main__':
    plot_cutout_overview([46339, 44283, 30804]) # Overview plot, just needs prepare_images() ran
    
    plot_cluster_summary([46339, 44283, 30804], cluster_method='kmeans') # Shows result of clustering, needs the full pipeline
    