from data_paths import pixel_sed_save_loc, read_saved_pixels, read_SPS_cat,  image_save_dir, get_cluster_save_path, check_and_make_dir, read_uncover_filters, get_wavelength, composite_sed_save_dir, composite_image_save_dir, read_redshifts
import sys
import os
import numpy as np
import pandas as pd
from astropy.io import ascii
from read_data import mosdef_df
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import interpolate
import initialize_mosdef_dirs as imd
from plot_vals import *
from helper_functions.helpers import delete_bad_values
from helper_functions.cross_cor_eqns import get_cross_cor

def get_composite_sed_save_path(cluster_method, norm_method='', distances='', id_dr3=-1):
    """Location to save the composite SEDs"""
    save_path = composite_sed_save_dir + f'{cluster_method}{norm_method}{distances}/'
    if id_dr3 >= 0:
        save_path = composite_sed_save_dir + f'{cluster_method}{norm_method}{distances}/{id_dr3}_composite_seds.csv'
    return save_path

def get_composite_image_save_path(cluster_method, norm_method='', distances='', id_dr3=-1, value=-1):
    """Location to save the plots of the composites"""
    check_and_make_dir(composite_image_save_dir + f'{cluster_method}{norm_method}{distances}/')
    save_path = composite_image_save_dir + f'{cluster_method}{norm_method}{distances}/{id_dr3}_composite_sed_{value}.pdf'
    return save_path

def get_composite_sed(id_dr3_list, cluster_method, norm_method='', distances='distances', run_filters=False):
    """Make composite SEDs. The parameters specify where to read the clustering info from"""
    plot_indiv = False
    
    for id_dr3 in id_dr3_list:
        # Read in the pixels and the clustering labels
        pixel_data = read_saved_pixels(id_dr3)
        cluster_savepath = get_cluster_save_path(cluster_method, norm_method=norm_method, distances=distances, id_dr3=id_dr3)
        clustered_data = np.load(cluster_savepath)

        pixel_seds = pixel_data['pixel_seds']
        bad_image_idxs = pixel_data['bad_image_idxs']
        filter_names = pixel_data['filter_names']
        masked_indicies = pixel_data['masked_indicies']
        cluster_values = clustered_data['cluster_values']
        unique_values = set(cluster_values)

        # Removing flagged images
        pixel_seds = delete_bad_values(pixel_seds, bad_image_idxs)
        filter_names = delete_bad_values(filter_names, bad_image_idxs)

        filt_dict = read_uncover_filters()
        waves = [get_wavelength(filt_dict, 'f_'+name) for name in filter_names] 

        composite_df = pd.DataFrame(zip(filter_names, waves), columns=['filter_name', 'wavelength'])

        # Figure setup
        fig, axs = plt.subplots(len(unique_values), 3, figsize=(15, 5 * len(unique_values)))

        # Loop through each clustering ID
        for row, value in enumerate(unique_values):
            pixel_idxs = np.where(cluster_values == value)[0]
            cluster_pixels = pixel_seds.T[pixel_idxs]
            pixel_locations = masked_indicies[:, pixel_idxs]
            
            # Arbitrarily pick a target pixel - here we are taking the brightest one
            target_pixel = cluster_pixels[np.argmax(np.sum(cluster_pixels, axis=1))]

            # Normalize all pixles to the target
            normalized_pixels = np.apply_along_axis(normalize_pixels, 1, cluster_pixels, target_pixel=target_pixel)

            # Make a composite as the mean of all the normalized pixels
            composite_sed = np.mean(normalized_pixels, axis=0)
            composite_df[f'composite_sed_group{value}'] = composite_sed

            # Saving and plotting
            save_info = [cluster_method, norm_method, distances]
            if plot_indiv:
                vis_composite_sed(id_dr3, value, waves, normalized_pixels, pixel_locations, composite_sed, save_info)     
            vis_composite_sed(id_dr3, value, waves, normalized_pixels, pixel_locations, composite_sed, save_info, axarr=axs[row, :])     

        # Save the composite
        composite_sed_dir = get_composite_sed_save_path(cluster_method=cluster_method, norm_method=norm_method, distances=distances)
        check_and_make_dir(composite_sed_dir)
        composite_sed_path = get_composite_sed_save_path(cluster_method=cluster_method, norm_method=norm_method, distances=distances, id_dr3=id_dr3)
        composite_df.to_csv(composite_sed_path)

        image_save_path = get_composite_image_save_path(cluster_method, norm_method, distances, id_dr3=id_dr3, value='overview')
        for ax in axs.flat:
            ax.tick_params(labelsize=12)
        for ax in axs[:-1, :].flat:   
            ax.set_xlabel("")
        fig.savefig(image_save_path, bbox_inches='tight') 
    return

def normalize_pixels(pixel, target_pixel):
    """Uses cross correlation to compute how to normalize each pixel to a target"""
    norm_factor = get_cross_cor(target_pixel, pixel)[0]

    if norm_factor < 0:
        sys.exit(f'Normalization is less than zero')

    norm_pixel = pixel * norm_factor

    return norm_pixel

def vis_composite_sed(id_dr3, value, waves, normalized_pixels, pixel_locations, composite_sed, save_info, axarr=['']):
    """Composite SED plotting"""
    image_filter = 'f444w'
    cluster_method, norm_method, distances = save_info
    
    redshift_df = read_redshifts()
    redshift = redshift_df[redshift_df['id']==id_dr3]['z_50'].iloc[0]
    rest_waves = waves / (1+redshift)

    pixel_data = read_saved_pixels(id_dr3)
    filter_names = pixel_data['filter_names']
    image_cutouts = pixel_data['image_cutouts']

    image_filt_idx = np.argmax(filter_names==image_filter)
    image_to_plot = image_cutouts[image_filt_idx]
    image_pixels = np.zeros((image_to_plot.shape[0], image_to_plot.shape[1], 4))  # 4 channels (RGBA), initial values are all zero
    
    # Plot setup
    plot_indiv = False
    if len(axarr) < 2:
        fig, axarr = plt.subplots(1, 3, figsize=(18, 6)) 
        plot_indiv = True 
    ax_image = axarr[0]
    ax_sed = axarr[1] 
    ax_similarity = axarr[2]


    plot_similarity_cluster(value, normalized_pixels, composite_sed, ax_similarity)

    
    # Colormap
    cmap = plt.get_cmap('plasma')
    norm = colors.Normalize(vmin=0, vmax=normalized_pixels.shape[0]-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # SED scatter plot
    for i in range(normalized_pixels.shape[0]):
        color = sm.to_rgba(i)  
        pixel_row, pixel_col = pixel_locations[:, i]
        image_pixels[pixel_row, pixel_col] = color
        ax_sed.scatter(rest_waves, normalized_pixels[i, :], color=color, s=12, alpha=0.5)
    
    # Image overlay
    ax_image.imshow(image_to_plot, cmap='gray', origin='lower')
    ax_image.imshow(image_pixels, cmap=cmap, origin='lower', alpha=1)

    # Composite
    ax_sed.plot(rest_waves, composite_sed, ls='', marker='o', markersize=6, color='black', mec='black', zorder=2)

    # Cleanup
    ax_sed.set_ylim(-0.2*np.max(composite_sed), 1.2 * np.max(composite_sed))
    ax_sed.set_xlim(np.min(rest_waves)-200, np.max(rest_waves)+2000)
    ax_sed.set_xlabel('Rest Wavlength ($\AA$)', fontsize=14)
    ax_sed.set_ylabel('Flux', fontsize=14)
    ax_sed.set_ylim(np.min(normalized_pixels)*0.95, np.max(normalized_pixels)*1.05)
    ax_image.axis('off')

    image_save_path = get_composite_image_save_path(cluster_method, norm_method, distances, id_dr3=id_dr3, value=value)

    for ax in axarr:
        scale_aspect(ax)

    if plot_indiv:
        fig.savefig(image_save_path, bbox_inches='tight')
    plt.close('all')




def plot_similarity_cluster(value, normalized_pixels, composite_sed, ax):
    """Plots a similarity figure, which uses cross-correlation to show how similar eachpixel is to the composite"""
    axisfont=14
    ticksize=12

    print(f'Computing Similarity for Cluster {value}')

    cross_cor_vals = np.apply_along_axis(get_cross_cor, 1, normalized_pixels, sed_2=composite_sed)
    cross_cor_sim_values = [1-cross_cor_vals[i][1] for i in range(len(cross_cor_vals))]

    bins = np.arange(0, 1.05, 0.05)
    hist = ax.hist(cross_cor_sim_values, bins=bins, color='black')
    
    mean_sim_to_composite = np.mean(cross_cor_sim_values)
    median_sim_to_composite = np.median(cross_cor_sim_values)
    std_sim_to_composite = np.std(cross_cor_sim_values)
        
    ax.vlines(median_sim_to_composite, 0, 1000, color='orange')
    ax.vlines(median_sim_to_composite-2*std_sim_to_composite, 0, 1000, color='red')
    ax.text(0.38, 0.94, f'Median: {median_sim_to_composite:0.3f}', transform=ax.transAxes, horizontalalignment='right', fontsize=12)
    ax.text(0.38, 0.88, f'2$\\sigma$: {2*std_sim_to_composite:0.3f}', transform=ax.transAxes, horizontalalignment='right', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(hist[0])+1)
    ax.set_xlabel('Similarity to Composite', fontsize=axisfont)
    ax.set_ylabel('Number of pixels', fontsize=axisfont)
    ax.tick_params(labelsize=ticksize)
    
    return 
   

def scale_aspect(ax):
    """Makes plot a square"""
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    ydiff = np.abs(ylims[1]-ylims[0])
    xdiff = np.abs(xlims[1]-xlims[0])
    ax.set_aspect(xdiff/ydiff)