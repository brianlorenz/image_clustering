# Read in a galaxy by ID and transforms the image and mask with pixels as [20, 1] arrays
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS
from data_paths import read_supercat, read_segmap, find_image_path, pixel_sed_save_loc, sed_save_loc, uncover_filters_info
import numpy as np 
import pandas as pd
from sedpy import observate
import pickle




def prepare_images(id_dr3_list, snr_thresh=3):
    """Reads in images from the catalogs and stores selected pixels in a format for clustering
    
    Parameters:
    id_dr3_list (list): List of ids to cluster on. Currently set up for id_dr3s fromt he UNCOVER/MegaScience Catalogs
    snr_thresh (float): Minimum signal-to-noise ratio needed to include a pixel. Currently checks against the median SNR from all images
    """
    # Read in necessary catalogs
    print ('Reading catalogs and images...')
    supercat_df = read_supercat()
    filter_list = get_filt_cols(supercat_df)

    # Read in image files
    image_dict = {}
    segmap, segmap_wcs = read_segmap()
    image_dict['segmap'] = segmap
    image_dict['segmap_wcs'] = segmap_wcs
    for filt in filter_list:
        image, wht_image, wcs, wht_wcs = load_image(filt)
        image_dict[filt] = image
        image_dict[filt+'_wcs'] = wcs
        image_dict[filt+'_wht'] = wht_image
        image_dict[filt+'_wht_wcs'] = wht_wcs


    for id_dr3 in id_dr3_list:
        print(f'Making cutouts for id_dr3 = {id_dr3}...')
        # Read in the image
        cutout_list, noise_cutout_list, boolean_segmap, obj_segmap_sizematch = images_and_segmap(id_dr3, supercat_df, image_dict, filter_list)
        cutout_arr = np.array(cutout_list) # shape is (n_images, cutout_y_size, cutout_x_size)
        noise_cutout_arr = np.array(noise_cutout_list)


        # Mask the image to only get galaxy pixels - currently using segmap
        median_snr = np.median(cutout_arr/noise_cutout_arr, axis=0)
        snr_masked = median_snr > snr_thresh # SNR thresh defined above
        mask = np.logical_and(snr_masked, boolean_segmap)
        masked_indicies = np.where(mask)

        pixel_seds = cutout_arr[:, masked_indicies[0], masked_indicies[1]] # shape is (n_images, n_pixels) where n_pixels are the number of pixels in the segmap
        # To access a particular pixel's SED, it is: 
        #       pixel_seds[:, pixel_id]
        # To access a pixel's coordinates back on the image, it is
        #       [masked_indicies[0][pixel_id], masked_indicies[1][pixel_id]]

        # Get the average SED from the super catalog
        supercat_row = supercat_df[supercat_df['id'] == id_dr3].iloc[0]
        sed_fluxes = [supercat_row[f'f_{filt_name}'] for filt_name in filter_list]
        err_sed_fluxes = [supercat_row[f'e_{filt_name}'] for filt_name in filter_list]
        np.savez(sed_save_loc + f'{id_dr3}_sed.npz', sed=sed_fluxes, err_sed=err_sed_fluxes) 

        bad_image_idxs = flag_blank_images(cutout_arr, sed_fluxes)

        np.savez(pixel_sed_save_loc + f'{id_dr3}_pixels.npz', pixel_seds=pixel_seds, mask=mask, masked_indicies=masked_indicies, image_cutouts=cutout_arr, noise_cutouts=noise_cutout_arr, boolean_segmap=boolean_segmap, obj_segmap=obj_segmap_sizematch.data, filter_names=np.array(filter_list), snr_thresh=snr_thresh, bad_image_idxs=bad_image_idxs) 

def images_and_segmap(id_dr3, supercat_df, image_dict, filter_list): 
    """Read in the segmap and images in each fiilter

    Parameters:
    id_dr3 (int): ID of the object, currently configured for UNCOVER/MegaScience
    supercat_df (pd.DataFrame): see read_supercat() in data_paths.py
    image_dict (dictionary): Contains all image data in dictionary form, generated in prepare_images()
    filter_list (list): List of filter names. See get_filt_cols()
    """
    # Get an astropy SkyCoord object cenetered on the galaxy
    obj_skycoord = get_coords(id_dr3, supercat_df)

    # Read in segmap, and use it to determine the size needed for future images
    obj_segmap = get_cutout_segmap(image_dict, obj_skycoord, size=(250,250))
    boolean_segmap = obj_segmap.data==id_dr3
    cutout_size = find_cutout_size(boolean_segmap)
    obj_segmap_sizematch = get_cutout_segmap(image_dict, obj_skycoord, size=cutout_size)
    boolean_segmap_sizematch = obj_segmap_sizematch.data==id_dr3

    cutout_list = []
    noise_cutout_list = []
    for filt in filter_list:
        image_cutout, wht_image_cutout = get_cutout(image_dict, obj_skycoord, filt, size=cutout_size)
        noise_image = 1/np.sqrt(wht_image_cutout.data)
        cutout_list.append(image_cutout.data)
        noise_cutout_list.append(noise_image)

    return cutout_list, noise_cutout_list, boolean_segmap_sizematch, obj_segmap_sizematch

def load_image(filt):
    """Opens the image .fits file and reads the data. Configured to find the image off of the filter name and find_image_path()"""
    image_str, wht_image_str = find_image_path(filt)
    with fits.open(image_str) as hdu:
        image = hdu[0].data
        wcs = WCS(hdu[0].header)
    with fits.open(wht_image_str) as hdu_wht:
        wht_image = hdu_wht[0].data
        wht_wcs = WCS(hdu_wht[0].header)  
    return image, wht_image, wcs, wht_wcs

def get_cutout(image_dict, obj_skycoord, filt, size = (100, 100)):
    """Takes the image data and returns a cutout of the requested size"""
    image = image_dict[filt]
    wcs = image_dict[filt+'_wcs']
    wht_image = image_dict[filt+'_wht']
    wht_wcs = image_dict[filt+'_wht_wcs']
    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs)
    wht_cutout = Cutout2D(wht_image, obj_skycoord, size, wcs=wht_wcs)
    return cutout, wht_cutout

def get_cutout_segmap(image_dict, obj_skycoord, size = (100, 100)):
    """Cutout of the segmentation map"""
    segmap_cutout = Cutout2D(image_dict['segmap'], obj_skycoord, size, wcs=image_dict['segmap_wcs'])
    return segmap_cutout

def get_coords(id_dr3, supercat_df):
    """Gets the ra/dec SkyCoord object for a given ID"""
    row = supercat_df[supercat_df['id']==id_dr3]
    obj_ra = row['ra'].iloc[0] * u.deg
    obj_dec = row['dec'].iloc[0] * u.deg
    obj_skycoord = SkyCoord(obj_ra, obj_dec)
    return obj_skycoord

def find_cutout_size(arr, pixel_buffer=10):
    """Given the boolean segmap array, find the minimum size needed to capture the whole galaxy in the image
    
    Parameters:
    arr (np.array): True where segmap=obj_id, False elsewhere
    pixel_buffer (int): Number of pixels to include on beyond the selected pixels.
        e.g. if all needed pixels are contained in a 30x30 cutout, then pixel_buffer=10 will return a 40x40 centered on the object
    
    Returns:
    cutout_size (float): Suggested size of cutout, contains full galaxy with a pixel_buffer/2 buffer on each side """
    # Any True in each row/column
    row_any = arr.any(axis=1)
    col_any = arr.any(axis=0)

    # Find first and last row with any True
    row_indices = np.where(row_any)[0]
    col_indices = np.where(col_any)[0]

    first_row = row_indices[0]
    last_row = row_indices[-1]
    first_col = col_indices[0]
    last_col = col_indices[-1]

    if first_row == 0 or first_col == 0 or last_row == len(arr) or last_col == len(arr):
        print('Need larger segmap size!!!') # The segmap is larger than the image size in this case

    middle_pixel = len(arr)/2
    row_dim_max = np.max([middle_pixel - first_row, last_row - middle_pixel])
    col_dim_max = np.max([middle_pixel - first_col, last_col - middle_pixel])
    single_dim_max = np.max([row_dim_max, col_dim_max])
    min_side = single_dim_max*2 + pixel_buffer 
    cutout_size = (min_side, min_side)

    return cutout_size

def get_filt_cols(df, skip_wide_bands=False):
    """Searches supercat_df columns to find the names of all jwst filters"""
    filt_cols = [col[2:] for col in df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    if skip_wide_bands ==  True:
        filt_cols = [col for col in filt_cols if 'w' not in col]
    return filt_cols

def uncover_read_filters():
    """Pulls the filters from supercat, then returns a dict with lots of useful filter info
    """
    supercat_df = read_supercat() # This is the fits file that has all of the medium band flux info
    filt_cols = get_filt_cols(supercat_df) # Grabs a list of the filter names from the columns of supercat
    
    sedpy_filts = [] 
    uncover_filt_dict = {}

    for filt in filt_cols: # Loops through each filter to read the info, then store it in the list and dict above
        filt = 'f_'+filt
        filtname = filt
        filt = filt.replace('f_', 'jwst_') 
        
        # Most names are jwst_f999m. But a few of the bands are from other telescopes
        try: 
            sedpy_filt = observate.load_filters([filt])
        except:
            try:
                filt = filt.replace('jwst_', 'wfc3_ir_')
                sedpy_filt = observate.load_filters([filt])
            except:
                filt = filt.replace('wfc3_ir_', 'acs_wfc_')
                sedpy_filt = observate.load_filters([filt])

        uncover_filt_dict[filtname+'_blue'] = sedpy_filt[0].blue_edge 
        uncover_filt_dict[filtname+'_red'] = sedpy_filt[0].red_edge
        uncover_filt_dict[filtname+'_wave_eff'] = sedpy_filt[0].wave_effective # This is the wavelength of the filter
        uncover_filt_dict[filtname+'_width_eff'] = sedpy_filt[0].effective_width
        uncover_filt_dict[filtname+'_width_rect'] = sedpy_filt[0].rectangular_width

        sedpy_filts.append(sedpy_filt[0])

    with open(uncover_filters_info, "wb") as f:
        pickle.dump(uncover_filt_dict, f) # Saves the filter info for faster access
    
    return uncover_filt_dict, sedpy_filts # Returns the dictionary and list with all the info saved


def flag_blank_images(image_cutouts, sed):
    """Finds which images are entirely empty, and flags them to be skipped in the clustering"""
    blank_image_idxs = []
    for i in range(len(image_cutouts)):
        if np.sum(image_cutouts[i]) == 0 or pd.isnull(sed[i]):
            blank_image_idxs.append(i)
    return blank_image_idxs

# Example usage of just this code
if __name__ == '__main__':
    uncover_read_filters()
    prepare_images([46339, 44283, 30804], snr_thresh=2)