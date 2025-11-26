# Clustering Pixels by SED shape

This code is an exploration for clustering individual pixels into groups based on their SED shape. It is currently set up to work with the UNCOVER/MegaScience catalogs and images. 

# SETUP

data_paths.py - Go here and configure the clustering and catalog folders to match your own device. Make sure that you download and update the information for the various catalogs. Currently this needs files that look like this:
    
SUPER_CATALOG - UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits

SPS_Catalog - UNCOVER_v5.3.0_LW_SUPER_SPScatalog_spsv1.0.fits

SEGMAP .fits file - UNCOVER_v5.2.0_SEGMAP.fits

psf_matched images =  psf_matched/uncover_v7.1_abell2744clu_f160w_bcgs_sci_f444w-matched.fits



main_clustering.py - This is the main pipeline to run everything. Start looking at the code here, and it will connect to all of the other functions. This is currently set up to accept id_dr3s from UNCOVER/MegaScience. 

# Project Status

I think there are a few main problems to tackle:

## How do we determine which pixels should be included for clustering? 

Currently this is implemented as requiring a median SNR abouve some threshold (3) as well as overlap with the galaxy's segmentation map. We may want to expand this to include nearby soures if their redshifts match. All of this filtering is done in prepare_images.py

## How should we cluster the pixel SEDs? 

First, I think there's a question of metric - how do we determine the "distance" between two pixels for the purposes of clustering. If you simply cluster the pixels directly from how they are measured, you just end up clustering by brightness. Instead, we want to cluster by SED shape. There are two main approaches that I saw:
    
### Pixel Normalization
Normalize the pixels to remove the differences in brightness, then cluster using a standard distance measurement (euclidean metric). The "standard" distance would be the usual euclidean distance calcuation of sqrt((x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2 + ...) for n dimensions. 

The question would then be how best to normalize the pixels. One idea that I tried was the L2 norm (where you make the length of the n-dimensional vector equal to 1). Another was to measure a median SED from the galaxy, then scale it to each pixel and subtract it. It is unclear ot me which is more effective. These implementations are in helper_functions/normalize.py

I'm sure there are other options for normalzing the pixels, so this would be a great area to explore.

### Custom Metrics
Instead of normalizing and using a Euclidean metric, you can define a new distance measurement (metric) entirely. This has any number of possiblities - you could imagine weighting more strongly by certain features in an SED (Halpha, Balmer break, etc), or by smoothing the SED shape and finding some distance between the smoothed polynomials. So many things to explore here. The ones implemented so far are:

Cross-correlation - this runs a calculation between two pixels and returns a value from 0 to 1 based on how similar their shapes are. Then, we can cluster based on this similarity values

Weighted-norm - where each pixel is not weighted equally in the calculation of distance. 

Both of these implementations are in distances.py, and it's not clear which is more effective.

For all of the clustering algorithms, if you want to use this method, you need to change your input matrix to include the distance measurements, then pass metric='precomputed' to the clustering function. This is not implemnted for all of them yet, but it is in DBSCAN and Agglomerative clustering

It is unclear to me which will be more productive between pixel normalization and custom metrics. I think the custom metrics offer a lot more flexibility , but will probably require more work to make sure it is working properly.

### Once you have decided on the best way to prepare the pixel SEDs for clustering, what is the best algorithm to cluster them? 

I have tried 6 different methods here, and again it is unclear to me which is most effective. Lots of pros and cons to all of them. In particular, I think determining how to figure out the required number of clusters is really important. Most of the methods require specifying the number of clusters as an input, and this would likely change from galaxy to galaxy


# Composite SEDs

This is one of the spots where I left off - currently, I'm generating composite SEDs for each of the clusters and then viewing their similarities for just one clustering method (kmeans). I was just starting to get this implemented when I stopped working on this project, so there is much more to explore with evaluating the composite SEDs for a variety of methods (should be easy to run, just haven't done it yet).

# Next Steps

My suggestions for next steps would be to find a galaxy that has very clear clumps and to run these methods and that. So far, I have been working with relatively uniform galaxies - I think taking an extreme example would be useful

Additionally, you could consider simulated a galaxy - generating SEDs at each pixel with FSPS, and having a few known clusters in mind. Then, you could see if the methods can reporduce the clusters you put into it, as well as seeing what happens if you ask the code to make more or fewer clusters than you input. 