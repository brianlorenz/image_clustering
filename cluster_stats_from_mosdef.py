# Plots some statistics about each cluser
# plot_similarity can plot the similarities between each pair of galaxies
# in a cluster and each galaxy and its SED

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from read_data import mosdef_df
from mosdef_obj_data_funcs import read_sed, read_mock_sed, read_mock_composite_sed
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from cross_correlate import get_cross_cor
import shutil



def plot_similarity_cluster(groupID, zobjs, similarity_matrix, axis_obj='False'):
    print(f'Computing Similarity for Cluster {groupID}')
    galaxies = zobjs[zobjs['cluster_num'] == groupID]
    similarities = []
    similarities_composite = []
    median_similarities = []
    num_galaxies = len(galaxies)
    for i in range(num_galaxies):
        for j in range(num_galaxies - i):
            if i != j:
                similarities.append(
                    similarity_matrix[galaxies.iloc[i]['new_index'], galaxies.iloc[j]['new_index']])
        gal_sims = []
        for j in range(num_galaxies):
            #Find and save the mean similarity
            if i != j:
                gal_sims.append(similarity_matrix[galaxies.iloc[i]['new_index'], galaxies.iloc[j]['new_index']])
        median_similarities.append(np.median(gal_sims))
    mock_composite_sed = read_mock_composite_sed(groupID)
    for i in range(num_galaxies):
        mock_sed = read_mock_sed(
            galaxies.iloc[i]['field'], galaxies.iloc[i]['v4id'])
        similarities_composite.append(
            1 - get_cross_cor(mock_composite_sed, mock_sed)[1])

    #galaxies['similarity_composite'] = similarities_composite
    zobjs.loc[zobjs['cluster_num'] == groupID,
                'similarity_composite'] = similarities_composite
    zobjs.loc[zobjs['cluster_num'] == groupID,
                'median_similarity_to_group'] = median_similarities

    galaxies = zobjs.loc[zobjs['cluster_num'] == groupID]

    axisfont = 14
    ticksize = 12
    ticks = 8

    # Figure for just the galaixes in that cluster
    if axis_obj == 'False':
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        ax = axis_obj

    bins = np.arange(0, 1.05, 0.05)
    ax.hist(similarities, bins=bins, color='black')

    ax.set_xlim(-0.05, 1.05)
    
    mean_sim = np.mean(similarities)
    mean_sim_to_composite = np.mean(similarities_composite)
    median_sim_to_composite = np.median(similarities_composite)
    std_sim_to_composite = np.std(similarities_composite)
    
    if axis_obj == 'False':
        ax.set_xlabel('Similarity', fontsize=axisfont)
        ax.set_ylabel('Number of pairs', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)
        
        ax.text(0.1, 0.9, f'Avg. similarity {mean_sim}', transform=ax.transAxes)
        imd.check_and_make_dir(imd.cluster_similarity_plots_dir)
        fig.savefig(imd.cluster_similarity_plots_dir + f'/{groupID}_similarity.pdf')
        plt.close()

        # Figure for the correlation with the composite:
        fig, ax = plt.subplots(figsize=(8, 7))

        bins = np.arange(0, 1.05, 0.05)
        ax.hist(similarities_composite, bins=bins, color='black')
        ax.vlines(median_sim_to_composite, 0, 10, color='orange')
        ax.vlines(median_sim_to_composite-2*std_sim_to_composite, 0, 10, color='red')
        ax.text(0.1, 0.9, f'Median similarity to composite {median_sim_to_composite}', transform=ax.transAxes)
        ax.text(0.1, 0.8, f'2 Std similarity to composite {2*std_sim_to_composite}', transform=ax.transAxes)

        # Flag galaxies for removal
        zeros = np.zeros(len(galaxies))
        galaxies['deletion_flag'] = zeros
        galaxies = galaxies.reset_index(drop=True)
        std_cutoff = median_sim_to_composite-2*std_sim_to_composite
        for i in range(len(galaxies)):
            gal_similarity = galaxies.iloc[i]['similarity_composite']
            if gal_similarity < 0.8:
                galaxies.loc[i, 'deletion_flag'] = 1

        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel('Similarity to Composite', fontsize=axisfont)
        ax.set_ylabel('Number of galaxies', fontsize=axisfont)
        ax.tick_params(labelsize=ticksize, size=ticks)
        imd.check_and_make_dir(imd.cluster_similarity_plots_dir)
        imd.check_and_make_dir(imd.cluster_similarity_composite_dir)
        fig.savefig(imd.cluster_similarity_composite_dir  + f'/{groupID}_similarity_composite.pdf')
        plt.close()

        # Also, save the values between each galaxy and the composite
        galaxies.to_csv(imd.cluster_similarity_composite_dir + f'/{groupID}_similarity_composite.csv', index=False)
        return mean_sim, mean_sim_to_composite
    
    else:
        return mean_sim, mean_sim_to_composite



def plot_all_similarity(n_clusters):
    """Create a histogram of similarities between individual galaxies in each cluser, and also to the composite SED

    Parameters:
    n_clusters (int): Number of clusters

    Returns:
    """

    similarity_matrix = ascii.read(
        imd.cluster_dir + '/similarity_matrix.csv').to_pandas().to_numpy()
    zobjs = ascii.read(
        imd.cluster_dir + '/zobjs_clustered.csv', data_start=1).to_pandas()
    zobjs['new_index'] = zobjs.index

    groupIDs = []
    mean_sims = []
    mean_sim_to_composites = []
    for groupID in range(n_clusters):
        mean_sim, mean_sim_to_composite = plot_similarity_cluster(groupID, zobjs, similarity_matrix)
        mean_sims.append(mean_sim)
        groupIDs.append(groupID)
        mean_sim_to_composites.append(mean_sim_to_composite)
    sim_df = pd.DataFrame(zip(groupIDs, mean_sims, mean_sim_to_composites), columns=['groupID', 'mean_sim', 'mean_sim_to_composite'])
    sim_df.to_csv(imd.cluster_similarity_plots_dir+'/composite_similarities.csv', index=False)

def remove_dissimilar_gals(n_clusters):
    """Removes galaxies from our sample that are not similar to their clusters"""
    
    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    removed_gal_df = ascii.read(imd.loc_removed_gal_df).to_pandas()
    zobjs = ascii.read(imd.cluster_dir + '/zobjs_clustered.csv', data_start=1).to_pandas()
    group_len = []
    for groupID in range(n_clusters):
        similarities_df = ascii.read(imd.cluster_similarity_composite_dir + f'/{groupID}_similarity_composite.csv').to_pandas()
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        images_list = os.listdir(f'/Users/brianlorenz/mosdef/Clustering/{groupID}/')
        group_len.append(len(group_df))
        for i in range(len(group_df)):
            field = group_df.loc[i]['field']
            v4id = group_df.loc[i]['v4id']
            gal_df_row = np.logical_and(filtered_gal_df['field']==field, filtered_gal_df['v4id']==v4id)
            
            if len(filtered_gal_df[gal_df_row]) == 0:
                breakpoint()


        for i in range(len(similarities_df)):
            if similarities_df.loc[i]['deletion_flag'] == 1:
                field = similarities_df.loc[i]['field']
                v4id = similarities_df.loc[i]['v4id']

                remove_row = np.logical_and(group_df['field']==field, group_df['v4id']==v4id)
                remove_row_idx = group_df[remove_row].index[0]
                group_df = group_df.drop(remove_row_idx)

                # Remove from filtered and move to removed
                gal_df_row = np.logical_and(filtered_gal_df['field']==field, filtered_gal_df['v4id']==v4id)
                gal_df_row_idx = filtered_gal_df[gal_df_row].index[0]
                filtered_gal_df[gal_df_row]
                removed_gal_df.append(filtered_gal_df[gal_df_row])
                filtered_gal_df = filtered_gal_df.drop(gal_df_row_idx)
                
                zobjs_row = np.logical_and(zobjs['field']==field, zobjs['v4id']==v4id)
                zobjs_row_idx = zobjs[zobjs_row].index[0]
                zobjs = zobjs.drop(zobjs_row_idx)

                filename = f'{field}_{v4id}_mock.pdf'
                cluster_num = similarities_df.loc[i]["cluster_num"]
                # breakpoint()
                os.remove(imd.cluster_dir + f'/{cluster_num}/' + filename)

                similarities_df = similarities_df.drop(i)

        group_df.to_csv(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv', index=False)
        similarities_df.to_csv(imd.cluster_similarity_composite_dir + f'/{groupID}_similarity_composite.csv', index=False)

    zobjs.to_csv(imd.cluster_dir + '/zobjs_clustered.csv', index=False)
    filtered_gal_df.to_csv(imd.loc_filtered_gal_df, index=False)
    removed_gal_df.to_csv(imd.loc_removed_gal_df, index=False)

def remove_flagged_seds(n_clusters):
    """Removes galaxies from our sample that are flagged by the SED algorithm"""
    
    filtered_gal_df = ascii.read(imd.loc_filtered_gal_df).to_pandas()
    removed_gal_df = ascii.read(imd.loc_removed_gal_df).to_pandas()
    zobjs = ascii.read(imd.cluster_dir + '/zobjs_clustered.csv', data_start=1).to_pandas()
    flagged_objs = zobjs[zobjs['flag_sed']>0]
    affected_groups = set(flagged_objs['cluster_num'].to_list())
    for groupID in affected_groups:
        gals_in_group = flagged_objs[flagged_objs['cluster_num']==groupID]
        group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv').to_pandas()
        
        for i in range(len(gals_in_group)):
            field = gals_in_group.iloc[i]['field']
            v4id = gals_in_group.iloc[i]['v4id']

            remove_row = np.logical_and(group_df['field']==field, group_df['v4id']==v4id)
            remove_row_idx = group_df[remove_row].index[0]
            group_df = group_df.drop(remove_row_idx)

            # Remove from filtered and move to removed
            gal_df_row = np.logical_and(filtered_gal_df['field']==field, filtered_gal_df['v4id']==v4id)
            gal_df_row_idx = filtered_gal_df[gal_df_row].index[0]
            filtered_gal_df[gal_df_row]
            removed_gal_df.append(filtered_gal_df[gal_df_row])
            filtered_gal_df = filtered_gal_df.drop(gal_df_row_idx)
            
            zobjs_row = np.logical_and(zobjs['field']==field, zobjs['v4id']==v4id)
            zobjs_row_idx = zobjs[zobjs_row].index[0]
            zobjs = zobjs.drop(zobjs_row_idx)

            filename = f'{field}_{v4id}_mock.pdf'
            os.remove(imd.cluster_dir + f'/{groupID}/' + filename)

        group_df.to_csv(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv', index=False)
    
    zobjs.to_csv(imd.cluster_dir + '/zobjs_clustered.csv', index=False)
    filtered_gal_df.to_csv(imd.loc_filtered_gal_df, index=False)
    removed_gal_df.to_csv(imd.loc_removed_gal_df, index=False)

# remove_flagged_seds(20)
# remove_dissimilar_gals(20)
# plot_all_similarity(20)