import numpy as np


def delete_bad_values(arr, bad_image_idxs):
    """Removes the indices from arr that are indicated by bad_image_idxs
    
    Parameters:
    arr (array): Array to remove the indicies from
    bad_image_idxs (list of ints): index numbers to drop

    Returns:
    arr (array): Imput array with the indices dropped
    """
    if len(bad_image_idxs) > 0:
        arr = np.delete(arr, bad_image_idxs, axis=0)
    return arr

