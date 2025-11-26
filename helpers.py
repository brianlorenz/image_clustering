import numpy as np


def delete_bad_values(arr, bad_image_idxs):
    if len(bad_image_idxs) > 0:
        arr = np.delete(arr, bad_image_idxs, axis=0)
    return arr

