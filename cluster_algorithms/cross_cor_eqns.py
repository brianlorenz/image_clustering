import numpy as np


def get_cross_cor(sed_1, sed_2):
    """Compare two SEDs with the same wavelengths

    Parameters:
    mock_sed_1 (pd.DataFrame): read the SED, then put it into a dataframe and directly into this funciton
    mock_sed_2 (pd.DataFrame): read the SED, then put it into a dataframe and directly into this funciton

    Returns:
    a12 (float): Normalization factor, where f1 = a12*f2
    b12 (float): correlation factor, from 0 to 1, where 0 is identical
    """

    f1 = sed_1
    f2 = sed_2

    a12 = np.sum(f1 * f2) / np.sum(f2**2)
    b12 = np.sqrt(np.sum((f1 - a12 * f2)**2) / np.sum(f1**2))
    return a12, b12

