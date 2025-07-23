import numpy as np
from numpy import ndarray
from scipy import ndimage
from typing import Optional

from ..utils import stats, eis

def normalize_residuals(z_meas, z_pred, norm="modulus"):
    # Calculate residuals
    z_err = z_meas - z_pred
    
    if norm == "modulus":
        norm = np.abs(z_meas)
        z_err = 100 * z_err / norm
    else:
        z_err = z_err / norm
    
    return z_err

def get_outliers(z_err_norm: ndarray, n_iter: int = 2, p_thresh: float = 1e-4, n_sigma: Optional[float] = None, std_sample_fraction=0.6):
    
    outlier_mask = np.zeros(len(z_err_norm), dtype=bool)
    
    # for i in range(n_iter):
    #     # Estimate robust metrics excluding outliers identified so far
    #     std_r = stats.robust_std(z_err_norm.real[~outlier_mask])
    #     std_i = stats.robust_std(z_err_norm.imag[~outlier_mask])
        
    #     # Get portion of distribution more extreme than y_err, assuming normally distributed errors
    #     prob_r = stats.outer_cdf_normal(z_err_norm.real, 0, std_r)
    #     prob_i = stats.outer_cdf_normal(z_err_norm.imag, 0, std_i)
        
    #     outlier_mask = (prob_r < p_thresh) | (prob_i < p_thresh)
        
    for i in range(n_iter):
        # Estimate robust metrics excluding outliers identified so far
        # Use modulus of error (distance in cartesian plane)
        
        # Get std of real and imaginary error components
        std = stats.robust_std(eis.complex_vector_to_concat(z_err_norm[~outlier_mask]), sample_fraction=std_sample_fraction)
        
        if n_sigma is None:
            # The squared error modulus (e_r ** 2 + e_i ** 2) should follow a chi-squared distribution
            # Get portion of distribution more extreme than y_err
            prob = stats.outer_cdf_chi2(np.abs(z_err_norm) ** 2, scale=std ** 2, k=2)
            outlier_mask = (prob < p_thresh)
        else:
            # Simply find errors larger than n_sigma * std
            outlier_mask = np.abs(z_err_norm) > std * n_sigma
        
        
    return np.where(outlier_mask)[0]
        
        
def get_limits(f_fit: ndarray, outlier_index: ndarray, max_num_outliers: int = 2, return_index: bool = False):
    """Get frequency limits of valid data.

    :param ndarray f_fit: frequencies
    :param ndarray outlier_index: indices of outliers identified by KK test
    :param int max_num_outliers: maximum number of allowable outliers inside the valid range
    :param bool return_index: if True, return indices corresponding to maximum and minimum frequencies.
        If False, only return f_min and f_max. Defaults to False
    :return Tuple[float, float]: Frequency limits: (f_min, f_max)
    """
    # Sort descending
    sort_index = np.argsort(f_fit)[::-1]
    f_fit = f_fit[sort_index]
    # outlier_index = outlier_index[sort_index[::-1]]
    outlier_index = [sort_index.tolist().index(i) for i in outlier_index]
    
    # outlier_mask = np.zeros(len(f_fit), dtype = bool)
    # outlier_mask[outlier_index] = True
    
    is_outlier = np.zeros(len(f_fit))
    is_outlier[outlier_index] = 1
    # Spread badness to nearest neighbors
    badness = ndimage.uniform_filter1d(is_outlier, size=3)
    
    # Find the first frequency where there is a clean point neighbored by at least 1 other clean point
    clean_index = np.where(badness == 0)[0]
    
    i_left = clean_index[0]
    i_right = clean_index[-1]
    # print(i_left, i_right)
    
    
    # Check how many bad points are inside the limits
    num_bad_inside = np.sum(is_outlier[i_left:i_right])
    
    # If there are too many outliers inside the clean range, move the bounds in
    if num_bad_inside > max_num_outliers:
        num_to_remove = num_bad_inside - max_num_outliers
        # Number of outliers that will be removed by moving each boundary inwards
        from_left = np.cumsum(is_outlier[i_left:i_right + 1])
        from_right = np.cumsum(is_outlier[i_left:i_right + 1][::-1])
        # print(from_left, from_right)
        # 2D grid of possible combinations of left and right boundary movements
        ll, rr = np.meshgrid(from_left, from_right)
        # Total number of outliers that will be removed by the combination of boundary movements
        tot_removed = ll + rr
        # Find movements that will satisfy the number of allowable outliers inside bounds
        index = np.argwhere(tot_removed >= num_to_remove)
        # Find combination that minimizes reduction in window size (maximizes clean frquency range)
        r, l = index[np.argmin(np.sum(index, axis=1))]
        # print(l, r)
        i_left = i_left + l
        i_right = i_right - r
        
    # If the chosen bounds are at outliers, move them to the next clean point
    if is_outlier[i_left] == 1:
        i_left = np.min(clean_index[clean_index >= i_left])
    if is_outlier[i_right] == 1:
        i_right = np.max(clean_index[clean_index <= i_right])
        
    
    f_max = f_fit[i_left]
    f_min = f_fit[i_right]
    
    if return_index:
        return (f_min, f_max), (i_left, i_right)
    else:
        return f_min, f_max

def trim_data(frequencies: ndarray, z: ndarray, f_min: float, f_max: float):
    mask = (frequencies <= f_max) & (frequencies >= f_min)
    
    return frequencies[mask], z[mask]