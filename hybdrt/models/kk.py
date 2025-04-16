import numpy as np
from numpy import ndarray
from scipy import ndimage

from ..utils import stats

def normalize_residuals(z_meas, z_pred, norm="modulus"):
    # Calculate residuals
    z_err = z_meas - z_pred
    
    if norm == "modulus":
        norm = np.abs(z_meas)
        z_err = 100 * z_err / norm
    else:
        z_err = z_err / norm
    
    return z_err

def get_outliers(z_err_norm: ndarray, n_iter: int = 2, p_thresh: float = 1e-4):
    
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
        std = stats.robust_std(np.abs(z_err_norm)[~outlier_mask])

        # Get portion of distribution more extreme than y_err, assuming normally distributed errors
        prob = stats.outer_cdf_normal(np.abs(z_err_norm), 0, std)
        
        outlier_mask = (prob < p_thresh)
        
    return np.where(outlier_mask)[0]
        
        
def get_limits(f_fit, outlier_index: ndarray):
    
    outlier_mask = np.zeros(len(f_fit), dtype = bool)
    outlier_mask[outlier_index] = True
    
    badness = np.zeros(len(f_fit))
    badness[outlier_index] = 1
    # Spread badness to nearest neighbors
    badness = ndimage.uniform_filter1d(badness, size=3)
    
    # Find the first frequency where there is a clean point neighbored by at least 1 other clean point
    # clean_index = np.where((badness < 1) & (~outlier_mask))[0]
    clean_index = np.where(badness == 0)[0]
    f_max = np.max(f_fit[clean_index])
    f_min = np.min(f_fit[clean_index])
    
    return f_min, f_max

def trim_data(frequencies: ndarray, z: ndarray, f_min: float, f_max: float):
    mask = (frequencies <= f_max) & (frequencies >= f_min)
    
    return frequencies[mask], z[mask]