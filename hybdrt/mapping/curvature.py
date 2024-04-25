import numpy as np
from scipy import signal, ndimage

from hybdrt.utils import stats
from ..filters import std_filter
from . import ndx


# -----------------------------------------------
# Peak prob
# -----------------------------------------------
def peak_prob_1d(arrays_1d, nonneg, sign, height, prominence):
    # Unpack arrays
    f, fxx, f_sigma, fxx_sigma = arrays_1d
    if nonneg and sign != 0:
        # This captures both the standard nonneg case and the series_neg case
        # peak_indices = peaks.find_peaks_simple(fxx, order=2, height=0, prominence=prominence, **kw)
        peak_indices, peak_info = signal.find_peaks(-sign * fxx, height=height, prominence=prominence)
        # peak_indices = peaks.find_peaks_compound(fx, fxx, **kw)
    else:
        # Find positive and negative peaks separately
        peak_index_list = []
        peak_info_list = []
        for peak_sign in [-1, 1]:
            peak_index, peak_info = signal.find_peaks(-peak_sign * fxx, height=height, prominence=prominence)
            # Limit to peaks that are positive in the direction of the current sign
            pos_index = peak_sign * f[peak_index] > 0
            # print(pos_index)
            peak_index = peak_index[pos_index]
            peak_info = {k: v[pos_index] for k, v in peak_info.items()}

            peak_index_list.append(peak_index)
            peak_info_list.append(peak_info)
        # Concatenate pos and neg peaks
        peak_indices = np.concatenate(peak_index_list)
        peak_info = {k: np.concatenate([pi[k] for pi in peak_info_list]) for k in peak_info.keys()}

        # Sort ascending
        sort_index = np.argsort(peak_indices)
        peak_indices = peak_indices[sort_index]
        peak_info = {k: v[sort_index] for k, v in peak_info.items()}

    # if prob_thresh is None:
    #     prob_thresh = 0.25

    # Use prominence or height, whichever is smaller
    min_prom = np.minimum(peak_info['prominences'], peak_info['peak_heights'])

    # Evaluate peak confidence
    curv_prob = 1 - stats.cdf_normal(0, min_prom, fxx_sigma[peak_indices])
    f_prob = 1 - stats.cdf_normal(0, np.sign(f[peak_indices]) * f[peak_indices], f_sigma[peak_indices])

    probs = np.minimum(curv_prob, f_prob)

    out = np.zeros(len(f))
    out[peak_indices] = probs

    return out
