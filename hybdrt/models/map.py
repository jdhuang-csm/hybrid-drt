import numpy as np
from scipy import ndimage


def make_peak_map(drt_array, pfrt_array, pfrt_thresh=None, filter_drt=False, drt_sigma=None,
                  filter_pfrt=False, pfrt_sigma=None):
    if filter_drt:
        if drt_sigma is None:
            drt_sigma = np.ones(np.ndim(drt_array))
            drt_sigma[-1] = 0
        drt_array = ndimage.gaussian_filter(drt_array, sigma=drt_sigma)

    if filter_pfrt:
        if pfrt_sigma is None:
            pfrt_sigma = 2
        pfrt_array = ndimage.gaussian_filter(pfrt_array, sigma=pfrt_sigma)

    #