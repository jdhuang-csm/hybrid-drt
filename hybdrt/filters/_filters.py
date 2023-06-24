import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold

from ._scifilters import empty_gaussian_filter1d, empty_gaussian_filter, gaussian_laplace1d


def rms_filter(a, size, empty=False, **kw):
    # Get mean of squared deviations
    a2 = a ** 2
    a2_mean = ndimage.uniform_filter(a2, size, **kw)

    if empty:
        # Determine kernel volume
        if np.isscalar(size):
            ndim = np.ndim(a)
            n = size ** ndim
        else:
            n = np.prod(size)
        a2_mean -= a2 / n
        a2_mean *= n / (n - 1)

    # Small negatives may arise due to precision loss
    a2_mean[a2_mean < 0] = 0

    return a2_mean ** 0.5


def std_filter(a, size, mask=None, **kw):
    if mask is None:
        a_mean = ndimage.uniform_filter(a, size, **kw)
        var = ndimage.uniform_filter((a - a_mean) ** 2, size, **kw)
    else:
        a_mean = masked_filter(a, mask, ndimage.uniform_filter, size=size, **kw)
        var = masked_filter((a - a_mean) ** 2, mask, ndimage.uniform_filter, size=size, **kw)

    # Small negatives may arise due to precision loss
    var[var < 0] = 0

    return var ** 0.5


def iqr_filter(a, size, **kw):
    q1 = ndimage.percentile_filter(a, 25, size=size, **kw)
    q3 = ndimage.percentile_filter(a, 75, size=size, **kw)
    return q3 - q1


def gaussian_kernel_scale(sigma, truncate=4.0, empty=False):
    sigma2 = sigma * sigma
    radius = int(float(sigma) * truncate + 0.5)
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    if empty:
        phi_x[x == 0] = 0
    return phi_x.sum()


def rog_filter(a, sigma_loc, sigma_glob, mask=None, median_pad=0.1, median_size=None, mode='reflect'):
    """
    Ratio of Gaussians
    :param a:
    :param sigma_loc:
    :param sigma_glob:
    :param mask:
    :param median_pad:
    :param median_size:
    :param mode:
    :return:
    """
    if mask is not None:
        local_scale = masked_filter(a ** 2, mask, sigma=sigma_loc, mode=mode)
    else:
        local_scale = ndimage.gaussian_filter(a ** 2, sigma_loc, mode=mode)

    if median_size is not None:
        local_scale += median_pad * ndimage.median_filter(local_scale, median_size, mode=mode)
    else:
        if mask is not None:
            local_scale += median_pad * np.median(a[mask > 0] ** 2)
        else:
            local_scale += median_pad * np.median(a ** 2)

    local_scale = local_scale ** 0.5

    if mask is not None:
        local_scale = np.nan_to_num(local_scale, nan=1)
        global_scale = np.exp(masked_filter(np.log(local_scale), mask, sigma=sigma_glob, mode=mode))
    else:
        global_scale = np.exp(ndimage.gaussian_filter(np.log(local_scale), sigma=sigma_glob, mode=mode))

    scaled = a * global_scale / local_scale
    if mask is not None:
        # Fill masked vales with filtered rescaled values
        out = scaled.copy()
        # print(masked_filter(scaled, mask, sigma=sigma_glob, mode=mode)[mask == 0])
        out[mask == 0] = masked_filter(scaled, mask, sigma=sigma_glob, mode=mode)[mask == 0]
        return out
    else:
        return scaled


def signed_hysteresis_threshold(a, low, high):
    """
    Apply hysteresis threshold to negative and positive portions of image separately
    :param a:
    :param low:
    :param high:
    :return:
    """
    thresh = np.zeros(a.shape, dtype=bool)

    for sign in [1, -1]:
        a_sign = a.copy()
        # Mask values of the opposite sign
        mask = a_sign * sign > 0
        a_sign[~mask] = 0

        sign_thresh = apply_hysteresis_threshold(a_sign * sign, low=low, high=high)
        thresh[mask] = sign_thresh[mask]
    return thresh


def flexible_hysteresis_threshold(a, low, high, structure=None):
    """
    Hysteresis threshold with optional structure argument to define connectivity
    :param a:
    :param low:
    :param high:
    :param structure:
    :return:
    """
    if low >= high:
        raise ValueError('low must be less than high')

    low_mask = a > low
    high_mask = a > high

    # Find components connected by low
    labels, count = ndimage.label(low_mask, structure=structure)

    # Check how many high points the low labels touch
    high_count = ndimage.sum_labels(high_mask, labels, index=np.arange(count + 1))
    touches_high = high_count > 0

    return touches_high[labels]


def masked_filter(a, mask, filter_func=None, **filter_kw):
    """
    Perform a masked/normalized filter operation on a. Only valid for linear filters
    :param ndarray a: array to filter
    :param ndarray mask: mask array indicating weight of each pixel in x_in; must match shape of x_in
    :param filter_func: filter function to apply. Defaults to gaussian_filter
    :param filter_kw: keyword args to pass to filter_func
    :return:
    """
    if filter_kw is None:
        if filter_func is None:
            sigma = np.ones(np.ndim(a))
            sigma[-1] = 0
            filter_kw = {'sigma': sigma}
        else:
            filter_kw = None
    if filter_func is None:
        filter_func = ndimage.gaussian_filter

    mask = mask.astype(float)

    x_filt = filter_func(a * mask, **filter_kw)
    mask_filt = filter_func(mask, **filter_kw)

    # print(np.sum(np.isnan(x_filt)), np.sum(np.isnan(mask_filt)), np.sum(mask_filt == 0))

    return x_filt / mask_filt


def nan_filter(a, filter_func, **filter_kw):
    mask = ~np.isnan(a)
    return masked_filter(np.nan_to_num(a), mask, filter_func, **filter_kw)


def iterate_gaussian_weights(a, init_weights=None, adaptive=False, iter=2, nstd=5, dev_rms_size=5,
                             nan_mask=None, **filter_kw):
    if init_weights is None:
        weights = np.ones(a.shape)
    else:
        weights = init_weights

    if nan_mask is not None:
        weights[nan_mask] = 0
    # prev_weights = weights.copy()

    # if adaptive:
    #     def empty_filter(a_in, **kw):
    #
    #         return nonuniform_gaussian_filter(a_in, sigma=sigma_list, empty=True, **kw)
    # else:
    #     empty_filter = empty_gaussian_filter

    # Determine weights based on deviation from empty filter values
    for i in range(iter):
        if adaptive:
            # Get sigma manually to use same sigma for a and mask
            sigmas = get_adaptive_sigmas(a, empty=True, weights=weights, **filter_kw)

            def filter_func(a_in, **kw):
                return adaptive_gaussian_filter(a_in, sigmas=sigmas, empty=True, **kw)
        else:
            filter_func = empty_gaussian_filter

        dev = a - masked_filter(a, weights, filter_func=filter_func, **filter_kw)
        # weights = np.exp(-(dev / (nstd * np.std(dev))) ** 6)
        # print('weights:', np.round(weights[:12], 3))
        # print('dev:    ', np.round(np.abs(dev[:12]), 3))

        # dev_rms = rms_filter(dev, size=dev_rms_size, empty=True)
        # if nan_mask is not None:
        #     dev_weights = (~nan_mask).astype(float)
        dev_rms = masked_filter(dev, weights, rms_filter, size=dev_rms_size, empty=True)
        weights = np.exp(-(dev / (nstd * dev_rms + 0.1 * np.std(dev))) ** 6)


        # print('dev_rms:', np.round(dev_rms[:12], 3))
        # weights = (weights * prev_weights) ** 0.5

        # prev_weights = weights.copy()

        if nan_mask is not None:
            weights[nan_mask] = 0

    return weights


def iterative_gaussian_filter(a, adaptive=False, iter=2, nstd=5, dev_rms_size=5, nan_mask=None, fill_nans=False,
                              **filter_kw):
    # Determine weights based on deviation from empty filter values
    weights = iterate_gaussian_weights(a, None, adaptive, iter, nstd, dev_rms_size=dev_rms_size, nan_mask=nan_mask,
                                       **filter_kw)

    if adaptive:
        # Get sigma manually to use same sigma for a and mask
        sigmas = get_adaptive_sigmas(a, empty=False, weights=weights, **filter_kw)

        def filter_func(a_in, **kw):
            return adaptive_gaussian_filter(a_in, sigmas=sigmas, **kw)

    else:
        filter_func = ndimage.gaussian_filter

    out = masked_filter(a, weights, filter_func=filter_func, **filter_kw)

    if nan_mask is not None and not fill_nans:
        out[nan_mask] = np.nan

    return out


# Nonuniform gaussian
# -------------------
def nonuniform_gaussian_filter1d(a, sigma, axis=-1, empty=False,
                                 mode='reflect', cval=0.0, truncate=4, order=0,
                                 sigma_node_factor=1.5, min_sigma=0.25):
    if np.max(sigma) > 0:
        sigma = np.maximum(sigma, 1e-8)
        # Get sigma nodes
        min_ls = max(np.min(np.log10(sigma)), np.log10(min_sigma))  # Don't go below min effective value
        max_ls = max(np.max(np.log10(sigma)), np.log10(min_sigma))
        num_nodes = int(np.ceil((max_ls - min_ls) / np.log10(sigma_node_factor))) + 1
        sigma_nodes = np.logspace(min_ls, max_ls, num_nodes)

        if np.min(sigma) < min_sigma:
            # If smallest sigma is below min effective value, insert dummy node at lowest value
            # This node will simply return the original array

            # Determine factor for uniform node spacing
            if len(sigma_nodes) > 1:
                factor = sigma_nodes[-1] / sigma_nodes[-2]
            else:
                factor = sigma_node_factor

            # Limit requested sigma values to 2 increments below min effective sigma
            # This will ensure that any sigma values well below min_sigma will not be filtered, while those
            # close to min_sigma will receive mixed-lengthscale filtering as intended
            sigma[sigma < min_sigma / (factor ** 2)] = min_sigma / (factor ** 2)

            # Insert as many sigma values as needed to get to lowest requested value (max 2 inserts)
            while sigma_nodes[0] > np.min(sigma) * 1.001:
                sigma_nodes = np.insert(sigma_nodes, 0, sigma_nodes[0] / factor)

        # print(sigma_nodes)
        if len(sigma_nodes) > 1:
            node_delta = np.log(sigma_nodes[-1] / sigma_nodes[-2])
        else:
            node_delta = 1

        def get_node_weights(x):
            # Tile x and nodes to same shape with extra axis
            tile_shape = np.ones(np.ndim(x) + 1, dtype=int)
            tile_shape[0] = len(sigma_nodes)
            # print('x:', x)
            x_tile = np.tile(x, tile_shape)
            node_tile = np.tile(sigma_nodes, (*x.shape, 1))
            node_tile = np.moveaxis(node_tile, -1, 0)

            nw = np.abs(np.log(x_tile / node_tile)) / node_delta
            nw[nw >= 1] = 1
            nw = 1 - nw
            # print('min weight:', np.min(nw))
            # print('max weight:', np.max(nw))
            # print('min weight sum:', np.min(np.sum(nw, axis=0)))
            # print('max weight sum:', np.max(np.sum(nw, axis=0)))
            return nw

        node_outputs = np.empty((len(sigma_nodes), *a.shape))

        for i in range(len(sigma_nodes)):
            if sigma_nodes[i] < min_sigma:
                # Sigma is below minimum effective value
                if empty:
                    # For empty filter, still need to apply filter to determine central value
                    node_outputs[i] = empty_gaussian_filter1d(a, sigma=min_sigma, axis=axis, mode=mode, cval=cval,
                                                              truncate=truncate, order=order)
                else:
                    # For standard filter, reduces to original array
                    node_outputs[i] = a
            else:
                if empty:
                    node_outputs[i] = empty_gaussian_filter1d(a, sigma=sigma_nodes[i], axis=axis, mode=mode, cval=cval,
                                                              truncate=truncate, order=order)
                else:
                    node_outputs[i] = ndimage.gaussian_filter1d(a, sigma=sigma_nodes[i], axis=axis, mode=mode,
                                                                cval=cval,
                                                                truncate=truncate, order=order)

        node_weights = get_node_weights(sigma)
        # print(node_weights.shape, node_outputs.shape)
        # print(np.sum(node_weights, axis=0))

        out = node_outputs * node_weights
        return np.sum(out, axis=0)

    else:
        # No filtering to perform on this axis
        return a


def nonuniform_gaussian_filter(a, sigma, empty=False,
                               mode='reflect', cval=0.0, truncate=4, order=0,
                               sigma_node_factor=1.5):
    axes = np.arange(np.ndim(a), dtype=int)

    # Apply sequence of 1d filters
    out = a
    for i, axis in enumerate(axes):
        out = nonuniform_gaussian_filter1d(out, sigma[i], axis, empty, mode, cval, truncate, order, sigma_node_factor)

    return out


# Adaptive Gaussian
# -----------------
def get_adaptive_sigma1d(a, axis=-1, presmooth_sigma=1, empty=False, weights=None,
                         curv_func=None, curv_kw=None, k_factor=1.0, max_sigma=5.0,
                         mode='reflect', cval=0.0, truncate=4.0):

    if max_sigma > 0:
        if curv_kw is None:
            if curv_func is None:
                curv_kw = {'curv_sigma': 1, 'mode': mode, 'cval': cval, 'truncate': truncate}
            else:
                curv_kw = {}

        if curv_func is None:
            def curv_func(a_in, curv_sigma=None, **kw):
                return gaussian_laplace1d(a_in, sigma=curv_sigma, axis=axis, **kw)

        if empty:
            filter_func = empty_gaussian_filter
        else:
            filter_func = ndimage.gaussian_filter

        if np.isscalar(presmooth_sigma):
            presmooth_sigma = [presmooth_sigma] * np.ndim(a)

        if np.max(presmooth_sigma) > 0:
            if weights is None:
                a_smooth = filter_func(a, sigma=presmooth_sigma, mode=mode, cval=cval, truncate=truncate)
            else:
                a_smooth = masked_filter(a, weights, filter_func, sigma=presmooth_sigma, mode=mode,
                                         cval=cval, truncate=truncate)
        else:
            a_smooth = a

        curv = curv_func(a_smooth, **curv_kw)
        # print(np.isnan(curv))
        curv /= (np.abs(a_smooth) + np.std(a_smooth))

        # print('std curv:', np.std(curv))
        if np.std(curv) == 0:
            sigma = np.ones(a.shape) * max_sigma
        else:
            curv /= np.std(curv)
            curv = ndimage.gaussian_filter(np.abs(curv), presmooth_sigma)

            # As k_factor increases, sigma becomes less sensitive to local curvature
            c = k_factor / (max_sigma ** 2)
            sigma = (k_factor / (np.abs(curv) + c)) ** 0.5
    else:
        sigma = np.zeros_like(a)

    # print(sigma)
    return sigma


def get_adaptive_sigmas(a, presmooth_sigma=None, empty=False, weights=None,
                        curv_func=None, curv_kw=None, k_factor=1.0, max_sigma=1.0,
                        mode='reflect', cval=0.0, truncate=4.0):
    axes = np.arange(np.ndim(a), dtype=int)

    if np.isscalar(k_factor):
        k_factor = [k_factor] * len(axes)
    if np.isscalar(max_sigma):
        max_sigma = [max_sigma] * len(axes)

    if presmooth_sigma is None:
        presmooth_sigma = max_sigma

    sigmas = []
    for i, axis in enumerate(axes):
        sigma = get_adaptive_sigma1d(a, axis, presmooth_sigma, empty, weights, curv_func, curv_kw,
                                     k_factor[i], max_sigma[i], mode, cval, truncate)
        sigmas.append(sigma)

    return sigmas


def adaptive_gaussian_filter1d(a, sigma=None, axis=-1, presmooth_sigma=1, empty=False,
                               curv_func=None, curv_kw=None, k_factor=1, max_sigma=1.0,
                               mode='reflect', cval=0.0, truncate=4, order=0, sigma_node_factor=1.5):

    # Determine sigma from curvature
    if sigma is None:
        sigma = get_adaptive_sigma1d(a, axis, presmooth_sigma, empty, None,
                                     curv_func, curv_kw, k_factor, max_sigma, mode,
                                     cval, truncate)

    return nonuniform_gaussian_filter1d(a, sigma, axis, empty, mode, cval, truncate, order, sigma_node_factor)


def adaptive_gaussian_filter(a, sigmas=None, presmooth_sigma=None, empty=False,
                             curv_func=None, curv_kw=None, k_factor=1, max_sigma=5,
                             mode='reflect', cval=0.0, truncate=4, order=0, sigma_node_factor=1.5):
    axes = np.arange(np.ndim(a), dtype=int)

    if np.isscalar(k_factor):
        k_factor = [k_factor] * len(axes)
    if np.isscalar(max_sigma):
        max_sigma = [max_sigma] * len(axes)

    if sigmas is None:
        sigmas = [None] * len(axes)

    if presmooth_sigma is None:
        presmooth_sigma = max_sigma

    # Apply sequence of 1d filters
    out = a
    for i, axis in enumerate(axes):
        if max_sigma[i] > 0:
            out = adaptive_gaussian_filter1d(out, sigmas[i], axis, presmooth_sigma, empty,
                                             curv_func, curv_kw, k_factor[i], max_sigma[i],
                                             mode, cval, truncate, order, sigma_node_factor)

    return out

# def adaptive_gaussian_filter1d(a, sigma=None, axis=-1, mode='reflect', cval=0.0, truncate=4):
#
#     radius = int(truncate * np.max(sigma))
#     filter_size = 2 * radius + 1
#
#     a_size = np.shape(a)[axis]
#     split_index = int(a_size / 2)
#     a_in = np.concatenate([a, sigma], axis=axis)
#
#     counter = 0
#     Problem: no way to access corresponding line of sigma with func
#
#     def func(x, out):
#         xlen = len(x) - (filter_size - 1)
#         alen = xlen / 2
#         aline = x[:alen + radius]
#         sigmaline = x[]
#         i = np.arange(len(x))
#         j = np.arange(-radius, len(x) + radius)
#         jj, ii = np.meshgrid(j, i)
#         jj, ss = np.meshgrid(j, sigma)
#         A = np.exp(-0.5 * ((ii - jj) / ss) ** 2)
#         A /= np.sum(A, axis=1)
#         out[:] = A @ x
#         counter += 1
#
#     ndimage.generic_filter1d(a, func, filter_size=filter_size, axis=axis, mode=mode, cval=cval)


def apply_filter(x_in, filter_func=None, filter_kw=None):
    # TODO: is this necessary?
    if filter_kw is None:
        if filter_func is None:
            sigma = np.ones(np.ndim(x_in))
            sigma[-1] = 0
            filter_kw = {'sigma': sigma}
        else:
            filter_kw = None
    if filter_func is None:
        filter_func = ndimage.gaussian_filter

    return filter_func(x_in, **filter_kw)
