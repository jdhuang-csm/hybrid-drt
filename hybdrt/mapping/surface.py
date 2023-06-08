import numpy as np
from scipy import signal, ndimage
from skimage import filters  # import sato, meijering, apply_hysteresis_threshold

from . import ndx
from ..filters import std_filter, signed_hysteresis_threshold, rms_filter
from ..models.peaks import estimate_peak_weight_distributions  # find_troughs as find_trough_indices
from ..utils import stats
from ..utils.array import nearest_index


def find_ridges_2d(p_ridge, distance=3, height=0.1, prominence=0.1, **kw):
    ridge_mask = np.zeros_like(p_ridge, dtype=bool)

    for i in range(p_ridge.shape[0]):
        peaks, _ = signal.find_peaks(p_ridge[i], distance=distance, height=height,
                                     prominence=prominence, **kw)
        ridge_mask[i, peaks] = 1

    return ridge_mask


def find_ridges(p_ridge, num_group_dims, **kw):
    # num_group_dims = np.ndim(p_ridge) - 2

    ridge_mask = ndx.filter_ndx(p_ridge, num_group_dims, mask_nans=False, by_group=True,
                                filter_func=find_ridges_2d, **kw)
    ridge_mask = np.nan_to_num(ridge_mask).astype(bool)
    return ridge_mask


# def find_troughs_1d(trough_prob, peak_mask):
#     trough_mask = np.zeros_like(peak_mask)
#
#     peaks = np.where(peak_mask > 0)[0]
#
#     if np.sum(peaks) > 0:
#         trough_mask[0] = 1
#         trough_mask[-1] = 1
#
#     for i, peak in enumerate(peaks[:-1]):
#         next_peak = peaks[i + 1]
#         trough_index = peak + 1 + np.argmax(trough_prob[peak + 1: next_peak])
#         trough_mask[trough_index] = 1
#
#     return trough_mask
#
#
# def find_troughs_2d(trough_prob, ridge_mask):
#     return np.stack([find_troughs_1d(trough_prob[i], ridge_mask[i]) for i in range(trough_prob.shape[0])],
#                     axis=0)


def find_troughs_1d(ridge_mask, f, p_trough):
    trough_mask = np.zeros_like(ridge_mask)

    peaks = np.where(ridge_mask)[0]

    # p_tot = p_trough * (1 - p_peak)

    for i, start_index in enumerate(peaks[:-1]):
        end_index = peaks[i + 1]

        # Determine peak signs
        left_sign = np.sign(f[start_index])
        right_sign = np.sign(f[end_index])

        if left_sign == right_sign:
            # Both peaks are same sign. Find max trough prob between peaks
            trough_index = start_index + np.argmax(p_trough[start_index:end_index])
            # # If max curvature is at an endpoint, go halfway between the max curvature and the midpoint of the peaks
            # if trough_index in (start_index, end_index):
            #     trough_index = int((start_index + end_index + 2 * trough_index) / 4)
        else:
            # Sign changes. Set the trough at the zero between the peaks
            zero_index = nearest_index(f[start_index:end_index], 0)
            trough_index = start_index + zero_index

        trough_mask[trough_index] = 1

    return trough_mask


# # TODO: is there any reason to have this 2d function? could just loop through array of arbitrary dimension
# def find_troughs_2d(f, fxx, ridge_mask):
#     return np.stack([find_troughs_1d(f[i], fxx[i], ridge_mask[i]) for i in range(f.shape[0])],
#                     axis=0)

def find_troughs_2d(ridge_mask, f, p_trough):
    return np.stack([find_troughs_1d(ridge_mask[i], f[i], p_trough[i]) for i in range(f.shape[0])],
                    axis=0)


def find_troughs(ridge_mask, f, p_trough, num_group_dims):
    it = np.nditer(f, op_axes=[list(np.arange(num_group_dims))], flags=['multi_index'])
    trough_mask = np.zeros_like(ridge_mask)

    for group in it:
        group_index = it.multi_index
        # trough_mask[group_index] = find_troughs_2d(f[group_index], fxx[group_index], ridge_mask[group_index])
        trough_mask[group_index] = find_troughs_2d(ridge_mask[group_index], f[group_index],
                                                   p_trough[group_index]
                                                   )
    return trough_mask


def integrate_ridges_1d(x, ridge_mask, trough_mask, tau=None, epsilon_factor=1.25, max_epsilon=1.25,
                        epsilon_uniform=None):
    # TODO: return FWHM
    if tau is None:
        # Arbitrary grid at 10 ppd
        tau = np.log10(1 + np.arange(len(x)) * 0.1)

    peaks = np.where(ridge_mask)[0]
    troughs = np.where(trough_mask)[0]

    weights = estimate_peak_weight_distributions(tau, x, None, peaks, basis_tau=tau, trough_indices=troughs,
                                                 epsilon_factor=epsilon_factor,
                                                 max_epsilon=max_epsilon,
                                                 epsilon_uniform=epsilon_uniform)

    f_peaks = x[None, :] * weights

    peak_area = np.zeros(len(x))
    peak_area[peaks] = np.sum(f_peaks, axis=1)

    return peak_area


def integrate_ridges_2d(x, ridge_mask, trough_mask, tau=None, epsilon_factor=1.25, max_epsilon=1.25,
                        epsilon_uniform=None):
    out = np.stack(
        [integrate_ridges_1d(x[i], ridge_mask[i], trough_mask[i], tau, epsilon_factor, max_epsilon, epsilon_uniform)
         for i in range(x.shape[0])],
        axis=0
    )
    return out


def integrate_ridges(x, ridge_mask, trough_mask, num_group_dims, tau=None,
                     epsilon_factor=1.25, max_epsilon=1.25, epsilon_uniform=None):
    it = np.nditer(x, op_axes=[list(np.arange(num_group_dims))], flags=['multi_index'])
    ridge_area = np.zeros(ridge_mask.shape)

    for group in it:
        group_index = it.multi_index
        ridge_area[group_index] = integrate_ridges_2d(x[group_index], ridge_mask[group_index],
                                                      trough_mask[group_index], tau=tau, epsilon_factor=epsilon_factor,
                                                      max_epsilon=max_epsilon, epsilon_uniform=epsilon_uniform
                                                      )
    return ridge_area


def coef_to_ridges(x, drtmd, num_group_dims, normalize=True,
                   ridge_filter=False, gmean_filter=False, ndx_filter=True, filter_kw=None,
                   std_size=5, std_baseline=0.1, ridge_repulse_distance=2,
                   hysteresis_threshold=True, thresh_low=0.2, thresh_high=0.75,
                   find_ridges_kw=None,
                   epsilon_factor=1.25, max_epsilon=1.25, epsilon_uniform=None
                   ):

    if ndx_filter and filter_kw is None:
        filter_kw = dict(
            iterative=True, iter=3, nstd=5, dev_rms_size=5,
            adaptive=True, impute=False, impute_groups=False,
            by_group=True,
            max_sigma=(0.5, 0.5),
            k_factor=(4, 2),
            presmooth_sigma=None,
            mode='nearest'
        )

    if find_ridges_kw is None:
        find_ridges_kw = {}

    if normalize:
        x_sum = np.nansum(np.abs(x), axis=-1)
        x_sum[x_sum == 0] = 1
        x_norm = x / x_sum[..., None]

    else:
        x_norm = x

    # First get ridge prob
    # Use normalized values for ridge prob
    f = drtmd.predict_drt(psi=None, x=x_norm, tau=drtmd.tau_supergrid)
    fx = drtmd.predict_drt(psi=None, x=x_norm, tau=drtmd.tau_supergrid, order=1)
    fxx = drtmd.predict_drt(psi=None, x=x_norm, tau=drtmd.tau_supergrid, order=2)

    cp = peak_prob(f, fx, fxx, std_size=std_size, std_baseline=std_baseline)
    tp = trough_prob(f, fx, fxx, std_size=std_size, std_baseline=std_baseline)
    p_ridge = cp * (1 - tp)
    p_trough = tp * (1 - cp)

    if ndx_filter:
        p_ridge = ndx.filter_ndx(p_ridge, num_group_dims, **filter_kw)
        p_trough = ndx.filter_ndx(p_trough, num_group_dims, **filter_kw)

    if ridge_filter:
        if gmean_filter:
            p_ridge *= ridge_prob_filter(p_ridge, num_group_dims)
            p_ridge **= 0.5
        else:
            p_ridge = ridge_prob_filter(p_ridge, num_group_dims)

    # calculate trough prob after filtering but before thresholding
    # p_trough = 1 - p_ridge

    if hysteresis_threshold:
        mask = filters.apply_hysteresis_threshold(p_ridge, thresh_low, thresh_high)
        p_ridge[~mask] = 0

        # mask = filters.apply_hysteresis_threshold(p_trough, thresh_low, thresh_high)
        # p_trough[~mask] = 0

    # if ndx_filter:
    #     p_ridge = ndx.filter_ndx(p_ridge, num_group_dims, **filter_kw)

    # Find ridges and troughs
    ridge_mask = find_ridges(p_ridge, num_group_dims, **find_ridges_kw)

    # Repel troughs from ridge locations
    if ridge_repulse_distance > 0:
        # radius = int(ridge_repulse_sigma * 4)
        # x_kernel = np.arange(-radius, radius + 1)
        # kernel = np.exp(-0.5 * (x_kernel / ridge_repulse_sigma) ** 6)
        # kernel /= np.sum(kernel)
        # print(x_kernel, kernel)
        # ridge_repulse = ndimage.convolve1d(ridge_mask.astype(float), kernel, axis=-1)

        # ridge_repulse = ndimage.gaussian_filter1d(ridge_mask.astype(float), sigma=ridge_repulse_sigma, axis=-1)
        # ridge_repulse /= np.max(ridge_repulse)

        ridge_repulse = ndimage.uniform_filter1d(ridge_mask.astype(float), size=2 * ridge_repulse_distance + 1,
                                                 axis=-1)
        ridge_repulse[ridge_repulse > 0] = 1

        p_trough *= (1 - ridge_repulse)

    if ridge_filter:
        # Filter p_trough after applying ridge_repulse
        if gmean_filter:
            p_trough *= ridge_prob_filter(p_trough, num_group_dims)
            p_trough **= 0.5
        else:
            p_trough = ridge_prob_filter(p_trough, num_group_dims)

    # if ndx_filter:
    #     p_trough = ndx.filter_ndx(p_trough, num_group_dims, **filter_kw)

    trough_mask = find_troughs(ridge_mask, f, p_trough, num_group_dims)

    # Integrate ridge area (unnormalized)
    ridge_area = integrate_ridges(x, ridge_mask, trough_mask, num_group_dims, tau=drtmd.tau_supergrid,
                                  epsilon_factor=epsilon_factor, max_epsilon=max_epsilon,
                                  epsilon_uniform=epsilon_uniform
                                  )

    return p_ridge, p_trough, ridge_mask, trough_mask, ridge_area


# -----------------------
# Probability functions
# -----------------------
def peak_prob(f, fx, fxx, std_size=5, f_var=None, fx_var=None, fxx_var=None, constrain_sign=False, std_baseline=0.1):
    # f = mrt.predict_drt(None, tau=mrt.tau_supergrid, x=x, order=0)
    # fx = mrt.predict_drt(None, tau=mrt.tau_supergrid, x=x, order=1)
    # fxx = mrt.predict_drt(None, tau=mrt.tau_supergrid, x=x, order=2)

    nan_mask = np.isnan(f)

    if f_var is None:
        f_std = std_filter(np.nan_to_num(f), size=std_size, mask=(~nan_mask).astype(float))
        f_std += std_baseline * np.std(f[~nan_mask])
        # f_std = (f_std ** 2 + (std_baseline * np.std(f[~nan_mask])) ** 2) ** 0.5
    else:
        f_std = f_var ** 0.5

    if fx_var is None:
        fx_std = std_filter(np.nan_to_num(fx), size=std_size, mask=(~nan_mask).astype(float))
        fx_std += std_baseline * np.std(fx[~nan_mask])
        # fx_std = (fx_std ** 2 + (std_baseline * np.std(fx[~nan_mask])) ** 2) ** 0.5
    else:
        fx_std = fx_var ** 0.5

    if fxx_var is None:
        fxx_std = std_filter(np.nan_to_num(fxx), size=std_size, mask=(~nan_mask).astype(float))
        fxx_std += std_baseline * np.std(fxx[~nan_mask])
        # fxx_std = (fxx_std ** 2 + (std_baseline * np.std(fxx[~nan_mask])) ** 2) ** 0.5
    else:
        fxx_std = fxx_var ** 0.5

    if constrain_sign:
        # Require the median curvature to have the opposite sign of the median DRT
        fxx_prob = 1 - 2 * stats.cdf_normal(0, -np.sign(f) * fxx, fxx_std)
        fxx_prob[fxx_prob < 0] = 0
    else:
        fxx_prob = 1 - stats.cdf_normal(0, -np.sign(f) * fxx, fxx_std)

    # Prob that fx is within 3 std of zero
    fx_prob = stats.cdf_normal(5 * fx_std, fx, fx_std) - stats.cdf_normal(-5 * fx_std, fx, fx_std)

    # f_prob = (1 - stats.cdf_normal(0, np.abs(f), f_std))  # Prob that f has apparent sign
    f_prob = (1 - stats.cdf_normal(1 * f_std, np.abs(f), f_std))  # Prob that f is more than 2 std away from zero

    cp = f_prob * fx_prob * fxx_prob
    # cp = fxx_prob * f_prob

    return cp


def trough_prob(f, fx, fxx, f_var=None, fx_var=None, fxx_var=None, std_size=5, std_baseline=0.1):
    nan_mask = np.isnan(f)

    if f_var is None:
        f_std = std_filter(np.nan_to_num(f), size=std_size, mask=(~nan_mask).astype(float))
        f_std += std_baseline * np.std(f[~nan_mask])
        # f_std = (f_std ** 2 + (std_baseline * np.std(f[~nan_mask])) ** 2) ** 0.5
    else:
        f_std = f_var ** 0.5

    if fx_var is None:
        fx_std = std_filter(np.nan_to_num(fx), size=std_size, mask=(~nan_mask).astype(float))
        fx_std += std_baseline * np.std(fx[~nan_mask])
        # fx_std = (fx_std ** 2 + (std_baseline * np.std(fx[~nan_mask])) ** 2) ** 0.5
    else:
        fx_std = fx_var ** 0.5

    if fxx_var is None:
        fxx_std = std_filter(np.nan_to_num(fxx), size=std_size, mask=(~nan_mask).astype(float))
        fxx_std += std_baseline * np.std(fxx[~nan_mask])
        # fxx_std = (fxx_std ** 2 + (std_baseline * np.std(fxx[~nan_mask])) ** 2) ** 0.5
    else:
        fxx_std = fxx_var ** 0.5

    # # Prob that f is within 3 std of zero
    # f_prob = stats.cdf_normal(3 * f_std, f, f_std) - stats.cdf_normal(-3 * f_std, f, f_std)

    # Prob that fx is within 3 std of zero
    fx_prob = stats.cdf_normal(5 * fx_std, fx, fx_std) - stats.cdf_normal(-5 * fx_std, fx, fx_std)
    # Exclude peaks
    # fx_prob[np.sign(fxx) == -np.sign(f)] = 0
    # fx_prob *= (1 - stats.cdf_normal(0, np.sign(fxx) * f, f_std))

    # Prob that curvature has same sign as function (excludes peaks)
    fxx_prob = (1 - stats.cdf_normal(0, np.sign(f) * fxx, fxx_std))

    # tp = f_prob * fx_prob * fxx_prob
    tp = fx_prob * fxx_prob
    return tp


def ridge_prob_filter(prob, num_group_dims, troughs=False, pad=3, sato=True, meijering=True,
                      aggregate='min'):
    if not max(sato, meijering):
        raise ValueError("At least one of sato or meijering must be set to True")

    if pad > 0:
        prob_pad = np.empty((*prob.shape[:-2], prob.shape[-2] + 2 * pad, prob.shape[-1]))
        # print(prob.shape, prob_pad.shape)
        prob_pad[..., pad:-pad, :] = prob
        prob_pad[..., :pad, :] = np.expand_dims(prob[..., 0, :], -2)
        prob_pad[..., -pad:, :] = np.expand_dims(prob[..., -1, :], -2)
    else:
        prob_pad = prob

    # Filter curv_prob
    probs = []
    if sato:
        cp_1 = ndx.filter_ndx(prob_pad, num_group_dims=num_group_dims, filter_func=filters.sato,
                              by_group=True, mask_nans=False, black_ridges=troughs,
                              sigmas=np.arange(0.25, 5, 1), mode='nearest'
                              )
        cp_1 = cp_1 / np.nanpercentile(cp_1, 99)
        cp_1[cp_1 > 1] = 1
        probs.append(cp_1)

    if meijering:
        cp_2 = ndx.filter_ndx(prob_pad, num_group_dims=num_group_dims, filter_func=filters.meijering,
                              by_group=True, mask_nans=False,
                              black_ridges=troughs, sigmas=np.arange(0.25, 5, 1), mode='nearest'
                              )
        cp_2 = cp_2 / np.nanpercentile(cp_2, 99)
        cp_2[cp_2 > 1] = 1
        probs.append(cp_2)

    if len(probs) > 1:
        probs = np.stack(probs, axis=0)
        if aggregate == 'gmean':
            cp_filt = np.prod(probs, axis=0) ** (1 / len(probs))
        else:
            cp_filt = getattr(np, aggregate)(probs, axis=0)

    else:
        cp_filt = probs[0]

    if pad > 0:
        cp_filt = cp_filt[..., pad:-pad, :]

    return cp_filt


def ridge_prob(f, fx, fxx, num_group_dims, subtract_troughs=True,
               std_baseline=0.1, std_size=5,
               ridge_filter=False, ndx_filter=True, filter_kw=None,
               hysteresis_threshold=True, thresh_low=0.2, thresh_high=0.75):
    rp = peak_prob(f, fx, fxx, std_size=std_size, std_baseline=std_baseline)

    if ndx_filter and filter_kw is None:
        filter_kw = dict(
            iterative=True, iter=3, nstd=5, dev_rms_size=5,
            adaptive=True, impute=True, impute_groups=True,
            max_sigma=(1,) * num_group_dims + (0.5, 0),
            k_factor=4,
            presmooth_sigma=None,
            mode='nearest'
        )

    if ridge_filter:
        rp = ridge_prob_filter(rp, num_group_dims)

    if subtract_troughs:
        tp = trough_prob(f, fx, fxx, std_baseline=std_baseline, std_size=std_size)
        if ridge_filter:
            tp = ridge_prob_filter(tp, num_group_dims)
        rp = rp - tp

    if ndx_filter:
        rp = ndx.filter_ndx(rp, num_group_dims, **filter_kw)

    if hysteresis_threshold:
        # Mask negatives
        rp[rp < 0] = 0
        # Apply hysteresis thresholds to ridges of same sign
        thresh = signed_hysteresis_threshold(rp * np.sign(f), thresh_low, thresh_high)
        rp[~thresh] = 0

    return rp

# def ridge_prob(f, fx, fxx, num_group_dims, subtract_troughs=True,
#                std_baseline=0.1, std_size=5,
#                ndx_filter=True, filter_kw=None,
#                hysteresis_threshold=True, thresh_low=0.2, thresh_high=0.75):
#     prob = peak_prob(f, fx, fxx, std_baseline=std_baseline, std_size=std_size)
#
#     if ndx_filter and filter_kw is None:
#         filter_kw = dict(
#             iterative=True, iter=3, nstd=5, dev_rms_size=5,
#             adaptive=True, impute=True, impute_groups=True,
#             max_sigma=(1,) * num_group_dims + (0.5, 0),
#             k_factor=4,
#             presmooth_sigma=None,
#             mode='nearest'
#         )
#
#     rp = ridge_prob_filter(prob, num_group_dims)
#     if hysteresis_threshold:
#         # Apply hysteresis thresholds to ridges of same sign
#         thresh = signed_hysteresis_threshold(rp * np.sign(f), thresh_low, thresh_high)
#         rp[~thresh] = 0
#
#     if ndx_filter:
#         rp = ndx.filter_ndx(rp, num_group_dims, **filter_kw)
#
#     if subtract_troughs:
#         # tp = trough_prob(f, fx, fxx, std_baseline=std_baseline, std_size=std_size)
#         # tp = ridge_prob_filter(tp, num_group_dims)
#         tp = ridge_prob_filter(prob, num_group_dims, troughs=True)
#
#         if hysteresis_threshold:
#             thresh = apply_hysteresis_threshold(tp, low=thresh_low, high=thresh_high)
#             tp[~thresh] = 0
#
#         if ndx_filter:
#             tp = ndx.filter_ndx(tp, num_group_dims, **filter_kw)
#
#         return rp - tp
#     else:
#         return rp
