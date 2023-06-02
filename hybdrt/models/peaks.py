import numpy as np
from scipy import signal
from ..matrices import basis

from .. import utils


def find_slope_peak_pairs(fx, **kw):
    # Find pairs of positive and negative peaks in the 1st derivative
    pos_peaks, _ = signal.find_peaks(fx, **kw)
    neg_peaks, _ = signal.find_peaks(-fx, **kw)

    if len(pos_peaks) == 0 and len(neg_peaks) == 0:
        pass
    elif len(pos_peaks) == 0:
        # Negative peak identified without positive peak.
        # Assume that an unidentified positive peak precedes the negative peak
        pos_peaks = np.array([0])
    elif len(neg_peaks) == 0:
        # POsitive peak identified without negative peak.
        # Assume that an unidentified negative peak follows the positive peak
        neg_peaks = np.array([-1])
    else:
        # If first peak is negative, assume that a positive peak precedes it but is not captured in the range
        if neg_peaks[0] < pos_peaks[0]:
            pos_peaks = np.insert(pos_peaks, 0, 0)
        # If last peak is positive, assume that a negative peak follows it but is not captured in the range
        if pos_peaks[-1] > neg_peaks[-1]:
            neg_peaks = np.append(neg_peaks, len(fx) - 1)

    return pos_peaks, neg_peaks


def find_peaks_simple(data, order, **kw):
    if order == 0:
        # Find peaks in the function
        f = data
        peaks, properties = signal.find_peaks(f, **kw)
    elif order == 1:
        fx, fxx = data

        if 'delta_fx' in kw:
            delta_fx_thresh = kw.pop('delta_fx')
        else:
            delta_fx_thresh = 0

        # Find pairs of positive and negative peaks in the 1st derivative
        pos_peaks, neg_peaks = find_slope_peak_pairs(fx, **kw)
        if len(pos_peaks) == 0:
            peaks = np.array([])
        else:
            # Only keep peak pairs which exceed the slope change threshold
            delta_fx = fx[pos_peaks] - fx[neg_peaks]
            pos_peaks = pos_peaks[delta_fx > delta_fx_thresh]
            neg_peaks = neg_peaks[delta_fx > delta_fx_thresh]

            # Find the minimum curvature value between each pair of 1st derivative peaks
            peaks = np.array([pos + np.argmin(fxx[pos:neg + 1]) for pos, neg in zip(pos_peaks, neg_peaks)])

    elif order == 2:
        fxx = data
        if 'height' not in kw.keys():
            kw['height'] = 0
        # Find negative peaks in the 2nd derivative of the function to identify shoulder peaks
        peaks, properties = signal.find_peaks(-fxx, **kw)
    else:
        raise ValueError(f'order must be in [0, 1, 2]. Received value {order}')

    return peaks


def find_peaks_compound(fx, fxx, order1_kw=None, order2_kw=None):
    # print(np.percentile(np.abs(fx), 100) * 0.05, np.percentile(np.abs(fxx), 100) * 0.05)
    if order1_kw is None:
        order1_kw = {'prominence':  1e-3 + np.percentile(np.abs(fx[~np.isinf(fx)]), 100) * 0.01,
                     'delta_fx': 1e-3 + np.percentile(np.abs(fxx[~np.isinf(fxx)]), 90) * 0.05}

    if order2_kw is None:
        order2_kw = {'prominence': 5e-3 + np.percentile(np.abs(fxx[~np.isinf(fxx)]), 100) * 0.01, 'height': 0}

    peaks_order1 = find_peaks_simple((fx, fxx), order=1, **order1_kw)

    peaks_order2 = find_peaks_simple(fxx, order=2, **order2_kw)

    peaks = np.intersect1d(peaks_order1, peaks_order2)

    return peaks


def find_troughs(f, fxx, peak_indices):
    # If left and right peaks have same sign, use existing logic (after accounting for sign)
    # If left and right peaks have different signs, find zero between them
    trough_indices = []
    f_mix = -(f - fxx)
    peak_indices = sorted(peak_indices)
    for i, start_index in enumerate(peak_indices[:-1]):
        end_index = peak_indices[i + 1]

        # Determine peak signs
        left_sign = np.sign(f[start_index])
        right_sign = np.sign(f[end_index])

        if left_sign == right_sign:
            # Both peaks are same sign
            sign = left_sign
            # print(start_index, np.min(sign * f[start_index:end_index]), min(sign * f[start_index], sign * f[end_index]))
            if np.min(sign * f[start_index:end_index]) < min(sign * f[start_index], sign * f[end_index]):
                # If there is a local minimum between the peaks, use this as the trough
                trough_index = start_index + np.argmin(sign * f[start_index:end_index])
                # print(start_index, 'local min')
            else:
                # If no local minimum, use the min of (f - fxx) to locate the trough
                trough_index = start_index + np.argmax(sign * f_mix[start_index:end_index])
                # If max curvature is at an endpoint, go halfway between the max curvature and the midpoint of the peaks
                if trough_index in (start_index, end_index):
                    trough_index = int((start_index + end_index + 2 * trough_index) / 4)
        else:
            # Sign changes. Set the trough at the zero between the peaks
            zero_index = utils.array.nearest_index(f[start_index:end_index], 0)
            trough_index = start_index + zero_index

        trough_indices.append(trough_index)

    return trough_indices


def estimate_peak_weight_distributions(tau, f, fxx, peak_indices, basis_tau, epsilon_factor=1.25, max_epsilon=1.25,
                                       epsilon_uniform=None, trough_indices=None):
    # print(len(peak_indices), len(trough_indices))
    if len(peak_indices) > 1:
        peak_indices = sorted(peak_indices)
        # Get RBF for weighting function
        rbf = basis.get_basis_func('gaussian')

        # Get weighting function for each peak
        peak_weights = np.empty((len(peak_indices), len(basis_tau)))
        # peak_iter = iter(peak_indices)
        # prev_index = 0
        # peak_index = next(peak_iter)

        if trough_indices is None:
            trough_indices = find_troughs(f, fxx, peak_indices)

        for i in range(len(peak_indices)):
            peak_index = peak_indices[i]
            # next_index = next(peak_iter, -1)
            if epsilon_uniform is None:
                # Estimate RBF length scale based on distance to next peak
                # l_epsilon = 2.5 / np.log(tau[peak_index] / tau[prev_index])
                # r_epsilon = 2.5 / np.log(tau[next_index] / tau[peak_index])
                # print(np.log(tau[prev_index]), np.log(tau[peak_index]), np.log(tau[next_index]))
                # print(l_epsilon, r_epsilon)
                if i == 0:
                    prev_index = 0
                else:
                    prev_index = trough_indices[i - 1]

                if i == len(peak_indices) - 1:
                    next_index = -1
                else:
                    next_index = trough_indices[i]

                l_epsilon = min(epsilon_factor / np.log(tau[peak_index] / tau[prev_index]), max_epsilon)
                r_epsilon = min(epsilon_factor / np.log(tau[next_index] / tau[peak_index]), max_epsilon)
                # print(np.log(tau[prev_index]), np.log(tau[peak_index]), np.log(tau[next_index]))
                # print(l_epsilon, r_epsilon)
            else:
                l_epsilon = epsilon_uniform
                r_epsilon = epsilon_uniform

            peak_weights[i, basis_tau < tau[peak_index]] = rbf(
                np.log(basis_tau[basis_tau < tau[peak_index]] / tau[peak_index]), l_epsilon
            )
            peak_weights[i, basis_tau >= tau[peak_index]] = rbf(
                np.log(basis_tau[basis_tau >= tau[peak_index]] / tau[peak_index]), r_epsilon
            )

            # prev_index = peak_index
            # peak_index = next_index

        # Normalize to total weight
        peak_weights /= np.sum(peak_weights, axis=0)
    else:
        peak_weights = np.ones((len(peak_indices), len(basis_tau)))

    return peak_weights


def squeeze_peak_coef(x_peak, basis_tau, squeeze_factor):
    # Identify peak center
    max_index = np.argmax(x_peak)
    tau_max = basis_tau[max_index]

    # Get original and squeezed basis grids in log space
    ln_tau = np.log(basis_tau)
    ln_tau_sqz = np.log(tau_max) + (ln_tau - np.log(tau_max)) / squeeze_factor

    # Interpolate new coefficients from squeezed space
    # Scale by squeeze factor to maintain peak area
    x_sqz = squeeze_factor * np.interp(ln_tau, ln_tau_sqz, x_peak)

    return x_sqz


def estimate_peak_params(tau, element_types, f=None, peak_indices=None, trough_indices=None, f_peaks=None):
    if f is not None and f_peaks is not None:
        raise ValueError('Only one of f or f_peaks should be provided')
    elif f is not None and peak_indices is None:
        raise ValueError('If f is provided, peak_indices must also be provided')
    elif f is None and f_peaks is None:
        raise ValueError('Either (f AND peak_indices) OR f_peaks must be provided')
    elif f is not None:
        num_peaks = len(peak_indices)
    else:
        num_peaks = len(f_peaks)

    # Check and format element_types
    if type(element_types) == list:
        # Different element type for each peak: ensure number of elements matches number of peaks
        if len(element_types) != num_peaks:
            raise ValueError(f'Length of element_types ({len(element_types)}) '
                             f'does not match length of peaks ({len(peak_indices)})')
    else:
        # Single element type: convert to list
        element_types = [element_types] * num_peaks

    peak_params = []

    r_tot = 0

    if f is not None:
        # Use f and peak indices to estimate peak parameters
        if trough_indices is None:
            trough_indices = [int(np.mean([peak_indices[i - 1], peak_indices[i]])) for i in range(1, len(peak_indices))]
        start_indices = [0] + trough_indices
        end_indices = np.array(trough_indices + [len(tau)]) + 1

        # print(start_indices, end_indices)

        # Estimate parameters for each peak
        for i, peak_index in enumerate(peak_indices):
            # 1. Estimate R. Integrate area from trough to trough (trough assumed to be halfway between peaks)
            start_index = start_indices[i]
            end_index = end_indices[i]

            r_k = np.trapz(f[start_index:end_index], x=np.log(tau[start_index:end_index]))
            r_tot += r_k
            # print(i, r_k)

            # 2. Estimate dispersion parameter for HN and ZARC elements
            if element_types[i] in ('HN', 'RQ'):
                beta_k = (2 / np.pi) * np.arctan2(2 * np.pi * abs(f[peak_index]), abs(r_k))

                if element_types[i] == 'HN':
                    r_left = abs(np.trapz(f[start_index:peak_index], x=np.log(tau[start_index:peak_index])))
                    r_right = abs(np.trapz(f[peak_index:end_index], x=np.log(tau[peak_index:end_index])))
                    if r_right >= r_left:
                        alpha_k = 0.99
                    else:
                        alpha_k = (r_right / r_left) ** ((1 - beta_k) / (2 * beta_k))
                    # print('alpha:', alpha_k)

                    # # Assume symmetric
                    # alpha_k = 0.99

                    params = [r_k, np.log(tau[peak_index]), alpha_k, beta_k]
                else:
                    params = [r_k, np.log(tau[peak_index]), beta_k]
            elif element_types[i] == 'RC':
                params = [r_k, np.log(tau[peak_index])]
            else:
                raise ValueError(f'Invalid element_type {element_types[i]}')

            # print(i, params)

            peak_params.append(params)
    else:
        for i, f_peak in enumerate(f_peaks):
            # 1. Find maximum of f_peak
            peak_index = np.argmax(np.abs(f_peak))
            # peak_index = peak_indices[i]

            # 2. Integrate peak area to get R
            r_k = np.trapz(f_peak, x=np.log(tau))
            r_tot += r_k
            # print('r:', r_k)

            # 3. Estimate dispersion parameter for HN and ZARC elements
            if element_types[i] in ('HN', 'RQ'):
                beta_k = (2 / np.pi) * np.arctan2(2 * np.pi * abs(f_peak[peak_index]), abs(r_k))
                # print('beta:', beta_k)

                if element_types[i] == 'HN':
                    r_left = abs(np.trapz(f_peak[:peak_index], x=np.log(tau[:peak_index])))
                    r_right = abs(np.trapz(f_peak[peak_index:], x=np.log(tau[peak_index:])))
                    if r_right >= r_left:
                        alpha_k = 0.99
                    else:
                        alpha_k = (r_right / r_left) ** ((1 - beta_k) ** 0.1 / (2 * beta_k))
                    # print('alpha:', alpha_k)

                    # # Assume symmetric
                    # alpha_k = 0.99

                    params = [r_k, np.log(tau[peak_index]), alpha_k, beta_k]
                else:
                    params = [r_k, np.log(tau[peak_index]), beta_k]
            elif element_types[i] == 'RC':
                params = [r_k, np.log(tau[peak_index])]
            else:
                raise ValueError(f'Invalid element_type {element_types[i]}')

            # print(i, params)

            peak_params.append(params)

    return peak_params


def min_peak_distances(new_peak_locations, base_peak_locations):
    """
    For each peak in new_peak_locations, return the minimum distance to a peak in base_peak_locations.
    Distances will be compared in the provided location space; if log spacing is desired (recommended),
    provide ln(tau) for peak locations.
    :param new_peak_locations: Locations of new peaks
    :param base_peak_locations: Locations of base peaks
    :return:
    """
    if len(base_peak_locations) == 0:
        return np.ones(len(new_peak_locations)) * np.inf
    else:
        def min_distance(peak, compare_peaks):
            return np.min(np.abs(peak - compare_peaks))

        min_dist = [min_distance(new_peak, base_peak_locations) for new_peak in new_peak_locations]

        return np.array(min_dist)


def index_closest_peaks(new_peak_locations, base_peak_locations):
    def argmin_distance(peak, compare_peaks):
        return np.argmin(np.abs(peak - compare_peaks))

    min_dist = [argmin_distance(new_peak, base_peak_locations) for new_peak in new_peak_locations]

    return np.array(min_dist)


def peak_similarity_index(new_peak_locations, base_peak_locations, epsilon=1):
    min_dist = min_peak_distances(new_peak_locations, base_peak_locations)
    rbf = basis.get_basis_func('gaussian')
    return rbf(min_dist, epsilon)


def find_new_peaks(new_peak_locations, base_peak_locations, distance_threshold=None):
    """
    Find peaks in new_peak_locations that are not contained in base_peak_locations
    :param new_peak_locations:
    :param base_peak_locations:
    :param distance_threshold:
    :return:
    """
    min_dist = min_peak_distances(new_peak_locations, base_peak_locations)

    num_new = len(new_peak_locations) - len(base_peak_locations)

    if num_new <= 0:
        # If new_peaks contains same number (or fewer) peaks than base_peaks, return peaks that drop below
        # the similarity threshold
        if distance_threshold is None:
            distance_threshold = 0.5

        new_index = np.where(min_dist > distance_threshold)
    else:
        if distance_threshold is None:
            # Return most dissimilar peaks
            sort_index = np.argsort(min_dist)
            new_index = sort_index[::-1][:num_new]
        else:
            # If threshold provided, return peaks below similarity threshold
            new_index = np.where(min_dist > distance_threshold)

    return new_index


def has_similar_peak(peak_location, compare_peak_locations, threshold=0.5, epsilon=1):
    sim_index = peak_similarity_index([peak_location], compare_peak_locations, epsilon)
    return sim_index[0] >= threshold


# def evalute_peak_prob()





