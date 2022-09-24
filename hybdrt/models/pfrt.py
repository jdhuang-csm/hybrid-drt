import numpy as np

from .. import utils


def identify_peaks(pf, threshold):
    """
    Get indices of peaks identified by pf at threshold
    :param ndarray pf: probability function of relaxation times
    :param threshold: probability function threshold
    :return:
    """
    index = np.where(pf >= threshold)[0]
    range_starts, range_ends = utils.array.find_contiguous_ranges(index)
    # Place each peak at maximum pf within peak range
    peak_index = np.array([start + np.argmax(pf[start:end]) for start, end in zip(range_starts, range_ends)])
    return peak_index


def get_peak_ranges(pf, min_prob):
    index = np.where(pf >= min_prob)[0]
    range_starts, range_ends = utils.array.find_contiguous_ranges(index)
    return range_starts, range_ends


def integrate_peaks(pf, min_prob):
    thresh_index = np.where(pf >= min_prob)[0]
    peak_starts, peak_ends = utils.array.find_contiguous_ranges(thresh_index)
    peak_index = identify_peaks(pf, min_prob)
    # print(peak_starts, peak_ends)
    peak_areas = np.array([np.trapz(pf[start - 1:end + 1]) for start, end in zip(peak_starts, peak_ends)])

    return peak_index, peak_areas


def rank_peaks(pf, min_prob, integrate=True):
    if integrate:
        peak_index, magnitudes = integrate_peaks(pf, min_prob)
    else:
        peak_index = identify_peaks(pf, min_prob)
        magnitudes = pf[peak_index]
    sort_index = np.argsort(magnitudes)[::-1]
    return peak_index[sort_index], magnitudes[sort_index]


def identify_candidate_peaks(candidate_pf, threshold, shift=False, **shift_kw):
    if shift:
        candidate_pf = shift_candidate_pfrt(candidate_pf, **shift_kw)

    # Get peaks in candidate
    range_starts, range_ends = get_peak_ranges(candidate_pf, threshold)

    # Place each candidate peak at maximum of tot_pf within peak range
    peak_index = np.array([start + np.argmax(candidate_pf[start:end]) for start, end in zip(range_starts, range_ends)])

    return peak_index


def shift_candidate_pfrt(candidate_pf, tot_pf=None, tot_thresh=None, tot_peak_ranges=None, tot_peak_indices=None):
    """
    Shift candidate peak probabilities to corresponding peak locations in total PFRT
    :param candidate_pf:
    :param tot_pf:
    :param tot_thresh:
    :param tot_peak_ranges:
    :param tot_peak_indices:
    :return:
    """
    if tot_pf is None:
        if tot_peak_ranges is None or tot_peak_indices is None:
            raise ValueError('If tot_pf is not provided, tot_peak_ranges and tot_peak_index must be provided')
    else:
        if tot_thresh is None:
            raise ValueError('If tot_pf is provided, tot_thresh must also be provided')

    # Get peaks in candidate
    thresh_index = np.where(candidate_pf > 0)[0]

    if tot_peak_ranges is None:
        tot_peak_ranges = get_peak_ranges(tot_pf, tot_thresh)

    if tot_peak_indices is None:
        tot_peak_indices = identify_peaks(tot_pf, tot_thresh)

    range_starts, range_ends = tot_peak_ranges

    tot_peak_match_index = [np.where((range_starts <= ti) & (range_ends >= ti))[0] for ti in thresh_index]

    def get_shift_index(match_index, cand_index):
        if len(match_index) == 1:
            return tot_peak_indices[match_index[0]]
        else:
            return cand_index

    shift_index = np.array([get_shift_index(mi, ti) for mi, ti in zip(tot_peak_match_index, thresh_index)])

    shift_pf = np.zeros(len(candidate_pf))
    shift_pf[shift_index] = candidate_pf[thresh_index]

    return shift_pf


def candidate_corr(target_peak_indices, candidate_pf):
    target_pf = np.zeros_like(candidate_pf)
    target_pf[target_peak_indices] = 1
    # nonzero_index = np.where((target_pf > 0) | (candidate_pf > 0))[0]
    # print(target_pf[nonzero_index], candidate_pf[nonzero_index])
    return np.corrcoef(target_pf, candidate_pf)[0, 1]


def get_matching_candidate(target_peak_indices, candidate_pfs, candidate_llh):
    match_quality = [candidate_corr(target_peak_indices, cand_pf) * cand_llh
                     for cand_pf, cand_llh in zip(candidate_pfs, candidate_llh)]
    return np.argmax(match_quality)


def select_candidates(tot_pf, candidate_pfs, candidate_llh, start_thresh=0.99, end_thresh=0.01, peak_thresh=1e-6):
    # Identify peaks in total PFRT
    tot_peak_ranges = get_peak_ranges(tot_pf, peak_thresh)
    tot_peak_indices = identify_peaks(tot_pf, peak_thresh)

    # Shift candidate PFRTs to align peaks with tot_pf
    shift_pfs = [shift_candidate_pfrt(cand_pf, tot_peak_ranges=tot_peak_ranges, tot_peak_indices=tot_peak_indices)
                 for cand_pf in candidate_pfs]

    # Rank total PFRT peaks
    ranked_peak_indices, peak_magnitudes = rank_peaks(tot_pf, peak_thresh)
    peak_magnitudes /= np.max(peak_magnitudes)

    # select_thresh = start_thresh
    include_index = np.where(peak_magnitudes >= start_thresh)[0]
    print(peak_magnitudes)
    if len(include_index) > 0:
        include_index = include_index[-1]
    else:
        include_index = 0

    target_peak_indices = []
    candidate_indices = []
    while include_index < len(peak_magnitudes) - 1:
        target_indices = ranked_peak_indices[:include_index + 1]
        candidate_index = get_matching_candidate(target_indices, shift_pfs, candidate_llh)
        target_peak_indices.append(target_indices)
        candidate_indices.append(candidate_index)

        # Incorporate next peak
        include_index += 1

        # Stop if we have reached peaks below desired threshold
        if peak_magnitudes[include_index] < end_thresh:
            break

    return target_peak_indices, candidate_indices




