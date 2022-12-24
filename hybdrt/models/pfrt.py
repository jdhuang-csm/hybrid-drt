import numpy as np

from .. import utils


def identify_peaks(pf, min_prob):
    """
    Get indices of peaks identified by pf at threshold
    :param ndarray pf: PFRT
    :param min_prob: probability threshold for identifying contiguous ranges
    :return:
    """
    # TODO: this should probably look for actual peaks, not contiguous ranges. Peak splitting seems likely
    range_starts, range_ends = get_peak_ranges(pf, min_prob)
    # Place each peak at maximum pf within peak range
    peak_index = np.array([start + np.argmax(pf[start:end]) for start, end in zip(range_starts, range_ends)])
    return peak_index


def get_peak_ranges(pf, min_prob):
    """
    Get start and end indices of peaks in the PFRT
    :param ndarray pf: PFRT to process
    :param float min_prob: probability threshold for identifying contiguous ranges
    :return:
    """
    index = np.where(pf >= min_prob)[0]
    range_starts, range_ends = utils.array.find_contiguous_ranges(index)
    return range_starts, range_ends


def integrate_peaks(pf, min_prob):
    """
    Get integrated areas of peaks in the PFRT
    :param ndarray pf: PFRT to process
    :param float min_prob: probability threshold for identifying contiguous ranges
    :return: peak_index, peak_areas
    """
    peak_starts, peak_ends = get_peak_ranges(pf, min_prob)
    peak_index = identify_peaks(pf, min_prob)
    # print(peak_starts, peak_ends)
    peak_areas = np.array([np.trapz(pf[start - 1:end + 1]) for start, end in zip(peak_starts, peak_ends)])

    return peak_index, peak_areas


def rank_peaks(pf, min_prob, integrate=True):
    """
    Rank peaks in the PFRT by their magnitudes
    :param ndarray pf: PFRT to process
    :param float min_prob: probability threshold for identifying contiguous ranges
    :param bool integrate: if True, rank peaks by their integrated area. If False, rank by peak height
    :return:
    """
    if integrate:
        peak_index, magnitudes = integrate_peaks(pf, min_prob)
    else:
        peak_index = identify_peaks(pf, min_prob)
        magnitudes = pf[peak_index]
    sort_index = np.argsort(magnitudes)[::-1]
    return peak_index[sort_index], magnitudes[sort_index]


def identify_candidate_peaks(candidate_pf, threshold, shift=False, **shift_kw):
    """
    Identify peaks in a candidate PFRT
    :param candidate_pf: candidate PFRT
    :param threshold: probability threshold for identifying contiguous ranges
    :param bool shift: if True, shift the candidate
    :param shift_kw:
    :return:
    """
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

    # Get nonzero values in candidate
    thresh_index = np.where(candidate_pf > 0)[0]

    if tot_peak_ranges is None:
        tot_peak_ranges = get_peak_ranges(tot_pf, tot_thresh)

    if tot_peak_indices is None:
        tot_peak_indices = identify_peaks(tot_pf, tot_thresh)

    range_starts, range_ends = tot_peak_ranges

    # Find indices where candidate nonzero values fall within a tot_pf peak range
    tot_peak_match_index = [np.where((range_starts <= ti) & (range_ends >= ti))[0] for ti in thresh_index]

    def get_shift_index(match_index, cand_index):
        if len(match_index) == 1:
            # If candidate peak matches a peak in tot_pf, place it at the corresponding tot_pf location
            return tot_peak_indices[match_index[0]]
        else:
            # No match - leave the candidate value in its original location
            return cand_index

    shift_index = np.array([get_shift_index(mi, ti) for mi, ti in zip(tot_peak_match_index, thresh_index)])

    # TODO: multiple candidate values could be placed at the same tot_pf peak location.
    #  Should either take max or sum
    shift_pf = np.zeros(len(candidate_pf))
    shift_pf[shift_index] = candidate_pf[thresh_index]

    return shift_pf


def candidate_corr(target_peak_indices, candidate_pf):
    """
    Get correlation of candidate PFRT to target peak model
    :param target_peak_indices: desired peak locations
    :param candidate_pf: candidate PFRT
    :return:
    """
    target_pf = np.zeros_like(candidate_pf)
    target_pf[target_peak_indices] = 1
    # nonzero_index = np.where((target_pf > 0) | (candidate_pf > 0))[0]
    # print(target_pf[nonzero_index], candidate_pf[nonzero_index])
    return np.corrcoef(target_pf, candidate_pf)[0, 1]


def get_matching_candidate(target_peak_indices, candidate_pfs, candidate_llh):
    """
    Get candidate PFRT that best matches target peak model.
    Match quality is determined by the product of LLH and candidate correlation to target
    :param target_peak_indices: desired peak locations
    :param candidate_pfs: list of candidate PFRTs
    :param candidate_llh: array of candidate LLH values
    :return:
    """
    match_quality = [candidate_corr(target_peak_indices, cand_pf) * cand_llh
                     for cand_pf, cand_llh in zip(candidate_pfs, candidate_llh)]
    return np.argmax(match_quality)


def select_candidates(tot_pf, candidate_pfs, candidate_llh, start_thresh=0.99, end_thresh=0.01, peak_thresh=1e-6):
    """

    :param tot_pf:
    :param candidate_pfs:
    :param candidate_llh:
    :param start_thresh:
    :param end_thresh:
    :param peak_thresh: probability threshold for identifying contiguous ranges in tot_pf
    :return:
    """
    # Identify peaks in total PFRT
    tot_peak_ranges = get_peak_ranges(tot_pf, peak_thresh)
    tot_peak_indices = identify_peaks(tot_pf, peak_thresh)

    # Shift candidate PFRTs to align peaks with tot_pf
    shift_pfs = [shift_candidate_pfrt(cand_pf, tot_peak_ranges=tot_peak_ranges, tot_peak_indices=tot_peak_indices)
                 for cand_pf in candidate_pfs]

    # Rank total PFRT peaks
    ranked_peak_indices, peak_magnitudes = rank_peaks(tot_pf, peak_thresh)
    peak_magnitudes /= np.max(peak_magnitudes)

    # Begin by including all peaks in tot_pf that exceed start_thresh
    include_index = np.where(peak_magnitudes >= start_thresh)[0]
    print(peak_magnitudes)
    if len(include_index) > 0:
        include_index = include_index[-1]
    else:
        include_index = 0

    # Incrementally decrease the threshold until all peaks are incorporated or end_thresh has been reached
    target_peak_indices = []
    candidate_indices = []
    while include_index < len(peak_magnitudes) - 1:
        # At each threshold, the target peak model includes all peaks in tot_pf that exceed the threshold
        # Find the best matching candidate PFRT
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




