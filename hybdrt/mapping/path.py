import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import scharr
import matplotlib.pyplot as plt

from ..filters import flexible_hysteresis_threshold, gaussian_laplace1d, iterative_gaussian_filter, \
    nonuniform_gaussian_filter1d
from ..utils.array import nearest_index


def find_path_2d(energy, start_coords, end_row_index, offset=2, offset_cost=0.1, momentum=0.1,
                 max_energy=np.inf, grad_strength=2, grad_sigma=2):
    i0, j0 = start_coords
    direction = np.sign(end_row_index - i0)
    energy = np.nan_to_num(energy)

    n_steps = abs(end_row_index - i0)
    j_coords = np.empty(n_steps + 1, dtype=int)
    j_coords[0] = j0

    # Add gradient contribution to energy to keep path in low-energy valleys
    if grad_strength > 0:
        if grad_sigma > 0:
            grad = np.abs(scharr(ndimage.gaussian_filter(energy, grad_sigma), axis=1))
        else:
            grad = np.abs(scharr(energy, axis=1))
        tot_energy = energy + grad_strength * grad
    else:
        tot_energy = energy

    i = i0
    j = j0
    prev_offset = 0
    offsets = np.arange(-offset, offset + 1, dtype=int)
    offset_costs = offset_cost * np.abs(offsets)  # ** 2
    end_i = end_row_index
    tot_cost = 0
    for n in range(n_steps):
        # Clip offsets that go past image edge
        offset_is_valid = (j + offsets >= 0) & (j + offsets < energy.shape[1])
        offsets_n = offsets[offset_is_valid]
        offset_costs_n = offset_costs[offset_is_valid]

        # Get energy and momentum cost of possible next steps
        next_e_tot = tot_energy[i + direction, j + offsets_n[0]:j + offsets_n[-1] + 1]
        next_e = energy[i + direction, j + offsets_n[0]:j + offsets_n[-1] + 1]
        # print(i, j)
        next_mc = momentum * np.abs(offsets_n - prev_offset)  # ** 2
        step_costs = next_e_tot + next_mc + offset_costs_n

        # Find best step
        min_index = np.argmin(step_costs)

        # Check if exceeded energy threshold
        if next_e[min_index] > max_energy:
            end_i = i
            j_coords = j_coords[:n + 1]
            break

        new_offset = offsets_n[min_index]
        i = i + direction
        j = j + new_offset
        j_coords[n + 1] = j
        tot_cost += step_costs[min_index]

        prev_offset = new_offset

    i_coords = np.arange(i0, end_i + direction, direction)
    return (i_coords, j_coords), tot_cost


def energy_from_prob(ridge_prob):
    # return -np.log(ridge_prob / (1 - ridge_prob))
    return -np.log(ridge_prob)


def find_paths_2d(ridge_prob, start_rows, end_rows, offset=2, offset_cost=0.1, momentum=0.1,
                  min_prob=0, grad_strength=2, grad_sigma=2,
                  **find_peaks_kw):
    paths = []
    costs = []
    energy = energy_from_prob(ridge_prob)

    if min_prob == 0:
        max_energy = np.inf
    else:
        max_energy = -np.log(min_prob / (1 - min_prob))

    for start_row, end_row in zip(start_rows, end_rows):
        peaks, _ = find_peaks(ridge_prob[start_row], **find_peaks_kw)
        for peak in peaks:
            start_coords = (start_row, peak)
            path, cost = find_path_2d(energy, start_coords, end_row, offset=offset, offset_cost=offset_cost,
                                      momentum=momentum, max_energy=max_energy, grad_strength=grad_strength,
                                      grad_sigma=grad_sigma)
            paths.append(path)
            costs.append(cost)

    return paths, costs


def find_starting_lines_3d(ridge_prob, start_row, max_slope=3, **find_peaks_kw):
    peaks, _ = find_peaks(ridge_prob[0, start_row, :].flatten(), **find_peaks_kw)
    num_slices = ridge_prob.shape[0]
    slope_inc = 1.0 / num_slices
    slopes = np.arange(-max_slope, max_slope + 0.1, slope_inc)
    log_prob = np.log(ridge_prob)

    col_indices = []
    for peak in peaks:
        lps = np.empty(len(slopes))
        for k, slope in enumerate(slopes):
            # col_index = np.round(peak + slope * np.arange(num_slices)).astype(int)
            col_index = columns_from_slope(peak, slope, num_slices)
            lp = np.sum(get_line_3d(log_prob, start_row, col_index))
            lps[k] = lp
            # print(slope, lp)
        best_slope = slopes[np.argmax(lps)]
        print('best_slope:', best_slope)
        col_index = np.round(peak + best_slope * np.arange(num_slices)).astype(int)
        col_indices.append(col_index)
    return col_indices


def get_line_3d(a, row, cols):
    return [a[i, row, cols[i]] for i in range(len(cols))]


def columns_from_slope(col, slope, num_slices):
    return np.round(col + slope * np.arange(num_slices)).astype(int)


def find_path_3d(energy, start_row, start_cols, end_row, *, offset=2, offset_cost=0.1, momentum=0.1,
                 slope_offset_cost=0.1, slope_momentum=0.1, max_slope=3,
                 max_energy=np.inf, grad_strength=2, grad_sigma=2):
    num_slices = energy.shape[0]
    direction = np.sign(end_row - start_row)
    energy = np.nan_to_num(energy)

    slope_inc = 1.0 / num_slices

    n_steps = abs(end_row - start_row)
    col_coords = np.empty((num_slices, n_steps + 1), dtype=int)
    col_coords[:, 0] = start_cols

    # Add gradient contribution to energy to keep path in low-energy valleys
    if grad_strength > 0:
        # if grad_sigma is not None:
        #     grad = np.abs(scharr(ndimage.gaussian_filter(energy, grad_sigma), axis=-1))
        # else:
        #     grad = np.abs(scharr(energy, axis=-1))
        grad = np.empty_like(energy)
        for i in range(num_slices):
            if grad_sigma is not None:
                grad[i] = np.abs(scharr(ndimage.gaussian_filter(energy[i], grad_sigma), axis=-1))
            else:
                grad[i] = np.abs(scharr(energy[i], axis=-1))
        tot_energy = energy + grad_strength * grad
    else:
        tot_energy = energy

    row = start_row + direction
    cols = start_cols
    slope = float(start_cols[-1] - start_cols[0]) / num_slices
    prev_offset = 0
    prev_slope_offset = 0
    offsets = np.arange(-offset, offset + 1, dtype=int)
    offset_costs = offset_cost * np.abs(offsets)  # ** 2
    end = end_row
    tot_cost = 0
    for n in range(n_steps):

        # Get possible slopes
        slopes = np.arange(slope - 2 * slope_inc, slope + 2 * slope_inc + 1e-10, slope_inc)
        slopes = slopes[np.abs(slopes) <= max_slope]

        slope_step_costs = np.abs(slopes - slope) * slope_offset_cost
        slope_momentum_costs = np.abs((slopes - slope) - prev_slope_offset) * slope_momentum

        # Get possible new column indices
        slope_energies = np.empty(len(slopes))
        slope_cols = np.empty((len(slopes), len(cols)), dtype=int)
        slope_offsets = np.empty(len(slopes), dtype=int)
        for k, test_slope in enumerate(slopes):
            slope_test_cols = columns_from_slope(cols[0], test_slope, num_slices)

            # Clip offsets that go past image edge
            offset_is_valid = (np.min(slope_test_cols) + offsets >= 0) & \
                              (np.max(slope_test_cols) + offsets < energy.shape[-1])
            offsets_k = offsets[offset_is_valid]
            offset_costs_k = offset_costs[offset_is_valid]

            test_energy = np.array(
                [get_line_3d(tot_energy, row, slope_test_cols + test_offset) for test_offset in offsets_k]
            )
            test_energy = np.sum(test_energy, axis=1)

            # Add momentum and offset costs
            test_energy += momentum * np.abs(offsets_k - prev_offset)
            test_energy += offset_costs_k

            min_index = np.argmin(test_energy)
            slope_energies[k] = test_energy[min_index]
            slope_cols[k] = slope_test_cols + offsets_k[min_index]
            slope_offsets[k] = offsets_k[min_index]

        # Add slope cost
        slope_energies += slope_step_costs + slope_momentum_costs
        slope_index = np.argmin(slope_energies)

        # Check if exceeded energy threshold
        next_energy = get_line_3d(energy, row, slope_cols[slope_index])
        if np.min(next_energy) > max_energy:
            end = row
            col_coords = col_coords[:, n + 1]
            break

        row = row + direction
        cols = slope_cols[slope_index]
        new_offset = slope_offsets[slope_index]
        col_coords[:, n + 1] = cols
        tot_cost += slope_energies[slope_index]

        prev_offset = new_offset

    row_coords = np.arange(start_row, end + direction, direction)
    return (row_coords, col_coords), tot_cost


def find_paths_3d(ridge_prob, start_row, end_row, offset=2, offset_cost=0.1, momentum=0.1,
                  slope_offset_cost=0.1, slope_momentum=0.1,
                  min_prob=0, grad_strength=2, grad_sigma=2, max_slope=3, **find_peaks_kw):
    start_lines = find_starting_lines_3d(ridge_prob, start_row, max_slope=max_slope, **find_peaks_kw)
    energy = energy_from_prob(ridge_prob + 1e-10)

    max_energy = energy_from_prob(min_prob)

    paths = []
    costs = []

    for start_line in start_lines:
        path, cost = find_path_3d(energy, start_row, start_line, end_row, offset=offset, offset_cost=offset_cost,
                                  momentum=momentum,
                                  slope_offset_cost=slope_offset_cost, slope_momentum=slope_momentum,
                                  max_slope=max_slope,
                                  max_energy=max_energy,
                                  grad_strength=grad_strength, grad_sigma=grad_sigma)
        paths.append(path)
        costs.append(cost)

    return paths, costs


# TODO: merge paths that are nearly the same
#  Compare costs for paths that cross?
#  Look for missed peaks - anything that exceeds prominence and is not in contact with the energy well of a path


def make_ridge_path_mask(ridge_prob, start_rows, end_rows, offset=2, offset_cost=0.1, momentum=0.1,
                         min_prob=0, grad_strength=2, grad_sigma=2, increment_labels=False,
                         **find_peaks_kw):
    if increment_labels:
        path_mask = np.zeros(ridge_prob.shape, dtype=int)
    else:
        path_mask = np.zeros(ridge_prob.shape, dtype=bool)

    num_group_dims = ridge_prob.ndim - 2
    it = np.nditer(ridge_prob, op_axes=[list(range(num_group_dims))], flags=['multi_index'])
    for _ in it:
        slice_index = it.multi_index
        if not np.all(np.isnan(ridge_prob[slice_index])):
            paths, costs = find_paths_2d(ridge_prob[slice_index], start_rows, end_rows, offset=offset,
                                         offset_cost=offset_cost, momentum=momentum, min_prob=min_prob,
                                         grad_strength=grad_strength, grad_sigma=grad_sigma, **find_peaks_kw)
            mask = paths_to_mask(path_mask[slice_index].shape, paths, increment_labels=increment_labels)
            path_mask[slice_index] = mask

    return path_mask


def path_energy_well(path_mask, energy, sigma=2):
    """
    Get the energy well surrounding a path.
    Well is defined by the region touching the path in which the curvature of the energy is positive
    :param path_mask:
    :param energy:
    :param sigma:
    :return:
    """
    curv = gaussian_laplace1d(energy, sigma=sigma, axis=1)
    well_depth = np.max(-energy) + 1
    elev = curv + well_depth * np.nan_to_num(path_mask)
    # Find connected regions along tau axis only
    structure = np.zeros((3, 3))
    structure[1] = 1
    well_mask = flexible_hysteresis_threshold(elev, 0, well_depth, structure=structure)
    return well_mask


def find_peaks_2d(ridge_prob, distance=3, height=0.1, prominence=0.1, **kw):
    peak_mask = np.zeros_like(ridge_prob, dtype=bool)

    for i in range(ridge_prob.shape[0]):
        peaks, _ = find_peaks(ridge_prob[i], distance=distance, height=height,
                              prominence=prominence, **kw)
        peak_mask[i, peaks] = 1

    return peak_mask


def find_missing_peaks(ridge_prob, *, paths=None, path_mask=None, **find_peaks_kw):
    if paths is None and path_mask is None:
        raise ValueError('Either paths or path_mask must be provided')
    elif paths is not None and path_mask is not None:
        raise ValueError('Only one of paths or path_mask should be provided')
    elif paths is not None:
        path_mask = paths_to_mask(ridge_prob.shape, paths)

    energy = energy_from_prob(ridge_prob)
    well_mask = path_energy_well(path_mask, energy)
    # print(np.sum(well_mask))

    # Find local maxima
    peak_mask = find_peaks_2d(ridge_prob, **find_peaks_kw)

    # Label connected peaks - only connect along tau axis to avoid connecting undetected peaks to distant wells
    structure = np.zeros((3, 3))
    structure[1] = 1
    peak_labels, peak_count = ndimage.label(peak_mask, structure=structure)

    # Find peaks that are not connected to existing path wells
    sums = ndimage.sum_labels(well_mask, peak_labels, index=np.arange(peak_count + 1))
    undetected = sums == 0
    # print(sums, undetected)

    return undetected[peak_labels] & peak_mask


def find_missing_paths(ridge_prob, missing_peak_mask, row_lim=None, **path_kwargs):
    # Group connected peaks
    peak_labels, num_peaks = ndimage.label(missing_peak_mask, structure=np.ones((3, 3)))

    energy = energy_from_prob(ridge_prob)

    if row_lim is None:
        row_lim = (0, len(ridge_prob) - 1)

    peak_paths = []
    peak_costs = []
    if num_peaks > 0:
        for label in np.unique(peak_labels)[1:]:
            # Start from any peak in the group
            start_coords = np.argwhere(peak_labels == label)[0]
            print(start_coords)
            start_row = start_coords[0]

            # If peak starts at one of the bounding rows, only consider the other endpoint
            if start_row == row_lim[0]:
                end_rows = row_lim[1:]
            elif start_row == row_lim[1]:
                end_rows = row_lim[:1]
            else:
                end_rows = row_lim

            paths = []
            pcost = 0
            for end_row in end_rows:
                print(end_row)
                path, cost = find_path_2d(energy, start_coords, end_row, **path_kwargs)
                paths.append(path)
                pcost += cost

            # Join the paths and sort by row index
            path_i = np.concatenate([p[0] for p in paths])
            path_j = np.concatenate([p[1] for p in paths])
            sort_index = np.argsort(path_i)
            path_i = path_i[sort_index]
            path_j = path_j[sort_index]

            # TODO: add cost for step to join paths
            peak_paths.append((path_i, path_j))
            peak_costs.append(pcost)

    return peak_paths, peak_costs


def paths_to_mask_3d(shape, paths, increment_labels=False, fill_nan=False):
    if increment_labels:
        output = np.zeros(shape, dtype=int)
        for i, path in enumerate(paths):
            for k, col_index in enumerate(path[1]):
                output[(np.ones(len(path[0]), dtype=int) * k, path[0], col_index)] = i + 1
    else:
        output = np.zeros(shape, dtype=bool)
        for path in paths:
            for k, col_index in enumerate(path[1]):
                output[(np.ones(len(path[0]), dtype=int) * k, path[0], col_index)] = 1

    if fill_nan:
        output = output.astype(float)
        output[output == 0] = np.nan

    return output


def paths_to_mask(shape, paths, increment_labels=False, fill_nan=False):
    if increment_labels:
        output = np.zeros(shape, dtype=int)
        for i, path in enumerate(paths):
            output[path] = i + 1
    else:
        output = np.zeros(shape, dtype=bool)
        for path in paths:
            output[path] = 1

    if fill_nan:
        output = output.astype(float)
        output[output == 0] = np.nan

    return output


def path_pair_metrics(path1, path2):
    _, index1, index2 = np.intersect1d(path1[0], path2[0], return_indices=True)

    j1 = np.array(path1[1][index1]).astype(float)
    j2 = np.array(path2[1][index2]).astype(float)

    corr = np.corrcoef(j1, j2)[0, 1]

    rss = np.sum((j1 - j2) ** 2) / len(j1)

    return (1 - corr) * rss


def find_bounding_troughs(trough_mask, path, path_ndim=3, tidy=False):
    if path_ndim == 3:
        row_index = path[0]
        col_indices = path[1]
        left_indices = np.empty_like(col_indices)
        right_indices = np.empty_like(col_indices)

        for i, col_index in enumerate(col_indices):
            # 2-d slice
            for j, row in enumerate(row_index):
                trough_index = np.where(trough_mask[i, row])[0]
                # Add start and end indices to ensure that a trough is found
                trough_index = np.unique(np.concatenate([trough_index, [0, trough_mask.shape[-1] - 1]]))
                left_indices[i, j] = trough_index[nearest_index(trough_index, col_index[j], -1)]
                right_indices[i, j] = trough_index[nearest_index(trough_index, col_index[j], 1)]

        if tidy:
            # Clean up indices
            for raw_index in (left_indices, right_indices):
                med = ndimage.median_filter(raw_index, size=(3, 5))
                bad_index = np.abs(raw_index - med) > 5
                raw_index[bad_index] = med[bad_index]
                # raw_index[...] = ndimage.gaussian_filter(raw_index, sigma=(0.5, 1))
                raw_index[...] = iterative_gaussian_filter(raw_index, sigma=(0.5, 1))

        return left_indices, right_indices


def integrate_paths(tau, f, rp, paths, troughs, width_sigma=1):
    # Set up for 3d only
    path_weights = np.zeros((len(paths), *f.shape))

    num_slices = f.shape[0]

    for k, (path, trough) in enumerate(zip(paths, troughs)):
        row_indices, path_indices = path
        left_indices, right_indices = trough

        k_mask = paths_to_mask_3d(f.shape, [path]).astype(float)
        f_path = k_mask * f * rp ** 0.5

        right_radius = right_indices - path_indices
        left_radius = path_indices - left_indices
        widths = 2 * np.minimum(left_radius, right_radius).astype(float)
        # widths = (right_indices - left_indices).astype(float)
        if width_sigma is not None:
            widths = ndimage.gaussian_filter(widths, sigma=width_sigma)
        sigmas = widths / 2
        # print(sigmas)
        sigmas = np.tile(sigmas, (f.shape[-1], 1, 1))
        sigmas = np.moveaxis(sigmas, 0, -1)

        # print(f_path.shape, widths.shape, sigmas.shape)

        path_weights[k] = nonuniform_gaussian_filter1d(f_path, sigmas, axis=-1)

    # Normalize weights
    weight_sum = np.sum(path_weights, axis=0)[None, :]
    weight_sum[weight_sum == 0] = 1
    norm_weights = path_weights / weight_sum

    path_dist = norm_weights * f[None, :]
    path_sizes = np.trapz(path_dist, x=np.log(tau), axis=-1)

    return path_dist, path_sizes


# ==================
# Visualization
# ==================
def plot_paths_and_troughs(paths, troughs, shape, slice_index=None, slice_axis=None, ax=None):
    path_mask = paths_to_mask_3d(shape, paths, fill_nan=True)

    trough_paths = []
    for k, path in enumerate(paths):
        row_index = path[0]
        trough_paths += [(row_index, troughs[k][0]), (row_index, troughs[k][1])]

    trough_mask = paths_to_mask_3d(shape, trough_paths, fill_nan=True)

    if slice_index is not None:
        path_mask = np.take(path_mask, slice_index, slice_axis)
        trough_mask = np.take(trough_mask, slice_index, slice_axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    ax.pcolormesh(path_mask, cmap='Reds', vmin=0, vmax=1)
    ax.pcolormesh(trough_mask, cmap='Blues', vmin=0, vmax=1)
