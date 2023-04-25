import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from itertools import combinations
from skimage.segmentation import join_segmentations, relabel_sequential
from copy import deepcopy

from ..utils.array import find_contiguous_ranges


def image_to_cloud(img, dim_grids, thresh=None, index=None, include_intensity=True, return_index=False):
    if len(dim_grids) != np.ndim(img):
        raise ValueError('dim_grids must match image dimensions')

    if index is None and thresh is None:
        raise ValueError('Either thresh or index must be provided')

    coord_mesh = np.meshgrid(*dim_grids, indexing='ij')

    if index is None:
        index = img > thresh

    values = [cm[index] for cm in coord_mesh]
    if include_intensity:
        values.append(img[index])

    cloud = np.stack(values, axis=0).T

    if return_index:
        return cloud, index
    else:
        return cloud


def cloud_to_image(cloud, index, fill_val=np.nan):
    img = np.empty(index.shape)
    img.fill(fill_val)

    img[index] = cloud

    return img


def coords_to_values(coords, dim_grids):
    # Coord should be a NxM array, rows are observations, columns are dimension coordinates
    if len(dim_grids) != coords.shape[1]:
        raise ValueError('dim_grids must match coords dimensions')

    output = np.zeros_like(coords)

    for i in range(coords.shape[1]):
        grid = dim_grids[i]
        output[:, i] = interp1d(np.arange(len(grid)), grid)(coords[:, i])

    return output


# def cluster_ridges(ridge_mask):
#
#     labels, num = ndimage.label(ridge_mask, structure=np.ones((3, ) * np.ndim(ridge_mask)))
#
#     labels_flat = labels[ridge_mask]
#     index = np.where(ridge_mask.flatten())[0]
#
#     return index, labels_flat


def find_clusters_to_split(labels, tau_axis=-1):
    split_labels = []
    for label in np.unique(labels):
        # Check for rows with 2 peaks in same cluster
        mask = labels == label
        row_sum = np.nansum(mask, axis=tau_axis)

        if np.nanmax(row_sum) > 1:
            split_labels.append(label)

    return split_labels


def find_mc_groups(labels, split_labels, tau_axis=-1, connectivity=2, fixed_split_indices=None):
    mc_groups = []

    # Make stucture for connected component labeling
    if connectivity == 1:
        structure = None
    elif connectivity == 2:
        structure = np.ones((3, 3))
    else:
        raise ValueError('Connectivity must be 1 or 2')

    for sl in split_labels:
        mask = labels == sl
        row_sum = np.nansum(mask, axis=tau_axis)

        # Find connected segments within 2d slices
        tmp = np.zeros_like(labels)
        tmp[mask] = 1
        tmp_labels = np.zeros_like(labels)
        it = np.nditer(tmp, op_axes=[np.arange(tmp.ndim - 2).tolist()], flags=['multi_index'])
        for i in it:
            index_2d = it.multi_index

            # Segment by connectivity
            slice_labels, _ = ndimage.label(tmp[index_2d], structure=structure)

            if fixed_split_indices is not None:
                split_indices = np.array(fixed_split_indices)
            else:
                split_indices = np.empty(0, dtype=int)

            # Find rows in which more than one peak exists
            conflict_indices = np.where(row_sum[index_2d] > 1)[0]
            if len(conflict_indices) > 0:
                start_indices, end_indices = find_contiguous_ranges(conflict_indices)
                # print(conflict_indices)
                # Clip at last index
                if end_indices[-1] == len(slice_labels):
                    end_indices = end_indices[:-1]

                print(index_2d, start_indices, end_indices)
                split_indices = np.unique(np.concatenate([split_indices, start_indices, end_indices]))
                # for j, (start, end) in enumerate(zip(start_indices, end_indices)):

            if len(split_indices) > 0:
                join_labels = slice_labels.copy()
                for j in split_indices:
                    # Offset to give a different label
                    join_labels[j:] = relabel_sequential(join_labels[j:], np.max(join_labels[j:]) + 1)[0]

                slice_labels = join_segmentations(slice_labels, join_labels)

            # Labels for each slice must be unique
            tmp_labels[index_2d], _, _ = relabel_sequential(slice_labels, np.max(tmp_labels) + 1)

        # # Find
        # # Make an array containing conflicting labels
        #
        # tmp[sum_mask & mask] = labels[sum_mask & mask]
        # # Find connected groups in the conflicting array
        # group_labels, _ = ndimage.label(tmp, structure=np.ones((3, ) * tmp.ndim))

        groups = [np.where(tmp_labels == li) for li in np.unique(tmp_labels)[1:]]
        print(sl, len(groups))
        mc_groups.append(groups)

    # At points where 2 peaks are in same row
    # At points where connectivity is broken
    # At OCV
    return mc_groups


def interaction_energy(image, group1, group2, c1, c2, tau_axis=-1, sigma=1, attraction=1, repulsion=10):
    """

    :param image: image array. Only used for shape
    :param group1: indices of first group. Should be in format returned by np.where
    :param group2: indices of second group
    :param c1: first group cluster label
    :param c2: second group cluster label
    :param tau_axis: axis corresponding to tau dimension
    :param sigma: attractive interaction lengthscale
    :return:
    """
    c1_arr = np.zeros(image.shape)
    c1_arr[group1] = c1
    c2_arr = np.zeros(image.shape)
    c2_arr[group2] = c2

    if c1 == c2:
        # Penalize each peak duplication within same row
        row_sum = np.sum((c1_arr > 0) | (c2_arr > 0), axis=tau_axis)
        repulse = np.sum(row_sum > 1) * repulsion

        # Attraction between nearby groups
        c1_spread = ndimage.gaussian_filter(c1_arr, sigma=sigma)
        # Normalize
        c1_spread *= attraction / np.max(c1_spread)
        attract = np.sum(c1_spread * c2_arr)

        return repulse - attract
    else:
        return 0


def interaction_matrix(image, groups, tau_axis=-1, sigma=1, attraction=1, repulsion=100):
    num_groups = len(groups)
    mat = np.zeros((num_groups, num_groups))

    for i, j in combinations(np.arange(num_groups), 2):
        if i != j:
            u = interaction_energy(image, groups[i], groups[j], 1, 1, tau_axis=tau_axis, sigma=sigma,
                                   attraction=attraction, repulsion=repulsion)
            mat[i, j] = u
            mat[j, i] = u

    return mat


def energy_delta(energy_mat, c0, change_index, new_val):
    # c1 = c0.copy()
    # c1[change_index] = new_val
    # return c1 @ energy_mat @ c1 - c0 @ energy_mat @ c0
    du = (energy_mat[change_index] @ c0) * (new_val - c0[change_index])
    return du


def accept_prob(du, temp):
    if du < 0:
        return 1
    else:
        return np.exp(-du / temp)


def test_step(du, temp, rng):
    p = accept_prob(du, temp)

    if p > rng.random():
        return True
    else:
        return False


def mc_anneal(image, groups, c0, temps, temp_n_iter, tau_axis=-1, sigma=1, attraction=1, repulsion=100,
              n_chains=1, energy_mat=None, rng=None, keep_samples=None):
    if energy_mat is None:
        energy_mat = interaction_matrix(image, groups, tau_axis=-tau_axis, sigma=sigma,
                                        attraction=attraction, repulsion=repulsion)

    if rng is None:
        rng = np.random.default_rng()

    chain_results = {
        'c_end': [],
        'c_best': [],
        'cum_du': [],
        'c_samples': [],
        'u_samples': []
    }
    du_min_tot = 0
    c_out = c0.copy()
    for n in range(n_chains):
        print('----------------------------')
        print(f'Running chain {n}...')
        print('----------------------------')
        c_best = c0.copy()
        cum_du = 0
        for i, (temp, n_iter) in enumerate(zip(temps, temp_n_iter)):
            print(f'Running {n_iter} iterations at temperature {temp}...')
            if i == len(temps) - 1:
                samples = keep_samples
            else:
                samples = None
            c_end, c_best, du, du_min, c_samples, u_samples = mc_optimize(
                image, groups, c_best, n_iter, temp,
                tau_axis=tau_axis, sigma=sigma, attraction=attraction,
                repulsion=repulsion, energy_mat=energy_mat, rng=rng,
                keep_samples=samples
            )
            cum_du += du_min

        print('Net energy change: {:.2f}'.format(cum_du))

        chain_results['c_end'].append(c_end)
        chain_results['c_best'].append(c_best)
        chain_results['cum_du'].append(cum_du)
        chain_results['c_samples'].append(c_samples)
        chain_results['u_samples'].append(u_samples)

        if cum_du < du_min_tot:
            c_out = c_best.copy()
            du_min_tot = deepcopy(cum_du)

    return c_out, chain_results


def mc_optimize(image, groups, c0, n_iter=100, temp=10, tau_axis=-1, sigma=1, attraction=1, repulsion=100,
                energy_mat=None, rng=None, keep_samples=None):
    if energy_mat is None:
        energy_mat = interaction_matrix(image, groups, tau_axis=-tau_axis, sigma=sigma,
                                        attraction=attraction, repulsion=repulsion)

    if rng is None:
        rng = np.random.default_rng()

    if keep_samples is not None:
        c_array = np.empty((keep_samples, len(c0)), dtype=int)
        u_array = np.empty(keep_samples)
        sample_start = n_iter - keep_samples
    else:
        c_array = None
        u_array = None
        sample_start = None

    c_i = c0.copy()
    cum_du = 0
    du_min = 0
    best_c = c0
    num_accepted = 0
    best_step = -1
    for i in range(n_iter):
        change_index = rng.integers(0, len(groups))
        new_val = c_i[change_index] * -1
        # print(change_index, new_val)

        du = energy_delta(energy_mat, c_i, change_index, new_val)

        if test_step(du, temp, rng):
            c_i[change_index] = new_val
            # print(i, change_index, du, 'accepted')
            # print(c_i)
            cum_du = cum_du + du
            num_accepted += 1
            if cum_du < du_min:
                best_c = c_i.copy()
                du_min = deepcopy(cum_du)
                best_step = i + 1
                # print(f'New minimum reached at step {i}')
        # else:
        #     print(i, du, 'rejected')

        # Keep samples at end of run
        if keep_samples is not None and i >= sample_start:
            c_array[i - sample_start] = c_i.copy()
            u_array[i - sample_start] = cum_du

    print('Accepted {:.0f} / {:.0f} steps ({:.1f} %)'.format(
        num_accepted, n_iter, 100 * num_accepted / n_iter)
    )
    print('Lowest energy {:.2f} reached at iteration {:.0f}'.format(du_min, best_step))

    return c_i, best_c, cum_du, du_min, c_array, u_array
