import numpy as np
from scipy import ndimage

from ..filters import gaussian_kernel_scale, empty_gaussian_filter, masked_filter


# def attraction_kernel(sigma, ndim=None, truncate=4.0):
#     if ndim is None:
#         ndim = len(sigma)
#
#     if np.isscalar(sigma):
#         sigma = (sigma,) * ndim
#
#     radius = [int(truncate * float(si) + 0.5) for si in sigma]
#     size = [2 * ri + 1 for ri in radius]
#
#     kernel = np.zeros(size)
#     kernel[tuple(radius)] = 1
#
#     kernel = ndimage.gaussian_filter(kernel, sigma=sigma, truncate=truncate)
#     kernel[..., radius[-1]] = 0
#     # kernel[tuple(radius)] = 0
#
#     return kernel


def count_extrema_row(extrema_mask, bound_mask, troughs=False):
    trough_indices = np.where(bound_mask)[0]
    count = np.zeros(len(extrema_mask))

    indices = np.unique(np.concatenate(([0], trough_indices, [len(extrema_mask)])))

    for i, start in enumerate(indices[:-1]):
        end = indices[i + 1]
        count[start:end] = np.sum(extrema_mask[start:end])

    if troughs:
        # Edges count as peaks - troughs should always be bounding
        window_mask = np.ones(len(extrema_mask), dtype=bool)
    else:
        window_mask = np.zeros(len(extrema_mask), dtype=bool)
        if len(trough_indices) > 1:
            window_mask[trough_indices[0]:trough_indices[-1]] = 1

    return count, window_mask


def count_extrema(extrema_mask, bound_mask, troughs=False):
    res = [count_extrema_row(extrema_mask[i], bound_mask[i], troughs=troughs) for i in range(extrema_mask.shape[0])]
    count = np.stack([r[0] for r in res], axis=0)
    window_mask = np.stack([r[1] for r in res], axis=0)
    return count, window_mask


def extremum_add_energy(count, window_mask):
    add_energy = np.zeros_like(count)
    # if 1+ peaks already exist in window, adding a peak increases energy
    full_mask = count > 0
    add_energy[full_mask] = count[full_mask]
    # if no peaks, adding a peak decreases energy
    empty_mask = (count == 0) & window_mask
    add_energy[empty_mask] = -1
    # if not between troughs, adding a peak slightly increases energy
    add_energy[~window_mask] += 0.5

    return add_energy


def extremum_remove_energy(count, window_mask):
    rem_energy = np.zeros_like(count)
    # If 1 peak & between troughs, removing it increases energy
    good_mask = (count == 1) & window_mask
    rem_energy[good_mask] = 1
    # If more than 1 peak, removing decreases energy
    rem_energy[count > 1] = -(count[count > 1] - 1)
    # If no peaks, can't remove
    # If not between troughs, removing a peak slightly decreases energy
    rem_energy[~window_mask] -= 0.5

    return rem_energy


def optimize_extrema_2d(ridge_mask, trough_mask, ridge_prob, trough_prob, max_energy_delta=0, max_iter=10,
                        attract_sigma=(5, 1), attraction=0.1, repulsion=10, lp_scale=1,
                        fixed_ridge_field=None, fixed_trough_field=None):
    rm_out = ridge_mask.copy()
    tm_out = trough_mask.copy()

    # Clip probability to prevent infinite log-prob
    ridge_prob = np.clip(ridge_prob, 1e-6, 1 - 1e-6)
    trough_prob = np.clip(trough_prob, 1e-6, 1 - 1e-6)
    ridge_lp = np.log(ridge_prob / (1 - ridge_prob)) * lp_scale
    trough_lp = np.log(trough_prob / (1 - trough_prob)) * lp_scale
    # ridge_lp = np.log(ridge_prob)
    # trough_lp = np.log(trough_prob)

    # attract_kernel = np.ones((3, 3))
    # attract_kernel[1] = 0

    att_ks = np.prod([gaussian_kernel_scale(s, empty=False) if s > 0 else 1 for s in attract_sigma])
    if fixed_ridge_field is None:
        fixed_ridge_field = 0
    if fixed_trough_field is None:
        fixed_trough_field = 0

    for i in range(max_iter):

        rcount, tmask = count_extrema(rm_out, tm_out)
        tcount, rmask = count_extrema(tm_out, rm_out, troughs=True)
        # rmask = np.ones(rmi.shape, dtype=bool)
        ridge_add_energy = extremum_add_energy(rcount, tmask) * repulsion
        ridge_remove_energy = extremum_remove_energy(rcount, tmask) * repulsion
        trough_add_energy = extremum_add_energy(tcount, rmask) * repulsion
        trough_remove_energy = extremum_remove_energy(tcount, rmask) * repulsion

        # Generate peak and ridge attraction fields
        # Ideally, field would not extend in tau dimension in same row,
        # but this would require constructing a customized 2d gaussian kernel with zeroed entries
        ridge_attraction_field = attraction * att_ks * ndimage.gaussian_filter(rm_out.astype(float),
                                                                             sigma=attract_sigma)
        ridge_attraction_field += fixed_ridge_field
        trough_attraction_field = attraction * att_ks * ndimage.gaussian_filter(tm_out.astype(float),
                                                                              sigma=attract_sigma)
        trough_attraction_field += fixed_trough_field

        # peak_attract = attraction * ndimage.convolve(rm_out.astype(float), attract_kernel)
        # trough_attraction_field = attraction * ndimage.convolve(rm_out.astype(float), attract_kernel)

        # Local peak/trough energy from log prob and attraction
        peak_energy = -(ridge_lp + ridge_attraction_field)
        trough_energy = -(trough_lp + trough_attraction_field)

        # Change in local energy
        ridge_delta_e = peak_energy * 2 * (0.5 - rm_out.astype(float))
        # Peak removal/addition energy
        ridge_delta_e += ridge_add_energy * (1 - rm_out.astype(float))
        ridge_delta_e += ridge_remove_energy * rm_out.astype(float)

        # Trough interaction energy
        trough_delta_e = trough_energy * 2 * (0.5 - tm_out.astype(float))
        # Trough removal/addition energy
        trough_delta_e += trough_add_energy * (1 - tm_out.astype(float))
        trough_delta_e += trough_remove_energy * tm_out.astype(float)
        # plot_x_2d(v_base, trough_delta_e, ax=axes[1, 3])

        r_index = np.argmin(ridge_delta_e, axis=-1)
        t_index = np.argmin(trough_delta_e, axis=-1)

        num_changed = 0
        for j in range(len(ridge_mask)):
            r_de = ridge_delta_e[j, r_index[j]]
            t_de = trough_delta_e[j, t_index[j]]
            #         print(r_de, t_de)
            if r_de <= t_de and r_de < max_energy_delta:
                rm_out[j, r_index[j]] = ~rm_out[j, r_index[j]]
                num_changed += 1
                # print(r_de)
                # if i > 2:
                #     print(f'changed peak at ({j}, {r_index[j]})')
            elif t_de < r_de and t_de < max_energy_delta:
                tm_out[j, t_index[j]] = ~tm_out[j, t_index[j]]
                num_changed += 1
                # print(r_de)
                # if i > 2:
                #     print(f'changed trough at ({j}, {t_index[j]})')

        if num_changed == 0:
            break

        # print(f'Changed {num_changed} rows at iteration {i}')

    return rm_out, tm_out


def optimize_extrema(ridge_mask, trough_mask, ridge_prob, trough_prob, attract_sigma=None,
                     max_energy_delta=0, max_iter=10,
                     attraction=1, repulsion=100, lp_scale=1, num_loops=1):
    num_group_dims = ridge_mask.ndim - 2
    it = np.nditer(ridge_mask, op_axes=[[i for i in range(num_group_dims)]], flags=['multi_index'])

    rm_out = ridge_mask.copy()
    tm_out = trough_mask.copy()

    if attract_sigma is None:
        attract_sigma = (1, ) * num_group_dims + (5, 1)

    att_ks = np.prod([gaussian_kernel_scale(s, empty=False) if s > 0 else 1 for s in attract_sigma])

    for n in range(num_loops):
        for _ in it:
            slice_index = it.multi_index
            if not np.all(np.isnan(ridge_prob[slice_index])):
                # Mask the extrema in the current slice.
                # Their contributions will be calculated within optimize_extrema_2d
                rm_masked = rm_out.astype(float)
                rm_masked[slice_index] = np.nan
                tm_masked = tm_out.astype(float)
                tm_masked[slice_index] = np.nan

                mask = np.isnan(ridge_prob) | np.isnan(rm_masked)

                ridge_field = masked_filter(np.nan_to_num(rm_masked), mask=~mask,
                                            filter_func=ndimage.gaussian_filter, sigma=attract_sigma
                                            )
                ridge_field *= attraction * att_ks
                trough_field = masked_filter(np.nan_to_num(tm_masked), mask=~mask,
                                             filter_func=ndimage.gaussian_filter, sigma=attract_sigma
                                             )
                trough_field *= attraction * att_ks

                rm_, tm_ = optimize_extrema_2d(rm_out[slice_index], tm_out[slice_index],
                                               ridge_prob[slice_index], trough_prob[slice_index],
                                               max_energy_delta, max_iter, attract_sigma[num_group_dims:],
                                               attraction, repulsion, lp_scale,
                                               fixed_ridge_field=ridge_field[slice_index],
                                               fixed_trough_field=trough_field[slice_index]
                                               )
                rm_out[slice_index] = rm_
                tm_out[slice_index] = tm_

    return rm_out, tm_out
