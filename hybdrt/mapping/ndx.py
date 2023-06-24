import numpy as np
from scipy import ndimage, signal, interpolate
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

from hybdrt.utils.array import rel_round, group_values
from hybdrt.filters._filters import iterative_gaussian_filter, adaptive_gaussian_filter, masked_filter, \
    get_adaptive_sigmas


# def make_peak_map(drt_array, pfrt_array, pfrt_thresh=None, filter_drt=False, drt_sigma=None,
#                   filter_pfrt=False, pfrt_sigma=None):
#     if filter_drt:
#         if drt_sigma is None:
#             drt_sigma = np.ones(np.ndim(drt_array))
#             drt_sigma[-1] = 0
#         drt_array = ndimage.gaussian_filter(drt_array, sigma=drt_sigma)
#
#     if filter_pfrt:
#         if pfrt_sigma is None:
#             pfrt_sigma = 2
#         pfrt_array = ndimage.gaussian_filter(pfrt_array, sigma=pfrt_sigma)
#
#     #


def resample(psi, psi_meas, x_meas, interp_class=None, interp_kw=None, remove_invariant=True):

    # Identify invariant dimensions
    # print(psi_meas.shape)
    if remove_invariant:
        dim_index = np.std(np.atleast_2d(psi_meas), axis=0) > 1e-8
    else:
        dim_index = np.ones(np.atleast_2d(psi).shape[1], dtype=bool)
    # print(dim_index)

    psi_meas_eff = np.atleast_2d(psi_meas)[:, dim_index]
    psi_eff = np.atleast_2d(psi)[:, dim_index]

    # Determine number of dimensions over which to resample
    ndim = np.sum(dim_index)
    # print('ndim:', ndim)

    if ndim == 1:
        if interp_kw is None:
            if interp_class is None:
                interp_kw = {'axis': 0}
            else:
                interp_kw = {}
        if interp_class is None:
            interp_class = interpolate.interp1d

        psi_eff = psi_eff.flatten()
        psi_meas_eff = psi_meas_eff.flatten()
        # print(psi_meas_eff.shape)
    else:
        if interp_kw is None:
            if interp_class is None:
                interp_kw = {'rescale': True}
            else:
                interp_kw = {}
        if interp_class is None:
            interp_class = interpolate.LinearNDInterpolator

    interp_func = interp_class(psi_meas_eff, x_meas, **interp_kw)

    return interp_func(psi_eff)


def assemble_ndx(x, psi, psi_dim_names, tau, sort_by=None, group_by=None, psi_precision=8, sort_dim_grids=None,
                 sort_dim_dist_thresh=None, impute=False):
    """
    Assemble n-dimensional array from 2-D array
    :param ndarray x: raw 2-D array of x values (N x M)
    :param ndarray psi: 2-D array of psi values (N x P)
    :param list psi_dim_names: list of names for psi dimensions (P)
    :param ndarray tau: tau vector (M)
    :param list sort_by: list of psi dimensions by which to sort x values within groups
    :param list group_by: list of psi dimensions by which to group x values
    :param int psi_precision: relative precision for psi; used for identifying distinct values
    :param list sort_dim_grids: list of grid values to use for each sort dimensions. If not specified, each sort dimension
    will be automatically segmented
    :param bool impute: if True, impute missing x values from nearest neighbor within groups
    :return:
    """
    # TODO: how will this be used?
    #  This provides an ND array that can be filtered to remove outliers, pad edges, in-fill, etc.
    #  There may be groups that are all nan - need to handle this with appropriate masking
    #  Also need to consider which x/psi are fed to this function - might want to limit to a single temperature.
    #  MUST avoid feeding duplicates to this function
    psi = rel_round(psi, psi_precision)

    if sort_by is None:
        sort_by = []

    if group_by is None:
        group_by = []

    if sort_dim_grids is None:
        sort_dim_grids = [None] * len(sort_by)

    if sort_dim_dist_thresh is None:
        sort_dim_dist_thresh = [None] * len(sort_by)

    # num_dims = max(1 + len(sort_by) + len(group_by), 2)

    shape = []
    dim_grid_values = []

    # Get unique values of group dimensions
    for dim in group_by:
        dim_vals = psi[:, psi_dim_names.index(dim)]
        unique_vals = np.unique(dim_vals)
        shape.append(len(unique_vals))
        dim_grid_values.append(unique_vals)

    # Determine number of groups
    if len(group_by) > 0:
        group_dims = psi[:, [psi_dim_names.index(d) for d in group_by]]
        group_dim_values = np.unique(group_dims, axis=0)
        # print('group_dim_values:', group_dim_values)
        psi_group_vals = psi[:, [psi_dim_names.index(d) for d in group_by]]
        num_groups = len(group_dim_values)
    else:
        num_groups = 1

    # Get grid values for sort dimensions
    sort_distance_thresholds = []
    for i, dim in enumerate(sort_by):
        grid_vals = sort_dim_grids[i]
        distance_thresh = sort_dim_dist_thresh[i]
        if grid_vals is None:
            dim_vals = psi[:, psi_dim_names.index(dim)].copy()
            if num_groups > 1:
                # Select grid values based on clusters in measured values
                # Allow 1/3 of groups to omit a grid value
                min_samples = max(num_groups - int(np.ceil(num_groups / 3)), 2)
                # print('min_samples:', min_samples)
                grid_vals, distance_thresh = segment_dimension(dim_vals, min_samples=min_samples,
                                                               return_distance_thresh=True)
            else:
                grid_vals = np.unique(dim_vals)
                if distance_thresh is None:
                    distance_thresh = np.min(np.diff(grid_vals))
        else:
            grid_vals = np.unique(grid_vals)
            if distance_thresh is None:
                distance_thresh = np.median(np.diff(grid_vals)) * 0.5

        # print(f'{dim} distance thresh:', distance_thresh)

        # print(dim, grid_vals)

        shape.append(len(grid_vals))
        dim_grid_values.append(grid_vals)
        sort_distance_thresholds.append(distance_thresh)

    #
    if len(sort_by) > 0:
        sort_dim_mesh = np.meshgrid(*dim_grid_values[len(group_by):][::-1])
        psi_interp_points = np.vstack([vals.flatten() for vals in sort_dim_mesh]).T
        # Scale to threshold distance in each dimension
        for i in range(len(sort_by)):
            psi_interp_points[:, i] /= sort_distance_thresholds[i]
        # print('psi_interp_shape:', psi_interp_points.shape)

        # Add a dummy column to allow use of NDInterpolator, which allows extrapolation
        if len(sort_by) == 1:
            psi_interp_points = np.hstack([psi_interp_points, np.ones((len(psi_interp_points), 1))])

        psi_sort_vals = psi[:, [psi_dim_names.index(d) for d in sort_by]]

    # # Append tau dimension
    # shape.append(len(tau))
    # dim_grid_values.append(tau)

    # print('dim_grid_values:', dim_grid_values)
    # print([len(dv) for dv in dim_grid_values])

    # Assemble psi array
    # if len(dim_grid_values) > 0:
    #     psi_mesh = np.meshgrid(*dim_grid_values)
    #     psi_mesh = [np.moveaxis(pm, 0, 1) for pm in psi_mesh]
    #     # print('psi_mesh shape:', psi_mesh[0].shape)
    # else:
    #     # No sorting or grouping performed
    #     psi_mesh = psi.copy()

    # Assemble x array
    x_out = np.empty((*shape, len(tau)))
    x_out.fill(np.nan)
    # print('x_out shape:', x_out.shape)
    if num_groups > 1:
        for i, group_vals in enumerate(group_dim_values):
            # Get indices of input psi corresponding to group
            in_group_index = [np.array_equal(pgv, group_vals) for pgv in psi_group_vals]
            meas_x = x[in_group_index]
            # print('meas_x shape:', meas_x.shape)

            # Get indices of output psi
            out_group_index = []
            for j, val in enumerate(group_vals):
                out_group_index.append(np.where(dim_grid_values[j] == val)[0][0])
            # print('out_group_index:', out_group_index)

            if len(sort_by) > 0:
                # Get coordinates of measured points in sort dimensions
                meas_points = psi_sort_vals[in_group_index].copy()
                # Scale to threshold distance in each dimension
                for j in range(len(sort_by)):
                    meas_points[:, j] /= sort_distance_thresholds[j]

                # Add a dummy column for ND interp
                if len(sort_by) == 1:
                    meas_points = np.hstack([meas_points, np.ones((len(meas_points), 1))])

                x_interp = resample(psi_interp_points, meas_points, meas_x, remove_invariant=False,
                                    interp_class=interpolate.NearestNDInterpolator)

                if not impute:
                    # Remove any interpolated values that exceed the neighbor distance threshold
                    nn_distance = np.min(cdist(psi_interp_points, meas_points), axis=1)
                    x_interp[nn_distance > 1.0] = np.nan

                # print('x_interp shape:', x_interp.shape)
                x_interp = x_interp.reshape([*sort_dim_mesh[0].shape, len(tau)])
                # print('x_interp reshaped:', x_interp.shape)
                x_out[tuple(out_group_index)] = x_interp

            else:
                x_out[tuple(out_group_index)] = meas_x
    else:
        if len(sort_by) > 0:
            # Get coordinates of measured points in sort dimensions
            meas_points = psi_sort_vals.copy()
            # Scale to threshold distance in each dimension
            for j in range(len(sort_by)):
                meas_points[:, j] /= sort_distance_thresholds[j]

            # Add a dummy column for ND interp
            if len(sort_by) == 1:
                meas_points = np.hstack([meas_points, np.ones((len(meas_points), 1))])

            x_interp = resample(psi_interp_points, meas_points, x, remove_invariant=False,
                                interp_class=interpolate.NearestNDInterpolator)

            if not impute:
                # Remove any interpolated values that exceed the neighbor distance threshold
                nn_distance = np.min(cdist(psi_interp_points, meas_points), axis=1)
                x_interp[nn_distance > 1.0] = np.nan

            x_out = x_interp
        else:
            # No grouping or sorting applied
            x_out = x

    return dim_grid_values, x_out


def filter_ndx(ndx, num_group_dims, impute=False, impute_groups=False, iterative=True, adaptive=False,
               mask_nans=True, filter_func=None, by_group=False,
               **filter_kw):
    if impute_groups and by_group:
        raise ValueError('Group imputation cannot be performed when filtering by group')

    # Identify nan entries
    nan_obs_index = np.isnan(ndx)

    # Identify nan groups
    # nan_group_index = np.all(np.isnan(ndx), axis=tuple(np.arange(-1, -(num_sort_dims + 2), -1)))
    nan_group_index = group_isnan(ndx, num_group_dims)

    # Set nans to zero to prevent spreading
    if mask_nans:
        ndx = np.nan_to_num(ndx)

    if by_group:
        # Filter within groups
        # num_group_dims = np.ndim(ndx) - (num_sort_dims + 1)
        it = np.nditer(ndx, op_axes=[list(np.arange(num_group_dims))], flags=['multi_index'])
        out = np.empty_like(ndx)
        for group in it:
            group_index = it.multi_index
            out[group_index] = _filter_ndx_sub(ndx[group_index], nan_obs_index[group_index],
                                               filter_func, filter_kw, mask_nans, iterative, adaptive)

    else:
        # Filter over all dimensions
        out = _filter_ndx_sub(ndx, nan_obs_index, filter_func, filter_kw, mask_nans, iterative, adaptive)

    # All nans will be filled. Reset nans based on imputation settings
    if impute:
        if not impute_groups:
            out[nan_group_index] = np.nan
    else:
        out[nan_obs_index] = np.nan

    return out


def _filter_ndx_sub(x_sub, nan_obs_index, filter_func, filter_kw, mask_nans, iterative, adaptive):
    """
    Sub-function for filtering N-D arrays
    :param x_sub:
    :param nan_obs_index:
    :param filter_func:
    :param filter_kw:
    :param mask_nans:
    :param iterative:
    :param adaptive:
    :return:
    """
    if filter_func is not None:
        if mask_nans:
            weights = (~nan_obs_index).astype(float)
            out = masked_filter(x_sub, weights, filter_func=filter_func, **filter_kw)
        else:
            out = filter_func(x_sub, **filter_kw)
    elif iterative:
        if mask_nans:
            nan_mask = nan_obs_index
        else:
            nan_mask = None
        out = iterative_gaussian_filter(x_sub, adaptive=adaptive, nan_mask=nan_mask, fill_nans=True, **filter_kw)
    else:
        if mask_nans:
            weights = (~nan_obs_index).astype(float)
        else:
            weights = None

        if adaptive:
            sigmas = get_adaptive_sigmas(x_sub, weights=weights, **filter_kw)

            def filter_func(a_in, **kw):
                return adaptive_gaussian_filter(a_in, sigmas=sigmas, **kw)

            # out = adaptive_gaussian_filter(ndx, weights=weights, **filter_kw)
            if mask_nans:
                out = masked_filter(x_sub, weights, filter_func=filter_func, **filter_kw)
            else:
                out = filter_func(x_sub, **filter_kw)
        else:
            if mask_nans:
                out = masked_filter(x_sub, weights, filter_func=ndimage.gaussian_filter, **filter_kw)
            else:
                out = ndimage.gaussian_filter(x_sub, **filter_kw)

    return out


def flatten_groups(ndx, num_group_dims):
    new_shape = (*ndx.shape[:num_group_dims], np.prod(ndx.shape[num_group_dims:]))
    return ndx.reshape(new_shape)


def group_isnan(ndx, num_group_dims):
    x_flat = flatten_groups(ndx, num_group_dims)
    return np.all(np.isnan(x_flat), axis=-1)


def group_nn_count(ndx, num_group_dims, axis=None):
    group_exists = ~group_isnan(ndx, num_group_dims)

    footprint = np.zeros((3, ) * num_group_dims)
    if axis is None:
        axis = np.arange(num_group_dims)
    elif np.isscalar(axis):
        axis = [axis]

    for axis in axis:
        ind1 = [1] * axis + [0] + [1] * (num_group_dims - 1 - axis)
        ind2 = ind1.copy()
        ind2[axis] = 2

        footprint[tuple(ind1)] = 1
        footprint[tuple(ind2)] = 1

    num_neighbors = ndimage.convolve(group_exists.astype(float), footprint, mode='constant')

    # if axis is not None:
    #     size = np.ones(num_group_dims, dtype=int)
    #     size[axis] = 3
    #     size = tuple(size)
    # else:
    #     size = (3, ) * num_group_dims

    # num_neighbors = ndimage.uniform_filter(group_exists.astype(float), size=size, mode='constant')
    # num_neighbors *= np.prod(size)
    # num_neighbors -= group_exists

    return num_neighbors


def segment_dimension(a, min_samples=3, return_distance_thresh=False):
    a = np.unique(a)

    # Estimate density of gaps in dimension
    diffs = np.diff(a)
    kde = KernelDensity(kernel='gaussian', bandwidth=np.percentile(diffs, 99) / 20)
    kde.fit(diffs[:, None])
    # print(diffs)

    # Find the end of the first cluster of diffs
    x = np.linspace(np.min(diffs), np.max(diffs), 1000)
    density = kde.score_samples(x[:, None])
    # fig, ax = plt.subplots()
    # ax.plot(x, density)
    # Find the first trough after the first peak
    first_peak_index = signal.argrelextrema(density, np.greater_equal)[0][0]
    print(first_peak_index)
    d_cluster = x[signal.argrelextrema(density[first_peak_index:], np.less_equal)[0][0] + first_peak_index]
    print('d_cluster:', d_cluster)

    # Cluster by density
    db = DBSCAN(eps=d_cluster, min_samples=min_samples)
    group_indices = db.fit_predict(a[:, None])

    # Sort by group
    sort_index = np.argsort(group_indices)
    a = a[sort_index]
    group_indices = group_indices[sort_index]

    # Get mean of each group (excluding outliers)
    a_groups = group_values(a, group_indices, sort=False)
    cluster_means = [np.mean(vals) for index, vals in zip(group_indices, a_groups) if index > -1]

    if return_distance_thresh:
        return np.sort(cluster_means), d_cluster
    else:
        return np.sort(cluster_means)





