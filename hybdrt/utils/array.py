import numpy as np


# Utility functions
# -----------------
def check_equality(a, b):
    """
    Convenience function for testing equality of arrays or dictionaries containing arrays
    :param dict or ndarray a: First object
    :param dict or ndarray b: Second object
    :return: bool
    """
    out = True
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        out = False

    return out


def rel_round(x, precision):
    """
    Round to relative precision
    :param ndarray x: array of numbers to round
    :param int precision : number of digits to keep
    :return: rounded array
    """
    try:
        # add 1e-30 for safety in case of zeros in x
        x_scale = np.floor(np.log10(np.array(np.abs(x)) + 1e-30))
        digits = (precision - x_scale).astype(int)
        # print(digits)
        if type(x) in (list, np.ndarray):
            if type(x) == list:
                x = np.array(x)
            shape = x.shape
            x_round = np.array([round(xi, di) for xi, di in zip(x.flatten(), digits.flatten())])
            x_round = np.reshape(x_round, shape)
        else:
            x_round = round(x, digits)
        return x_round
    except TypeError:
        return x


def is_subset(x, y, precision=10):
    """
    Check if x is a subset of y
    :param ndarray x: candidate subset array
    :param ndarray y: candidate superset array
    :param int precision: number of digits to compare. If None, compare exact values (or non-numeric values)
    :return: bool
    """
    if precision is None:
        # # Compare exact or non-numeric values
        # return np.min([xi in y for xi in x])
        set_x = set(x)
        set_y = set(y)
        return set_x.issubset(set_y)
    else:
        # Compare rounded values
        set_x = set(rel_round(x, precision))
        set_y = set(rel_round(y, precision))
        return set_x.issubset(set_y)
        # return np.min([rel_round(xi, precision) in rel_round(y, precision) for xi in x])


def get_intersection_index(x1, x2, precision=10):
    """
    Get indices which x1 and x2 intersect
    :param ndarray x1: first array
    :param ndarray x2: second array
    :param int precision: relative precision to match on
    :return: x1_index, x2_index
    """
    x_int, x1_index, x2_index = np.intersect1d(rel_round(x1, precision), rel_round(x2, precision),
                                                   return_indices=True)

    return x1_index, x2_index


def get_subset_index(subset, superset, precision=10):
    vals, sub_index, sup_index = np.intersect1d(rel_round(subset, precision), rel_round(superset, precision),
                                               return_indices=True)

    sort_index = np.argsort(sub_index)
    return sup_index[sort_index]


def weighted_quantile_2d(values, quantiles, sample_weight=None, axis=0,
                      values_sorted=False, old_style=False):
    """
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    Modified for multi-dim arrays from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    :param values: 2d array of values
    :param quantiles: scalar or array-like with quantile(s) needed
    :param sample_weight: 1d array-like of the same size as values.shape[axis]
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values, axis=axis)
        values = np.take_along_axis(values, sorter, axis=axis)
        sample_weight = np.array([sample_weight[index] for index in sorter])

    weighted_quantiles = np.cumsum(sample_weight, axis=axis) - 0.5 * sample_weight

    if old_style:
        # To be consistent with numpy.percentile
        weighted_quantiles -= np.take(weighted_quantiles, 0, axis=axis)
        weighted_quantiles /= np.take(weighted_quantiles, -1, axis=axis)
    else:
        weighted_quantiles /= np.sum(sample_weight, axis=axis)

    take_axis = (axis + 1) % 2

    quant_out = [
        np.array([
            np.interp(quantile, np.take(weighted_quantiles, i, take_axis), np.take(values, i, take_axis))
            for i in range(values.shape[take_axis])
        ])
        for quantile in quantiles
    ]

    return quant_out


def is_uniform(x):
    """
    Check if x is uniformly spaced
    :param ndarray x: input array
    :return: bool
    """
    xdiff = np.diff(x)
    if np.std(xdiff) / np.mean(xdiff) <= 0.01:
        return True
    else:
        return False


def is_log_uniform(x):
    """
    Check if x is uniformly log-distributed
    :param ndarray x: input array
    :return: bool
    """
    return is_uniform(np.log(x))


def apply_unit_step(times, t_step, func):
    out = np.zeros_like(times)
    out[times < t_step] = 0
    out[times >= t_step] = func(times)

    return out


def unit_step(t, ts=0.0):
    """
    Unit step function. Value is 0 before ts, 1 after ts
    :param ndarray t: times
    :param float ts: step time
    :return: array of step function values
    """
    out = np.zeros_like(t)
    out[t < ts] = 0
    out[t >= ts] = 1
    return out


def is_monotonic_ascending(x):
    """
    Check if array is monotonically increasing
    :param ndarray x: array to check
    :return:
    """
    x_diff = np.diff(x)
    if np.min(x_diff) >= 0:
        return True
    else:
        return False


def is_monotonic(x):
    """
    Check if array is monotonically increasing OR decreasing
    :param ndarray x: array to check
    :return:
    """
    return any([is_monotonic_ascending(x), is_monotonic_ascending(-x)])


def nearest_index(x_array, x_val):
    """
    Get index of x_array corresponding to value closest to x_val
    :param ndarray x_array: Array to index
    :param float x_val: Value to match
    :return:
    """
    return np.argmin(np.abs(x_array - x_val))


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def inctrapz(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Incremental integral via trapezoidal rule. Equivalent to diff(cumtrapz)
    :param y:
    :param x:
    :param dx:
    :param axis:
    :param initial:
    :return:
    """
    y = np.asarray(y)
    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        else:
            d = np.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    nd = len(y.shape)
    slice1 = tupleset((slice(None),) * nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),) * nd, axis, slice(None, -1))
    res = d * (y[slice1] + y[slice2]) / 2.0

    if initial is not None:
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res],
                             axis=axis)

    return res


def find_contiguous_ranges(indices):
    """Find start and end of contiguous ranges"""
    range_starts = np.insert(np.where(np.diff(indices) > 1)[0] + 1, 0, 0)
    range_ends = np.append(range_starts[1:], len(indices))
    return indices[range_starts], indices[range_ends - 1] + 1


def find_contiguous_centers(indices):
    """Find centers of contiguous ranges"""
    range_starts, range_ends = find_contiguous_ranges(indices)
    return [int(0.5 * (start + end - 1)) for start, end in zip(range_starts, range_ends)]



# def save_pickle(obj, file):
#     with open(file, 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#     print('Dumped pickle to {}'.format(file))
#
#
# def load_pickle(file):
#     with open(file, 'rb') as f:
#         return pickle.load(f)
