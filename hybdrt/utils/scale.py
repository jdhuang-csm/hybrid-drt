import numpy as np

prefix_map = {-2: r'$\mu$', -1: 'm', 0: '', 1: 'k', 2: 'M', 3: 'G'}


def get_scale_prefix(y):
    """Get unit scale (mu, m, k, M, G) for array"""
    # If data is complex, get max over real and imag components
    if np.max(np.abs(np.imag(y))) > 0:
        y = np.concatenate((y.real, y.imag))
    y_ord = np.floor(np.log10(np.max(np.abs(y))) / 3)
    prefix = prefix_map.get(y_ord, '')
    return prefix


def get_scale_factor(y):
    # If data is complex, get max over real and imag components
    if np.max(np.abs(np.imag(y))) > 0:
        y = np.concatenate((y.real, y.imag))
    y_ord = np.floor(np.log10(np.max(np.abs(y))) / 3)
    return 10 ** (3 * y_ord)


def get_scale_prefix_and_factor(y):
    prefix = get_scale_prefix(y)
    factor = get_scale_factor(y)
    return prefix, factor


def get_factor_from_prefix(prefix):
    pwr_map = {v: k for k, v in prefix_map.items()}
    pwr = pwr_map[prefix]
    return 10 ** (3 * pwr)


def get_common_scale_prefix(y_list, aggregate='max'):
    """
    Get common unit scale for multiple datasets
    Parameters:
        df_list: list of DataFrames
        aggregate: method for choosing common scale. Defaults to min (smallest scale)
    """
    rev_map = {v: k for k, v in prefix_map.items()}
    prefixes = [get_scale_prefix(y) for y in y_list]
    powers = [rev_map[p] for p in prefixes]
    common_power = getattr(np, aggregate)(powers)
    common_prefix = prefix_map.get(common_power, '')
    return common_prefix