import numpy as np
from scipy import ndimage
from pathlib import Path, WindowsPath

from .. import fileload as fl
from . import ndx
from ..preprocessing import outlier_prob
from ..utils import stats
from ..utils.eis import complex_vector_to_concat
from ..filters import masked_filter, iqr_filter, std_filter


def assemble_nddata(data_list, psi, psi_dim_names, data_type=None, truncate=False,
                    sort_by=None, group_by=None, sort_dim_grids=None, sort_dim_dist_thresh=None, impute=False):
    # Set data reader
    if data_type == 'chrono':
        def data_reader(file):
            t, i_sig, v_sig = fl.get_chrono_tuple(fl.read_chrono(file))
            return v_sig
    elif data_type == 'eis':
        def data_reader(file):
            f, z = fl.get_eis_tuple(fl.read_eis(file))
            return z
    elif data_type == 'hybrid':
        def data_reader(files):
            if files[0] is not None:
                t, i_sig, v_sig = fl.get_chrono_tuple(fl.read_chrono(files[0]))
            else:
                v_sig = []

            if files[1] is not None:
                f, z = fl.get_eis_tuple(fl.read_eis(files[1]))
            else:
                z = []

            return v_sig, z
    elif data_type is not None:
        data_reader = data_type
    else:
        # data_list must be a list of data arrays
        data_reader = None

    # Read data
    if type(data_list[0]) in (str, Path, WindowsPath, tuple):
        if data_type is None:
            raise ValueError('If data_list is a list of files, data_type must be provided')
        y_list = []
        for data in data_list:
            y_list.append(data_reader(data))
    else:
        y_list = data_list

    # Determine data vector size
    if truncate:
        # Truncate to shortest vector
        len_func = np.min
    else:
        # Pad to match longest vector
        len_func = np.max

    if data_type == 'hybrid':
        chrono_len = np.array([len(y[0]) for y in y_list])
        eis_len = np.array([len(y[1]) for y in y_list])
        chrono_len[chrono_len == 0] = np.max(chrono_len)
        eis_len[eis_len == 0] = np.max(eis_len)

        chrono_len = len_func(chrono_len)
        eis_len = len_func(eis_len)
        # grid_len = chrono_len * eis_len
    # elif np.iscomplex(y_list[0]):
    #     chrono_len, eis_len = None, None
    #     grid_len = 2 * len_func([len(y) for y in y_list])

        y_lists = [[y[0] for y in y_list], [y[1] for y in y_list]]
        grid_lens = [chrono_len, eis_len]
    else:
        grid_len = len_func([len(y) for y in y_list])

        y_lists = [y_list]
        grid_lens = [grid_len]


    # Format 2d array
    # if data_type == 'hybrid':
    #     y_lists = [[y[0] for y in y_list], [y[1] for y in y_list]]
    #     grid_lens = [chrono_len, eis_len]
    # else:
    #     y_lists = [y_list]
    #     grid_lens = [grid_len]

    y_out = []
    for y_list, grid_len in zip(y_lists, grid_lens):
        y_arr = np.empty((len(y_list), grid_len), dtype=y_list[0].dtype)
        y_arr.fill(np.nan)
        for i, y_i in enumerate(y_list):
            ylen = min(grid_len, len(y_i))
            y_arr[i, :ylen] = y_i[:ylen].copy()

            # if data_type == 'hybrid':
            #     v_i = y_list[i][0]
            #     z_i = y_list[i][1]
            #
            #     vlen = min(chrono_len, len(v_i))
            #     # zlen = min(eis_len, len(z_i))
            #
            #     y_arr[i, :vlen] = v_i[:vlen]
            #
            #     if eis_len <= len(z_i):
            #         y_eis = complex_vector_to_concat(z_i[:eis_len])
            #     else:
            #         # Pad with nans and convert complex to concatenated
            #         y_eis = np.empty(eis_len, dtype=complex)
            #         y_eis.fill(np.nan)
            #         y_eis[:len(z_i)] = z_i
            #         y_eis = complex_vector_to_concat(y_eis)
            #     y_arr[i, chrono_len:] = y_eis

        if y_arr.dtype == complex:
            y_arr = complex_vector_to_concat(y_arr, axis=-1)

        dim_grid_values, psi_mesh, ndy = ndx.assemble_ndx(
            y_arr, psi, psi_dim_names, tau=np.arange(y_arr.shape[-1]), sort_by=sort_by,
            group_by=group_by, sort_dim_grids=sort_dim_grids, sort_dim_dist_thresh=sort_dim_dist_thresh,
            impute=impute
        )

        y_out.append(ndy)

    if len(y_out) == 1:
        return dim_grid_values, psi_mesh, y_out[0]
    else:
        return dim_grid_values, psi_mesh, tuple(y_out)


def impute_nans(ndy, method='filter', filter_func=None, **filter_kw):
    if method == 'filter':
        nan_index = np.isnan(ndy)
        mask = (~nan_index).astype(float)

        # Filter to fill nans
        y_filt = masked_filter(np.nan_to_num(ndy), mask, filter_func=filter_func, **filter_kw)

        # Replace nans with filtered values
        y_out = ndy.copy()
        y_out[nan_index] = y_filt[nan_index]
    else:
        raise ValueError(f'Imputation method {method} not implemented')

    return y_out


def flag_outliers(ndy, filter_size, thresh=0.9, p_prior=0.01, impute=True, impute_kw=None):

    # Impute nans
    if impute and np.any(np.isnan(ndy)):
        if impute_kw is None:
            impute_kw = {'sigma': 0.5}
        y_filt = impute_nans(ndy, **impute_kw)
    else:
        y_filt = ndy

    # Calculate local center and spread w/robust metrics
    mu_in = ndimage.median_filter(y_filt, filter_size)
    sigma_in = iqr_filter(y_filt, size=filter_size) / 1.349
    sigma_in += 0.05 * stats.robust_std(np.nan_to_num(y_filt, np.nanmedian(y_filt))) + 1e-8
    sigma_out = np.abs(ndy - mu_in) + 1e-8

    p_out = outlier_prob(ndy, mu_in, sigma_in, sigma_out, p_prior)
    p_out = np.nan_to_num(p_out)

    return p_out > thresh


def flag_bad_obs(x_raw, x_filt, std_size=5, thresh=2, test_factor_correction=False, test_offset_correction=False,
                 return_rss=False, robust_std=True):

    if type(x_raw) in (list, tuple):
        x_raw_list = x_raw
        x_filt_list = x_filt
    else:
        x_raw_list = [x_raw]
        x_filt_list = [x_filt]

    bad_index = []
    rss_list = []
    for xri, xfi in zip(x_raw_list, x_filt_list):
        # Get local std
        # x_std = std_filter(xfi, size=std_size)
        xfi_tmp = xfi.copy()
        xfi_tmp[np.isnan(xfi_tmp)] = np.nanmedian(xfi_tmp)

        if robust_std:
            x_std = iqr_filter(xfi_tmp, size=std_size) / 1.349
            x_std += 0.1 * stats.robust_std(xfi[~np.isnan(xfi)])
        else:
            x_std = std_filter(xfi_tmp, size=std_size)
            x_std += 0.1 * np.std(xfi[~np.isnan(xfi)])
        # x_std += 0.1 * np.std(xfi)
        # x_std /= 2
        # print('x_std:', x_std)
        if np.any(np.isnan(x_std)):
            raise ValueError('x_std contains nans!')

        # Get RSS for each observation
        resid = np.nan_to_num((xri - xfi) / x_std)
        rss = np.sum(resid ** 2, axis=-1) / xri.shape[-1]
        rss_list.append(rss)

        # print('rss:', rss)

        # Flag observation if mean squared residual exceeds thresh std devs
        bad = np.zeros(xri.shape, dtype=bool)
        bad[rss >= thresh] = 1
        bad_index.append(bad)

    correct_funcs = []
    if test_factor_correction:
        correct_funcs.append(lambda x: factor_correction(*x))
    if test_offset_correction:
        correct_funcs.append(lambda x: offset_correction(*x))

    if len(correct_funcs) > 0:
        x_corrected = [xi.copy() for xi in x_raw_list]

        for cfunc in correct_funcs:
            x_test = [xi.copy() for xi in x_raw_list]

            # Apply corrections to obs for which all measurements are bad
            all_bad = np.all(np.concatenate(bad_index, axis=-1), axis=-1)
            print('all_bad:', np.where(all_bad))

            if np.any(all_bad):
                x_raw_in = tuple([xi[all_bad] for xi in x_raw_list])
                x_filt_in = tuple([xi[all_bad] for xi in x_filt_list])
                x_cor = cfunc((x_raw_in, x_filt_in))

                for i in range(len(x_test)):
                    x_test[i][all_bad] = x_cor[i]

            # Apply corrections to observations for which only one measurement is bad
            for i, bad in enumerate(bad_index):
                one_bad = ~all_bad & np.all(bad, axis=-1)
                print(i, 'one_bad:', np.where(one_bad))
                if np.any(one_bad):
                    x_cor = cfunc((x_raw_list[i][one_bad], x_filt_list[i][one_bad]))
                    x_test[i][one_bad] = x_cor

            # Check if the correction fixed any observations
            test_bad, test_rss = flag_bad_obs(x_test, x_filt_list, std_size=std_size, thresh=thresh,
                                              test_factor_correction=False, test_offset_correction=False,
                                              return_rss=True)

            if len(x_raw_list) == 1:
                test_bad = [test_bad]
                test_rss = [test_rss]

            # Insert corrected data & update bad index
            for i, (bi, tbi) in enumerate(zip(bad_index, test_bad)):
                # print(np.where(bi), np.where(tbi))
                fixed_index = bi & ~tbi
                x_corrected[i][np.where(fixed_index)] = x_test[i][np.where(fixed_index)]
                bad_index[i] = bi & ~fixed_index
                rss_list[i][np.any(fixed_index, axis=1)] = test_rss[i][np.any(fixed_index, axis=1)]
                print('fixed:', np.where(np.all(fixed_index, axis=-1)))

            # print(np.all(fixed_index, axis=1))

            # x_corrected[np.where(fixed_index)] = x_test[np.where(fixed_index)]
            # bad = bad & ~fixed_index

        if len(bad_index) == 1:
            if return_rss:
                return bad_index[0], x_corrected[0], rss_list[0]
            else:
                return bad_index[0], x_corrected[0]
        else:
            if return_rss:
                return bad_index, x_corrected, rss_list
            else:
                return bad_index, x_corrected
    else:
        if len(bad_index) == 1:
            if return_rss:
                return bad_index[0], rss_list[0]
            else:
                return bad_index[0]
        else:
            if return_rss:
                return bad_index, rss_list
            else:
                return bad_index


def factor_correction(x_raw, x_filt, x_floor=1e-6):
    # Apply same factor to all datasets
    if type(x_raw) in (list, tuple):
        x_raw_ = np.concatenate(x_raw, axis=-1)
        x_filt_ = np.concatenate(x_filt, axis=-1)
    else:
        x_raw_ = x_raw
        x_filt_ = x_filt

    test_index = np.abs(x_raw_) > x_floor

    # Get factor to match x_raw to x_filt
    factors = np.empty_like(x_raw_)
    factors.fill(np.nan)
    factors[test_index] = x_filt_[test_index] / x_raw_[test_index]
    factors = np.nanmedian(factors, axis=-1)
    # factors = np.nanmean(factors, axis=-1)
    print(factors)

    # Apply factor correction
    x_cor_ = x_raw_ * np.expand_dims(factors, axis=-1)

    # Split if concatenated
    if type(x_raw) in (list, tuple):
        x_cor = []
        i = 0
        for xi in x_raw:
            x_cor.append(x_cor_[:, i:i + xi.shape[-1]])
            i += xi.shape[-1]
    else:
        x_cor = x_cor_

    # print(x_cor)

    return x_cor


def offset_correction(x_raw, x_filt):
    # Test offsets independently
    if type(x_raw) in (list, tuple):
        x_raw_list = x_raw
        x_filt_list = x_filt
    else:
        x_raw_list = [x_raw]
        x_filt_list = [x_filt]

    x_cor = []
    for x_raw, x_filt in zip(x_raw_list, x_filt_list):
        offsets = x_raw - x_filt
        offsets = np.nanmedian(offsets, axis=-1)
        x_cor.append(x_raw + np.expand_dims(offsets, axis=-1))

    if len(x_cor) == 1:
        return x_cor[0]
    else:
        return x_cor





