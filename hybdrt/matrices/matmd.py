import numpy as np
# import time

from . import mat1d as mat1d
from .. import utils


# =========================
# Impedance matrices
# =========================

def construct_md_impedance_matrix(frequency_list, basis_tau, basis_type, epsilon, special_parameters,
                                  frequency_precision,
                                  integrate_method='trapz', integrate_points=1000, zga_params=None,
                                  interpolate_lookups=None):
    # t0 = time.time()
    num_obs = len(frequency_list)
    num_basis = len(basis_tau)
    num_special = len(special_parameters)
    num_params = num_basis + num_special

    # Concatenate frequencies from individual observations
    frequency_list = [f for f in frequency_list if f is not None]
    if len(frequency_list) > 0:
        f_flat = utils.array.rel_round(np.concatenate(frequency_list), frequency_precision)

        # One row for each unique (frequency, psi) combo, one column for each unique (tau, psi) combo
        zm = np.zeros((2 * len(f_flat), num_params * num_obs))
        # t1 = time.time()
        # print('setup time: {:.3f} s'.format(t1 - t0))

        # Get unique frequencies and index to map all frequencies to f_unique
        f_unique, inverse_index = np.unique(f_flat, return_inverse=True)

        # Sort frequencies descending to ensure same order as basis_tau - this is important for toeplitz check
        f_unique = f_unique[::-1]
        inverse_index = inverse_index[::-1]
        # t2 = time.time()
        # print('index time: {:.3f} s'.format(t2 - t1))

        # Construct reference matrix using f_unique
        if integrate_method == 'interp':
            z_re_lookup = interpolate_lookups['z_real']
            z_im_lookup = interpolate_lookups['z_imag']
        else:
            z_re_lookup, z_im_lookup = None, None
        zm_re_ref = mat1d.construct_impedance_matrix(f_unique, 'real', basis_tau, basis_type, epsilon,
                                                     frequency_precision - 1, integrate_method, integrate_points,
                                                     zga_params, z_re_lookup)
        zm_im_ref = mat1d.construct_impedance_matrix(f_unique, 'imag', basis_tau, basis_type, epsilon,
                                                     frequency_precision - 1, integrate_method, integrate_points,
                                                     zga_params, z_im_lookup)
        # t3 = time.time()
        # print('zm_ref calc time: {:.3f} s'.format(t3 - t2))

        # Map reference matrix elements to all frequencies, duplicating for repeated values
        zm_re_flat = zm_re_ref[inverse_index]
        zm_im_flat = zm_im_ref[inverse_index]
        # t4 = time.time()
        # print('zm_flat time: {:.3f} s'.format(t4 - t3))

        row_start = 0
        ref_row_start = 0
        for i, frequencies in enumerate(frequency_list):
            num_f = utils.md.get_data_tuple_length(frequencies)
            re_row_start = row_start
            re_row_end = row_start + num_f
            im_row_start = re_row_end
            im_row_end = im_row_start + num_f

            if num_f > 0:
                frequencies

                # Insert DRT impedance matrix for observation i
                zm[re_row_start:re_row_end, num_params * i + num_special:num_params * (i + 1)] = \
                    zm_re_flat[ref_row_start:ref_row_start + num_f]
                zm[im_row_start:im_row_end, num_params * i + num_special:num_params * (i + 1)] = \
                    zm_im_flat[ref_row_start:ref_row_start + num_f]

                # Insert special parameter response columns
                if 'R_inf' in special_parameters.keys():
                    zm[re_row_start:re_row_end, num_params * i + special_parameters['R_inf']['index']] = 1
                    zm[im_row_start:im_row_end, num_params * i + special_parameters['R_inf']['index']] = 0

                if 'inductance' in special_parameters.keys():
                    zv_induc = mat1d.construct_inductance_impedance_vector(frequencies)
                    zm[re_row_start:re_row_end, num_params * i + special_parameters['inductance']['index']] = zv_induc.real
                    zm[im_row_start:im_row_end, num_params * i + special_parameters['inductance']['index']] = zv_induc.imag

                if 'v_baseline' in special_parameters.keys():
                    zm[row_start:row_start + 2 * num_f, num_params * i + special_parameters['v_baseline']['index']] = 0

                if 'vz_offset' in special_parameters.keys():
                    # If vz_offset is active, column values will be updated during fitting
                    zm[row_start:row_start + 2 * num_f, num_params * i + special_parameters['vz_offset']['index']] = 0

            row_start += 2 * num_f
            ref_row_start += num_f
        # t5 = time.time()
        # print('column expansion: {:.2f} s'.format(t5 - t4))

        # print('Total zm {} construction time: {:.3f} s'.format(part, time.time() - t0))
    else:
        zm = np.zeros((0, num_params * num_obs))

    return zm


def construct_md_inductance_impedance_matrix(frequency_list):
    # Shouldn't need this anymore
    num_obs = len(frequency_list)

    # One row for each unique (frequency, psi) combo, one column for each psi value
    num_freq = np.sum([len(f) for f in frequency_list])
    zm = np.zeros((num_freq, num_obs), dtype=complex)

    start = 0
    for i in range(num_obs):
        num_f = len(frequency_list[i])
        end = start + num_f

        # Calculate and insert impedance vector for observation i
        zv_i = mat1d.construct_inductance_impedance_vector(frequency_list[i])
        zm[start:end, i] = zv_i

        start += num_f

    return zm


def construct_md_inf_impedance_matrix(frequency_list):
    # Shouldn't need this anymore
    num_obs = len(frequency_list)

    # One row for each unique (frequency, psi) combo, one column for each psi value
    num_freq = np.sum([len(f) for f in frequency_list])
    zm = np.zeros((num_freq, num_obs), dtype=complex)

    start = 0
    for i in range(num_obs):
        num_f = len(frequency_list[i])
        end = start + num_f

        # Insert impedance vector for observation i
        zm[start:end, i] = 1

        start += num_f

    return zm


# ====================================
# Chronopotentiometry matrices
# ====================================
def construct_md_response_matrix(data_list, step_info_list, step_model, basis_tau,
                                 basis_type, epsilon, special_parameters, op_mode='galv',
                                 integrate_method='trapz', integrate_points=1000, zga_params=None,
                                 interpolate_grids=None, smooth_inf_response=True):
    num_obs = len(data_list)
    num_basis = len(basis_tau)
    num_special = len(special_parameters)
    num_params = num_basis + num_special

    # One row for each unique (time, psi) combo, one column for each unique (tau, psi) combo
    num_times = np.sum([utils.md.get_data_tuple_length(data) for data in data_list])
    rm = np.zeros((num_times, num_params * num_obs))

    # Can't easily reuse response matrix for different observations - depends on times, step_times, AND step_sizes
    # Thus, calculate full response matrix for each observation
    row_start = 0
    for i, data in enumerate(data_list):
        num_t = utils.md.get_data_tuple_length(data)
        row_end = row_start + num_t

        if num_t > 0:
            # Unpack data and step info
            times, i_signal, v_signal = data
            step_times, step_sizes, tau_rise = step_info_list[i]

            # Determine input and response from chrono_mode
            input_signal, response_signal = utils.chrono.get_input_and_response(i_signal, v_signal, op_mode)

            # Calculate and insert DRT response matrix for observation i
            rm_drt, _ = mat1d.construct_response_matrix(basis_tau, times, step_model, step_times,
                                                        step_sizes, basis_type, epsilon, tau_rise,
                                                        op_mode, integrate_method, integrate_points, zga_params,
                                                        interpolate_grids)
            rm[row_start:row_end, num_params * i + num_special:num_params * (i + 1)] = rm_drt

            # Insert special parameter response columns
            if 'R_inf' in special_parameters.keys():
                rv_inf = mat1d.construct_inf_response_vector(times, step_model, step_times, step_sizes, tau_rise,
                                                             input_signal, smooth_inf_response, op_mode)
                rm[row_start:row_end, num_params * i + special_parameters['R_inf']['index']] = rv_inf

            if 'inductance' in special_parameters.keys():
                rv_induc = mat1d.construct_inductance_response_vector(times, step_model, step_times, step_sizes,
                                                                      tau_rise, op_mode)
                rm[row_start:row_end, num_params * i + special_parameters['inductance']['index']] = rv_induc

            if 'v_baseline' in special_parameters.keys():
                rm[row_start:row_end, num_params * i + special_parameters['v_baseline']['index']] = 1

            if 'vz_offset' in special_parameters.keys():
                # If vz_offset is active, column values will be updated during fitting
                rm[row_start:row_end, num_params * i + special_parameters['vz_offset']['index']] = 0

        row_start += num_t

    return rm


def construct_md_inductance_response_matrix(time_list, step_time_list, step_size_list, tau_rise_list, step_model,
                                            op_mode='galv'):
    # Shouldn't need this anymore
    num_obs = len(time_list)

    # One row for each unique (time, psi) combo, one column for each psi value
    num_times = np.sum([len(t) for t in time_list])
    rm = np.zeros((num_times, num_obs))

    start = 0
    for i in range(num_obs):
        num_t = len(time_list[i])
        end = start + num_t

        # Calculate and insert response vector for observation i
        rv_i = mat1d.construct_inductance_response_vector(time_list[i], step_model, step_time_list[i],
                                                          step_size_list[i], tau_rise_list[i], op_mode)
        rm[start:end, i] = rv_i

        start += num_t

    return rm


def construct_md_inf_response_matrix(time_list, input_signal_list, step_time_list, step_size_list, tau_rise_list,
                                     step_model, smooth, op_mode='galv'):
    # Shouldn't need this anymore
    num_obs = len(time_list)

    # One row for each unique (time, psi) combo, one column for each psi value
    num_times = np.sum([len(t) for t in time_list])
    rm = np.zeros((num_times, num_obs))

    start = 0
    for i in range(num_obs):
        num_t = len(time_list[i])
        end = start + num_t

        # Calculate and insert response vector for observation i
        rv_i = mat1d.construct_inf_response_vector(time_list[i], step_model, step_time_list[i], step_size_list[i],
                                                   tau_rise_list[i], input_signal_list[i], smooth, op_mode)
        rm[start:end, i] = rv_i

        start += num_t

    return rm


# =========================
# Penalty matrices
# =========================
def construct_md_integrated_derivative_matrix(num_obs, basis_grid, basis_type, order, epsilon, special_parameters,
                                              special_penalties,
                                              zga_params=None):
    # Get dimensions
    num_basis = len(basis_grid)
    num_special = len(special_parameters)
    num_params = num_basis + num_special

    # Get base matrix
    m_base = np.zeros((num_params, num_params))

    # Insert DRT penalty matrix
    m_drt = mat1d.construct_integrated_derivative_matrix(basis_grid, basis_type, order, epsilon, zga_params)
    m_base[num_special:, num_special:] = m_drt

    # Insert special parameter penalties
    for k, v in special_parameters.items():
        index = v['index']
        penalty = special_penalties[k]
        m_base[index, index] = penalty

    m = construct_block_diagonal(m_base, num_obs)

    return m_base, m


# =============================
# Variance estimation matrices
# =============================
def construct_md_chrono_var_matrix(data_list, step_info_list, vmm_epsilon, error_structure):
    
    num_times = np.sum([utils.md.get_data_tuple_length(data) for data in data_list])
    vmm = np.zeros((num_times, num_times))

    start_index = 0
    for i, data in enumerate(data_list):
        num_t = utils.md.get_data_tuple_length(data)

        if num_t > 0:
            times = data[0]
            step_times = step_info_list[0]
            vmm_i = mat1d.construct_chrono_var_matrix(times, step_times, vmm_epsilon, error_structure)
            vmm[start_index: start_index + num_t, start_index: start_index + num_t] = vmm_i
            start_index += num_t

    return vmm


def construct_md_eis_var_matrix(data_list, vmm_epsilon, reim_cor, error_structure):
    num_freq = np.sum([utils.md.get_data_tuple_length(data) for data in data_list])
    vmm = np.zeros((num_freq * 2, num_freq * 2))

    start_index = 0
    for i, data in enumerate(data_list):
        num_f = utils.md.get_data_tuple_length(data)

        if num_f > 0:
            frequencies = data[0]
            vmm_i = mat1d.construct_eis_var_matrix(frequencies, vmm_epsilon, reim_cor, error_structure)
            vmm[start_index: start_index + 2 * num_f, start_index: start_index + 2 * num_f] = vmm_i
            start_index += 2 * num_f

    return vmm


# ============================
# General
# ============================
def construct_block_diagonal(base_matrix, n):
    r = base_matrix.shape[0]
    c = base_matrix.shape[1]
    m_out = np.zeros((r * n, c * n))

    for i in range(n):
        m_out[i * r: (i + 1) * r, i * c: (i + 1) * c] = base_matrix.copy()

    return m_out


