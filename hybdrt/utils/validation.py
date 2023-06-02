import numpy as np

from hybdrt.utils.array import check_equality


def check_basis_type(basis_type):
    options = ['gaussian', 'beta', 'beta-rbf', 'Cole-Cole', 'step', 'delta', 'ramp', 'pwl', 'pwl_transformed', 'zga']
    if basis_type not in options:
        raise ValueError(f'Invalid basis_type {basis_type}. Options: {options}')


def check_step_model(step_model):
    options = ['ideal', 'expdecay']
    if step_model not in options:
        raise ValueError(f'Invalid step_model {step_model}. Options: {options}')


def check_penalty_type(penalty_type):
    options = ['discrete', 'integral']
    if penalty_type not in options:
        raise ValueError(f'Invalid penalty {penalty_type}. Options: {options}')


def check_error_structure(error_structure):
    options = [None, 'uniform']
    if error_structure not in options:
        raise ValueError(f'Invalid error structure {error_structure}. Options: {options}')


def check_eis_data(frequencies, z):
    if not check_equality(np.shape(frequencies), np.shape(z)):
        raise ValueError('frequencies and z must have same shape')


def check_chrono_data(times, i_signal, v_signal):
    if i_signal is not None:
        if not check_equality(np.shape(times), np.shape(i_signal)):
            raise ValueError('times and i_signal must have same shape')
    if v_signal is not None:
        if not check_equality(np.shape(times), np.shape(v_signal)):
            raise ValueError('times and v_signal must have same shape')


def check_md_data(psi_array, chrono_data_list, eis_data_list):
    # Get apparent number of observations for each dataset
    if chrono_data_list is not None:
        num_chrono = len(chrono_data_list)
        # Check each observation
        for i in range(num_chrono):
            if chrono_data_list[i] is not None:
                times, i_signal, v_signal = chrono_data_list[i]
                check_chrono_data(times, i_signal, v_signal)
    else:
        num_chrono = psi_array.shape[0]

    if eis_data_list is not None:
        num_eis = len(eis_data_list)
        # Check each observation
        for i in range(num_eis):
            if eis_data_list[i] is not None:
                frequencies, z = eis_data_list[i]
                check_eis_data(frequencies, z)
    else:
        num_eis = psi_array.shape[0]

    all_nums = [num_chrono, num_eis, psi_array.shape[0]]
    if not all(num == all_nums[0] for num in all_nums):
        raise ValueError('Number of observations in chrono data, EIS data, and psi_array must be the same')


def check_md_x_spec(*args):
    x_spec = [*args]
    num_provided = np.sum([xs is not None for xs in x_spec])
    if num_provided != 1:
        raise ValueError('One and only one of the following parameters must be provided: '
                         'psi_array, obs_indices, or x')


def check_ctrl_mode(ctrl_mode):
    options = ['galv', 'pot']
    if ctrl_mode not in options:
        raise ValueError(f'Invalid ctrl_mode {ctrl_mode}. Options: {options}')
    