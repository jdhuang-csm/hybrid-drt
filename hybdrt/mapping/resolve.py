# Functions for resolving initial DRT fits with coherent multi-observation optimization
import numpy as np
import cvxopt
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d, median_filter

from ..matrices.basis import construct_func_eval_matrix


def get_offset_pq(drt):
    """
    Get P matrix and q vector with data-dependent parameters removed and offset accordingly
    :param DRT drt: DRT instance
    :return:
    """
    p = drt.fit_parameters['p_matrix']
    q = drt.fit_parameters['q_vector']
    # x_drt = drt.fit_parameters['x'] / drt.coefficient_scale
    # x_dop = drt.fit_parameters['x_dop'] / (drt.coefficient_scale * drt.dop_scale_vector)

    # Identify data-dependent parameters to remove
    special_indices = [drt.special_qp_params[k]['index'] for k in drt.special_qp_params.keys()
                       if k in ['v_baseline', 'vz_offset']]
    num_remove = len(special_indices)
    x_remove = np.empty(num_remove)

    for k, v in drt.special_qp_params.items():
        if k == 'v_baseline':
            x_remove[v['index']] = drt.fit_parameters['v_baseline'] / drt.response_signal_scale \
                                   + drt.scaled_response_offset
        elif k == 'vz_offset':
            x_remove[v['index']] = drt.fit_parameters['vz_offset']

    # Get the vector capturing the interaction between removed (fixed) parameters and remaining (variable) parameters
    q_offset = x_remove @ p[:num_remove, num_remove:]

    # Get P and q excluding removed parameters
    p_trim = p[num_remove:, num_remove:]
    q_trim = q[num_remove:]

    return p_trim, q_trim + q_offset


def resize_pq(p, q, special_offset, tau_indices, match_tau_indices):
    num_drt = tau_indices[1] - tau_indices[0]
    match_num = match_tau_indices[1] - match_tau_indices[0]
    new_size = p.shape[0] + (match_num - num_drt)
    left_offset = tau_indices[0] - match_tau_indices[0]  # >= 0
    right_offset = tau_indices[1] - match_tau_indices[1]  # <= 0

    p_out = np.zeros((new_size, new_size))
    q_out = np.zeros(new_size)

    # Insert special parameter sub-arrays
    p_out[:special_offset, :special_offset] = p[:special_offset, :special_offset]
    q_out[:special_offset] = q[:special_offset]

    # Get DRT parameter sub-arrays
    p_drt = p[special_offset:, special_offset:]
    q_drt = q[special_offset:]

    # Insert DRT sub-arrays
    if left_offset >= 0 and right_offset <= 0:
        # Expand
        # print('expand')
        left = special_offset + left_offset
        right = new_size + right_offset

        # DRT block
        p_out[left:right, left:right] = p_drt
        q_out[left:right] = q_drt

        # DRT-special blocks
        p_out[left:right, :special_offset] = p[special_offset:, :special_offset]
        p_out[:special_offset, left:right] = p[:special_offset, special_offset:]

    elif left_offset < 0 and right_offset > 0:
        # Truncate
        # DRT block
        # print('truncate')
        p_out[special_offset:, special_offset:] = p_drt[-left_offset:-right_offset, -left_offset:-right_offset]
        q_out[special_offset:] = q_drt[-left_offset:-right_offset]

        # DRT-special block
        p_out[special_offset:, :special_offset] = p[-left_offset:-right_offset, :special_offset]
        p_out[:special_offset, special_offset:] = p[:special_offset, -left_offset:-right_offset]
    elif left_offset >= 0:
        # Expand left, truncate right
        # DRT block
        # print('expand-truncate')
        left = special_offset + left_offset
        p_out[left:, left:] = p_drt[:-right_offset, :-right_offset]
        q_out[left:] = q_drt[:-right_offset]

        # DRT-special block
        p_out[left:, :special_offset] = p[special_offset:, :special_offset]
        p_out[:special_offset, left:] = p[:special_offset, special_offset:]
    else:
        # Truncate left, expand right
        # DRT block
        # print('truncate-expand')
        right = new_size + right_offset
        p_out[:right, :right] = p_drt[-left_offset:, -left_offset:]
        q_out[:right] = q_drt[-left_offset:]

        # DRT-special block
        p_out[:right, :special_offset] = p[-left_offset:, :special_offset]
        p_out[:special_offset, :right] = p[:special_offset, -left_offset:]

    # print(np.max(np.abs(p_out - p)))
    # print(np.max(np.abs(q_out - q)))

    return p_out, q_out


def offset_special_dict(special_qp_params):
    shifted_dict = deepcopy(special_qp_params)

    del_index = {}
    for name in ['v_baseline', 'vz_offset']:
        if name in special_qp_params.keys():
            index = special_qp_params[name]['index']
            del_index[name] = index

    if len(del_index) > 0:
        # Remove entries from shifted_dict
        for name in ['v_baseline', 'vz_offset']:
            if name in shifted_dict.keys():
                del shifted_dict[name]

        # Shift indices of remaining special params
        for key in list(shifted_dict.keys()):
            index = shifted_dict[key]['index']
            shift = np.sum([special_qp_params[name].get('size', 1) if di < index else 0
                            for name, di in del_index.items()])
            shifted_dict[key]['index'] = index - shift
    return shifted_dict


def resolve_observations(obs_drt_list, obs_tau_indices, nonneg, obs_psi=None,
                         truncate=False, sigma=1, lambda_psi=1, unpack=False,
                         tau_filter_sigma=0, special_filter_sigma=0):
    # Determine tau grid to resolve
    if truncate:
        # Truncate to shortest grid covered by all obs
        left_index = np.max([oti[0] for oti in obs_tau_indices])
        right_index = np.min([oti[1] for oti in obs_tau_indices])
    else:
        # Expand to longest grid
        left_index = np.min([oti[0] for oti in obs_tau_indices])
        right_index = np.max([oti[1] for oti in obs_tau_indices])
    match_tau_indices = (left_index, right_index)

    # Get special parameters after removing data-dependent parameters
    special_dict = offset_special_dict(obs_drt_list[0].special_qp_params)
    special_offset = np.sum([v.get('size', 1) for v in special_dict.values()])

    p_list = []
    q_list = []

    for i, drt in enumerate(obs_drt_list):
        p, q = get_offset_pq(drt)
        p, q = resize_pq(p, q, special_offset, obs_tau_indices[i], match_tau_indices)
        p_list.append(p)
        q_list.append(q)

    nr = len(obs_drt_list)
    nc = len(q_list[0])

    # Make differentiation matrix
    # TODO: implement ND version with pairwise distances
    # epsilon = 1 / (np.sqrt(2) * sigma)
    # print('eps:', epsilon)
    # if obs_psi is None:
    #     obs_psi = np.arange(nr)
    # else:
    #     obs_psi = obs_psi.flatten()

    # Ly = construct_func_eval_matrix(obs_psi, order=2, epsilon=epsilon)
    # Ly[0, 0] *= 0.5
    # Ly[-1, -1] *= 0.5
    # Reflect penalty at edges
    Ly = gaussian_filter1d(np.eye(nr), sigma=sigma, mode='reflect', order=2)
    # factor = (1 / (np.sqrt(2 * np.pi) * sigma)) ** -1
    # Ly *= factor
    # print(Ly[:5, :5])

    # Apply the penalty to the rescaled coefficients (i.e. true scale)
    scale_vec = np.array([drt.coefficient_scale for drt in obs_drt_list])
    scale_smooth = gaussian_filter1d(median_filter(scale_vec, 3), 2)
    scale_mat = np.diag(scale_vec / scale_smooth)  # np.median(scale_vec))

    # param_scale = np.ones((nr, nc)) * (scale_vec / scale_smooth)[:, None]
    # print(param_scale)

    param_scale = np.ones(nc)
    if 'R_inf' in special_dict.keys():
        x_inf = np.array([drt.fit_parameters['R_inf'] / drt.coefficient_scale for drt in obs_drt_list])
        ohmic_scale = 5 * np.std(x_inf)
        param_scale[special_dict['R_inf']['index']] = ohmic_scale ** -2
    if 'x_dop' in special_dict.keys():
        x_dop = np.array([drt.fit_parameters['x_dop'] / (drt.coefficient_scale * drt.dop_scale_vector)
                          for drt in obs_drt_list])
        # Scale the penalty to the magnitude of the dop params
        dop_scales = np.std(x_dop, axis=0) + 0.1 * np.std(x_dop)
        # print(dop_scales)
        # dop_scales = np.stack([drt.dop_scale_vector for drt in obs_drt_list], axis=0)
        # dop_scale_smooth = gaussian_filter1d(median_filter(dop_scales, (3, 1)), 2, axis=0)
        # param_scale = np.hstack(((dop_scales / 100) ** 2, np.ones((nr, nc - special_offset))))
        # param_scale = np.hstack((np.ones((nr, special_offset)) * 1, np.ones((nr, nc - special_offset))))
        # param_scale = np.hstack((np.tile(dop_scales ** -2, (nr, 1)), np.ones((nr, nc - special_offset))))
        # print(param_scale[:, 0], param_scale[:, special_offset-1])
        # print(param_scale)

        dop_start = special_dict['x_dop']['index']
        dop_end = dop_start + special_dict['x_dop'].get('size', 1)
        param_scale[dop_start:dop_end] = dop_scales ** -2
    else:
        dop_start, dop_end = None, None

    Lys = Ly @ scale_mat
    My = Lys.T @ Lys

    #     My = hybdrt.matrices.mat1d.construct_integrated_derivative_matrix(np.arange(nr),
    #                                                                       order=2, epsilon=epsilon)

    p_matrix = np.zeros((nr * nc, nr * nc))
    M_full = np.zeros((nr * nc, nr * nc))

    if tau_filter_sigma > 0 or special_filter_sigma > 0:
        filter_mat = np.eye(nc)

        if special_filter_sigma > 0 and dop_start is not None:
            special_epsilon = 1 / (np.sqrt(2) * special_filter_sigma)
            filter_mat[dop_start:dop_end, dop_start:dop_end] = construct_func_eval_matrix(
                np.arange(dop_start, dop_end), epsilon=special_epsilon, order=0
            )
        # else:
        #     filter_mat[:special_offset, :special_offset] = np.eye(special_offset)

        if tau_filter_sigma > 0:
            tau_epsilon = 1 / (np.sqrt(2) * tau_filter_sigma)
            filter_mat[special_offset:, special_offset:] = construct_func_eval_matrix(
                np.arange(nc - special_offset), epsilon=tau_epsilon, order=0
            )
        # else:
        #     filter_mat[special_offset:, special_offset:] = np.eye(nc - special_offset)

        full_filter_mat = np.zeros_like(M_full)
    else:
        filter_mat = None
        full_filter_mat = None

    for i in range(nr):
        start = i * nc
        end = (i + 1) * nc
        p_matrix[start:end, start:end] = p_list[i]

    for i in range(nr):
        for j in range(nr):
            row_start = i * nc
            row_end = (i + 1) * nc
            col_start = j * nc
            col_end = (j + 1) * nc
            M_full[row_start:row_end, col_start:col_end] += np.diag(np.ones(nc) * param_scale * My[i, j]) * lambda_psi

            if i == j and filter_mat is not None:
                full_filter_mat[row_start:row_end, col_start:col_end] = filter_mat

    if full_filter_mat is not None:
        M_full = full_filter_mat @ M_full @ full_filter_mat

    p_matrix = p_matrix + M_full

    q_vector = np.concatenate(q_list)

    G = -np.eye(p_matrix.shape[1])

    if nonneg:
        h = np.zeros(p_matrix.shape[1])
    else:
        h = 10 * np.ones(p_matrix.shape[1])

    for sp in special_dict.values():
        if sp['nonneg']:
            start_index = sp['index']
            end_index = sp['index'] + sp.get('size', 1)
            for i in range(nr):
                h[start_index + i * nc:end_index + i * nc] = 0

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    p_matrix = cvxopt.matrix(p_matrix.T)
    q_vector = cvxopt.matrix(q_vector.T)

    res = cvxopt.solvers.qp(p_matrix, q_vector, G, h, initvals=None)
    x_opt = np.array(list(res['x'])).reshape((nr, nc))

    if unpack:
        x_drt, x_special = unpack_resolved_x(x_opt, obs_drt_list, special_dict)
        return x_drt, x_special, match_tau_indices
    else:
        return x_opt, match_tau_indices


def unpack_resolved_x(x, obs_drt_list, special_dict):
    special_offset = np.sum([v.get('size', 1) for v in special_dict.values()])

    x_drt_raw = x[:, special_offset:]
    coef_scale = np.array([drt.coefficient_scale for drt in obs_drt_list])
    x_drt = x_drt_raw * coef_scale[:, None]

    x_special = {}
    for key, info in special_dict.items():
        start_index = info['index']
        size = info.get('size', 1)
        end_index = start_index + size
        x_k = x[:, start_index:end_index]

        # Scale to true values
        x_k = x_k * coef_scale[:, None]

        if key == 'x_dop':
            dop_scale = np.array([drt.dop_scale_vector for drt in obs_drt_list])
            x_k = x_k * dop_scale

        if size == 1:
            x_k = x_k.flatten()

        x_special[key] = x_k

    return x_drt, x_special

