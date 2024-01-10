import numpy as np
from scipy.special import loggamma
import cvxopt
from copy import deepcopy

from hybdrt.utils.stats import log_pdf_gamma, pdf_normal
from .. import preprocessing as pp
# from . import peaks
# from ..matrices import basis

cvxopt.solvers.options['show_progress'] = False


def get_num_special(special_qp_params: dict):
    if len(special_qp_params) == 0:
        return 0
    else:
        return np.sum([qp.get('size', 1) for qp in special_qp_params.values()])

# ===============================================
# Analytical solutions for hyper-lambda approach
# ===============================================
def calculate_qp_l2_matrix(hypers, rho_vector, dop_rho_vector, penalty_matrices, s_vectors,
                           penalty_type, special_qp_params, nonlin: bool = False):
    """
    Calculate lml matrix (Lambda^1/2 @ M @ Lambda^1/2)
    :param dop_rho_vector:
    :param special_qp_params:
    :param dict hypers: dict of hyperparameters
    :param ndarray rho_vector:
    :param dict penalty_matrices: list of l2 (M) matrices
    :param list s_vectors: list of lambda vectors
    :param str penalty_type:
    :return:
    """
    num_special = get_num_special(special_qp_params)

    derivative_weights = hypers['derivative_weights']
    l2_lambda_0 = hypers['l2_lambda_0']

    if 'x_dop' in special_qp_params.keys():
        dop_start = special_qp_params['x_dop']['index']
        dop_end = dop_start + special_qp_params['x_dop']['size']
        dop_l2_lambda_0 = hypers['dop_l2_lambda_0']
        dop_derivative_weights = hypers['dop_derivative_weights']
    else:
        dop_start, dop_end = None, None
        dop_l2_lambda_0, dop_derivative_weights = None, None

    # Sum weighted matrices for all derivative orders
    if penalty_type == 'integral':
        l2_mat = np.zeros_like(penalty_matrices['m0'])
        
        if nonlin:
            M = int(l2_mat.shape[0] / 2)
        
        for k, d_weight in enumerate(derivative_weights):
            if d_weight > 0:
                sv = s_vectors[k]
                sm = np.diag(sv ** 0.5)
                m_k = penalty_matrices[f'm{k}'].copy()

                # Multiply DRT sub-matrix by derivative weight and rho
                d_factor = l2_lambda_0 * d_weight * rho_vector[k]
                if nonlin:
                    m_k[num_special:M, num_special:M] *= d_factor
                    m_k[M + num_special:, M + num_special:] *= d_factor
                else:
                    m_k[num_special:, num_special:] *= d_factor

                # Multiply DOP sub-matrix by derivative weight and rho
                if 'x_dop' in special_qp_params.keys():
                    dop_factor = dop_l2_lambda_0 * dop_derivative_weights[k] * dop_rho_vector[k]
                    
                    if nonlin:
                        m_k[dop_start:dop_end, dop_start:dop_end] *= dop_factor
                        m_k[M + dop_start:M + dop_end, M + dop_start:M + dop_end] *= dop_factor
                    else:
                        m_k[dop_start:dop_end, dop_start:dop_end] *= dop_factor

                # l2_mat += d_weight * rho_vector[k] * (sm @ m_k @ sm)
                l2_mat += sm @ m_k @ sm
                
        # Add nonlinear cross matrix
        m1_nl = penalty_matrices.get('m1_nl', None)
        if m1_nl is not None:
            l2_mat += m1_nl
        # l2_mat *= 2  # multiply by 2 for exponential prior
        # return l2_lambda_0 * l2_mat
        return l2_mat
    elif penalty_type == 'discrete':
        l2_mat = np.zeros((penalty_matrices['l0'].shape[1], penalty_matrices['l0'].shape[1]))
        for k, d_weight in enumerate(derivative_weights):
            if d_weight > 0:
                sv = s_vectors[k]
                sm = np.diag(sv)
                l_k = penalty_matrices[f'l{k}']
                l2_mat += d_weight * rho_vector[k] * l_k.T @ sm @ l_k

        return l2_lambda_0 * l2_mat


def calculate_md_qp_l2_matrix(derivative_weights, rho_diagonals, penalty_matrices, s_vectors, l2_lambda_0_diagonal,
                              penalty_type):
    # Apply rho_diagonals to penalty matrices
    if penalty_type == 'integral':
        scaled_penalty_matrices = {f'm{k}': rho_diagonals[k] @ penalty_matrices[f'm{k}'] @ rho_diagonals[k]
                                   for k in range(len(derivative_weights))}
    else:
        scaled_penalty_matrices = {f'l{k}': rho_diagonals[k] @ penalty_matrices[f'l{k}']
                                   for k in range(len(derivative_weights))}

    l2_mat = calculate_qp_l2_matrix(hypers, np.ones(len(derivative_weights)), dop_rho_vector,
                                    scaled_penalty_matrices, s_vectors, penalty_type, special_qp_params)
    # Apply l2_lambda_0 data vector
    l2_mat = l2_lambda_0_diagonal @ l2_mat @ l2_lambda_0_diagonal

    return l2_mat


def solve_hyper_l1_lambda(x, hl_beta, lambda_0):
    return lambda_0 * hl_beta / (hl_beta + x)


def solve_hyper_l2_lambda(m, x, lv, hl_beta, lambda_0):
    xm = np.diag(x)
    lm = np.diag(lv ** 0.5)
    xlm = xm @ lm @ m @ xm
    xlm = xlm - np.diag(np.diagonal(xlm))
    c = np.sum(xlm, axis=0)

    a = hl_beta / 2
    b = 0.5 * (2 * a - 2) / lambda_0
    d = x ** 2 * np.diagonal(m) + 2 * b
    # 2nd numerator term should be divided by 2
    lv = (c ** 2 - np.sign(c) * c * np.sqrt(4 * d * (2 * a - 2) + c ** 2) + 2 * d * (2 * a - 2)) / (2 * d ** 2)
    return lv


def solve_hyper_l2_lambda_discrete(l, x, hl_beta, lambda_0):
    lx2 = (l @ x) ** 2
    # lam = np.ones(self.A_re.shape[1]) #*lambda_0
    lam = 1 / (lx2 / (hl_beta - 1) + 1 / lambda_0)
    # if dist_type == 'series':
    #     # add ones for R_ohmic and inductance
    #     lam = np.hstack(([1, 1], lam))
    return lam


# ======================================
# Analytical solutions for QPHB approach
# ======================================
def get_data_factor(n_eff, ppd_eff):
    factor = np.sqrt((n_eff / (71 * np.sqrt(2)))) * (10 * np.sqrt(2) / ppd_eff)
    # factor = (10 * np.sqrt(2) / ppd_eff)

    return factor


def get_data_factor_from_data(times, step_times, frequencies):
    if times is not None:
        chrono_num = len(times[times >= step_times[0]])
    else:
        chrono_num = 0

    if frequencies is not None:
        eis_num = np.sqrt(2) * len(frequencies)
    else:
        eis_num = 0

    num_decades = pp.get_num_decades(frequencies, times, step_times)
    tot_num = eis_num + chrono_num
    tot_ppd = (tot_num - 1) / num_decades

    return get_data_factor(tot_num, tot_ppd)


def get_default_hypers(eff_hp, fit_dop, nu_basis_type):
    if eff_hp:
        s_alpha = np.array([5, 10, 25])  # np.array([1.5, 2.5, 25])
        rho_alpha = np.array([0.15, 0.2, 0.25])  # np.array([0.1, 0.15, 0.2])
        iw_alpha = None  # 1.5
        iw_beta = None  # 0.5 * data_factor ** 2
    else:
        s_alpha = np.array([1.05, 1.15, 2.5])
        rho_alpha = np.array([0.05, 0.1, 0.05])
        iw_alpha = None
        iw_beta = None

    hypers = dict(
        rp_scale=14,
        derivative_weights=np.array([1.5, 1.0, 0.5]),
        sigma_ds=np.array([1, 1000, 1000]),
        l1_lambda_0=0,
        l2_lambda_0=142,  # 284 * data_factor ** -1,
        iw_alpha=iw_alpha,
        iw_beta=iw_beta,
        s_alpha=s_alpha,
        s_0=np.ones(3),
        rho_alpha=rho_alpha,
        rho_0=np.ones(3),
        outlier_p=None,
    )

    if fit_dop:
        # if nu_basis_type == 'delta':
        #     hypers['dop_l2_lambda_0'] = 100
        #     hypers['dop_l1_lambda_0'] = 0
        #     hypers['dop_derivative_weights'] = np.array([1, 0, 0])
        #     hypers['dop_s_alpha'] = np.array([2, 2, 2])
        #     hypers['dop_rho_alpha'] = np.array([0.15, 0.2, 0.25])
        #     hypers['dop_s_0'] = np.ones(3)
        #     hypers['dop_rho_0'] = np.ones(3)
        #     hypers['dop_sigma_ds'] = np.array([1, 1000, 1000])
        # else:
        hypers['dop_l2_lambda_0'] = 100
        hypers['dop_l1_lambda_0'] = 0
        hypers['dop_derivative_weights'] = np.array([0.5, 1.0, 0.5])
        hypers['dop_s_alpha'] = np.array([5, 10, 25])
        hypers['dop_rho_alpha'] = np.array([0.15, 0.2, 0.25])
        hypers['dop_s_0'] = np.ones(3)
        hypers['dop_rho_0'] = np.ones(3)
        hypers['dop_sigma_ds'] = np.array([1, 1000, 1000])

    return hypers


# def solve_s(m_k, x, sv_k, rho_k, alpha, beta):
#     """
#     Determine optimal values of local penalty scale parameters, s_km
#     :param rho_k:
#     :param ndarray m_k: integrated penalty matrix
#     :param ndarray x: coefficient vector
#     :param ndarray sv_k: previous vector of local penalty scale parameters
#     :param float alpha: effective alpha hyperparameter of gamma prior on s_dm. Must be > 1
#     :param float beta: effective beta hyperparameter of gamma prior on s_dm. Must be > 0
#     :return:
#     """
#     xm = np.diag(x)
#     sm = np.diag(sv_k ** 0.5)
#     # xsm = xm @ sm @ m @ xm
#     # xsm = xsm - np.diag(np.diagonal(xsm))
#     # b1 = np.sum(xsm, axis=0)
#     # b2 = np.sum(xsm, axis=1)
#     # b = (b1 + b2) / 2
#     xsm = xm @ m_k @ xm @ sm
#     xsm = xsm - np.diag(np.diagonal(xsm))
#     b = np.sum(xsm, axis=0)
#     b *= rho_k
#
#     # Exponential prior
#     d = x ** 2 * rho_k * np.diagonal(m_k) + beta
#     sv_k = (2 * b ** 2 - 2 * abs(b) * np.sqrt(4 * d * (alpha - 1) + b ** 2) + 4 * d * (alpha - 1)) / (
#             4 * d ** 2)
#     return sv_k


# def solve_s(m_k, x, sv_k, rho_k, alpha, beta, g_mat, sigma_s):
#     """
#     Determine optimal values of local penalty scale parameters, s_km
#     :param rho_k:
#     :param ndarray m_k: integrated penalty matrix
#     :param ndarray x: coefficient vector
#     :param ndarray sv_k: previous vector of local penalty scale parameters
#     :param float alpha: effective alpha hyperparameter of gamma prior on s_dm. Must be > 1
#     :param float beta: effective beta hyperparameter of gamma prior on s_dm. Must be > 0
#     :return:
#     """
#     xm = np.diag(x)
#     sm = np.diag(sv_k ** 0.5)
#     xsm = xm @ m_k @ xm @ sm
#     xsm = xsm - np.diag(np.diagonal(xsm))
#     b = np.sum(xsm, axis=0)
#     b *= rho_k
#
#     # gm = g_mat - np.diag(np.diag(g_mat))
#     gs = g_mat @ sm
#     # gs = gs - np.diag(np.diag(gs))
#     c = np.sum(gs, axis=0)
#     c /= (2 * sigma_s ** 2)
#     b = b + c
#
#     # Exponential prior
#     d = x ** 2 * rho_k * np.diagonal(m_k) + beta + np.diag(g_mat) / (2 * sigma_s ** 2)
#     sv_k = (2 * b ** 2 - 2 * abs(b) * np.sqrt(4 * d * (alpha - 1) + b ** 2) + 4 * d * (alpha - 1)) / (
#             4 * d ** 2)
#     return sv_k


def solve_s(pm_k, x, sv_k, rho_k, alpha, beta, g_mat, sigma_ds, penalty_type, x_offset=0):
    if penalty_type == 'integral':
        xm = np.diag(x + x_offset)  # np.sign(x) * np.abs(x) ** 0.5)
        gamma = rho_k * xm @ pm_k @ xm + g_mat / (2 * sigma_ds ** 2) + beta * np.eye(len(x))

        um = np.diag(sv_k ** 0.5)
        # gzd = gamma - np.diag(np.diag(gamma))
        # b = gzd @ sv_k ** 0.5
        gu = gamma @ um
        np.fill_diagonal(gu, 0)

        if np.max(np.abs(gu)) > 1e-10:
            # gu is not diagonal. Quadratic solution
            b = np.sum(gu, axis=1)
            u_hat = (-b + np.sign(b) * np.sqrt(b ** 2 + 4 * np.diag(gamma) * (alpha - 1))) / (2 * np.diag(gamma))
            s_hat = u_hat ** 2
        else:
            # gu is diagonal
            s_hat = (alpha - 1) / (np.diag(gamma))
    elif penalty_type == 'discrete':
        if np.max(np.abs(g_mat)) > 1e-10:
            lx2 = rho_k * (pm_k @ x) ** 2
            g_mat_zd = g_mat.copy()
            np.fill_diagonal(g_mat_zd, 0)
            g_diag = np.diag(g_mat)
            a = beta + 0.5 * lx2 + (1 / (2 * sigma_ds ** 2)) * g_diag
            b = (1 / (2 * sigma_ds ** 2)) * (g_mat_zd @ (sv_k ** 0.5))

            # s_hat = (-b + np.sqrt(b ** 2 + 4 * g_diag * (alpha - 0.5) / sigma_s ** 2)) / (2 * g_diag / sigma_s ** 2)
            u_hat = (-b + np.sign(b) * np.sqrt(b ** 2 + 4 * a * (alpha - 0.5))) / (2 * a)
            s_hat = u_hat ** 2
        else:
            s_hat = (alpha - 0.5) / (0.5 * rho_k * (pm_k @ x) ** 2 + beta)

    s_hat[np.isnan(s_hat)] = 1

    return s_hat


# def solve_s(m_k, x, sv_k, rho_k, alpha, beta):
#     """
#     Determine optimal values of local penalty scale parameters, s_dm
#     :param rho_k:
#     :param ndarray m_k: integrated penalty matrix
#     :param ndarray x: coefficient vector
#     :param ndarray sv_k: previous vector of local penalty scale parameters
#     :param float alpha: effective alpha hyperparameter of gamma prior on s_dm. Must be > 1
#     :param float beta: effective beta hyperparameter of gamma prior on s_dm. Must be > 0
#     :return:
#     """
#     xm = np.diag(x)
#     sm = np.diag(sv_k)
#     xsm = xm @ sm @ m_k @ xm
#     xsm = xsm - np.diag(np.diagonal(xsm))
#     b = np.sum(xsm, axis=1)
#
#     a = 2 * rho_k * x ** 2 * np.diag(m_k)
#
#     # Exponential prior
#     sv_k = (-(2 * b + beta) + np.sqrt((2 * b + beta) ** 2 + 4 * (alpha - 1) * a)) / (2 * a)
#     sv_k[a == 0] = 1
#
#     return sv_k


def solve_rho(penalty_matrix, x, sv, alpha, beta, xmx_norm, penalty_type):
    """
    Determine optimal value of p_d (weight of derivative of order d)
    :param ndarray penalty_matrix: integrated penalty matrix
    :param ndarray x: coefficient vector
    :param ndarray sv: vector of local penalty scale parameters
    :param float alpha: effective alpha hyperparameter of gamma prior on p_d. Must be > 1
    :param float beta: effective beta hyperparameter of gamma prior on p_d. Must be > 0
    :param xmx_norm: normalizing value of scalar x.T @ S @ M @ S @ x. Determined from ordinary ridge coefficients
    :param str penalty_type: derivative penalty type (integral or discrete)
    :return: float
    """
    if penalty_type == 'integral':
        sm = np.diag(sv ** 0.5)
        xsmsx = x.T @ sm @ penalty_matrix @ sm @ x
        # print('xsmsx:', xsmsx)
        return alpha / (xsmsx / xmx_norm + beta)
    elif penalty_type == 'discrete':
        sm = np.diag(sv)
        xlslx = x.T @ penalty_matrix.T @ sm @ penalty_matrix @ x
        return alpha / (xlslx / xmx_norm + beta)


def calculate_md_xmx_norm_array(x, penalty_matrices, derivative_weights, qp_mat_offset, batch_size,
                                params_per_obs):
    num_drt = params_per_obs - qp_mat_offset
    xmx_norm_array = np.empty((batch_size, len(derivative_weights)))
    for k in range(len(derivative_weights)):
        m = penalty_matrices[f'm{k}']
        drt_indices = np.concatenate([i * params_per_obs + np.arange(qp_mat_offset, params_per_obs, dtype=int)
                                      for i in range(batch_size)])

        x_drt = x[drt_indices].reshape((batch_size, num_drt))
        m_drt = m[drt_indices][:, drt_indices]
        xmx_k = [x_drt[i].T @ m_drt[i * num_drt:(i + 1) * num_drt, i * num_drt:(i + 1) * num_drt] @ x_drt[i]
                 for i in range(batch_size)]
        xmx_norm_array[:, k] = xmx_k

    return xmx_norm_array


def solve_convex_opt(wrv, wrm, l2_matrix, l1v, nonneg, special_params, init_vals=None,
                     fixed_x_index=None, fixed_x_values=None, include_fixed_cov=True,
                     curvature_constraint=None, nonlin: bool = False):
    """
    Solve convex optimization problem. Used for ridge_fit method
    :param ndarray wrv: weighted response vector. wrv = W @ y
    :param ndarray wrm: weighted response matrix. wrm = W @ rm
    :param ndarray l2_matrix: penalty matrix. ll2l = Lambda @ M @ Lambda
    :param ndarray l1v: L1 (LASSO) penalty vector
    :param bool nonneg: if True, constrain x >= 0. If False, allow negative values
    :param ndarray special_params: dict of special parameters that are always non-negative or unbounded
    :param init_vals: initial parameter values
    :param ndarray fixed_x_index: indices of parameters to fix during optimization
    :param ndarray fixed_x_values: values of fixed parameters
    :param bool include_fixed_cov: if True, account for covariance of active parameters with fixed parameters.
    If False,ignore covariance with fixed parameters
    :return:
    """
    if fixed_x_index is not None:
        if include_fixed_cov:
            p_matrix = (wrm.T @ wrm + l2_matrix)
            q_vector = (-wrm.T @ wrv + l1v)
            # Account for covariance of active parameters with fixed parameters
            for index, value in zip(fixed_x_index, fixed_x_values):
                q_vector += 0.5 * value * (p_matrix[index] + p_matrix[:, index])

            q_vector = np.delete(q_vector, fixed_x_index)
            p_matrix = np.delete(np.delete(p_matrix, fixed_x_index, axis=0), fixed_x_index, axis=1)
        else:
            # # Delete rows of wrm and lml, entries of l1_lambda_vector prior to multiplication for efficiency
            # wrm = np.delete(wrm, fixed_x_index, axis=1)
            # lml = np.delete()
            # p_matrix =
            pass
    else:
        # print(wrm.shape, wrv.shape, np.asarray(l1v).shape)
        p_matrix = (wrm.T @ wrm + l2_matrix)
        q_vector = (-wrm.T @ wrv + l1v)
        
    # print(p_matrix.shape, q_vector.shape)
    if nonlin:
        M = int(wrm.shape[0] / 2)

    G = -np.eye(p_matrix.shape[1])
    if nonneg:
        # coefficients must be >= 0
        h = np.zeros(p_matrix.shape[1])
        # V_baseline and vz_factor can be negative
        for sp in special_params.values():
            if not sp['nonneg']:
                end_index = sp['index'] + sp.get('size', 1)
                h[sp['index']:end_index] = 10
                if nonlin:
                    h[M + sp['index']:M + end_index] = 10
        # h[special_indices['unbnd']] = 10
        # print(h)
    # print(h)
    else:
        # coefficients can be positive or negative
        h = 10 * np.ones(p_matrix.shape[1])
        # HFR and inductance must still be non-negative
        # h[special_indices['nonneg']] = 0
        for sp in special_params.values():
            if sp['nonneg']:
                end_index = sp['index'] + sp.get('size', 1)
                h[sp['index']:end_index] = 0
                if nonlin:
                    h[M + sp['index']:M + end_index] = 0
                    
        # print(h)

    if curvature_constraint is not None:
        dm, limit = curvature_constraint
        G = np.vstack((G, dm))
        h = np.concatenate((h, limit))

    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    p_matrix = cvxopt.matrix(p_matrix.T)
    # print('rank(p_matrix):', np.linalg.matrix_rank(p_matrix))
    q_vector = cvxopt.matrix(q_vector.T)

    return cvxopt.solvers.qp(p_matrix, q_vector, G, h, initvals=init_vals)


# =============================
# GP assist
# =============================
def solve_gp_rho(gp_omega, gp_mu, x, sv, alpha, beta):
    dx = x - gp_mu
    s_diag = np.diag(sv ** 0.5)
    nume = alpha - 0.5
    deno = 0.5 * dx @ s_diag @ gp_omega @ s_diag @ dx + beta

    return nume / deno


def solve_gp_s(gp_omega, gp_mu, x, sv, alpha, beta):
    """
    Determine optimal values of local penalty scale parameters, s_dm
    :param ndarray m: integrated penalty matrix
    :param ndarray x: coefficient vector
    :param ndarray sv: previous vector of local penalty scale parameters
    :param float alpha: effective alpha hyperparameter of gamma prior on s_dm. Must be > 1
    :param float beta: effective beta hyperparameter of gamma prior on s_dm. Must be > 0
    :return:
    """
    dx = x - gp_mu
    dx_diag = np.diag(dx)
    s_diag = np.diag(sv ** 0.5)
    xsm = dx_diag @ s_diag @ gp_omega @ dx_diag
    xsm = xsm - np.diag(np.diagonal(xsm))
    b = np.sum(xsm, axis=1)

    # Normal prior
    d = dx ** 2 * np.diagonal(gp_omega) + 2 * beta
    sv = (2 * b ** 2 - 2 * abs(b) * np.sqrt(d * (2 * alpha - 1) + b ** 2) + d * (2 * alpha - 1)) / (d ** 2)
    return sv


# =================================
# QPHB iteration
# =================================
def is_converged(x_in, x_out, x_atol, x_rtol):
    x_delta = x_out - x_in
    x_in = x_in + 1e-15  # avoid division by zero
    if np.max(np.abs(x_delta / x_in)) <= x_rtol or np.max(np.abs(x_delta)) <= x_atol:
        return True
    else:
        return False


def iterate_qphb(x_in, s_vectors, rho_vector, dop_rho_vector, rv, weights, est_weights, out_tvt,
                 rm, vmm, penalty_matrices, penalty_type, l1_lambda_vector,
                 hypers, eff_hp,
                 xmx_norms, dop_xmx_norms, fixed_x_index, fixed_x_values, curvature_constraint,
                 nonneg, special_qp_params, x_rtol, max_hp_iter, history, nonlin: bool = False):
    # Unpack hyperparameter dict
    # l2_lambda_0 = hypers['l2_lambda_0']
    derivative_weights = hypers['derivative_weights']
    rho_alpha = hypers['rho_alpha']
    rho_0 = hypers['rho_0']
    s_alpha = hypers['s_alpha']
    s_0 = hypers['s_0']
    # outlier_lambda = hypers['outlier_lambda']
    outlier_p = hypers['outlier_p']
    sigma_ds = hypers['sigma_ds']
    
    if nonlin:
        M = int(len(x_in) / 2)

    # if 'dop' in special_qp_params.keys():
    #     dop_start = special_qp_params['dop']['index']
    #     dop_end = dop_start + special_qp_params['dop']['size']
    #
    #     # dop_l2_lambda_0 = hypers['dop_l2_lambda_0']
    #     dop_derivative_weights = hypers['dop_derivative_weights']
    #     dop_rho_alpha = hypers['dop_rho_alpha']
    #     dop_rho_0 = hypers['dop_rho_0']
    #     dop_s_alpha = hypers['dop_s_alpha']
    #     dop_s_0 = hypers['dop_s_0']
    #     dop_sigma_ds = hypers['dop_sigma_ds']
    # if outlier_alpha is not None:
    #     outlier_args = (outlier_alpha, outlier_beta, outlier_variance, 2)
    # else:
    #     outlier_args = None
    k_range = len(derivative_weights)

    # Apply weights to rm and rv
    # wv = format_chrono_weights(rv, weights)
    wm = np.diag(weights)
    wrm = wm @ rm
    wrv = wm @ rv

    # Make l2_matrix matrix for each derivative order
    # l2_matrices = [penalty_matrices[f'pm_k{n}'] for n in range(k_range)]
    # l2_matrices = [lambdas[k] * penalty_matrices[f'pm_k{n}'] for n in range(3)]
    l2_matrix = calculate_qp_l2_matrix(hypers, rho_vector, dop_rho_vector, penalty_matrices,
                                       s_vectors, penalty_type, special_qp_params, nonlin=nonlin)
    # print('l2_matrix', np.sum(l2_matrix))

    # x_offsets = [0., 0.0, 0.0, 0.]
    # for k in range(len(derivative_weights)):
    #     if x_offsets[k] > 0:
    #         sv = s_vectors[k]
    #         sm = np.diag(sv ** 0.5)
    #         m = penalty_matrices[f'm{k}']
    #         ramp_length = 20
    #         x_offset = np.zeros(len(x_in)) + x_offsets[k]
    #         rbf = basis.get_basis_func('gaussian')
    #         x_offset[:ramp_length] = x_offsets[k] * rbf(ramp_length - np.arange(ramp_length), 0.1)
    #         x_offset[-ramp_length:] = x_offsets[k] * rbf(np.arange(ramp_length), 0.1)
    #
    #         sms = 2 * l2_lambda_0 * derivative_weights[k] * rho_vector[k] * sm @ m @ sm
    #         l1_lambda_vector = l1_lambda_vector + sms @ x_offset

    # Solve the ridge problem with QP: optimize x
    cvx_result = solve_convex_opt(wrv, wrm, l2_matrix, l1_lambda_vector, nonneg, special_qp_params,
                                  fixed_x_index=fixed_x_index, fixed_x_values=fixed_x_values,
                                  curvature_constraint=curvature_constraint, nonlin=nonlin)
    x = np.array(list(cvx_result['x']))

    if fixed_x_index is not None:
        # insert_index = [index - i for i, index in enumerate(fixed_x_index)]
        # print(fixed_x_index, insert_index)
        for index, value in zip(fixed_x_index, fixed_x_values):
            x = np.insert(x, index, value)

    # Get number of special (non-DRT) parameters in x vector
    num_special = get_num_special(special_qp_params)

    # lambdas = np.zeros(3)
    # l_alphas = [10, 5, 1]

    # For each derivative order, update the global penalty strength and the local penalty scale vector
    s_vectors = s_vectors.copy()
    rho_vector = rho_vector.copy()
    if dop_rho_vector is not None:
        dop_rho_vector = dop_rho_vector.copy()

    # # *TMP*
    # g_drt = np.eye(len(x) - num_special)
    # np.fill_diagonal(g_drt[:, 1:], -0.5)
    # np.fill_diagonal(g_drt[1:], -0.5)
    # g_drt[0, 1] = -1
    # g_drt[-1, -2] = -1
    # g = np.zeros((len(x), len(x)))
    # g[num_special:, num_special:] = g_drt
    # g_mat = g.T @ g
    # xm = np.diag(np.sign(x) * np.abs(x) ** 0.5)
    # g_mat = xm @ g_mat @ xm

    # g_mat = 0
    # for k in range(k_range):
    #     g_mat += penalty_matrices[f'pm_k{k}'] * rho_vector[k] * derivative_weights[k]
    # xm = np.diag(np.sign(x) * np.abs(x) ** 0.1)
    # g_mat = xm @ g_mat @ xm
    # g_mat = penalty_matrices['m2'] * rho_vector[2]
    # xm = np.diag(x + 0.1)xm

    it = 0
    # nu_vectors = [np.zeros(len(x_in))] * len(derivative_weights)
    while it < max_hp_iter:
        s_converged = [False] * k_range
        rho_converged = [False] * k_range
        rho_in = rho_vector.copy()
        for k, d_weight in enumerate(derivative_weights):
            if d_weight > 0:
                # Get penalty matrix
                if penalty_type == 'integral':
                    pm_k = penalty_matrices[f'm{k}']
                else:
                    pm_k = penalty_matrices[f'l{k}']

                sv_in = s_vectors[k]
                # Solve for lambda scale vector
                s_k_alpha = s_alpha[k]
                if np.shape(s_0) == (k_range,) or type(s_0) == list:
                    s_k_0 = s_0[k]
                else:
                    s_k_0 = s_0

                # Set beta such that mode of gamma distribution for s is s_k_0
                if penalty_type == 'integral':
                    s_k_beta = (s_k_alpha - 1) / s_k_0
                else:
                    s_k_beta = (s_k_alpha - 0.5) / s_k_0

                if not eff_hp:
                    rho_k_eff = rho_vector[k]
                else:
                    rho_k_eff = 1

                # if k == 0:
                #     xm = np.diag(np.sign(x) * np.abs(x) ** 0.5)
                #     g_mat = penalty_matrices['m1'].copy()
                #     g_mat = xm @ g_mat @ xm
                # else:
                #     g_mat = 0
                #
                # sv_out = solve_s(pm_k, x, sv_in, rho_k_eff, s_k_alpha, s_k_beta, g_mat, sigma_ds=sigma_ds[k],
                #                  penalty_type=penalty_type, x_offset=0) #x_offsets[k])
                #
                # # Handle numerical instabilities that may arise for large lambda_0 and small hl_alpha
                # sv_out[sv_out <= 0] = 1e-15
                # sv_out[:num_special] = s_k_0
                # s_vectors[k] = sv_out
                # s_converged[k] = is_converged(sv_in, sv_out, np.mean(s_k_0) * 5e-2, 1e-2)

                x_drt = x[num_special:]
                pm_drt = pm_k[num_special:, num_special:]
                sv_drt = sv_in[num_special:]

                if k == 0:
                    xm = np.diag(np.sign(x_drt) * np.abs(x_drt) ** 0.5)
                    g_mat_drt = penalty_matrices['m1'][num_special:, num_special:]
                    g_mat_drt = xm @ g_mat_drt @ xm
                else:
                    g_mat_drt = 0

                sv_out = solve_s(pm_drt, x_drt, sv_drt, rho_k_eff, s_k_alpha, s_k_beta, g_mat_drt,
                                 sigma_ds=sigma_ds[k], penalty_type=penalty_type, x_offset=0)

                # Handle numerical instabilities that may arise for large lambda_0 and small hl_alpha
                sv_out[sv_out <= 0] = 1e-15
                s_vectors[k][num_special:] = sv_out
                
                # TODO: temp fix
                if nonlin:
                    s_vectors[k][M: M + num_special] = s_k_0

                s_converged[k] = is_converged(sv_drt, sv_out, np.mean(s_k_0) * 5e-2, 1e-2)

                # calculate global derivative strength
                rho_k_alpha = rho_alpha[k]
                if np.shape(rho_0) == (k_range,):
                    rho_k_0 = rho_0[k]
                else:
                    rho_k_0 = rho_0
                rho_k_beta = rho_k_alpha / rho_k_0

                # Only include DRT penalty in rho calculation
                # x_drt = x[num_special:]
                # pm_drt = pm_k[num_special:, num_special:]
                # sv_drt = sv_out[num_special:]
                rho_vector[k] = solve_rho(pm_drt, x_drt, sv_out, rho_k_alpha, rho_k_beta, xmx_norms[k], penalty_type)

                rho_converged[k] = is_converged(rho_in[k], rho_vector[k], rho_k_0 * 5e-2, 1e-2)

                # sm = np.diag(sv_drt ** 0.5)
                # l_beta = l_alphas[k] / 200
                # lambdas[k] = l_alphas[k] / (l_beta + d_weight * rho_vector[k] * x_drt @ sm @ m_drt @ sm @ x_drt)

        # print(it, rho_vector)
        # rho_converged = is_converged(rho_in, rho_vector, rho_0 * 5e-2, 1e-2)

        if np.min(rho_converged) and np.min(s_converged):
            # print(f'hp update converged in {it + 1} iterations')
            break
        else:
            it += 1
    # print('xmx norms', xmx_norms)
    # print('lambdas', lambdas)

    # DOP
    # --------------------
    if 'x_dop' in special_qp_params.keys():
        dop_start = special_qp_params['x_dop']['index']
        dop_end = dop_start + special_qp_params['x_dop']['size']

        # dop_l2_lambda_0 = hypers['dop_l2_lambda_0']
        dop_derivative_weights = hypers['dop_derivative_weights']
        dop_rho_alpha = hypers['dop_rho_alpha']
        dop_rho_0 = hypers['dop_rho_0']
        dop_s_alpha = hypers['dop_s_alpha']
        dop_s_0 = hypers['dop_s_0']
        dop_sigma_ds = hypers['dop_sigma_ds']

        it = 0
        while it < max_hp_iter:
            s_converged = [False] * k_range
            rho_converged = [False] * k_range
            rho_in = dop_rho_vector.copy()
            for k, d_weight in enumerate(dop_derivative_weights):
                if d_weight > 0:
                    # Get penalty matrix
                    if penalty_type == 'integral':
                        pm_k = penalty_matrices[f'm{k}']
                    else:
                        pm_k = penalty_matrices[f'l{k}']

                    sv_in = s_vectors[k]
                    # Solve for lambda scale vector
                    s_k_alpha = dop_s_alpha[k]
                    if np.shape(dop_s_0) == (k_range,) or type(dop_s_0) == list:
                        s_k_0 = dop_s_0[k]
                    else:
                        s_k_0 = dop_s_0

                    # Set beta such that mode of gamma distribution for s is s_k_0
                    if penalty_type == 'integral':
                        s_k_beta = (s_k_alpha - 1) / s_k_0
                    else:
                        s_k_beta = (s_k_alpha - 0.5) / s_k_0

                    if not eff_hp:
                        rho_k_eff = dop_rho_vector[k]
                    else:
                        rho_k_eff = 1

                    # if k == 0:
                    #     xm = np.diag(np.sign(x) * np.abs(x) ** 0.5)
                    #     g_mat = penalty_matrices['m1'].copy()
                    #     g_mat = xm @ g_mat @ xm
                    # else:
                    #     g_mat = 0

                    # sv_out = solve_s(pm_k, x, sv_in, rho_k_eff, s_k_alpha, s_k_beta, g_mat, sigma_ds=sigma_ds[k],
                    #                  penalty_type=penalty_type, x_offset=x_offsets[k])

                    x_dop = x[dop_start:dop_end]
                    pm_dop = pm_k[dop_start:dop_end, dop_start:dop_end]
                    sv_dop = sv_in[dop_start:dop_end]
                    # print(sv_dop)

                    g_mat_dop = penalty_matrices.get(f'gmat{k}_dop', 0)
                    # if k == 0:
                    #     xm = np.diag(np.sign(x_dop * np.abs(x_dop) ** 0.5)
                    #     g_mat_drt = penalty_matrices['m1'].copy()[num_special:, num_special:]
                    #     # if penalty_type == 'discrete':
                    #     #     g_mat *= 0.2
                    #     g_mat_drt = xm @ g_mat_drt @ xm
                    #     # if penalty_type == 'discrete':
                    #     #     g_mat += 0.2 * np.diag(x) @ penalty_matrices[f'm{k}'] @ np.diag(x)
                    # else:
                    #     g_mat_drt = 0

                    sv_out = solve_s(pm_dop, x_dop, sv_dop, rho_k_eff, s_k_alpha, s_k_beta, g_mat_dop,
                                     sigma_ds=dop_sigma_ds[k], penalty_type=penalty_type, x_offset=0)
                    # print(pm_dop)

                    # Handle numerical instabilities that may arise for large lambda_0 and small hl_alpha
                    sv_out[sv_out <= 0] = 1e-15
                    s_vectors[k][dop_start:dop_end] = sv_out

                    s_converged[k] = is_converged(sv_dop, sv_out, np.mean(s_k_0) * 5e-2, 1e-2)

                    # calculate global derivative strength
                    rho_k_alpha = dop_rho_alpha[k]
                    if np.shape(dop_rho_0) == (k_range,):
                        rho_k_0 = dop_rho_0[k]
                    else:
                        rho_k_0 = dop_rho_0
                    rho_k_beta = rho_k_alpha / rho_k_0

                    # Only include DOP penalty in rho calculation
                    # x_dop = x[dop_start:dop_end]
                    # pm_drt = pm_k[dop_start:dop_end, dop_start:dop_end]
                    # sv_drt = sv_out[num_special:]
                    # print('x_dop:', x_dop)
                    dop_rho_vector[k] = solve_rho(pm_dop, x_dop, sv_out, rho_k_alpha, rho_k_beta,
                                                  dop_xmx_norms[k],
                                                  penalty_type)

                    rho_converged[k] = is_converged(rho_in[k], dop_rho_vector[k], rho_k_0 * 5e-2, 1e-2)

                    # sm = np.diag(sv_drt ** 0.5)
                    # l_beta = l_alphas[k] / 200
                    # lambdas[k] = l_alphas[k] / (l_beta + d_weight * rho_vector[k] * x_drt @ sm @ m_drt @ sm @ x_drt)

            # print(it, rho_vector)
            # rho_converged = is_converged(rho_in, rho_vector, rho_0 * 5e-2, 1e-2)

            if np.min(rho_converged) and np.min(s_converged):
                # print(f'hp update converged in {it + 1} iterations')
                break
            else:
                it += 1
    # ---------
    # END DOP

    # Estimate weights
    weights, outlier_t, out_tvt = estimate_weights(x, rv, vmm, rm, est_weights, None, None,
                                                   out_tvt, outlier_p)
    # print(weights[0], est_weights[0])

    # Calculate cost for diagnostics
    # P = wrm.T @ wrm + l2_matrix
    # q = (-wrm.T @ wrv + l1_lambda_vector)
    # cost = 0.5 * x.T @ P @ x + q.T @ x

    # lp = evaluate_posterior_lp(x, penalty_matrices, penalty_type, hypers, l1_lambda_vector, rho_vector, s_vectors,
    #                            weights, rm, rv, xmx_norms)

    if history is not None:
        history.append(
            {'x': x.copy(),
             # 'l1_lambda_vector': l1_lambda_vector.copy(),
             # 'l2_lambda_0': l2_lambda_0,
             's_vectors': s_vectors.copy(),
             'rho_vector': rho_vector.copy(),
             'dop_rho_vector': deepcopy(dop_rho_vector),
             'weights': weights.copy(),
             'outlier_t': outlier_t,
             'fun': cvx_result['primal objective'],
             # 'cost': cost,
             # 'lp': lp,
             'cvx_result': cvx_result,
             # 'nu_vectors': nu_vectors.copy()
             }
        )

    # check for convergence
    x_atol = np.mean(x_in) * 1e-3  # absolute tolerance
    converged = is_converged(x_in, x, x_atol, x_rtol)

    return x, s_vectors, rho_vector, dop_rho_vector, weights, outlier_t, out_tvt, cvx_result, converged


# def iterate_qphb2(x_in, s_vectors, rho_vector, rv, weights, est_weights, outlier_t,
#                   rm, vmm, penalty_matrices, penalty_type, l1_lambda_vector,
#                   hypers, eff_hp,
#                   xmx_norms, fixed_x_index, fixed_x_values,
#                   nonneg, special_qp_params, x_rtol, max_hp_iter, history):
#     # Unpack hyperparameter dict
#     l2_lambda_0 = hypers['l2_lambda_0']
#     derivative_weights = hypers['derivative_weights']
#     rho_alpha = hypers['rho_alpha']
#     rho_0 = hypers['rho_0']
#     s_alpha = hypers['s_alpha']
#     s_0 = hypers['s_0']
#     # outlier_lambda = hypers['outlier_lambda']
#     outlier_p = hypers['outlier_p']
#     # if outlier_alpha is not None:
#     #     outlier_args = (outlier_alpha, outlier_beta, outlier_variance, 2)
#     # else:
#     #     outlier_args = None
#     k_range = len(derivative_weights)
#
#     # Apply weights to rm and rv
#     # wv = format_chrono_weights(rv, weights)
#     wm = np.diag(weights)
#     wrm = wm @ rm
#     wrv = wm @ rv
#
#     # Make l2_matrix matrix for each derivative order
#     l2_matrix = calculate_qp_l2_matrix(np.array(derivative_weights), rho_vector, penalty_matrices, s_vectors,
#                                        l2_lambda_0, penalty_type, special_qp_params, dop_params)
#     # print('l2_matrix', np.sum(l2_matrix))
#
#     # Solve the ridge problem with QP: optimize x
#     cvx_result = solve_convex_opt(wrv, wrm, l2_matrix, l1_lambda_vector, nonneg, special_qp_params,
#                                   fixed_x_index=fixed_x_index, fixed_x_values=fixed_x_values)
#     # init_vals={'x': cvxopt.matrix(x0)})
#     x = np.array(list(cvx_result['x']))
#
#     if fixed_x_index is not None:
#         # insert_index = [index - i for i, index in enumerate(fixed_x_index)]
#         # print(fixed_x_index, insert_index)
#         for index, value in zip(fixed_x_index, fixed_x_values):
#             x = np.insert(x, index, value)
#
#     # Get number of special (non-DRT) parameters in x vector
#     num_special = len(special_qp_params)
#
#     # lambdas = np.zeros(3)
#     # l_alphas = [10, 5, 1]
#
#     # For each derivative order, update the global penalty strength and the local penalty scale vector
#     s_vectors = s_vectors.copy()
#     rho_vector = rho_vector.copy()
#
#     it = 0
#     while it < max_hp_iter:
#         rho_converged = [False] * k_range
#         rho_in = rho_vector.copy()
#         sv_in = s_vectors[0].copy()
#
#         # First calculate overall s_vector
#         m_tot = 0
#         for k in range(k_range):
#             if penalty_type == 'integral':
#                 m_tot += penalty_matrices[f'm{k}'] * derivative_weights[k] * rho_vector[k]
#             else:
#                 m_tot += penalty_matrices[f'l{k}'] * (derivative_weights[k] * rho_vector[k]) ** 0.5
#
#         # Solve for lambda scale vector
#         s_k_alpha = s_alpha[0]
#         if np.shape(s_0) == (k_range,):
#             s_k_0 = s_0[0]
#         else:
#             s_k_0 = s_0
#
#         # Set beta such that mode of gamma distribution for s is s_k_0
#         if penalty_type == 'integral':
#             s_k_beta = (s_k_alpha - 1) / s_k_0
#         else:
#             s_k_beta = (s_k_alpha - 0.5) / s_k_0
#
#         # if not eff_hp:
#         #     rho_k = rho_vector[0]
#         # else:
#         #     rho_k = 1
#
#         xm = np.diag(np.sign(x) * np.abs(x) ** 0.5)
#         if penalty_type == 'discrete':
#             sm = np.diag(sv_in)
#             g_mat = penalty_matrices['l1'].T @ sm @ penalty_matrices['l1']
#             g_mat = xm @ g_mat @ xm
#         else:
#             g_mat = xm @ penalty_matrices['m1'] @ xm
#         # g_mat = penalty_matrices['m1']
#         # g_mat *= 1e-10
#
#         sv_out = solve_s(1 * m_tot / m_tot[-1, -1], x, sv_in, 1, s_k_alpha, s_k_beta, g_mat, 1, penalty_type)
#
#         # Handle numerical instabilities that may arise for large lambda_0 and small hl_alpha
#         sv_out[sv_out <= 0] = 1e-15
#         for k in range(k_range):
#             s_vectors[k] = sv_out
#
#         s_converged = is_converged(sv_in, sv_out, s_k_0 * 5e-2, 1e-2)
#
#         # Next calculate rho_vector
#         for k, d_weight in enumerate(derivative_weights):
#             if d_weight > 0:
#                 if penalty_type == 'integral':
#                     pm_k = penalty_matrices[f'm{k}']
#                 else:
#                     pm_k = penalty_matrices[f'l{k}']
#
#                 # calculate derivative strength
#                 rho_k_alpha = rho_alpha[k]
#                 if np.shape(rho_0) == (k_range,):
#                     rho_k_0 = rho_0[k]
#                 else:
#                     rho_k_0 = rho_0
#                 rho_k_beta = rho_k_alpha / rho_k_0
#
#                 # Only include DRT penalty in rho calculation
#                 x_drt = x[num_special:]
#                 m_drt = pm_k[num_special:, num_special:]
#                 sv_drt = sv_out[num_special:]
#                 rho_vector[k] = solve_rho(m_drt, x_drt, sv_drt, rho_k_alpha, rho_k_beta, xmx_norms[k], penalty_type)
#
#                 rho_converged[k] = is_converged(rho_in[k], rho_vector[k], rho_k_0 * 5e-2, 1e-2)
#
#                 # sm = np.diag(sv_drt ** 0.5)
#                 # l_beta = l_alphas[k] / 200
#                 # lambdas[k] = l_alphas[k] / (l_beta + d_weight * rho_vector[k] * x_drt @ sm @ m_drt @ sm @ x_drt)
#
#         # print(it, rho_vector)
#         # rho_converged = is_converged(rho_in, rho_vector, rho_0 * 5e-2, 1e-2)
#
#         if np.min(rho_converged) and s_converged:
#             # print(f'hp update converged in {it + 1} iterations')
#             break
#         else:
#             it += 1
#     # print('xmx norms', xmx_norms)
#     # print('lambdas', lambdas)
#
#     ############# original cvx location
#
#     # Estimate weights
#     weights, outlier_t = estimate_weights(x, rv, vmm, rm, est_weights, None, None, outlier_t, outlier_p)
#     # print(weights[0], est_weights[0])
#
#     # Calculate cost for diagnostics
#     P = wrm.T @ wrm + l2_matrix
#     q = (-wrm.T @ wrv + l1_lambda_vector)
#     cost = 0.5 * x.T @ P @ x + q.T @ x
#
#     if history is not None:
#         history.append(
#             {'x': x.copy(),
#              'l1_lambda_vector': l1_lambda_vector.copy(),
#              'l2_lambda_0': l2_lambda_0,
#              's_vectors': s_vectors.copy(),
#              'rho_vector': rho_vector.copy(),
#              'weights': weights.copy(),
#              'outlier_t': outlier_t,
#              'fun': cvx_result['primal objective'],
#              'cost': cost,
#              'cvx_result': cvx_result,
#              }
#         )
#
#     # check for convergence
#     x_atol = np.mean(x_in) * 1e-3
#     converged = is_converged(x_in, x, x_atol, x_rtol)
#
#     return x, s_vectors, rho_vector, weights, outlier_t, cvx_result, converged


def iterate_md_qphb(x_in, s_vectors, rho_array, rho_diagonals, data_vector, weights, est_weights, outlier_t,
                    rm, vmm, penalty_matrices, penalty_type, l1_lambda_vector,
                    l2_lambda_0_diagonal, qphb_hypers,  # derivative_weights, rho_alpha, rho_0, s_alpha, s_0,
                    gp_args,
                    xmx_norm_array, fixed_x_index, fixed_x_values,
                    batch_size, params_per_obs, special_qp_params, x_rtol, max_hp_iter, history):
    # what needs to be different for MD?
    # 1: calculate rho_vector independently for each observation (can/should these be linked?)
    # 2: Add a term to SMS for GP covariance across psi
    # 3:

    derivative_weights = qphb_hypers['derivative_weights']
    rho_alpha = qphb_hypers['rho_alpha']
    rho_0 = qphb_hypers['rho_0']
    s_alpha = qphb_hypers['s_alpha']
    s_0 = qphb_hypers['s_0']
    nonneg = qphb_hypers['nonneg']
    outlier_p = qphb_hypers['outlier_p']
    k_range = len(derivative_weights)

    # Apply weights to rm and rv
    # wv = format_chrono_weights(rv, weights)
    wm = np.diag(weights)
    wrm = wm @ rm
    wrv = wm @ data_vector

    if gp_args is not None:
        gp_omega, gp_mu, gp_rho_diag, gp_s_vector, gp_rho_alpha, gp_rho_0, gp_s_alpha, gp_s_0, gp_frac = gp_args
        # print('min s', np.min(gp_s_vector))
        s_diag = np.diag(gp_s_vector ** 0.5)
        # print(gp_omega.shape)
        gp_lambda_0_diag = np.sqrt(gp_frac) * l2_lambda_0_diagonal
        gp_sms = gp_lambda_0_diag @ gp_rho_diag @ s_diag @ gp_omega @ s_diag @ gp_rho_diag @ gp_lambda_0_diag
        gp_q_vector = -gp_mu.T @ gp_sms
    else:
        gp_sms = 0
        gp_q_vector = 0
        gp_frac = 0

    # Make l2_matrix matrix for each derivative order
    # l2_matrices = [rho_diagonals[k] @ penalty_matrices[f'm{k}'] @ rho_diagonals[k]
    #                for k in range(len(derivative_weights))]
    # l2_matrix = calculate_qp_l2_matrix(derivative_weights, rho_vector, l2_matrices, s_vectors, l2_lambda_0, penalty_type)
    # # gphb_lambda_0_diag = ((1 - gp_frac) * l2_lambda_0_diagonal) ** 0.5
    # l2_matrix = l2_lambda_0_diagonal @ l2_matrix @ l2_lambda_0_diagonal

    l2_matrix = calculate_md_qp_l2_matrix(derivative_weights, rho_diagonals, penalty_matrices, s_vectors,
                                          l2_lambda_0_diagonal, penalty_type)

    # Solve the ridge problem with QP: optimize x
    # Multiply l2_matrix by 2 due to exponential prior
    cvx_result = solve_convex_opt(wrv, wrm, l2_matrix + gp_sms, l1_lambda_vector + gp_q_vector, nonneg,
                                  special_qp_params, fixed_x_index=fixed_x_index, fixed_x_values=fixed_x_values)
    # init_vals={'x': cvxopt.matrix(x0)})
    x = np.array(list(cvx_result['x']))

    if fixed_x_index is not None:
        # insert_index = [index - i for i, index in enumerate(fixed_x_index)]
        # print(fixed_x_index, insert_index)
        for index, value in zip(fixed_x_index, fixed_x_values):
            x = np.insert(x, index, value)

    # Get number of special (non-DRT) parameters in x vector
    num_special = get_num_special(special_qp_params)

    # lambdas = np.zeros(3)
    # l_alphas = [10, 5, 1]

    # For each derivative order, update the global penalty strength and the local penalty scales
    s_vectors = s_vectors.copy()
    rho_array = rho_array.copy()

    it = 0
    while it < max_hp_iter:
        s_converged = [False] * k_range
        rho_converged = [False] * k_range
        rho_in = rho_array.copy()
        for k, d_weight in enumerate(derivative_weights):
            if d_weight > 0:
                m = penalty_matrices[f'm{k}']
                sv_in = s_vectors[k]
                # Solve for lambda scale vector
                s_k_alpha = s_alpha[k]
                if np.shape(s_0) == (k_range,):
                    s_k_0 = s_0[k]
                else:
                    s_k_0 = s_0
                s_k_beta = (s_k_alpha - 1) / s_k_0  # Ensure mode of gamma distribution for s is 1
                sv_out = solve_s(m, x, sv_in, 1, s_k_alpha, s_k_beta, g_mat, 1, penalty_type)

                # Handle numerical instabilities that may arise for large lambda_0 and small hl_alpha
                # sv_out[sv_out <= 0] = 1e-15
                s_vectors[k] = sv_out

                s_converged[k] = is_converged(sv_in, sv_out, s_k_0 * 5e-2, 1e-2)

                # calculate derivative strength
                rho_k_alpha = rho_alpha[k]
                if np.shape(rho_0) == (k_range,):
                    rho_k_0 = rho_0[k]
                else:
                    rho_k_0 = rho_0
                rho_k_beta = rho_k_alpha / rho_k_0

                # Only include DRT penalty in rho calculation
                for i in range(batch_size):
                    start_index = i * params_per_obs
                    end_index = start_index + params_per_obs
                    x_drt = x[start_index + num_special:end_index]
                    m_drt = m[start_index + num_special:end_index, start_index + num_special:end_index]
                    sv_drt = sv_out[start_index + num_special:end_index]
                    rho_array[i, k] = solve_rho(m_drt, x_drt, sv_drt, rho_k_alpha, rho_k_beta, xmx_norm_array[i, k],
                                                penalty_type)

                rho_converged[k] = is_converged(rho_in[:, k], rho_array[:, k], rho_k_0 * 5e-2, 1e-2)

                # sm = np.diag(sv_drt ** 0.5)
                # l_beta = l_alphas[k] / 200
                # lambdas[k] = l_alphas[k] / (l_beta + d_weight * rho_vector[k] * x_drt @ sm @ m_drt @ sm @ x_drt)

        if gp_args is not None:
            gp_omega, gp_mu, gp_rho_diag, gp_s_vector, gp_rho_alpha, gp_rho_0, gp_s_alpha, gp_s_0, gp_frac = gp_args
            gp_s_beta = (gp_s_alpha - 0.5) / gp_s_0
            gp_rho_beta = (gp_rho_alpha - 0.5) / gp_rho_0

            # Calculate full GP s vector
            gp_sv_out = solve_gp_s(gp_omega, gp_mu, x, gp_s_vector, gp_s_alpha, gp_s_beta)
            # gp_sv_out[gp_sv_out <= 0] = 1e-15

            # Calculate gp_rho for each observation in batch
            gp_rho_out = np.zeros(batch_size)
            for i in range(batch_size):
                start_index = i * params_per_obs
                end_index = start_index + params_per_obs
                x_i = x[start_index:end_index]
                gp_omega_i = gp_omega[start_index:end_index, start_index:end_index]
                gp_mu_i = gp_mu[start_index:end_index]
                sv_i = gp_sv_out[start_index:end_index]
                gp_rho_out[i] = solve_gp_rho(gp_omega_i, gp_mu_i, x_i, sv_i, gp_rho_alpha, gp_rho_beta)

            gp_hypers_out = (gp_rho_out, gp_sv_out)
        else:
            gp_hypers_out = None

        if rho_converged and np.min(s_converged):
            # print(f'hp update converged in {it + 1} iterations')
            break
        else:
            it += 1

    # Estimate weights
    weights, outlier_t = estimate_weights(x, data_vector, vmm, rm, est_weights, None, None, outlier_t, outlier_p)
    # print(weights[0], est_weights[0])

    # Calculate cost for diagnostics
    P = wrm.T @ wrm + l2_matrix
    q = (-wrm.T @ wrv + l1_lambda_vector)
    cost = 0.5 * x.T @ P @ x + q.T @ x

    if history is not None:
        history.append(
            {'x': x.copy(),
             # 'l1_lambda_vector': l1_lambda_vector.copy(),
             # 'l2_lambda_0': l2_lambda_0_,
             's_vectors': s_vectors.copy(),
             'rho_array': rho_array.copy(),
             'weights': weights.copy(),
             'outlier_t': outlier_t.copy(),
             'fun': cvx_result['primal objective'],
             'cost': cost,
             'cvx_result': cvx_result,
             'gp_hypers': gp_hypers_out
             }
        )

    # check for convergence
    x_atol = np.mean(x_in) * 1e-3
    converged = is_converged(x_in, x, x_atol, x_rtol)

    return x, s_vectors, rho_array, weights, outlier_t, gp_hypers_out, cvx_result, converged


# ==============================
# Credible interval construction
# ==============================
def calculate_pq(rm, rv, penalty_matrices, penalty_type, hypers, l1_lambda_vector, rho_vector, dop_rho_vector,
                 s_vectors, weights, special_qp_params):
    """
    Calculate P and q for quadratic programming
    :param dop_rho_vector:
    :param special_qp_params:
    :param penalty_type:
    :param rm:
    :param rv:
    :param penalty_matrices:
    :param derivative_weights:
    :param l2_lambda_0:
    :param l1_lambda_vector:
    :param rho_vector:
    :param s_vectors:
    :param weights:
    :return:
    """
    # l2_matrices = [penalty_matrices[f'm{n}'] for n in range(len(derivative_weights))]
    l2_matrix = calculate_qp_l2_matrix(hypers, rho_vector, dop_rho_vector, penalty_matrices,
                                       s_vectors, penalty_type, special_qp_params)

    wm = np.diag(weights)
    wrm = wm @ rm
    wrv = wm @ rv

    p_matrix = l2_matrix + wrm.T @ wrm
    q_vector = -wrm.T @ wrv + l1_lambda_vector

    return p_matrix, q_vector


def calculate_md_pq(rm, rv, penalty_matrices, penalty_type, derivative_weights, l2_lambda_0_diagonal, l1_lambda_vector,
                    rho_diagonals,
                    s_vectors, weights):
    l2_matrix = calculate_md_qp_l2_matrix(derivative_weights, rho_diagonals, penalty_matrices, s_vectors,
                                          l2_lambda_0_diagonal, penalty_type)

    wm = np.diag(weights)
    wrm = wm @ rm
    wrv = wm @ rv

    p_matrix = l2_matrix + wrm.T @ wrm
    q_vector = -wrm.T @ wrv + l1_lambda_vector

    return p_matrix, q_vector


# def evaluate_hessian(p_matrix, q_vector, x):
#     # u = p_matrix @ x + q_vector.T
#     # return np.outer(u, u) - p_matrix  # curvature of pdf, not log pdf
#     return -p_matrix


def evaluate_gradient(p_matrix, q_vector, x):
    return p_matrix @ x + q_vector.T


def get_raw_hyperparams(hypers, rho_vector, xmx_norms):
    "Get raw hyperparameters from effective"
    # Unpack hyperparameter dict
    l2_lambda_0 = hypers['l2_lambda_0']
    derivative_weights = hypers['derivative_weights']
    rho_alpha = hypers['rho_alpha']
    rho_0 = hypers['rho_0']
    s_alpha = hypers['s_alpha']
    s_0 = hypers['s_0']

    rho_alpha_raw = xmx_norms * l2_lambda_0 * derivative_weights * rho_alpha
    rho_beta = rho_alpha / rho_0
    rho_beta_raw = xmx_norms * l2_lambda_0 * derivative_weights * rho_beta

    s_alpha_raw = l2_lambda_0 * derivative_weights * rho_vector * (s_alpha - 1) + 1
    s_beta = (s_alpha - 1) / s_0
    s_beta_raw = l2_lambda_0 * derivative_weights * rho_vector * s_beta

    return rho_alpha_raw, rho_beta_raw, s_alpha_raw, s_beta_raw


def get_eff_hyperparams(rho_alpha_raw, rho_beta_raw, s_alpha_raw, s_beta_raw, rho_vector, l2_lambda_0,
                        xmx_norms, derivative_weights):
    rho_alpha = rho_alpha_raw / (xmx_norms * l2_lambda_0 * derivative_weights) + 1
    rho_beta = rho_beta_raw / (xmx_norms * l2_lambda_0 * derivative_weights)
    rho_0 = rho_alpha / rho_beta

    s_alpha = (s_alpha_raw - 1) / (l2_lambda_0 * derivative_weights * rho_vector) + 1
    s_beta = s_beta_raw / (l2_lambda_0 * derivative_weights * rho_vector)
    s_0 = (s_alpha - 1) / s_beta

    return rho_alpha, rho_0, s_alpha, s_0


def evaluate_posterior_lp(x, penalty_matrices, penalty_type, hypers, l1_lambda_vector, rho_vector, dop_rho_vector,
                          s_vectors, weights, rm, rv, xmx_norms, special_qp_params):
    p_matrix, q_vector = calculate_pq(rm, rv, penalty_matrices, penalty_type, hypers, l1_lambda_vector, rho_vector,
                                      dop_rho_vector, s_vectors, weights, special_qp_params)

    wm = np.diag(weights)
    wrv = wm @ rv

    lp_x = -0.5 * (x.T @ p_matrix @ x) - q_vector.T @ x - 0.5 * wrv.T @ wrv

    # Convert effective hyperparameters to raw (true) hyperparameters
    rho_alpha_raw, rho_beta_raw, s_alpha_raw, s_beta_raw = get_raw_hyperparams(hypers, rho_vector, xmx_norms)

    # rho prior
    active_index = np.where(hypers['derivative_weights'] > 0)
    lp_rho = log_pdf_gamma(rho_vector[active_index], rho_alpha_raw[active_index],
                           rho_beta_raw[active_index], True)
    lp_rho = np.sum(lp_rho)  # / rho_alpha_raw)

    # s prior
    lp_s = 0
    # print('s params:', s_alpha_raw, s_beta_raw)
    for k in range(len(rho_vector)):
        if hypers['derivative_weights'][k] > 0:
            lp_s_vec = log_pdf_gamma(s_vectors[k], s_alpha_raw[k], s_beta_raw[k], True)
            lp_s += np.sum(lp_s_vec)  # - len(s_vectors[k]) * np.log(s_alpha_raw[k] * s_beta_raw[k])#/ s_alpha_raw[k]

    # print('eval time: {:.3f} s'.format(time.time() - start))
    # print(lp_x, lp_rho, lp_s)

    return lp_x + lp_rho + lp_s


def evaluate_lml(x_hat, penalty_matrices, penalty_type, hypers, l1_lambda_vector, rho_vector, dop_rho_vector, s_vectors,
                 weights, rm, rv, special_qp_params, alpha_0=1, beta_0=1):
    """
    Evaluate log-marginal likelihood (evidence) assuming fixed hyperparameters (s, rho)
    :param x_hat:
    :param penalty_matrices:
    :param penalty_type:
    :param hypers:
    :param l1_lambda_vector:
    :param rho_vector:
    :param s_vectors:
    :param weights:
    :param rm:
    :param rv:
    :return:
    """
    # Get posterior precision matrix (P)
    p_matrix, q_vector = calculate_pq(rm, rv, penalty_matrices, penalty_type, hypers, l1_lambda_vector, rho_vector,
                                      dop_rho_vector, s_vectors, weights, special_qp_params)
    # q_vector = p_matrix @ x_hat  # equivalent
    # q_vector = -wrm.T @ wrv

    det_sign, log_det_p = np.linalg.slogdet(p_matrix)
    if det_sign < 0:
        raise ValueError('Determinant of posterior covariance matrix is negative - not PSD')

    # Get prior precision matrix
    wrm = np.diag(weights) @ rm
    # waaw = wrm.T @ wrm
    # omega = p_matrix - waaw  # prior precision matrix
    omega = calculate_qp_l2_matrix(hypers, rho_vector, dop_rho_vector, penalty_matrices,
                                   s_vectors, penalty_type, special_qp_params)
    # print(omega)

    det_sign, log_det_omega = np.linalg.slogdet(omega)
    if det_sign < 0:
        raise ValueError('Determinant of prior covariance matrix is negative - not PSD')

    wrv = np.diag(weights) @ rv

    # lml = 0.5 * (
    #         log_det_omega - log_det_p
    #         - wrv.T @ wrv
    #         + x_hat.T @ p_matrix @ x_hat
    #         # - x_hat @ wrm.T @ wrm @ x_hat
    #         # - 2 * q_vector.T @ x_hat
    # ) + np.sum(np.log(weights))
    # print(0.5 * (log_det_p_inv - log_det_omega_inv), 0.5 * x_hat.T @ p_matrix @ x_hat - 0.5 * wrv.T @ wrv)
    # print(-0.5 * (x_hat @ wrm.T @ wrm @ x_hat - 2 * wrv.T @ wrm @ x_hat + wrv.T @ wrv) + np.sum(np.log(weights)))
    # print(-0.5 * (x_hat @ wrm.T @ wrm @ x_hat + 2 * q_vector.T @ x_hat + wrv.T @ wrv) + np.sum(np.log(weights)))
    # print(wrv.T @ wrv - x_hat.T @ wrm.T @ wrm @ x_hat)
    # print(wrv.T @ wrv - x_hat.T @ p_matrix @ x_hat)
    # lml = 0.5 * (log_det_omega_inv - log_det_p_inv) + np.sum(np.log(weights)) - np.log(0.5 * (wrv.T @ wrv - x_hat.T @ p_matrix @ x_hat))

    # mu_n = np.linalg.inv(p_matrix) @ wrm.T @ wrv
    # print((mu_n - x_hat) / np.median(x_hat))

    alpha = len(rv) / 2 + alpha_0
    beta = 0.5 * (wrv.T @ wrv - x_hat.T @ wrm.T @ wrm @ x_hat - x_hat.T @ omega @ x_hat) + beta_0
    # print('beta:', beta)
    # print('xMx:', x_hat.T @ omega @ x_hat)
    # print('y2 - Ax2:', wrv.T @ wrv - x_hat.T @ wrm.T @ wrm @ x_hat )
    lml = 0.5 * (log_det_omega - log_det_p) + np.sum(np.log(weights)) \
          + loggamma(alpha) - loggamma(alpha_0) + alpha_0 * np.log(beta_0) - alpha * np.log(beta)

    return lml


def evaluate_rss(x_hat, rm, rv, weights):
    wm = np.diag(weights)
    wrm = wm @ rm
    wrv = wm @ rv

    return x_hat @ wrm.T @ wrm @ x_hat - 2 * wrv.T @ wrm @ x_hat + wrv.T @ wrv


def evaluate_llh(x_hat, rm, rv, weights, marginalize_weights=True, alpha_0=2, beta_0=1, include_constants=False):
    wm = np.diag(weights)
    wrm = wm @ rm
    wrv = wm @ rv

    if marginalize_weights:
        alpha_n = alpha_0 - 1 + len(rv) / 2
        beta_n = beta_0 + 0.5 * (x_hat @ wrm.T @ wrm @ x_hat - 2 * wrv.T @ wrm @ x_hat + wrv.T @ wrv)
        llh = alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n) + loggamma(alpha_n) - loggamma(alpha_0)

        # alpha_n = - 1 + len(rv) / 2
        # beta_n = + 0.5 * (x_hat @ wrm.T @ wrm @ x_hat - 2 * wrv.T @ wrm @ x_hat + wrv.T @ wrv)
        # llh = - alpha_n * np.log(beta_n) + loggamma(alpha_n)
    else:
        llh = -0.5 * (x_hat @ wrm.T @ wrm @ x_hat - 2 * wrv.T @ wrm @ x_hat + wrv.T @ wrv)

    # Add sum of log weights
    llh += np.sum(np.log(weights))

    if include_constants:
        llh -= 0.5 * len(rv) * np.log(2 * np.pi)

    return llh


def evaluate_md_posterior_lp(x, derivative_weights, penalty_matrices, penalty_type, l2_lambda_0_vector,
                             l2_lambda_0_diagonal,
                             l1_lambda_vector, rho_diagonals, rho_array, s_vectors, weights, rm, rv, rho_alpha, rho_0,
                             s_alpha, s_0, xmx_norm_array):
    p_matrix, q_vector = calculate_md_pq(rm, rv, penalty_matrices, penalty_type, derivative_weights,
                                         l2_lambda_0_diagonal, l1_lambda_vector, rho_diagonals, s_vectors, weights)

    wm = np.diag(weights)
    wrv = wm @ rv

    lp_x = -0.5 * (x.T @ p_matrix @ x) - q_vector.T @ x - 0.5 * wrv.T @ wrv

    # Tile arrays to consistent shape
    num_obs = len(l2_lambda_0_vector)
    num_params = int(len(x) / num_obs)
    rho_alpha_nk = np.tile(rho_alpha, (num_obs, 1))
    s_alpha_nk = np.tile(s_alpha, (num_obs, 1))
    d_weight_nk = np.tile(derivative_weights, (num_obs, 1))
    l2_lambda_0_nk = np.tile(l2_lambda_0_vector, (len(derivative_weights), 1)).T

    # Convert effective hyperparameters to raw (true) hyperparameters
    rho_alpha_raw, rho_beta_raw, s_alpha_raw, s_beta_raw = get_raw_hyperparams(rho_alpha_nk, rho_0, s_alpha_nk, s_0,
                                                                               rho_array, l2_lambda_0_nk,
                                                                               xmx_norm_array, d_weight_nk)

    # rho prior
    lp_rho = log_pdf_gamma(rho_array, rho_alpha_raw, rho_beta_raw, True)
    lp_rho = np.sum(lp_rho)  # / rho_alpha_raw)

    # s prior
    lp_s = 0

    for k in range(len(derivative_weights)):
        s_alpha_raw_rep = np.repeat(s_alpha_raw[:, k], num_params)
        s_beta_raw_rep = np.repeat(s_beta_raw[:, k], num_params)
        lp_s_vec = log_pdf_gamma(s_vectors[k], s_alpha_raw_rep, s_beta_raw_rep, True)
        lp_s += np.sum(lp_s_vec)  # - len(s_vectors[k]) * np.log(s_alpha_raw[k] * s_beta_raw[k])#/ s_alpha_raw[k]

    # print('eval time: {:.3f} s'.format(time.time() - start))
    # print(lp_x, lp_rho, lp_s)

    return lp_x + lp_rho + lp_s


# def optimize_lp_semi_fixed(x_in, fixed_x_index, fixed_prior, s_vectors, rho_vector, rv, weights, est_weights,
#                            rm, vmm, penalty_matrices, l1_lambda_vector,
#                            l2_lambda_0, derivative_weights, rho_alpha, rho_0, s_alpha, s_0,
#                            w_alpha, w_beta, xmx_norms,
#                            nonneg, special_qp_params, xtol, history, max_iter):
#     x = x_in.copy()
#     fixed_x_values = x[fixed_x_index]
#
#     if fixed_prior:
#         # Determine current raw hyperparameters
#         rho_alpha_raw, rho_beta_raw, s_alpha_raw, s_beta_raw = get_raw_hyperparams(rho_alpha, rho_0, s_alpha, s_0,
#                                                                                    rho_vector, l2_lambda_0, xmx_norms,
#                                                                                    derivative_weights)
#
#         # print('init rho_vector:', rho_vector)
#     it = 0
#     while it < max_iter:
#         x, s_vectors, rho_vector, weights, cvx_result, converged = iterate_qphb(x, s_vectors, rho_vector, rv, weights,
#                                                                                 est_weights, outlier_variance, rm, vmm,
#                                                                                 penalty_matrices, penalty_type,
#                                                                                 l1_lambda_vector, l2_lambda_0,
#                                                                                 derivative_weights, rho_alpha, rho_0,
#                                                                                 s_alpha, s_0, xmx_norms, fixed_x_index,
#                                                                                 fixed_x_values, nonneg)
#
#         if fixed_prior:
#             # Update effective hyperparameters to maintain original raw hyperparameter values
#             rho_alpha, rho_0, s_alpha, s_0 = get_eff_hyperparams(rho_alpha_raw, rho_beta_raw, s_alpha_raw, s_beta_raw,
#                                                                  rho_vector, l2_lambda_0, xmx_norms, derivative_weights)
#             # print('effective hyperparams:', rho_alpha, rho_0, s_alpha, s_0)
#             # print('rho_vector:', rho_vector)
#
#         # print(m, it, 'delta x:', (x - x_prev) / x_prev)
#         if converged:
#             break
#         else:
#             it += 1
#
#     lp_new = evaluate_posterior_lp(x, derivative_weights, penalty_type, penalty_matrices, l2_lambda_0, l1_lambda_vector,
#                                    rho_vector, s_vectors, weights, rm, rv)
#
#     return x, lp_new


# =================================
# Variance estimation and weighting
# =================================
def solve_init_weight_scale(w_scale_est, alpha, beta):
    if alpha is not None:
        b = (1 / 2 - alpha + 1)
        s_hat = (-b + np.sqrt(b ** 2 + 2 * beta * (w_scale_est ** -2))) / (2 * beta)
        w_hat = s_hat ** -0.5
    else:
        # Do nothing
        w_hat = deepcopy(w_scale_est)
    return w_hat


def solve_outlier_variance(s_base, resid, alpha, beta):
    # s_out = (-0.5 + np.sqrt(0.25 + 2 * outlier_lambda * resid ** 2)) / (2 * outlier_lambda) - s_base
    s_out = (
                    np.sqrt(
                        8 * alpha * beta * resid ** 2 - 8 * alpha * resid ** 2 * s_base + beta ** 2
                        + 10 * beta * resid ** 2 - 2 * beta * s_base + resid ** 4 - 10 * resid ** 2 * s_base
                        + s_base ** 2
                    )
                    - 4 * alpha * s_base - beta + resid ** 2 - 5 * s_base
            ) / (2 * (2 * alpha + 3))
    s_out = np.maximum(s_out, 0)
    s_out[np.isnan(s_out)] = 0
    return s_out


def solve_outlier_t(vmm, resid, outlier_p):
    """
    Solve for outlier t vector. outlier_t = 1 - outlier_probability
    :param vmm: variance estimation matrix
    :param t_in: current t vector
    :param resid: residuals
    :param outlier_p: prior probability of any point being an outlier (Bernoulli)
    :return:
    """
    # tvt = outlier_tvt(vmm, t_in)

    s_bar = vmm @ resid ** 2

    pdf_in = pdf_normal(resid, 0, np.sqrt(s_bar))
    pdf_out = pdf_normal(resid, 0, np.abs(resid))
    t_out = 1 - outlier_p * pdf_out / ((1 - outlier_p) * pdf_in + outlier_p * pdf_out)
    # Don't allow points with *smaller* residuals than expected by s_bar to be treated as outliers
    t_out[np.sqrt(s_bar) > np.abs(resid)] = 1

    # t_out = t_new
    # t_out = t_in * t_new

    return t_out


def outlier_tvt(vmm, outlier_t):
    """
    Get TVT + (I-T) matrix for variance estimation with outliers
    :param vmm:
    :param outlier_t:
    :return:
    """
    # sqrt_t_diag = np.diag(outlier_t ** 0.5)
    # t_diag = np.diag(1 - outlier_t)
    # tvt = sqrt_t_diag @ vmm @ sqrt_t_diag + t_diag

    # Broadcast multiplication - equivalent to above, but 5x faster
    sqrt_t = outlier_t ** 0.5
    tvt = np.multiply(np.multiply(sqrt_t[:, None], vmm), sqrt_t[None, :])
    tvt += np.diag(1 - outlier_t)

    return tvt


# Chrono weight estimation
# ---------------------------
# TODO: replace with 1d filter if data is uniformly spaced or RSS if error_structure=='uniform'
#  Requires splitting hybrid data vector into EIS and chrono vectors
def estimate_weights(x, y, vmm, rm, est_weights=None, w_alpha=None, w_beta=None,
                     out_tvt=None, outlier_p=None, var_floor=None):
    resid = rm @ x - y

    # Get outlier_t vector
    if outlier_p is not None:
        outlier_t = solve_outlier_t(vmm, resid, outlier_p)
        out_tvt = outlier_tvt(vmm, outlier_t)
        vmm_eff = out_tvt
    else:
        outlier_t = np.ones(len(y))
        out_tvt = None
        vmm_eff = vmm

    # Estimate error variance vector from weighted mean of residuals
    if var_floor is None:
        var_floor = np.var(y) * 1e-7
        # print('var floor:', var_floor)
    s_hat = vmm_eff @ resid ** 2
    s_hat[s_hat < var_floor] = var_floor

    # Convert variance to weights
    w_hat = s_hat ** -0.5

    if est_weights is not None:
        # To ensure convergence (avoid poor initial fit resulting in weights going to zero), average current weights
        # with initial estimate from overfitted ridge

        # As current weights approach initial estimate (indicating better fit), give them more weight
        # scale_current = np.mean(s_hat) ** -0.5
        # scale_est = np.mean(est_weights ** -2) ** -0.5
        # frac_current = 0.5
        # frac_current = scale_current / (scale_current + scale_est)
        frac_current = w_hat / (w_hat + est_weights)
        # frac_current = 1 - np.exp(-2 * scale_current / scale_est)
        # print('frac_current', frac_current)
        frac_est = 1 - frac_current
        # Take mean of current and initial weight estimates
        w_hat = (frac_current * w_hat + frac_est * est_weights)

    if w_alpha is not None and w_beta is not None:
        # Apply prior
        w_scale = np.mean(w_hat)
        # w_beta = w_0 ** 2 * (w_alpha - 1.5)
        w_hat = w_hat * solve_init_weight_scale(w_scale, w_alpha, w_beta) / w_scale

    return w_hat, outlier_t, out_tvt


# def _estimate_weights(resid, error_structure, outlier_t, uniform_spacing):
#     if error_structure == 'uniform':
#         # Inlier contribution
#         s_hat = np.sum((outlier_t * resid) ** 2) / len(resid)
#         s_hat = np.ones(len(resid)) *
#
#         # Add outlier contribution
#         s_hat += (1 - outlier_t) * resid ** 2
#
#         w_hat = s_hat ** -0.5


def initialize_weights(hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector, s_vectors, rv, rm,
                       vmm, nonneg, special_qp_params):
    # Calculate L2 penalty matrix (SMS)
    # l2_matrices = [penalty_matrices[f'm{n}'] for n in range(len(derivative_weights))]
    # Apply very small penalty strength for overfit
    # iw_hypers = hypers.copy()
    # iw_hypers['l2_lambda_0'] = 1e-4
    # # iw_hypers['l2_lambda_0'] = 1e-10
    # if 'dop_l2_lambda_0' in hypers.keys():
    #     dop_drt_ratio = hypers['dop_l2_lambda_0'] / hypers['l2_lambda_0']
    #     iw_hypers['dop_l2_lambda_0'] = dop_drt_ratio * iw_hypers['l2_lambda_0']

    l2_matrix = calculate_qp_l2_matrix(hypers, rho_vector, dop_rho_vector, penalty_matrices,
                                       s_vectors, penalty_type, special_qp_params)

    iw_alpha = hypers['iw_alpha']
    iw_beta = hypers['iw_beta']
    outlier_p = hypers['outlier_p']
    l1_lambda_0 = hypers['l1_lambda_0']

    if outlier_p is not None:
        outlier_t = np.ones(vmm.shape[0])
        est_weights = np.ones(vmm.shape[0])
        out_tvt = outlier_tvt(vmm, outlier_t)
        # Repeat the initial weights calculation to remove outliers
        for i in range(2):
            # Solve the ridge problem with QP: optimize x
            # Multiply l2_matrix by 2 due to exponential prior
            w_diag = np.diag(est_weights)
            cvx_result = solve_convex_opt(w_diag @ rv, w_diag @ rm, l2_matrix, l1_lambda_0, nonneg, special_qp_params)
            x_overfit = np.array(list(cvx_result['x']))
            # print(x_overfit)
            # print(rm @ x_overfit)

            # Get weight structure
            if i == 0:
                # Do not include self in variance estimate for initial outlier weighting
                vmm_base = vmm - np.diag(np.diag(vmm))
                vm_rowsum = 1 - np.diag(vmm)
                vmm_base /= vm_rowsum[:, None]
                vmm = vmm_base

            for j in range(2):
                # Repeat weight calculation to allow outlier_t to converge
                est_weights, outlier_t, out_tvt = estimate_weights(x_overfit, rv, vmm, rm, est_weights=None,
                                                                  out_tvt=out_tvt,
                                                                  outlier_p=outlier_p)

        # print('init outlier prob:', 1 - outlier_t)

    else:
        cvx_result = solve_convex_opt(rv, rm, l2_matrix, l1_lambda_0, nonneg, special_qp_params)
        x_overfit = np.array(list(cvx_result['x']))
        est_weights, outlier_t, out_tvt = estimate_weights(x_overfit, rv, vmm, rm,
                                                           est_weights=None, outlier_p=outlier_p)

    # Get global weight scale
    # Need to average variance, not weights
    # est_var_scale = np.mean(est_weights ** -2)
    # est_weight_scale = est_var_scale ** -0.5
    # est_weight_scale = np.mean(est_weights)

    # # Solve for initial weight scale (global)
    # # iw_beta = iw_0 ** 2 * (iw_alpha - 1.5)
    # init_weight_scale = solve_init_weight_scale(est_weight_scale, iw_alpha, iw_beta)
    #
    # # Get initial weight vector
    # init_weights = est_weights * init_weight_scale / est_weight_scale

    # Local
    init_weights = solve_init_weight_scale(est_weights, iw_alpha, iw_beta)

    return est_weights, init_weights, x_overfit, outlier_t


def estimate_x_rp(hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector, s_vectors, rv, rm,
                  nonneg, special_qp_params, l2_lambda_0=1e-4, l1_lambda_0=1e-3):
    """
    Estimate coefficients for Rp estimation
    :param dop_rho_vector:
    :param penalty_matrices:
    :param penalty_type:
    :param derivative_weights:
    :param rho_vector:
    :param s_vectors:
    :param rv:
    :param rm:
    :param nonneg:
    :param special_qp_params:
    :param l2_lambda_0:
    :param l1_lambda_0:
    :return:
    """
    # Calculate L2 penalty matrix (SMS)
    # Apply small penalty strength
    rp_hypers = hypers.copy()
    rp_hypers['l2_lambda_0'] = l2_lambda_0
    if 'dop_l2_lambda_0' in hypers.keys():
        dop_drt_ratio = hypers['dop_l2_lambda_0'] / hypers['l2_lambda_0']
        rp_hypers['dop_l2_lambda_0'] = dop_drt_ratio * l2_lambda_0

    l2_matrix = calculate_qp_l2_matrix(rp_hypers, rho_vector, dop_rho_vector, penalty_matrices,
                                       s_vectors, penalty_type, special_qp_params)

    cvx_result = solve_convex_opt(rv, rm, l2_matrix, l1_lambda_0, nonneg, special_qp_params)
    x_rp = np.array(list(cvx_result['x']))

    return x_rp

# def initialize_md_weights(penalty_matrices, derivative_weights, rho_diagonals, s_vectors, rv, rm,
#                           vmm, l1_lambda_vector, nonneg, special_qp_params,
#                           iw_alpha, iw_beta, outlier_lambda):
#     # Calculate L2 penalty matrix (SMS)
#     l2_matrices = [rho_diagonals[k] @ penalty_matrices[f'm{k}'] @ rho_diagonals[k]
#                    for k in range(len(derivative_weights))]
#     sms = calculate_sms(derivative_weights, l2_matrices, s_vectors)
#     sms *= 1e-6  # Apply very small penalty strength for overfit
#
#     # Solve the ridge problem with QP: optimize x
#     # Multiply sms by 2 due to exponential prior
#     cvx_result = solve_convex_opt(rv, rm, 2 * sms, l1_lambda_vector, nonneg, special_qp_params)
#     x_overfit = np.array(list(cvx_result['x']))
#     # print(x_overfit)
#     # print(rm @ x_overfit)
#
#     # Get weight structure
#     # if outlier_lambda is not None:
#     #     # Iterate twice to ensure outlier variance subtracted from base variance
#     #     outlier_variance = None
#     #     for i in range(2):
#     #         est_weights, outlier_variance = estimate_weights(x_overfit, rv, vmm, rm, est_weights=None,
#     #                                                          outlier_lambda=outlier_lambda,
#     #                                                          outlier_variance=outlier_variance)
#     # else:
#     #     est_weights, outlier_variance = estimate_weights(x_overfit, rv, vmm, rm, est_weights=None,
#     #                                                      outlier_lambda=outlier_lambda)
#     est_weights, outlier_variance = estimate_weights(x_overfit, rv, vmm, rm, est_weights=None,
#                                                      outlier_lambda=outlier_lambda,
#                                                      outlier_variance=outlier_variance)
#
#     # Get global weight scale
#     # Need to average variance, not weights
#     # est_var_scale = np.mean(est_weights ** -2)
#     # est_weight_scale = est_var_scale ** -0.5
#     # est_weight_scale = np.mean(est_weights)
#
#     # # Solve for initial weight scale (global)
#     # # iw_beta = iw_0 ** 2 * (iw_alpha - 1.5)
#     # init_weight_scale = solve_init_weight_scale(est_weight_scale, iw_alpha, iw_beta)
#     #
#     # # Get initial weight vector
#     # init_weights = est_weights * init_weight_scale / est_weight_scale
#
#     # Local
#     init_weights = solve_init_weight_scale(est_weights, iw_alpha, iw_beta)
#
#     return est_weights, init_weights, x_overfit

#
# # EIS weight estimation
# # -------------------------
# def estimate_eis_weights(x, z, vmm, a_re, a_im, est_weights=None, error_structure=None):
#     resid_re = a_re @ x - z.real
#     resid_im = a_im @ x - z.imag
#
#     n = len(z)
#     # Concatenate residuals
#     resid = np.concatenate((resid_re, resid_im))
#     # Estimate error variance vectors from weighted mean of residuals
#     s_hat = vmm @ resid ** 2  # concatenated variance vector
#     # s_hat_re = s_hat[:n]
#     # s_hat_im = s_hat[n:]
#
#     # Convert variance to weights
#     w_hat = s_hat ** -0.5
#     # w_hat_re = s_hat_re ** -0.5
#     # w_hat_im = s_hat_im ** -0.5
#
#     if est_weights is not None:
#         # To ensure convergence (avoid poor initial fit resulting in weights going to zero), average current weights
#         # with initial estimate from overfitted ridge
#
#         # As current weights approach initial estimate (indicating better fit), give them more weight
#         scale_current = np.mean(s_hat) ** -0.5
#         scale_est = np.mean(est_weights ** -2) ** -0.5
#         # frac_current = 0.5
#         frac_current = scale_current / (scale_current + scale_est)
#         # print('frac_current', frac_current)
#         frac_est = 1 - frac_current
#
#         # Take mean of current and initial weight estimates
#         # w_hat_re = (frac_current * w_hat_re + frac_est * est_weights.real)
#         # w_hat_im = (frac_current * w_hat_im + frac_est * est_weights.imag)
#         w_hat = frac_current * w_hat + frac_est * est_weights
#
#     return w_hat
#
#
# def initialize_eis_weights(penalty_matrices, derivative_weights, p_vector, s_vectors, z_scaled,
#                            a_re, a_im, vmm, l1_lambda_vector, nonneg, error_structure,
#                            iw_alpha, iw_0):
#     # Calculate L2 penalty matrix (SMS)
#     l2_matrices = [penalty_matrices[f'm{n}'] for n in range(len(derivative_weights))]
#     sms = calculate_sms(np.array(derivative_weights) * p_vector, l2_matrices, s_vectors)
#     sms *= 1e-6  # Apply very small penalty strength for overfit
#
#     # Solve the ridge problem with QP: optimize x
#     # Multiply sms by 2 due to exponential prior
#     cvx_result = solve_convex_opt(z_scaled.real, z_scaled.imag, a_re, a_im, 2 * sms, l1_lambda_vector, nonneg)
#     x_overfit = np.array(list(cvx_result['x']))
#
#     # Get weight structure
#     est_weights = estimate_weights(x_overfit, z_scaled, vmm, a_re, a_im, est_weights=None,
#                                    error_structure=error_structure)
#
#     # Get global weight scale
#     w_scale_re = np.mean(est_weights.real)
#     w_scale_im = np.mean(est_weights.imag)
#     est_weight_scale = w_scale_re + 1j * w_scale_im
#
#     # s_scale_re = np.mean(est_weights.real) ** -2
#     # s_scale_im = np.mean(est_weights.imag) ** -2
#
#     # Solve for initial weight scale
#     iw_beta = iw_0 ** 2 * (iw_alpha - 1.5)
#     init_weight_scale_re = solve_init_weight_scale(w_scale_re, iw_alpha, iw_beta)
#     init_weight_scale_im = solve_init_weight_scale(w_scale_im, iw_alpha, iw_beta)
#
#     # Get initial weight vector (or scalar)
#     init_weights = est_weights.real * init_weight_scale_re / w_scale_re \
#                    + 1j * est_weights.imag * init_weight_scale_im / w_scale_im
#
#     return est_weights, init_weights
