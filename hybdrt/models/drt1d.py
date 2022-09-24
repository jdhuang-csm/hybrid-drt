import inspect
import itertools
import time
import warnings
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal

from hybdrt import utils, preprocessing as pp
from ..utils import stats
from hybdrt.matrices import mat1d, basis
from . import qphb, peaks, elements, pfrt
from ..evaluation import get_similarity_function
from .drtbase import DRTBase, format_chrono_weights, format_eis_weights
from hybdrt.plotting import get_transformed_plot_time, add_linear_time_axis, plot_eis, plot_distribution


class DRT(DRTBase):
    def __init__(self, **init_kw):
        super().__init__(**init_kw)

        self.candidate_dict = None
        self.candidate_df = None
        self.best_candidate_dict = None
        self.best_candidate_df = None
        self.discrete_candidate_dict = None
        self.discrete_candidate_df = None
        self.discrete_reordered_candidates = None
        self.discrete_model_kwargs = None

        self.pfrt_result = None

    @property
    def num_data(self):
        if self.fit_type.find('hybrid') >= 0:
            num_data = len(self.get_fit_times()) + 2 * len(self.get_fit_frequencies())
        elif self.fit_type.find('eis') >= 0:
            num_data = 2 * len(self.get_fit_frequencies())
        else:
            num_data = len(self.get_fit_times())
        return num_data

    @property
    def num_independent_data(self):
        if self.fit_type.find('hybrid') >= 0:
            num_data = len(self.get_fit_times()) + len(self.get_fit_frequencies())
        elif self.fit_type.find('eis') >= 0:
            num_data = len(self.get_fit_frequencies())
        else:
            num_data = len(self.get_fit_times())
        return num_data

    # # Ridge fit
    # # ----------------------------
    # def ridge_fit(self, times, i_signal, v_signal, step_times=None, nonneg=True, scale_signal=True,
    #               offset_baseline=True,
    #               offset_steps=False,
    #               downsample=True, downsample_kw=None, smooth_inf_response=True,
    #               # basic fit control
    #               l2_lambda_0=2, l1_lambda_0=0.1, weights=None, R_inf_scale=100, inductance_scale=1e-4,
    #               penalty_type='integral', derivative_weights=[0, 0, 1],
    #               # hyper-lambda options
    #               hyper_l2_lambda=True, hl_l2_beta=100,
    #               hyper_l1_lambda=False, hl_l1_beta=2.5,
    #               # optimization control
    #               xtol=1e-3, max_iter=20):
    #
    #     # Define special parameters included in quadratic programming parameter vector
    #     self.special_qp_params = {
    #         'v_baseline': {'index': 0, 'nonneg': False},
    #         'R_inf': {'index': 1, 'nonneg': True}
    #     }
    #
    #     if self.fit_inductance:
    #         self.special_qp_params['inductance'] = {'index': len(self.special_qp_params), 'nonneg': True}
    #
    #     # Process data and calculate matrices for fit
    #     sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, frequencies=None, z=None,
    #                                                step_times=step_times, downsample=downsample,
    #                                                downsample_kw=downsample_kw, offset_steps=offset_steps,
    #                                                smooth_inf_response=smooth_inf_response, scale_data=scale_signal,
    #                                                rp_scale=7, penalty_type=penalty_type,
    #                                                derivative_weights=derivative_weights)
    #     sample_times, sample_i, sample_v, response_baseline, _ = sample_data
    #     rm_drt, induc_rv, inf_rv, _, _, drt_penalty_matrices = matrices  # partial matrices
    #
    #     # if penalty_type == 'discrete':
    #     #	  # Convert differentiation matrices to L2 norm matrices
    #     #	  m0_p = m0_p @ m0_p
    #     #	  m1_p = m1_p @ m1_p
    #     #	  m2_p = m2_p @ m2_p
    #
    #     # print('m2_p sum:', np.sum(m2_p))
    #
    #     # Offset voltage baseline
    #     if offset_baseline:
    #         response_offset = -response_baseline
    #     else:
    #         response_offset = 0
    #     scaled_response_signal = self.scaled_response_signal + response_offset
    #
    #     # Add columns to rm for v_baseline, R_inf, and inductance
    #     rm = np.empty((rm_drt.shape[0], rm_drt.shape[1] + 3))
    #     rm[:, 0] = 1  # v_baseline
    #     if self.fit_inductance:
    #         rm[:, 1] = induc_rv * inductance_scale  # inductance response
    #     else:
    #         rm[:, 1] = 0
    #     rm[:, 2] = inf_rv  # R_inf response. only works for galvanostatic mode
    #     rm[:, 3:] = rm_drt  # DRT response
    #
    #     # print('rm:', rm[0])
    #     # print('rank(rm):', np.linalg.matrix_rank(rm))
    #
    #     # Construct L2 penalty matrices
    #     penalty_matrices = {}
    #     if penalty_type == 'integral':
    #         for n in range(0, 3):
    #             # Get penalty matrix for DRT coefficients
    #             m_drt = drt_penalty_matrices[f'm{n}']
    #
    #             # Add rows/columns for v_baseline, inductance, and R_inf
    #             m = np.zeros((m_drt.shape[0] + 3, m_drt.shape[1] + 3))
    #
    #             # No penalty applied to v_baseline (m[0, 0] = 0)
    #             # Insert penalties for R_inf and inductance
    #             m[1, 1] = 1  # inductance
    #             m[2, 2] = 1 / R_inf_scale
    #             m[3:, 3:] = m_drt  # DRT penalty matrix
    #
    #             penalty_matrices[f'm{n}'] = m
    #     elif penalty_type == 'discrete':
    #         for n in range(0, 3):
    #             # Get penalty matrix for DRT coefficients
    #             l_drt = drt_penalty_matrices[f'l{n}']
    #
    #             # Add rows/columns for v_baseline, inductance, and R_inf
    #             l = np.zeros((l_drt.shape[0] + 3, l_drt.shape[1] + 3))
    #
    #             # No penalty applied to v_baseline (m[0, 0] = 0)
    #             # Insert penalties for R_inf and inductance
    #             l[1, 1] = 1  # inductance
    #             l[2, 2] = (1 / R_inf_scale) ** 0.5
    #             l[3:, 3:] = l_drt  # DRT penalty matrix
    #             penalty_matrices[f'l{n}'] = l
    #
    #             # Calculate norm matrix
    #             penalty_matrices[f'm{n}'] = l.T @ l
    #
    #     # Indicate indices of special variables - always unbounded (v_baseline) or always non-neg (inductance, R_inf)
    #     special_indices = {'nonneg': np.array([1, 2], dtype=int), 'unbnd': np.array([0], dtype=int)}
    #
    #     # l2_matrices = []
    #     # for m_p in [m0_p, m1_p, m2_p]:
    #     #     m = np.zeros((m_p.shape[0] + 3, m_p.shape[1] + 3))
    #     #     # No penalty applied to v_baseline (m[0, 0] = 0)
    #     #     # Insert penalties for R_inf and inductance
    #     #     m[1, 1] = 1
    #     #     m[2, 2] = 1 / R_inf_scale
    #     #     m[3:, 3:] = m_p  # DRT penalty matrix
    #     #     l2_matrices.append(m)
    #
    #     # Construct lambda vectors
    #     l1_lambda_vector = np.zeros(rm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
    #     l1_lambda_vector[3:] = l1_lambda_0  # Remaining entries set to l1_lambda_0 - for DRT coefficients
    #     l2_lv = np.ones(rm.shape[1])  # First 3 entries are one - for v_baseline, R_inf, and inductance
    #     l2_lv[3:] = l2_lambda_0  # Remaining entries set to l2_lambda_0 - for DRT coefficients
    #     l2_lambda_vectors = [l2_lv.copy()] * 3  # need one vector for each order of the derivative
    #
    #     # print('lml:', lml)
    #
    #     # Get weight vector
    #     wv = format_chrono_weights(scaled_response_signal, weights)
    #     # print(wv)
    #     wm = np.diag(wv)
    #     # print('wm:', wm)
    #     # Apply weights to rm and signal
    #     wrm = wm @ rm
    #     wrv = wm @ scaled_response_signal
    #     # print('wrv:', wrv)
    #
    #     if hyper_l1_lambda or hyper_l2_lambda:
    #         self.ridge_iter_history = []
    #         it = 0
    #
    #         x = np.zeros(rm.shape[1]) + 1e-6
    #
    #         # dv_dlnt = np.ones(rm.shape[1])
    #
    #         # P = wrm.T @ wrm + lml
    #         # q = (-wrm.T @ wrv + l1_lambda_vector)
    #         # cost = 0.5 * x.T @ P @ x + q.T @ x
    #
    #         while it < max_iter:
    #
    #             x_prev = x.copy()
    #
    #             x, l2_lambda_vectors, cvx_result, converged = self._iterate_hyper_ridge(x_prev, 3, 1, nonneg,
    #                                                                                     penalty_type, hyper_l1_lambda,
    #                                                                                     hyper_l2_lambda, wrv, wrm,
    #                                                                                     l1_lambda_vector,
    #                                                                                     penalty_matrices,
    #                                                                                     l2_lambda_vectors,
    #                                                                                     derivative_weights, hl_l1_beta,
    #                                                                                     l1_lambda_0, hl_l2_beta,
    #                                                                                     l2_lambda_0, xtol)
    #
    #             if converged:
    #                 break
    #             elif it == max_iter - 1:
    #                 warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')
    #
    #             it += 1
    #
    #     else:
    #         # Ordinary ridge fit
    #         # # Make lml matrix for each derivative order
    #         # lml = np.zeros_like(l2_matrices[0])
    #         # for n, d_weight in enumerate(derivative_weights):
    #         #	  l2_lv = l2_lambda_vectors[n]
    #         #	  lm = np.diag(l2_lv ** 0.5)
    #         #	  m = l2_matrices[n]
    #         #	  lml += d_weight * lm @ m @ lm
    #
    #         # Make lml matrix for each derivative order
    #         l2_matrices = [penalty_matrices[f'm{n}'] for n in range(0, 3)]
    #         lml = qphb.calculate_qp_l2_matrix(derivative_weights, rho_vector, l2_matrices, l2_lambda_vectors,
    #                                           l2_lambda_0, penalty_type)
    #
    #         cvx_result = qphb.solve_convex_opt(wrv, wrm, lml, l1_lambda_vector, nonneg, special_indices)
    #
    #     # Store final cvxopt result
    #     self.cvx_result = cvx_result
    #
    #     # Extract model parameters
    #     x_out = np.array(list(cvx_result['x']))
    #     self.fit_parameters = {'x': x_out[3:] * self.coefficient_scale,
    #                            'R_inf': x_out[2] * self.coefficient_scale,
    #                            'v_baseline': (x_out[0] - response_offset) * self.response_signal_scale,
    #                            'v_sigma_tot': None,
    #                            'v_sigma_res': None}
    #
    #     if self.fit_inductance:
    #         self.fit_parameters['inductance'] = x_out[1] * self.coefficient_scale * inductance_scale
    #     else:
    #         self.fit_parameters['inductance'] = 0
    #
    #     self.fit_type = 'ridge'

    def _add_special_qp_param(self, name, nonneg):
        options = ['R_inf', 'v_baseline', 'inductance', 'vz_offset']
        if name not in options:
            raise ValueError('Invalid special QP parameter {name}. Options: {options}')

        self.special_qp_params[name] = {'index': len(self.special_qp_params), 'nonneg': nonneg}

    def _qphb_fit_core(self, times, i_signal, v_signal, frequencies, z, step_times=None,
                       nonneg=True, scale_data=True, update_scale=False,
                       # chrono args
                       offset_steps=True, offset_baseline=True, downsample=False, downsample_kw=None,
                       smooth_inf_response=True,
                       # penalty settings
                       v_baseline_penalty=1e-6, R_inf_penalty=1e-6, inductance_penalty=1e-6, inductance_scale=1e-5,
                       penalty_type='integral',
                       # error structure
                       chrono_error_structure='uniform', eis_error_structure=None,
                       chrono_vmm_epsilon=0.1, eis_vmm_epsilon=0.25, eis_reim_cor=0.25,
                       # Hybrid settings
                       vz_offset=False, vz_offset_scale=0.05,
                       eis_weight_factor=None, chrono_weight_factor=None,
                       # Prior hyperparameters
                       eff_hp=True, weight_factor=1,
                       # rp_scale=14, l2_lambda_0=None, l1_lambda_0=0.0, derivative_weights=None,
                       # iw_alpha=None, iw_beta=None,
                       # s_alpha=None, s_0=None,
                       # rho_alpha=None, rho_0=None,
                       # optimization control
                       xtol=1e-2, max_iter=50,
                       peak_locations=None,
                       **kw):

        # Check inputs
        utils.validation.check_chrono_data(times, i_signal, v_signal)
        utils.validation.check_eis_data(frequencies, z)
        for err_struct in [chrono_error_structure, eis_error_structure]:
            utils.validation.check_error_structure(err_struct)
        utils.validation.check_penalty_type(penalty_type)

        # Determine data type
        if times is None:
            data_type = 'eis'
        elif frequencies is None:
            data_type = 'chrono'
        else:
            data_type = 'hybrid'

        # Define special parameters included in quadratic programming parameter vector
        self.special_qp_params = {}

        self._add_special_qp_param('R_inf', True)

        if times is not None:
            self._add_special_qp_param('v_baseline', False)

        if self.fit_inductance:
            self._add_special_qp_param('inductance', True)

        if vz_offset and data_type == 'hybrid':
            self._add_special_qp_param('vz_offset', False)

        # Get preprocessing hyperparameters. Won't know data factor until chrono data has been downsampled
        pp_hypers = qphb.get_default_hypers(1, eff_hp)
        pp_hypers.update(kw)

        # Process data and calculate matrices for fit
        sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, frequencies, z,
                                                   step_times=step_times, downsample=downsample,
                                                   downsample_kw=downsample_kw, offset_steps=offset_steps,
                                                   smooth_inf_response=smooth_inf_response,
                                                   scale_data=scale_data, rp_scale=pp_hypers['rp_scale'],
                                                   penalty_type=penalty_type,
                                                   derivative_weights=pp_hypers['derivative_weights'])

        sample_times, sample_i, sample_v, response_baseline, z_scaled = sample_data
        rm_drt, induc_rv, inf_rv, zm_drt, induc_zv, drt_penalty_matrices = matrices  # partial matrices

        # Get data factor
        data_factor = qphb.get_data_factor_from_data(times, self.step_times, frequencies)
        if self.print_diagnostics:
            print('data factor:', data_factor)

        # Get default hyperparameters and update with user-specified values
        qphb_hypers = qphb.get_default_hypers(data_factor, eff_hp)
        qphb_hypers.update(kw)

        # Store fit kwargs for reference (after prep_for_fit creates self.fit_kwargs)
        self.fit_kwargs.update(qphb_hypers)
        self.fit_kwargs['nonneg'] = nonneg
        self.fit_kwargs['eff_hp'] = eff_hp
        self.fit_kwargs['penalty_type'] = penalty_type

        if self.print_diagnostics:
            print('lambda_0, iw_beta:', qphb_hypers['l2_lambda_0'], qphb_hypers['iw_beta'])

        # Format matrices for QP fit
        rm, zm, penalty_matrices = self._format_qp_matrices(rm_drt, inf_rv, induc_rv, zm_drt, induc_zv,
                                                            drt_penalty_matrices, v_baseline_penalty, R_inf_penalty,
                                                            inductance_penalty, vz_offset_scale, inductance_scale,
                                                            penalty_type, qphb_hypers['derivative_weights'])

        # Construct hybrid response-impedance matrix
        if rm is None:
            rzm = zm
        elif zm is None:
            rzm = rm
        else:
            rzm = np.vstack((rm, zm))

        # Make a copy for vz_offset calculation
        if data_type == 'hybrid' and vz_offset:
            rzm_vz = rzm.copy()
            # Remove v_baseline from rzm_vz - don't want to scale the baseline, only the delta
            rzm_vz[:, self.special_qp_params['v_baseline']['index']] = 0
        else:
            rzm_vz = None

        # Offset voltage baseline
        if times is not None:
            if offset_baseline:
                self.scaled_response_offset = -response_baseline
            else:
                self.scaled_response_offset = 0
            # print('scaled_response_offset:', self.scaled_response_offset * self.response_signal_scale)
            rv = self.scaled_response_signal + self.scaled_response_offset
        else:
            rv = None

        # Flatten impedance vector
        if frequencies is not None:
            zv = np.concatenate([z_scaled.real, z_scaled.imag])
        else:
            zv = None

        # Construct hybrid response-impedance vector
        if times is None:
            rzv = zv
        elif frequencies is None:
            rzv = rv
        else:
            rzv = np.concatenate([rv, zv])

        # Construct lambda vectors
        l1_lambda_vector = np.zeros(rzm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        l1_lambda_vector[self.get_qp_mat_offset():] = qphb_hypers['l1_lambda_0']

        # Initialize rho and s vectors at prior mode
        k_range = len(qphb_hypers['derivative_weights'])
        rho_vector = qphb_hypers['rho_0'].copy()
        s_vectors = [np.ones(rzm.shape[1]) * qphb_hypers['s_0'][k] for k in range(k_range)]

        # Initialize x near zero
        x = np.zeros(rzm.shape[1]) + 1e-6

        # Construct matrices for variance estimation
        if times is not None:
            chrono_vmm = mat1d.construct_chrono_var_matrix(sample_times, self.step_times,
                                                           chrono_vmm_epsilon,
                                                           chrono_error_structure)
        else:
            chrono_vmm = None

        if frequencies is not None:
            eis_vmm = mat1d.construct_eis_var_matrix(frequencies, eis_vmm_epsilon, eis_reim_cor,
                                                     eis_error_structure)
        else:
            eis_vmm = None

        # Stack variance matrices
        if chrono_vmm is None:
            vmm = eis_vmm
        elif eis_vmm is None:
            vmm = chrono_vmm
        else:
            vmm = np.zeros((len(rzv), len(rzv)))
            vmm[:len(sample_times), :len(sample_times)] = chrono_vmm
            vmm[len(sample_times):, len(sample_times):] = eis_vmm

        # Initialize data weight (IMPORTANT)
        # ----------------------------------
        # Initialize chrono and eis weights separately
        if times is not None:
            chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = \
                qphb.initialize_weights(penalty_matrices, penalty_type, qphb_hypers['derivative_weights'], rho_vector,
                                        s_vectors, rv, rm, chrono_vmm, nonneg, self.special_qp_params,
                                        qphb_hypers.get('chrono_iw_alpha', qphb_hypers['iw_alpha']),
                                        qphb_hypers.get('chrono_iw_beta', qphb_hypers['iw_beta']),
                                        qphb_hypers['outlier_p'])

            chrono_weight_scale = np.mean(chrono_est_weights ** -2) ** -0.5
        else:
            chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = None, None, None, None
            chrono_weight_scale = None

        if frequencies is not None:
            eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = \
                qphb.initialize_weights(penalty_matrices, penalty_type, qphb_hypers['derivative_weights'], rho_vector,
                                        s_vectors, zv, zm, eis_vmm, nonneg, self.special_qp_params,
                                        qphb_hypers.get('eis_iw_alpha', qphb_hypers['iw_alpha']),
                                        qphb_hypers.get('eis_iw_beta', qphb_hypers['iw_beta']),
                                        qphb_hypers['outlier_p'])

            eis_weight_scale = np.mean(eis_est_weights ** -2) ** -0.5
        else:
            eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = None, None, None, None
            eis_weight_scale = None

        # est_weights, init_weights, x_overfit = qphb.initialize_weights(penalty_matrices, derivative_weights,
        #                                                                rho_vector,
        #                                                                s_vectors, rzv, rzm, vmm,
        #                                                                l1_lambda_vector,
        #                                                                nonneg, self.special_qp_params,
        #                                                                eis_iw_alpha,
        #                                                                eis_iw_beta)
        #
        # chrono_est_weights = est_weights[:len(rv)]
        # eis_est_weights = est_weights[len(rv):]
        # chrono_init_weights = init_weights[:len(rv)]
        # eis_init_weights = init_weights[len(rv):]

        # eis_est_weights = eis_est_weights_raw * eis_weight_factor
        # eis_init_weights = eis_init_weights_raw * eis_weight_factor

        # Get weight factors
        if data_type == 'hybrid':
            if eis_weight_factor is None:
                eis_weight_factor = (chrono_weight_scale / eis_weight_scale) ** 0.25

            if chrono_weight_factor is None:
                chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.25

            if self.print_diagnostics:
                print('w_eis / w_ci:', eis_weight_scale / chrono_weight_scale)
                print('eis weight factor:', eis_weight_factor)
                print('chrono weight factor:', chrono_weight_factor)

            est_weights = np.concatenate([chrono_est_weights, eis_est_weights])
            init_weights = np.concatenate([chrono_init_weights, eis_init_weights])
            outlier_t = np.concatenate([chrono_outlier_t, eis_outlier_t])
        elif data_type == 'eis':
            est_weights = eis_est_weights
            init_weights = eis_init_weights
            outlier_t = eis_outlier_t
        else:
            est_weights = chrono_est_weights
            init_weights = chrono_init_weights
            outlier_t = chrono_outlier_t

        weights = init_weights

        if self.print_diagnostics:
            if chrono_init_weights is not None:
                print('Est chrono weight:', np.mean(chrono_est_weights))
                print('Initial chrono weight:', np.mean(chrono_init_weights))
            if eis_init_weights is not None:
                print('Est EIS weight:', np.mean(eis_est_weights))
                print('Initial EIS weight:', np.mean(eis_init_weights))
        # ------------------------------

        # Initialize xmx_norms at 1
        xmx_norms = np.ones(k_range)
        # xmx_norms = [10, 0.2, 0.1]

        # TEST: curvature constraints
        if peak_locations is not None:
            drt_curv_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), None, self.tau_basis_type,
                                                               self.tau_epsilon, 2, self.zga_params)
            curv_matrix = np.zeros((len(x) - self.get_qp_mat_offset(), len(x)))
            curv_matrix[:, self.get_qp_mat_offset():] = drt_curv_matrix
            peak_indices = np.array([utils.array.nearest_index(np.log(self.basis_tau), np.log(pl))
                                     for pl in peak_locations])
            curv_spread_func = get_similarity_function('gaussian')
        else:
            curv_matrix = None
            peak_indices = None
            curv_spread_func = None

        self.qphb_history = []
        it = 0
        # fixed_prior = False

        while it < max_iter:

            x_in = x.copy()

            # Apply chrono/eis weight adjustment factors
            if data_type == 'hybrid':
                weights[:len(rv)] *= chrono_weight_factor
                weights[len(rv):] *= eis_weight_factor

            # Apply overall weight scaling factor
            if it > 0:
                weights = weights * weight_factor

            # TEST: enforce curvature constraint
            if peak_locations is not None and it > 5:
                curv = curv_matrix @ x_in
                peak_curv = curv[peak_indices]
                curv_limit = [2.5 * pc * curv_spread_func(np.log(self.basis_tau / pl), 1.5, 2)
                              for pc, pl in zip(peak_curv, peak_locations)]
                curv_limit = np.sum(curv_limit, axis=0)
                # curv_limit = 0.5 * (curv_limit + curv)
                curvature_constraint = (-curv_matrix, -curv_limit)
            else:
                curvature_constraint = None

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and scale_data and update_scale:
                # Get scale factor
                rp = np.sum(x[self.get_qp_mat_offset():]) * np.pi ** 0.5 / self.tau_epsilon
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                if self.print_diagnostics:
                    print('Iter {} scale factor: {:.3f}'.format(it, scale_factor))
                # Update data and qphb parameters to reflect new scale
                x_in *= scale_factor
                x_overfit_chrono *= scale_factor
                x_overfit_eis *= scale_factor
                rzv *= scale_factor
                xmx_norms *= scale_factor  # shouldn't this be scale_factor ** 2?
                est_weights /= scale_factor
                init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            # Perform actual QPHB operation
            x, s_vectors, rho_vector, weights, outlier_t, cvx_result, converged = \
                qphb.iterate_qphb(x_in, s_vectors, rho_vector, rzv, weights, est_weights, outlier_t, rzm, vmm,
                                  penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers, eff_hp, xmx_norms,
                                  None, None, curvature_constraint, nonneg, self.special_qp_params, xtol, 1,
                                  self.qphb_history)

            # Normalize to ordinary ridge solution
            if it == 0:
                # Only include DRT penalty in XMX norms
                x_drt = x[len(self.special_qp_params):]
                xmx_norms = np.array([x_drt.T @ drt_penalty_matrices[f'm{k}'] @ x_drt for k in range(k_range)])
                if self.print_diagnostics:
                    print('xmx', xmx_norms)

            # Update vz_offset column
            if data_type == 'hybrid' and vz_offset:
                # Update the response matrix with the current predicted y vector
                # vz_offset offsets chrono and eis predictions
                y_hat = rzm_vz @ x
                vz_sep = y_hat.copy()
                vz_sep[len(rv):] *= -1  # vz_offset > 0 means EIS Rp smaller chrono Rp
                rzm[:, self.special_qp_params['vz_offset']['index']] = vz_sep

            if converged:
                break
            elif it == max_iter - 1:
                warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

            it += 1

        # Store the scaled weights for reference
        scaled_weights = weights.copy()
        if data_type == 'hybrid':
            # weights from iterate_qphb does not have weight factors applied
            scaled_weights[:len(rv)] *= chrono_weight_factor
            scaled_weights[len(rv):] *= eis_weight_factor

        # Store QPHB diagnostic parameters
        # TODO: should scaled_weights or weights be used here? weights seems to make sense...
        p_matrix, q_vector = qphb.calculate_pq(rzm, rzv, penalty_matrices, penalty_type, qphb_hypers,
                                               l1_lambda_vector, rho_vector, s_vectors, weights)

        # post_lp = qphb.evaluate_posterior_lp(x, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
        #                                      rho_vector, s_vectors, weights, rzm, rzv, xmx_norms)

        self.qphb_params = {'est_weights': est_weights.copy(),
                            'init_weights': init_weights.copy(),
                            'weights': scaled_weights.copy(),  # scaled weights
                            'true_weights': weights.copy(),  # unscaled weights
                            'data_factor': data_factor,
                            'chrono_weight_factor': chrono_weight_factor,
                            'eis_weight_factor': eis_weight_factor,
                            'xmx_norms': xmx_norms.copy(),
                            'x_overfit_chrono': x_overfit_chrono,
                            'x_overfit_eis': x_overfit_eis,
                            'p_matrix': p_matrix,
                            'q_vector': q_vector,
                            'rho_vector': rho_vector,
                            's_vectors': s_vectors,
                            'outlier_t': outlier_t,
                            'vmm': vmm,
                            'l1_lambda_vector': l1_lambda_vector,
                            # 'posterior_lp': post_lp,
                            'rm': rzm,
                            'rv': rzv,
                            'penalty_matrices': penalty_matrices,
                            }

        if self.print_diagnostics:
            print('rho:', rho_vector)

        # Store final cvxopt result
        self.cvx_result = cvx_result

        # Get sigma vector from (unscaled) weights
        sigma_vec = (weights ** -1)
        if data_type == 'hybrid':
            v_sigma = sigma_vec[:len(rv)] * self.response_signal_scale
            z_sigma = utils.eis.concat_vector_to_complex(sigma_vec[len(rv):]) * self.impedance_scale
        elif data_type == 'eis':
            z_sigma = utils.eis.concat_vector_to_complex(sigma_vec) * self.impedance_scale
            v_sigma = None
        else:
            v_sigma = sigma_vec * self.response_signal_scale
            z_sigma = None

        # Extract model parameters
        x_out = np.array(list(cvx_result['x']))
        self.fit_parameters = self.extract_qphb_parameters(x_out)
        self.fit_parameters['v_sigma_tot'] = v_sigma
        self.fit_parameters['v_sigma_res'] = None
        self.fit_parameters['z_sigma_tot'] = z_sigma

        self.fit_type = f'qphb_{data_type}'

    def fit_chrono(self, times, i_signal, v_signal, step_times=None,
                   nonneg=True, scale_data=True, update_scale=False,
                   offset_baseline=True, offset_steps=True,
                   downsample=False, downsample_kw=None, smooth_inf_response=True,
                   error_structure='uniform', vmm_epsilon=0.1,
                   **kwargs):

        self._qphb_fit_core(times, i_signal, v_signal, None, None, step_times=step_times,
                            nonneg=nonneg, scale_data=scale_data, update_scale=update_scale,
                            offset_steps=offset_steps, offset_baseline=offset_baseline,
                            downsample=downsample, downsample_kw=downsample_kw, smooth_inf_response=smooth_inf_response,
                            chrono_error_structure=error_structure, chrono_vmm_epsilon=vmm_epsilon, **kwargs
                            )

    def fit_eis(self, frequencies, z, nonneg=True, scale_data=True, update_scale=False,
                error_structure=None, vmm_epsilon=0.25, vmm_reim_cor=0.25, **kwargs):

        self._qphb_fit_core(None, None, None, frequencies, z, nonneg=nonneg,
                            scale_data=scale_data, update_scale=update_scale,
                            eis_error_structure=error_structure, eis_vmm_epsilon=vmm_epsilon, eis_reim_cor=vmm_reim_cor,
                            **kwargs)

    def fit_hybrid(self, times, i_signal, v_signal, frequencies, z, step_times=None,
                   nonneg=True, scale_data=True, update_scale=False,
                   # chrono parameters
                   offset_steps=True, offset_baseline=True,
                   downsample=False, downsample_kw=None, smooth_inf_response=True,
                   # vz offset
                   vz_offset=False, vz_offset_scale=0.05,
                   # Error structure
                   chrono_error_structure='uniform', eis_error_structure=None,
                   chrono_vmm_epsilon=0.1, eis_vmm_epsilon=0.25, eis_reim_cor=0.25,
                   eis_weight_factor=None, chrono_weight_factor=None, **kwargs):

        self._qphb_fit_core(times, i_signal, v_signal, frequencies, z, step_times=step_times,
                            nonneg=nonneg, scale_data=scale_data, update_scale=update_scale,
                            offset_steps=offset_steps, offset_baseline=offset_baseline,
                            downsample=downsample, downsample_kw=downsample_kw, smooth_inf_response=smooth_inf_response,
                            chrono_error_structure=chrono_error_structure, eis_error_structure=eis_error_structure,
                            chrono_vmm_epsilon=chrono_vmm_epsilon, eis_vmm_epsilon=eis_vmm_epsilon, eis_reim_cor=eis_reim_cor,
                            vz_offset=vz_offset, vz_offset_scale=vz_offset_scale,
                            eis_weight_factor=eis_weight_factor, chrono_weight_factor=chrono_weight_factor, **kwargs
                            )

    def qphb_fit_chrono(self, times, i_signal, v_signal, step_times=None, nonneg=True, scale_data=True,
                        offset_baseline=True, offset_steps=True, update_scale=False,
                        downsample=False, downsample_kw=None, smooth_inf_response=True,
                        # basic fit control
                        v_baseline_penalty=0, R_inf_penalty=0, inductance_penalty=0,
                        inductance_scale=1e-5,
                        penalty_type='integral', error_structure='uniform',
                        # Prior hyperparameters
                        vmm_epsilon=0.1, reduce_factor=1, info_factor=1, weight_factor=1,
                        eff_hp=True,
                        # l2_lambda_0=None, l1_lambda_0=0.0,
                        # rp_scale=14, derivative_weights=[1.5, 1.0, 0.5],
                        # iw_alpha=1.50, iw_beta=None,
                        # s_alpha=[1.5, 2.5, 25], s_0=1,
                        # rho_alpha=[1.1, 1.15, 1.2], rho_0=1,
                        # w_alpha=None, w_beta=None,
                        # optimization control
                        xtol=1e-2, max_iter=50, **kw):

        utils.validation.check_error_structure(error_structure)
        utils.validation.check_penalty_type(penalty_type)

        # # Format list/scalar arguments into arrays
        # derivative_weights = np.array(derivative_weights)
        # k_range = len(derivative_weights)
        # if np.shape(s_alpha) == ():
        #     s_alpha = [s_alpha] * k_range
        #
        # if np.shape(rho_alpha) == ():
        #     rho_alpha = [rho_alpha] * k_range
        #
        # s_alpha = np.array(s_alpha)
        # rho_alpha = np.array(rho_alpha)

        # Define special parameters included in quadratic programming parameter vector
        self.special_qp_params = {
            'v_baseline': {'index': 0, 'nonneg': False},
            'R_inf': {'index': 1, 'nonneg': True}
        }

        if self.fit_inductance:
            self.special_qp_params['inductance'] = {'index': len(self.special_qp_params), 'nonneg': True}

        # # If rp_scale is to be set automatically, pass a temporary value of 1 to prep_for_fit
        # # rp_scale can only be determined after downsampling
        # if rp_scale is None:
        #     rp_scale_tmp = 1
        # else:
        #     rp_scale_tmp = rp_scale

        # Get preprocessing hyperparameters - not dependent on data factor
        pp_hypers = qphb.get_default_hypers(1, eff_hp)
        pp_hypers.update(kw)

        # Process data and calculate matrices for fit
        sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, frequencies=None, z=None,
                                                   step_times=step_times, downsample=downsample,
                                                   downsample_kw=downsample_kw, offset_steps=offset_steps,
                                                   smooth_inf_response=smooth_inf_response, scale_data=scale_data,
                                                   rp_scale=pp_hypers['rp_scale'], penalty_type=penalty_type,
                                                   derivative_weights=pp_hypers['derivative_weights'])
        sample_times, sample_i, sample_v, response_baseline, _ = sample_data
        rm_drt, induc_rv, inf_rv, _, _, drt_penalty_matrices = matrices  # partial matrices

        # Store fit kwargs for reference (after prep_for_fit creates self.fit_kwargs)
        # self.fit_kwargs.update(
        #     {
        #         'nonneg': nonneg,
        #         'derivative_weights': derivative_weights,
        #         'rho_alpha': rho_alpha,
        #         'rho_0': rho_0,
        #         's_alpha': s_alpha,
        #         's_0': s_0,
        #         'w_alpha': w_alpha,
        #         'w_beta': w_beta
        #     }
        # )

        # Set lambda_0 and iw_0 based on sample size and density
        ppd = pp.get_time_ppd(sample_times, self.step_times)
        num_post_step = len(sample_times[sample_times >= self.step_times[0]])
        n_eff = num_post_step  # / 10
        ppd_eff = ppd * info_factor  # / 10
        print('ppd:', ppd)

        data_factor = qphb.get_data_factor(n_eff, ppd_eff)
        # data_factor = (np.sqrt(n_eff / 142) * (20 / ppd_eff))
        print('data_factor:', data_factor)

        # Get default hyperparameters and update with user-specified values
        qphb_hypers = qphb.get_default_hypers(data_factor, eff_hp)
        qphb_hypers.update(kw)

        self.fit_kwargs.update(qphb_hypers)
        self.fit_kwargs['nonneg'] = nonneg

        # if l2_lambda_0 is None:
        #     # lambda_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     # l2_lambda_0 = 142 * np.sqrt(len(frequencies) / 71) * (10 / ppd)
        #     l2_lambda_0 = 142 * (reduce_factor * data_factor) ** -1  # * np.sqrt(n_eff / 142) * (20 / ppd_eff)  #
        #
        # if iw_beta is None:
        #     # iw_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     # iw_beta = (4 ** 2 * 0.05) * (np.sqrt(len(frequencies) / 71) * (10 / ppd)) ** 2
        #     iw_beta = 0.5 * (data_factor * reduce_factor) ** 2  # * (np.sqrt(n_eff / 142) * (20 / ppd_eff)) ** 2 #
        print('lambda_0, iw_beta:', qphb_hypers['l2_lambda_0'], qphb_hypers['iw_beta'])

        #
        # if l2_lambda_0 is None:
        #     # lambda_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     l2_lambda_0 = 142 * np.sqrt(num_post_step / 201) * (34 / ppd)
        #
        # if iw_beta is None:
        #     # iw_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     iw_beta = 0.005 * (np.sqrt(num_post_step / 201) * (34 / ppd)) ** 2
        # print('lambda_0, iw_beta:', l2_lambda_0, iw_beta)

        # Offset voltage baseline
        if offset_baseline:
            self.scaled_response_offset = -response_baseline
        else:
            self.scaled_response_offset = 0
        rv = self.scaled_response_signal + self.scaled_response_offset

        # Update rp_scale
        # if rp_scale is None:
        #     # rp_scale = 14 * data_factor ** -1
        #     num_decades = pp.get_num_decades(None, sample_times, self.step_times)
        #     print('num decades:', num_decades)
        #     rp_scale = 14 * (num_decades / 7) ** 0.5
        #     rv *= rp_scale
        #     self.scaled_response_offset *= rp_scale
        #     self.update_data_scale(rp_scale)
        # print('rp_scale:', rp_scale)

        # Format matrices for QP fit
        rm, _, penalty_matrices = self._format_qp_matrices(rm_drt, inf_rv, induc_rv, None, None, drt_penalty_matrices,
                                                           v_baseline_penalty, R_inf_penalty, inductance_penalty, None,
                                                           inductance_scale, penalty_type,
                                                           qphb_hypers['derivative_weights'])

        # Construct l1 lambda vectors
        l1_lambda_vector = np.zeros(rm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        l1_lambda_vector[self.get_qp_mat_offset():] = qphb_hypers['l1_lambda_0']

        # Initialize s and rho vectors at prior mode
        k_range = len(qphb_hypers['derivative_weights'])
        rho_vector = np.ones(k_range) * qphb_hypers['rho_0']
        s_vectors = [np.ones(rm.shape[1]) * qphb_hypers['s_0']] * k_range

        # Initialize x near zero
        x = np.zeros(rm.shape[1]) + 1e-6

        # Construct matrix for variance estimation
        vmm = mat1d.construct_chrono_var_matrix(sample_times, self.step_times, vmm_epsilon, error_structure)
        # print(vmm[0])

        # Initialize data weight (IMPORTANT)
        # ----------------------------------
        est_weights, init_weights, x_overfit, outlier_t = qphb.initialize_weights(penalty_matrices, penalty_type,
                                                                                  qphb_hypers['derivative_weights'],
                                                                                  rho_vector, s_vectors, rv, rm, vmm,
                                                                                  nonneg, self.special_qp_params,
                                                                                  qphb_hypers['iw_alpha'],
                                                                                  qphb_hypers['iw_beta'],
                                                                                  qphb_hypers['outlier_p'])

        weights = init_weights
        # print(est_weights[0], weights[0])

        print('Initial weight:', np.mean(weights))
        # print('Initial Rp:', np.sum(x_overfit[3:]) * (np.pi ** 0.5 / self.tau_epsilon) * self.coefficient_scale)
        # print('Initial R_inf:', x_overfit[2] * self.coefficient_scale)

        # Initialize xmx_norms at 1
        xmx_norms = [1] * k_range

        self.qphb_history = []
        it = 0

        # est_weights *= weight_factor

        # print(est_weights[0], weights[0])

        while it < max_iter:

            x_in = x.copy()

            weights *= weight_factor

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and scale_data and update_scale:
                # Get scale factor
                rp = np.sum(x[len(self.special_qp_params):]) * np.pi ** 0.5 / self.tau_epsilon
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                # print('scale factor:', scale_factor)
                # Update data and qphb parameters to reflect new scale
                # self.scaled_response_offset *= scale_factor
                x_in *= scale_factor
                x_overfit *= scale_factor
                rv *= scale_factor
                xmx_norms *= scale_factor
                est_weights /= scale_factor
                init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            # print(it, weights[0], weight_factor)
            x, s_vectors, rho_vector, weights, outlier_t, cvx_result, converged = qphb.iterate_qphb(x_in, s_vectors,
                                                                                                    rho_vector, rv,
                                                                                                    weights,
                                                                                                    est_weights,
                                                                                                    outlier_t, rm, vmm,
                                                                                                    penalty_matrices,
                                                                                                    penalty_type,
                                                                                                    l1_lambda_vector,
                                                                                                    qphb_hypers, eff_hp,
                                                                                                    xmx_norms, None,
                                                                                                    None, None, nonneg,
                                                                                                    self.special_qp_params,
                                                                                                    xtol, 1,
                                                                                                    self.qphb_history)

            # Normalize to ordinary ridge solution
            if it == 0:
                # Only include DRT penalty in XMX norms
                x_drt = x[len(self.special_qp_params):]
                xmx_norms = np.array([x_drt.T @ drt_penalty_matrices[f'm{n}'] @ x_drt for n in range(k_range)])
                print('xmx', xmx_norms)
                # self.xmx_norms = xmx_norms

            if converged:
                break
            elif it == max_iter - 1:
                warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

            it += 1

        # Store scaled weights
        scaled_weights = weights * weight_factor
        cov_weights = weights / (weight_factor ** 4)

        # Store QPHB diagnostic parameters
        p_matrix, q_vector = qphb.calculate_pq(rm, rv, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                               rho_vector, s_vectors, scaled_weights)

        p_matrix_cov, _ = qphb.calculate_pq(rm, rv, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                            rho_vector, s_vectors, cov_weights)

        post_lp = qphb.evaluate_posterior_lp(x, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                             rho_vector, s_vectors, weights, rm, rv, xmx_norms)

        self.qphb_params = {'est_weights': est_weights.copy(),
                            'init_weights': init_weights.copy(),
                            'weights': scaled_weights.copy(),
                            'true_weights': weights.copy(),
                            'xmx_norms': xmx_norms.copy(),
                            'x_overfit': x_overfit,
                            'p_matrix': p_matrix,
                            'p_matrix_cov': p_matrix_cov,
                            'q_vector': q_vector,
                            'rho_vector': rho_vector,
                            's_vectors': s_vectors,
                            'outlier_t': outlier_t,
                            'vmm': vmm,
                            'l1_lambda_vector': l1_lambda_vector,
                            'posterior_lp': post_lp,
                            'rm': rm,
                            'rv': rv,
                            'penalty_matrices': penalty_matrices,
                            }

        # Store final cvxopt result
        self.cvx_result = cvx_result

        # Get sigma vector from weights
        # w_vec = format_chrono_weights(rv, weights)
        sigma_vec = (weights ** -1) * self.response_signal_scale

        # Extract model parameters
        x_out = np.array(list(cvx_result['x']))
        self.fit_parameters = self.extract_qphb_parameters(x_out)
        self.fit_parameters['v_sigma_tot'] = sigma_vec
        self.fit_parameters['v_sigma_res'] = None
        self.fit_parameters['z_sigma_tot'] = None
        # special_indices = {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}
        # x_out = np.array(list(cvx_result['x']))
        # self.fit_parameters = {'x': x_out[len(self.special_qp_params):] * self.coefficient_scale,
        #                        'R_inf': x_out[special_indices['R_inf']] * self.coefficient_scale,
        #                        'v_baseline': (x_out[special_indices['v_baseline']] - scaled_response_offset) \
        #                                      * self.response_signal_scale,
        #                        'v_sigma_tot': sigma_vec,
        #                        'v_sigma_res': None}
        #
        # if self.fit_inductance:
        #     self.fit_parameters['inductance'] = x_out[special_indices['inductance']] \
        #                                         * self.coefficient_scale * inductance_scale
        # else:
        #     self.fit_parameters['inductance'] = 0

        self.fit_type = 'qphb_chrono'

    def qphb_fit_eis(self, frequencies, z, nonneg=True, scale_data=True, update_scale=False,
                     # basic fit control
                     R_inf_penalty=1e-6, inductance_penalty=1e-6,
                     inductance_scale=1e-5,
                     penalty_type='integral', error_structure=None,
                     # Prior hyperparameters
                     vmm_epsilon=0.25, vmm_reim_cor=0.25, eff_hp=True,
                     weight_factor=1,
                     # rp_scale=14, l2_lambda_0=None, l1_lambda_0=0.0, derivative_weights=None,
                     # iw_alpha=None, iw_beta=None,
                     # s_alpha=None, s_0=None,
                     # rho_alpha=None, rho_0=None,
                     # optimization control
                     xtol=1e-2, max_iter=50,
                     peak_locations=None,
                     adjust_factor=1, fixed_s_vectors=None,  # remove these args
                     **kw):

        utils.validation.check_eis_data(frequencies, z)
        utils.validation.check_error_structure(error_structure)
        utils.validation.check_penalty_type(penalty_type)

        ppd = pp.get_ppd(frequencies)
        ppd_eff = np.sqrt(2) * ppd
        n_eff = np.sqrt(2) * len(frequencies)

        data_factor = qphb.get_data_factor(n_eff, ppd_eff)
        if self.print_diagnostics:
            print('data factor:', data_factor)

        # Get default hyperparameters and update with user-specified values
        qphb_hypers = qphb.get_default_hypers(data_factor, eff_hp)
        qphb_hypers.update(kw)

        # if rp_scale is None:
        #     # print(np.sqrt((n_eff / (71 * np.sqrt(2)))), np.sqrt(10 * np.sqrt(2) / ppd_eff))
        #     num_decades = pp.get_num_decades(frequencies, None, None)
        #     rp_scale = 14 * (num_decades / 7) ** 0.5
        #     # rp_scale = 14 * data_factor ** -1  # (np.sqrt(n_eff / (71 * np.sqrt(2))) * np.sqrt(10 * np.sqrt(2) / ppd_eff)) ** -1
        # print('rp_scale:', rp_scale)

        # if s_alpha is None:
        #     s_alpha = 1 + np.array([0.5, 1.5, 24]) * data_factor  # (np.sqrt((n_eff / 142)) * (20 / ppd_eff))
        # print('s_alpha:', s_alpha)

        # Format list/scalar arguments into arrays
        # derivative_weights = np.array(derivative_weights)
        # k_range = len(derivative_weights)
        # if np.shape(s_alpha) == ():
        #     s_alpha = [s_alpha] * k_range
        #
        # if np.shape(rho_alpha) == ():
        #     rho_alpha = [rho_alpha] * k_range
        #
        # s_alpha = np.array(s_alpha)
        # rho_alpha = np.array(rho_alpha)

        # Define special parameters included in quadratic programming parameter vector
        self.special_qp_params = {
            'R_inf': {'index': 0, 'nonneg': True}
        }

        if self.fit_inductance:
            self.special_qp_params['inductance'] = {'index': len(self.special_qp_params), 'nonneg': True}

        # Process data and calculate matrices for fit
        sample_data, matrices = self._prep_for_fit(None, None, None, frequencies, z, step_times=None, downsample=False,
                                                   downsample_kw=None, offset_steps=False, smooth_inf_response=False,
                                                   scale_data=scale_data, rp_scale=qphb_hypers['rp_scale'],
                                                   penalty_type=penalty_type,
                                                   derivative_weights=qphb_hypers['derivative_weights'])
        _, _, _, _, z_scaled = sample_data
        _, _, _, zm_drt, induc_zv, drt_penalty_matrices = matrices

        # Store fit kwargs for reference (after prep_for_fit creates self.fit_kwargs)
        self.fit_kwargs.update(qphb_hypers)
        self.fit_kwargs['nonneg'] = nonneg
        self.fit_kwargs['eff_hp'] = eff_hp
        self.fit_kwargs['penalty_type'] = penalty_type
        #     {
        #         'nonneg': nonneg,
        #         'derivative_weights': derivative_weights,
        #         'rho_alpha': rho_alpha,
        #         'rho_0': rho_0,
        #         's_alpha': s_alpha,
        #         's_0': s_0,
        #         # 'w_alpha': w_alpha,
        #         # 'w_beta': w_beta
        #     }
        # )

        # Set lambda_0 and iw_0 based on sample size and density
        # ppd = get_ppd(frequencies)
        # ppd_eff = 2 * ppd
        # n_eff = 2 * len(frequencies)

        # if l2_lambda_0 is None:
        #     # lambda_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     l2_lambda_0 = 142 * data_factor ** -1  # (np.sqrt(n_eff / 142) * (20 / ppd_eff))
        #     # l2_lambda_0 = 142 * (np.sqrt(n_eff / 142) * (20 / ppd_eff)) ** -2
        #
        # if iw_beta is None:
        #     # iw_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     iw_beta = 0.5 * data_factor ** 2  # (np.sqrt(n_eff / 142) * (20 / ppd_eff)  )  ** 2
        #     # iw_beta = 0.5 * (14 / rp_scale) ** 2
        if self.print_diagnostics:
            print('lambda_0, iw_beta:', qphb_hypers['l2_lambda_0'], qphb_hypers['iw_beta'])

        # Format matrices for QP fit
        _, zm, penalty_matrices = self._format_qp_matrices(None, None, None, zm_drt, induc_zv, drt_penalty_matrices, 0,
                                                           R_inf_penalty, inductance_penalty, None, inductance_scale,
                                                           penalty_type, qphb_hypers['derivative_weights'])

        # Concatenate real and imag impedance
        zv = np.concatenate([z_scaled.real, z_scaled.imag])

        # Construct l1 lambda vectors
        l1_lambda_vector = np.zeros(zm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        l1_lambda_vector[self.get_qp_mat_offset():] = qphb_hypers['l1_lambda_0']  # Remaining entries set to l1_lambda_0

        # Initialize rho and s vectors at prior mode
        k_range = len(qphb_hypers['derivative_weights'])
        rho_vector = qphb_hypers['rho_0'].copy()
        s_vectors = [np.ones(zm.shape[1]) * qphb_hypers['s_0'][k] for k in range(k_range)]

        # Initialize x near zero
        x = np.zeros(zm.shape[1]) + 1e-6

        # Construct matrix for variance estimation
        vmm = mat1d.construct_eis_var_matrix(frequencies, vmm_epsilon, vmm_reim_cor, error_structure)

        # Initialize data weight (IMPORTANT)
        # ----------------------------------
        est_weights, init_weights, x_overfit, outlier_t = qphb.initialize_weights(penalty_matrices, penalty_type,
                                                                                  qphb_hypers['derivative_weights'],
                                                                                  np.ones(k_range),
                                                                                  [np.ones(zm.shape[1])] * k_range,
                                                                                  zv, zm, vmm,
                                                                                  nonneg, self.special_qp_params,
                                                                                  qphb_hypers['iw_alpha'],
                                                                                  qphb_hypers['iw_beta'],
                                                                                  qphb_hypers['outlier_p'])

        # init_weights *= 0.5
        weights = init_weights

        if self.print_diagnostics:
            print('Est weight:', np.mean(est_weights ** -2) ** -0.5)
            print('Initial weight:', np.mean(weights ** -2) ** -0.5)

        # Initialize xmx_norms at 1
        xmx_norms = np.ones(k_range)
        # xmx_norms = [10, 0.2, 0.1]

        # TEST: curvature constraints
        if peak_locations is not None:
            drt_curv_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), None, self.tau_basis_type,
                                                               self.tau_epsilon, 2, self.zga_params)
            curv_matrix = np.zeros((len(x) - self.get_qp_mat_offset(), len(x)))
            curv_matrix[:, self.get_qp_mat_offset():] = drt_curv_matrix
            peak_indices = np.array([utils.array.nearest_index(np.log(self.basis_tau), np.log(pl))
                                     for pl in peak_locations])
            curv_spread_func = get_similarity_function('gaussian')
        else:
            curv_matrix = None
            peak_indices = None
            curv_spread_func = None

        self.qphb_history = []
        it = 0
        # fixed_prior = False

        while it < max_iter:

            x_in = x.copy()

            if it > 0:
                weights = weights * weight_factor

            if peak_locations is not None and it > 5:
                curv = curv_matrix @ x_in
                peak_curv = curv[peak_indices]
                curv_limit = [2.5 * pc * curv_spread_func(np.log(self.basis_tau / pl), 1.5, 2)
                              for pc, pl in zip(peak_curv, peak_locations)]
                curv_limit = np.sum(curv_limit, axis=0)
                # curv_limit = 0.5 * (curv_limit + curv)
                curvature_constraint = (-curv_matrix, -curv_limit)
            else:
                curvature_constraint = None

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and scale_data and update_scale:
                # Get scale factor
                rp = np.sum(x[self.get_qp_mat_offset():]) * np.pi ** 0.5 / self.tau_epsilon
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                # print('rp, scale factor:', rp, scale_factor)
                # Update data and qphb parameters to reflect new scale
                x_in *= scale_factor
                x_overfit *= scale_factor
                z_scaled *= scale_factor
                zv *= scale_factor
                xmx_norms *= scale_factor  # shouldn't this be squared?
                est_weights /= scale_factor
                init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            # if it <= 5:
            #     est_weights_in = est_weights
            # else:
            #     est_weights_in = None

            x, s_vectors, rho_vector, weights, outlier_t, cvx_result, converged = \
                qphb.iterate_qphb(x_in, s_vectors, rho_vector, zv, weights, est_weights, outlier_t, zm, vmm,
                                  penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers, eff_hp, xmx_norms,
                                  None, None, curvature_constraint, nonneg, self.special_qp_params, xtol, 1,
                                  self.qphb_history)

            if fixed_s_vectors is not None:
                s_vectors = fixed_s_vectors.copy()

            # Normalize to ordinary ridge solution
            if it == 0:
                # Only include DRT penalty in XMX norms
                x_drt = x[len(self.special_qp_params):]
                xmx_norms = np.array([x_drt.T @ drt_penalty_matrices[f'm{k}'] @ x_drt for k in range(k_range)])
                if self.print_diagnostics:
                    print('xmx', xmx_norms)

            rho_vector[np.where(qphb_hypers['derivative_weights'] > 0)] **= adjust_factor

            if converged:
                break
            elif it == max_iter - 1:
                warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

            it += 1

        # Store QPHB diagnostic parameters
        # l2_matrices = [penalty_matrices[f'm{n}'] for n in range(3)]
        # sms = qphb.calculate_sms(np.array(derivative_weights) * rho_vector, l2_matrices, s_vectors)
        # sms *= l2_lambda_0
        #
        # wm = np.diag(weights)
        # wrm = wm @ zm
        # wrv = wm @ zv
        #
        # p_matrix = 2 * sms + wrm.T @ wrm
        # q_vector = -wrm.T @ wrv + l1_lambda_vector

        p_matrix, q_vector = qphb.calculate_pq(zm, zv, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                               rho_vector, s_vectors, weights)

        # post_lp = qphb.evaluate_posterior_lp(x, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
        #                                      rho_vector, s_vectors, weights, zm, zv, xmx_norms)

        self.qphb_params = {'est_weights': est_weights.copy(),
                            'init_weights': init_weights.copy(),
                            'weights': weights.copy(),
                            'data_factor': data_factor,
                            'xmx_norms': xmx_norms.copy(),
                            'x_overfit': x_overfit,
                            'p_matrix': p_matrix,
                            'q_vector': q_vector,
                            'rho_vector': rho_vector,
                            's_vectors': s_vectors,
                            'outlier_t': outlier_t,
                            'vmm': vmm,
                            'l1_lambda_vector': l1_lambda_vector,
                            # 'posterior_lp': post_lp,
                            'rm': zm,
                            'rv': zv,
                            'penalty_matrices': penalty_matrices,
                            'hypers': qphb_hypers
                            }

        if self.print_diagnostics:
            print('rho:', rho_vector)

        # Store final cvxopt result
        self.cvx_result = cvx_result

        # Get sigma vector from weights
        sigma_vec = (weights[:len(z)] ** -1 + 1j * weights[len(z):] ** -1) * self.impedance_scale

        # Extract model parameters
        x_out = np.array(list(cvx_result['x']))
        self.fit_parameters = self.extract_qphb_parameters(x_out)
        self.fit_parameters['v_sigma_tot'] = None
        self.fit_parameters['v_sigma_res'] = None
        self.fit_parameters['z_sigma_tot'] = sigma_vec

        self.fit_type = 'qphb_eis'

    def _continue_from_init(self, qphb_hypers, x_init, rv, rm, vmm, rho_vector, s_vectors, outlier_t,
                            penalty_matrices, xmx_norms,
                            est_weights, weights, l1_lambda_vector,
                            nonneg=True, update_scale=False, weight_factor=1,
                            penalty_type='integral', eff_hp=True, xtol=1e-2, max_iter=10, min_iter=2, **kw):

        # Update hyperparameters with user-specified values
        qphb_hypers = qphb_hypers.copy()
        qphb_hypers.update(kw)

        x = x_init.copy()
        continue_history = []
        it = 0

        while it < max_iter:

            x_in = x.copy()

            weights = weights * weight_factor

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and update_scale:
                # Get scale factor
                rp = np.sum(x[self.get_qp_mat_offset():]) * np.pi ** 0.5 / self.tau_epsilon
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                # print('rp, scale factor:', rp, scale_factor)
                # Update data and qphb parameters to reflect new scale
                x_in *= scale_factor
                # x_overfit *= scale_factor
                # z_scaled *= scale_factor
                rv *= scale_factor
                xmx_norms *= scale_factor  # shouldn't this be squared?
                est_weights /= scale_factor
                # init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            x, s_vectors, rho_vector, weights, outlier_t, cvx_result, converged = \
                qphb.iterate_qphb(x_in, s_vectors, rho_vector, rv, weights, est_weights, outlier_t, rm, vmm,
                                  penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers, eff_hp, xmx_norms,
                                  None, None, None, nonneg, self.special_qp_params, xtol, 1, continue_history)

            # # Normalize to ordinary ridge solution
            # if it == 0:
            #     # Only include DRT penalty in XMX norms
            #     x_drt = x[len(self.special_qp_params):]
            #     xmx_norms = np.array([x_drt.T @ drt_penalty_matrices[f'm{k}'] @ x_drt for k in range(k_range)])
            #     if self.print_diagnostics:
            #         print('xmx', xmx_norms)

            # rho_vector[np.where(qphb_hypers['derivative_weights'] > 0)] **= adjust_factor

            if converged and it >= min_iter - 1:
                break
            elif it == max_iter - 1:
                warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

            it += 1

        return continue_history

    # --------------------------------
    # Discrete-continuous framework
    # --------------------------------
    def _generate_candidates_s0(self, multiplier, steps, xtol, max_iter, **kw):
        """
        Generate candidate distributions by varying the regularization strength unidirectionally starting from the
        baseline fit obtained by a qphb_fit method. The local regularization strength mode (s_0) is multiplied by
        multiplier, while the global regularization strength (l2_lambda_0) is divided by the multiplier.
        Works best for finding additional peaks (s0_multiplier > 1).
        Less effective for suppressing peaks (s0_multiplier < 1). Use _generate_candidates_weight instead.
        :param float multiplier: Factor by which to multiply the local regularization strength mode (s_0) at each
        step. If s0_multiplier > 1, the generated candidates will be less regularized than the baseline solution.
        If s0_multiplier < 1, the generated candidates will be more regularized than the baseline solution.
        :param int steps: Number of steps to take in s_0 space. The final s_0 value will be s0_multiplier ** s0_steps
        times the baseline s_0 value listed in self.qphb_params
        :param float xtol: Relative parameter tolerance for convergence
        :param int max_iter: Maximum QPHB iterations to perform at each s_0 value
        :param kw: Keyword arguments to pass to _continue_from_init; these will update the qphb_hypers dict
        :return:
        """
        # When s0_multiplier < 1, better to start each run from last s_in
        # When s0_multiplier > 1, better to start each run from default s_in

        x_in = self.qphb_history[-1]['x'].copy()
        rho_in = self.qphb_params['rho_vector'].copy()
        s_in = self.qphb_params['s_vectors'].copy()
        weights_in = self.qphb_params['weights'].copy()

        history = []
        hypers = []

        # Starting from default solution, incrementally adjust s0 to promote/suppress peaks
        for i in range(1, steps + 1):
            s0_factor = multiplier ** i  # Cumulative factor

            if multiplier > 1:
                # s_in = [np.ones_like(s_vec) * s0_factor for s_vec in s_in]
                s_in = self.qphb_params['s_vectors'].copy()
                s_in = [s_vec * s0_factor for s_vec in s_in]  # Start from scaled default s_vectors
            else:
                s_in = [s_vec * multiplier for s_vec in s_in]  # Increment factor from last step

            new_hypers = {'s_0': self.fit_kwargs['s_0'] * s0_factor,
                          'l2_lambda_0': self.fit_kwargs['l2_lambda_0'] * (s0_factor ** -1)
                          }

            hist = self._continue_from_init(self.qphb_params['hypers'], x_in,
                                            self.qphb_params['rv'], self.qphb_params['rm'], self.qphb_params['vmm'],
                                            rho_in, s_in,  # self.qphb_params['s_vectors'],
                                            self.qphb_params['outlier_t'],
                                            self.qphb_params['penalty_matrices'], self.qphb_params['xmx_norms'],
                                            self.qphb_params['est_weights'], weights_in,
                                            self.qphb_params['l1_lambda_vector'],
                                            nonneg=self.fit_kwargs['nonneg'], update_scale=False,
                                            penalty_type=self.fit_kwargs['penalty_type'],
                                            eff_hp=self.fit_kwargs['eff_hp'], xtol=xtol, max_iter=max_iter,
                                            **new_hypers, **kw
                                            )
            x_in = hist[-1]['x'].copy()
            rho_in = hist[-1]['rho_vector'].copy()
            s_in = hist[-1]['s_vectors'].copy()
            weights_in = hist[-1]['weights'].copy()

            history += hist
            hypers += [new_hypers] * len(hist)

        candidate_x = [h['x'] for h in history]

        return candidate_x, history, hypers

    def _generate_candidates_weights(self, multiplier, steps, xtol, max_iter, **kw):
        """
        Generate candidate distributions by varying the regularization strength unidirectionally starting from the
        baseline fit obtained by a qphb_fit method. The local regularization strength mode (s_0) is multiplied by
        s0_multiplier, while the global regularization strength (l2_lambda_0) is divided by the multiplier.
        Works best for finding additional peaks (s0_multiplier > 1).
        Less effective for suppressing peaks (s0_multiplier < 1). Use _generate_candidates_weight instead.
        :param float s0_multiplier: Factor by which to multiply the local regularization strength mode (s_0) at each
        step. If s0_multiplier > 1, the generated candidates will be less regularized than the baseline solution.
        If s0_multiplier < 1, the generated candidates will be more regularized than the baseline solution.
        :param int s0_steps: Number of steps to take in s_0 space. The final s_0 value will be s0_multiplier ** s0_steps
        times the baseline s_0 value listed in self.qphb_params
        :param float xtol: Relative parameter tolerance for convergence
        :param int max_iter: Maximum QPHB iterations to perform at each s_0 value
        :param kw: Keyword arguments to pass to _continue_from_init; these will update the qphb_hypers dict
        :return:
        """
        # When s0_multiplier < 1, better to start each run from last s_in
        # When s0_multiplier > 1, better to start each run from default s_in

        x_in = self.qphb_history[-1]['x'].copy()
        rho_in = self.qphb_params['rho_vector'].copy()
        s_in = self.qphb_params['s_vectors'].copy()
        weights_in = self.qphb_params['weights'].copy()

        history = []
        hypers = []

        # Starting from default solution, incrementally adjust s0 to promote/suppress peaks
        for i in range(1, steps + 1):
            weight_factor = multiplier ** i  # Cumulative factor

            # if multiplier > 1:
            #     # Start from default s_vectors
            s_in = self.qphb_params['s_vectors'].copy()
            # weights_in = weights_in * multiplier

            new_hypers = {'weight_factor': weight_factor}

            hist = self._continue_from_init(self.qphb_params['hypers'], x_in,
                                            self.qphb_params['rv'], self.qphb_params['rm'], self.qphb_params['vmm'],
                                            rho_in, s_in,  # self.qphb_params['s_vectors'],
                                            self.qphb_params['outlier_t'],
                                            self.qphb_params['penalty_matrices'], self.qphb_params['xmx_norms'],
                                            self.qphb_params['est_weights'], weights_in,
                                            self.qphb_params['l1_lambda_vector'],
                                            nonneg=self.fit_kwargs['nonneg'], update_scale=False,
                                            penalty_type=self.fit_kwargs['penalty_type'],
                                            eff_hp=self.fit_kwargs['eff_hp'], xtol=xtol, max_iter=max_iter,
                                            **new_hypers, **kw
                                            )
            x_in = hist[-1]['x'].copy()
            rho_in = hist[-1]['rho_vector'].copy()
            s_in = hist[-1]['s_vectors'].copy()
            weights_in = hist[-1]['weights'].copy()

            history += hist
            hypers += [new_hypers] * len(hist)

        candidate_x = [h['x'] for h in history]

        return candidate_x, history, hypers

    def generate_candidates(self, s0_multiplier=4, s0_steps=2, weight_multiplier=0.5, weight_steps=3,
                            include_qphb_history=True,
                            xtol=1e-2, max_iter=10, llh_kw=None, find_peaks_kw=None,
                            **kw):
        """
        Generate candidate distributions by varying the regularization strength unidirectionally starting from the
        baseline fit obtained by a qphb_fit method.
        :param float s0_multiplier: Factor by which to multiply the local regularization strength mode (s_0) at each
        step. If s0_multiplier > 1, the generated candidates will be less regularized than the baseline solution.
        If s0_multiplier < 1, the generated candidates will be more regularized than the baseline solution.
        :param int s0_steps: Number of steps to take in s_0 space. The final s_0 value will be s0_multiplier ** s0_steps
        times the baseline s_0 value listed in self.qphb_params
        :param float xtol: Relative parameter tolerance for convergence
        :param int max_iter: Maximum QPHB iterations to perform at each s_0 value
        :param dict llh_kw: Keyword arguments to pass to self.evaluate_llh for evaluation of candidate solutions
        :param dict find_peaks_kw: Keyword arguments to pass to self.find_peaks for evaluation of candidate solutions
        :param kw: Keyword arguments to pass to _continue_from_init; these will update the qphb_hypers dict
        :return:
        """
        # Get candidates from default fit history
        if include_qphb_history:
            qphb_x = [h['x'] for h in self.qphb_history]
            qphb_history = self.qphb_history
        else:
            qphb_x = [self.qphb_history[-1]['x']]
            qphb_history = self.qphb_history[-1:]

        # Suppress/promote peaks by decreasing/increasing s_0
        down_x, down_history, down_hypers = self._generate_candidates_weights(weight_multiplier, weight_steps,
                                                                              xtol, max_iter, **kw)
        up_x, up_history, up_hypers = self._generate_candidates_s0(s0_multiplier, s0_steps, xtol, max_iter, **kw)

        # Get relevant hyperparameters for default solution
        hypers_keys = list(down_hypers[0].keys()) + list(up_hypers[0].keys())
        default_hypers = [{k: self.fit_kwargs.get(k, None) for k in hypers_keys}] * len(qphb_x)

        # Gather all candidates
        candidate_history = qphb_history + up_history + down_history
        candidate_hypers = default_hypers + up_hypers + down_hypers
        candidate_x = np.array(qphb_x + up_x + down_x)

        # Evaluate llh
        if llh_kw is None:
            llh_kw = {}
        cand_weights = [qphb.estimate_weights(x, self.qphb_params['rv'], self.qphb_params['vmm'],
                                              self.qphb_params['rm'])[0]
                        for x in candidate_x]
        candidate_llh = np.array([self.evaluate_llh(weights, x, **llh_kw)
                                  for x, weights in zip(candidate_x, cand_weights)])
        # candidate_llh = np.array([self.evaluate_llh(self.qphb_params['est_weights'], x, **llh_kw) for x in candidate_x])

        # Identify peaks in candidate distributions
        # TODO: consider models with same num_peaks but different peak locations. How can this be easily flagged?
        # TODO: Might need to cluster peaks
        # TODO: for each peak (cluster), consider frequency of appearance and/or llh distribution?
        if find_peaks_kw is None:
            find_peaks_kw = {}
        candidate_peak_tau = [self.find_peaks(x=self.extract_qphb_parameters(x)['x'], **find_peaks_kw)
                              for x in candidate_x]
        candidate_num_peaks = np.array([len(pt) for pt in candidate_peak_tau])

        # # Don't include candidates with no peaks (usually from 1st iteration with low weights)
        # include_index = np.where(candidate_num_peaks > 0)

        # Evaluate BIC
        num_special = self.get_qp_mat_offset()
        candidate_bic = np.array(
            [stats.bic(num_special + num_peaks * 4, self.num_independent_data, llh)
             for num_peaks, llh in zip(candidate_num_peaks, candidate_llh)]
        )

        # Store results
        self.candidate_dict = {
            'x': candidate_x,
            'peak_tau': candidate_peak_tau,
            'num_peaks': candidate_num_peaks,
            'llh': candidate_llh,
            'bic': candidate_bic,
            'history': candidate_history,
            'hypers': candidate_hypers
        }

        best_bic = np.min(candidate_bic)
        best_llh = np.max(candidate_llh)

        self.candidate_df = pd.DataFrame(
            np.vstack((candidate_num_peaks, candidate_llh, candidate_bic,
                       candidate_llh - best_llh, candidate_bic - best_bic)).T,
            columns=['num_peaks', 'llh', 'bic', 'rel_llh', 'rel_bic']
        )

        # Get best candidate for each num_peaks
        unique_num_peaks = np.unique(candidate_num_peaks)
        self.best_candidate_dict = {}
        best_indices = np.empty(len(unique_num_peaks), dtype=int)
        for i, num_peaks in enumerate(unique_num_peaks):
            model_index = np.where(candidate_num_peaks == num_peaks)
            llh_vals = candidate_llh[model_index]
            best_index = np.where((candidate_num_peaks == num_peaks) & (candidate_llh == np.max(llh_vals)))
            best_indices[i] = best_index[0][0]
            self.best_candidate_dict[num_peaks] = {
                'x': candidate_x[best_index][0],
                'llh': candidate_llh[best_index][0],
                'bic': candidate_bic[best_index][0],
                'peak_tau': candidate_peak_tau[best_index[0][0]],
                'history': candidate_history[best_index[0][0]],
                'hypers': candidate_hypers[best_index[0][0]]
            }

        self.best_candidate_df = pd.DataFrame(
            np.vstack((candidate_num_peaks[best_indices], candidate_num_peaks[best_indices],
                       candidate_llh[best_indices], candidate_bic[best_indices],
                       candidate_llh[best_indices] - best_llh, candidate_bic[best_indices] - best_bic)).T,
            columns=['model_id', 'num_peaks', 'llh', 'bic', 'rel_llh', 'rel_bic']
        )

        return self.candidate_dict.copy()

    def convert_candidate_to_discrete(self, candidate_num_peaks, model_init_kw=None,
                                      **fit_kw):
        start = time.time()

        candidate_info = self.get_candidate(candidate_num_peaks, 'continuous')
        peak_tau = candidate_info['peak_tau']
        candidate_x = candidate_info['x']

        # Convert peak positions to indices

        tau_eval = self.get_tau_eval(10)
        peak_indices = np.array([utils.array.nearest_index(tau_eval, pt) for pt in peak_tau])

        # Initialize discrete model from candidate DRT
        if model_init_kw is None:
            model_init_kw = {}
        dem = elements.DiscreteElementModel.from_drt(self, candidate_x, tau_eval, peak_indices, **model_init_kw)
        # dem.drt_estimates['eis_weights'] = utils.eis.complex_vector_to_concat(self.predict_sigma('eis')) ** -1

        if self.fit_type.find('eis') > -1:
            dem.fit_eis(self.get_fit_frequencies(), self.z_fit, from_drt=True, **fit_kw)
        else:
            pass

        if self.print_diagnostics:
            print('{}-peak candidate conversion time: {:.3f} s'.format(candidate_num_peaks, time.time() - start))

        return dem

    def create_discrete_models(self, candidates=None, max_num_peaks=10, model_init_kw=None, llh_kw=None, lml_kw=None,
                               **fit_kw):
        start = time.time()

        if self.print_diagnostics:
            print('Beginning discrete model creation')

        if max_num_peaks is None:
            max_num_peaks = np.inf

        if candidates is None:
            candidates = [k for k in self.best_candidate_dict.keys() if 0 < k <= max_num_peaks]

        if llh_kw is None:
            llh_kw = {}
        if lml_kw is None:
            lml_kw = {}

        self.discrete_model_kwargs = {
            'model_init_kw': model_init_kw,
            'llh_kw': llh_kw,
            'lml_kw': lml_kw,
            'fit_kw': fit_kw
        }

        # Generate discrete models
        self.discrete_candidate_dict = {}
        num_candidates = len(candidates)
        discrete_llh = np.empty(num_candidates)
        discrete_lml = np.empty(num_candidates)
        discrete_bic = np.empty(num_candidates)
        for i, candidate in enumerate(candidates):
            dem = self.convert_candidate_to_discrete(candidate, model_init_kw, **fit_kw)
            weights = dem.weights
            llh = dem.evaluate_llh(weights=weights, **llh_kw)
            lml = dem.estimate_lml(weights=weights, **lml_kw, **llh_kw)
            bic = dem.evaluate_bic(weights=weights, **llh_kw)
            discrete_llh[i] = llh
            discrete_lml[i] = lml
            discrete_bic[i] = bic
            self.discrete_candidate_dict[candidate] = {
                'model': dem,
                'llh': llh,
                'bic': bic,
                'lml': lml,
                'peak_tau': dem.get_peak_tau(),
                'time_constants': dem.get_time_constants()
            }

        # Get best metrics across models
        best_llh = np.max(discrete_llh)
        best_lml = np.max(discrete_lml)
        best_bic = np.min(discrete_bic)

        # Fill in metrics relative to best
        for i, candidate in enumerate(candidates):
            self.discrete_candidate_dict[candidate]['rel_llh'] = discrete_llh[i] - best_llh
            self.discrete_candidate_dict[candidate]['rel_bic'] = discrete_bic[i] - best_bic
            self.discrete_candidate_dict[candidate]['rel_lml'] = discrete_lml[i] - best_lml

        self.discrete_candidate_df = pd.DataFrame(
            np.vstack([candidates, np.array(candidates).astype(int), discrete_llh, discrete_bic, discrete_lml,
                       discrete_llh - best_llh, discrete_bic - best_bic, discrete_lml - best_lml]).T,
            columns=['model_id', 'num_peaks', 'llh', 'bic', 'lml', 'rel_llh', 'rel_bic', 'rel_lml']
        )

        if self.print_diagnostics:
            print('Discrete models created in {:.3f} s'.format(time.time() - start))

        return self.discrete_candidate_dict.copy()

    def dual_fit_eis(self, frequencies, z, qphb_kw=None, generate_kw=None, discrete_kw=None):
        start_time = time.time()

        if qphb_kw is None:
            qphb_kw = {}
        self.qphb_fit_eis(frequencies, z, **qphb_kw)

        if generate_kw is None:
            generate_kw = {}
        self.generate_candidates(**generate_kw)

        if discrete_kw is None:
            discrete_kw = {}
        self.create_discrete_models(**discrete_kw)

        if self.print_diagnostics:
            print('------------------------------')
            print('Dual fit completed in {:.3f} s'.format(time.time() - start_time))
            print('------------------------------')

    def sort_discrete_by_llh(self, start_from_model=None):
        """
        Sort discrete candidates by improvement in log-likelihood per peak
        :return:
        """
        if start_from_model is None:
            start_from_model = self.discrete_candidate_df['model_id'].values[0]

        start_index = self.discrete_candidate_df[self.discrete_candidate_df['model_id'] == start_from_model].index[0]
        model_ids = self.discrete_candidate_df.loc[start_index:, 'model_id'].values

        cand_llh = self.discrete_candidate_df.loc[start_index:, 'llh'].values
        cand_num_peaks = self.discrete_candidate_df.loc[start_index:, 'num_peaks'].values

        delta_llh = np.diff(cand_llh) / np.diff(cand_num_peaks)  # per-peak improvement
        # lml_change = np.diff(cand_lml)

        cand_peak_ln_tau = [np.log(v['time_constants']) for k, v in self.discrete_candidate_dict.items() if
                            k in model_ids]
        added_peak_index = [peaks.find_new_peaks(cand_peak_ln_tau[i], cand_peak_ln_tau[i - 1])
                            for i in range(1, len(cand_peak_ln_tau))]
        added_peak_tau = [np.exp(cand_peak_ln_tau[i + 1][index]) for i, index in enumerate(added_peak_index)]

        # Insert peaks included in simplest model
        added_peak_index = [np.arange(cand_num_peaks[0], dtype=int)] + added_peak_index
        added_peak_tau = [np.exp(cand_peak_ln_tau[0])] + added_peak_tau
        print(delta_llh)
        delta_llh = np.insert(delta_llh, 0, np.max(delta_llh) + 1)  # improvement undefined for 1st model
        # lml_change = np.insert(lml_change, 0, np.max(lml_change) + 1)

        # Sort by largest improvement
        sort_index = np.argsort(delta_llh)[::-1]

        return cand_num_peaks[sort_index], [added_peak_index[i] for i in sort_index], \
               [added_peak_tau[i] for i in sort_index], delta_llh[sort_index]

    def search_for_better_discrete(self, min_num_peaks=1, max_num_peaks=None, start_from_best=True,
                                   p2p_distance_threshold=0.5):
        if start_from_best:
            start_from_model = self.discrete_candidate_df.loc[self.discrete_candidate_df['lml'].argmax(), 'model_id']
        else:
            start_from_model = None

        sorted_num_peaks, added_peak_index, added_peak_tau, delta_llh = self.sort_discrete_by_llh(start_from_model)

        if max_num_peaks is None:
            max_num_peaks = np.inf

        # Get basic model info
        init_model = self.discrete_candidate_dict[sorted_num_peaks[0]]['model']
        first_drt_element = init_model.drt_elements[0]
        offset_model_string = init_model.model_string[:init_model.model_string.find(first_drt_element)]
        drt_element_type, _ = elements.parse_element_string(first_drt_element)
        params_per_element = len(elements.element_parameters(drt_element_type)[0])
        drt_param_start = init_model.parameter_indices[init_model.element_names.index(first_drt_element)][0]
        print('drt_param_start:', drt_param_start)

        def make_model_string(num_peaks):
            drt_string = '-'.join([f'{drt_element_type}{k + 1}' for k in range(num_peaks)])
            return f'{offset_model_string}{drt_string}'

        def find_best_candidate_with_peak(peak_tau_scalar):
            """Find best discrete candidate containing the target peak"""
            # Get all models containing target peak
            match_dict = {k: v for k, v in self.discrete_candidate_dict.items()
                          if peaks.has_similar_peak(np.log(peak_tau_scalar), np.log(v['time_constants']),
                                                    threshold=0.5, epsilon=2
                                                    )
                          }
            # Get matching model with best llh
            best_match_id = list(match_dict.keys())[np.argmax([v['llh'] for v in match_dict.values()])]
            return best_match_id

        def get_peak_estimate_info(peak_tau_array):
            num_params = drt_param_start + len(peak_tau_array) * params_per_element
            init_values = np.empty(num_params)
            lb = np.empty(num_params)
            ub = np.empty(num_params)
            rss = np.empty(num_params)

            # Take offset param estimates from best candidate (based on llH)
            offset_model_id = self.discrete_candidate_df.loc[self.discrete_candidate_df['llh'].argmax(), 'model_id']
            offset_model = self.discrete_candidate_dict[offset_model_id]['model']
            init_values[:drt_param_start] = offset_model.drt_estimates['init_values'][:drt_param_start]
            offset_bounds = offset_model.parameter_bounds[:drt_param_start]
            lb[:drt_param_start], ub[:drt_param_start] = elements.flatten_bounds(offset_bounds)
            rss[:drt_param_start] = offset_model.drt_estimates['rss']

            for k, peak_tau in enumerate(peak_tau_array):
                param_start_index = drt_param_start + k * params_per_element
                param_end_index = drt_param_start + (k + 1) * params_per_element

                # Get the best model containing the peak
                best_model_id = find_best_candidate_with_peak(peak_tau)
                model_dict = self.discrete_candidate_dict[best_model_id]
                model = model_dict['model']
                print('matching model peak tau:', model_dict['time_constants'])

                # Find the matching peak
                peak_match_index = np.argmin(np.abs(np.log(peak_tau) - np.log(model_dict['time_constants'])))
                print(k, best_model_id, peak_tau, peak_match_index)

                # Get the init parameter values of the corresponding peak
                element_name = f'{drt_element_type}{peak_match_index + 1}'
                print(model.get_element_parameter_values(element_name))
                print('init values before update:', init_values)
                print(element_name)
                init_values[param_start_index:param_end_index] = \
                    model.get_element_parameter_values(element_name, x=model.drt_estimates['init_values'])
                print('init values after update:', init_values)
                bounds = model.get_element_bounds(element_name)
                lb[param_start_index:param_end_index], ub[param_start_index:param_end_index] = \
                    elements.flatten_bounds(bounds)
                rss[param_start_index:param_end_index] = model.drt_estimates['rss']

            return init_values, lb, ub, rss

        def make_test(base_peak_tau, base_params, base_lb, base_ub, base_rss, add_peak_index, add_peak_tau,
                      source_num_peaks):
            print('add_peak_tau:', add_peak_tau)
            # Determine tau-ordered position of new peak
            new_element_sort = np.where(add_peak_tau < base_peak_tau)
            if len(new_element_sort[0]) > 0:
                new_element_position = new_element_sort[0][0]
            else:
                new_element_position = len(base_peak_tau)

            # print(new_element_position)

            new_peak_tau = np.insert(base_peak_tau, new_element_position, add_peak_tau)
            new_num_peaks = len(new_peak_tau)

            # Compare to existing candidate with same num peaks
            orig_candidate = self.discrete_candidate_dict.get(new_num_peaks, None)
            if orig_candidate is not None:
                # Check if test peaks are the same as original candidate peaks
                different_peaks = peaks.find_new_peaks(np.log(new_peak_tau), np.log(orig_candidate['time_constants']),
                                                       distance_threshold=p2p_distance_threshold)

                print('New peaks, orig peaks:', new_peak_tau, orig_candidate['time_constants'])
                print('diff peaks:', different_peaks)
                if len(different_peaks[0]) == 0:
                    # Test candidate contains the same peaks as the base candidate.
                    # Update test_peak_tau with the base candidate's peak_tau,
                    # since the base candidate is already optimized

                    # new_peak_tau = orig_candidate['time_constants']
                    # new_params = orig_candidate['model'].drt_estimates['init_values'].copy()  #
                    # # new_params = orig_candidate['model'].parameter_values.copy()
                    # new_rss = np.ones(len(new_params)) * orig_candidate['model'].drt_estimates['rss']
                    # new_lb, new_ub = elements.flatten_bounds(orig_candidate['model'].parameter_bounds)

                    new_params, new_lb, new_ub, new_rss = get_peak_estimate_info(new_peak_tau)

                    test_is_different = False
                else:
                    # Test candidate is different than the base candidate
                    test_is_different = True
            else:
                # No base candidate with same num_peaks exists
                test_is_different = True

            if test_is_different:
                # add_element_name = f'{drt_element_type}{add_peak_index + 1}'
                # # print(add_element_name)
                # source_model = self.discrete_candidate_dict[source_num_peaks]['model']
                # add_params = source_model.get_element_parameter_values(add_element_name,
                #                                                        x=source_model.drt_estimates['init_values'])
                # print('add_params:', add_params)
                # add_rss = np.ones(len(add_params)) * source_model.drt_estimates['rss']
                # add_lb, add_ub = elements.flatten_bounds(source_model.get_element_bounds(add_element_name))
                # insert_index = drt_param_start + new_element_position * params_per_element
                # new_params = np.insert(base_params, insert_index, add_params)
                # new_lb = np.insert(base_lb, insert_index, add_lb)
                # new_ub = np.insert(base_ub, insert_index, add_ub)
                # new_rss = np.insert(base_rss, insert_index, add_rss)

                new_params, new_lb, new_ub, new_rss = get_peak_estimate_info(new_peak_tau)

            print('new_peak_tau:', new_peak_tau)
            print('test is different:', test_is_different)

            return new_peak_tau, new_params, new_lb, new_ub, new_rss, test_is_different

        # Start from the simplest model and incrementally add peaks with largest llh improvement
        test_peak_tau = added_peak_tau[0]
        test_params = init_model.parameter_values.copy()
        test_lb, test_ub = elements.flatten_bounds(init_model.parameter_bounds)
        test_rss = np.ones(len(test_params)) * init_model.drt_estimates['rss']
        test_candidates = []

        for i in range(1, len(sorted_num_peaks)):
            print(i, sorted_num_peaks[i], added_peak_tau[i])
            if len(added_peak_tau[i]) == 0:
                pass
            elif len(added_peak_tau[i]) == 1:
                test_peak_tau, test_params, test_lb, test_ub, test_rss, is_different = \
                    make_test(test_peak_tau, test_params, test_lb, test_ub, test_rss, added_peak_index[i][0],
                              added_peak_tau[i][0], sorted_num_peaks[i])

                test_num_peaks = len(test_peak_tau)

                if is_different and test_num_peaks >= min_num_peaks:
                    test_candidates.append({
                        'num_peaks': test_num_peaks,
                        'peak_tau': test_peak_tau,
                        'init_values': test_params,
                        'bounds': elements.pair_bounds(test_lb, test_ub),
                        'init_val_rss': test_rss,
                        'model_string': make_model_string(test_num_peaks)
                    })

                print('test_num_peaks:', test_num_peaks)
                if test_num_peaks >= max_num_peaks:
                    break
            else:
                print('2+ peaks added')
                num_added = len(added_peak_tau[i])
                iter_index = np.arange(num_added)
                for num_new in range(1, num_added + 1):
                    test_num_peaks = len(test_peak_tau) + num_new
                    print('test_num_peaks:', test_num_peaks)
                    print(test_peak_tau)
                    combos = itertools.combinations(iter_index, num_new)
                    for combo in combos:
                        # Add all peaks in combo
                        tmp_peak_tau, tmp_params, tmp_lb, tmp_ub, tmp_rss = test_peak_tau, test_params, test_lb, test_ub, test_rss
                        for index in combo:
                            tmp_peak_tau, tmp_params, tmp_lb, tmp_ub, tmp_rss, tmp_is_diff = \
                                make_test(tmp_peak_tau, tmp_params, tmp_lb, tmp_ub, tmp_rss, added_peak_index[i][index],
                                          added_peak_tau[i][index], sorted_num_peaks[i])
                        if tmp_is_diff and test_num_peaks >= min_num_peaks:
                            test_candidates.append({
                                'num_peaks': test_num_peaks,
                                'peak_tau': tmp_peak_tau,
                                'init_values': tmp_params,
                                'bounds': elements.pair_bounds(tmp_lb, tmp_ub),
                                'init_val_rss': tmp_rss,
                                'model_string': make_model_string(test_num_peaks)
                            })

                    if test_num_peaks >= max_num_peaks:
                        break

                # Continue from final combo, which will include all new peaks
                test_peak_tau, test_params, test_lb, test_ub, test_rss = tmp_peak_tau, tmp_params, tmp_lb, tmp_ub, tmp_rss

        self.discrete_reordered_candidates = test_candidates.copy()

        return test_candidates

    def create_reordered_discrete_models(self, min_num_peaks=None, max_num_peaks=None, start_from_best=True,
                                         search_kw=None,
                                         append_models=True):
        if min_num_peaks is None:
            min_num_peaks = 1

        if max_num_peaks is None:
            best_num = self.discrete_candidate_df.loc[self.discrete_candidate_df['lml'].argmax(), 'num_peaks']
            max_num_peaks = best_num + 3

        if search_kw is None:
            search_kw = {}

        test_candidates = self.search_for_better_discrete(start_from_best=start_from_best, min_num_peaks=min_num_peaks,
                                                          max_num_peaks=max_num_peaks,
                                                          **search_kw)

        test_models = []
        for cand_info in test_candidates:
            start = time.time()
            dem = elements.DiscreteElementModel(cand_info['model_string'])
            print(cand_info['init_values'])
            dem.drt_estimates = {
                'init_values': cand_info['init_values'],
                'eis_weights': self.qphb_params['est_weights'] / self.impedance_scale,
                'rss': cand_info['init_val_rss']
            }
            dem.set_bounds(cand_info['bounds'])

            if self.fit_type.find('eis') > -1:
                dem.fit_eis(self.get_fit_frequencies(), self.z_fit, from_drt=True,
                            **self.discrete_model_kwargs['fit_kw'])
            else:
                pass

            if self.print_diagnostics:
                print('Fitted test candidate with {} peaks in {:.3f} s'.format(cand_info['num_peaks'],
                                                                               time.time() - start))

            test_models.append(dem)

        if append_models:
            # Add new models to discrete candidate dict and dataframe
            # Use same kwargs for evaluation to ensure consistency with original models
            llh_kw = self.discrete_model_kwargs['llh_kw']
            lml_kw = self.discrete_model_kwargs['lml_kw']
            # Generate discrete models
            num_candidates = len(test_models)
            discrete_llh = np.empty(num_candidates)
            discrete_lml = np.empty(num_candidates)
            discrete_bic = np.empty(num_candidates)
            model_ids = np.empty(num_candidates)
            for i, dem in enumerate(test_models):
                llh = dem.evaluate_llh(**llh_kw)
                lml = dem.estimate_lml(**lml_kw, **llh_kw)
                bic = dem.evaluate_bic(**llh_kw)
                discrete_llh[i] = llh
                discrete_lml[i] = lml
                discrete_bic[i] = bic
                test_num_peaks = test_candidates[i]['num_peaks']
                num_existing_models = len([v for v in self.discrete_candidate_dict.values()
                                           if len(v['time_constants']) == test_num_peaks]
                                          )
                model_id = test_num_peaks + 0.1 * num_existing_models
                model_ids[i] = model_id

                self.discrete_candidate_dict[model_id] = {
                    'model': dem,
                    'llh': llh,
                    'bic': bic,
                    'lml': lml,
                    'peak_tau': dem.get_peak_tau(),
                    'time_constants': dem.get_time_constants()
                }

            cand_num_peaks = np.array([cand['num_peaks'] for cand in test_candidates]).astype(int)
            append_df = pd.DataFrame(
                np.vstack([model_ids, cand_num_peaks, discrete_llh, discrete_bic, discrete_lml,
                           discrete_llh, discrete_bic, discrete_lml]).T,  # placeholders for relative metrics
                columns=['model_id', 'num_peaks', 'llh', 'bic', 'lml', 'rel_llh', 'rel_bic', 'rel_lml']
            )

            self.discrete_candidate_df = pd.concat((self.discrete_candidate_df, append_df), ignore_index=True)
            self.discrete_candidate_df.sort_values('model_id', ignore_index=True, inplace=True)

            # Get best metrics across models
            best_llh = self.discrete_candidate_df['llh'].max()
            best_lml = self.discrete_candidate_df['lml'].max()
            best_bic = self.discrete_candidate_df['bic'].min()

            # Update relative metrics in dataframe
            self.discrete_candidate_df['rel_llh'] = self.discrete_candidate_df['llh'] - best_llh
            self.discrete_candidate_df['rel_lml'] = self.discrete_candidate_df['lml'] - best_lml
            self.discrete_candidate_df['rel_bic'] = self.discrete_candidate_df['bic'] - best_bic

            # Fill in metrics relative to best in dict
            for candidate in self.discrete_candidate_dict.keys():
                self.discrete_candidate_dict[candidate]['rel_llh'] = \
                    self.discrete_candidate_dict[candidate]['llh'] - best_llh
                self.discrete_candidate_dict[candidate]['rel_lml'] = \
                    self.discrete_candidate_dict[candidate]['lml'] - best_lml
                self.discrete_candidate_dict[candidate]['rel_bic'] = \
                    self.discrete_candidate_dict[candidate]['bic'] - best_bic

        return test_models

    def plot_candidate_distribution(self, candidate_id, candidate_type, mark_peaks=False, mark_peaks_kw=None,
                                    tau=None, **kw):
        candidate_info = self.get_candidate(candidate_id, candidate_type)

        if candidate_type == 'continuous':
            candidate_x = self.extract_qphb_parameters(candidate_info['x'])['x']

            if mark_peaks_kw is None:
                mark_peaks_kw = {'peak_tau': candidate_info['peak_tau']}
            return self.plot_distribution(tau=tau, x=candidate_x, mark_peaks=mark_peaks, mark_peaks_kw=mark_peaks_kw,
                                          **kw)
        else:
            dem = candidate_info['model']

            if tau is None:
                tau = self.get_tau_eval(20)

            return dem.plot_distribution(tau, mark_peaks=mark_peaks, mark_peaks_kw=mark_peaks_kw, **kw)

    def predict_candidate_distribution(self, candidate_id, candidate_type, tau=None, **kw):
        candidate_info = self.get_candidate(candidate_id, candidate_type)

        if candidate_type == 'continuous':
            candidate_x = self.extract_qphb_parameters(candidate_info['x'])['x']

            return self.predict_distribution(tau=tau, x=candidate_x, **kw)
        else:
            dem = candidate_info['model']

            if tau is None:
                tau = self.get_tau_eval(20)

            return dem.predict_distribution(tau, **kw)

    def plot_candidate_eis_fit(self, candidate_id, candidate_type, **kw):
        candidate_info = self.get_candidate(candidate_id, candidate_type)
        if candidate_type == 'continuous':
            candidate_x_raw = candidate_info['x']
            return self.plot_eis_fit(predict_kw={'x': candidate_x_raw}, **kw)
        else:
            dem = candidate_info['model']
            return dem.plot_eis_fit(**kw)

    def evaluate_norm_bayes_factors(self, candidate_type, criterion=None, candidate_id=None):
        cand_df = self.get_candidate_df(candidate_type)

        if criterion is None:
            criterion = 'bic'

        if candidate_id is None:
            return stats.norm_bayes_factors(cand_df[criterion].values, criterion)
        else:
            cand_index = np.where(cand_df['model_id'] == candidate_id)
            bf = stats.norm_bayes_factors(cand_df[criterion].values, criterion)
            return bf[cand_index]

    def evaluate_bayes_factor(self, candidate_id_1, candidate_id_2, candidate_type='discrete', criterion=None):
        if criterion is None:
            criterion = 'bic'

        cand_1_info = self.get_candidate(candidate_id_1, candidate_type)
        cand_2_info = self.get_candidate(candidate_id_2, candidate_type)

        return stats.bayes_factor(cand_1_info[criterion], cand_2_info[criterion], criterion)

    def plot_norm_bayes_factors(self, candidate_type, criterion=None, ax=None, **kw):
        factors = self.evaluate_norm_bayes_factors(candidate_type, criterion)

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        cand_df = self.get_candidate_df(candidate_type)

        ax.plot(cand_df['num_peaks'].values, factors, **kw)

        ax.set_xticks(cand_df['num_peaks'].values)
        ax.set_xlabel('$Q$')
        ax.set_ylabel('$B_Q$')

        return ax

    def get_candidate_df(self, candidate_type):
        if candidate_type == 'continuous':
            return self.best_candidate_df
        elif candidate_type == 'discrete':
            return self.discrete_candidate_df
        elif candidate_type == 'pfrt':
            return self.pfrt_candidate_df
        else:
            raise ValueError(f'Invalid candidate_type {candidate_type}')

    def get_candidate(self, candidate_num_peaks, candidate_type):
        if candidate_type == 'continuous':
            if self.best_candidate_dict is None:
                raise ValueError('Candidates must be first be generated using generate_candidates')
            else:
                try:
                    candidate_info = self.best_candidate_dict[candidate_num_peaks]
                    return candidate_info
                except KeyError:
                    raise ValueError(f'No candidate with {candidate_num_peaks} peaks exists')
        elif candidate_type == 'discrete':
            if self.discrete_candidate_dict is None:
                raise ValueError('Discrete candidates must be first be created using create_discrete_models')
            else:
                try:
                    candidate_info = self.discrete_candidate_dict[candidate_num_peaks]
                    return candidate_info
                except KeyError:
                    raise ValueError(f'No candidate with {candidate_num_peaks} peaks exists')
        elif candidate_type == 'pfrt':
            if self.pfrt_candidate_dict is None:
                raise ValueError('Discrete candidates must be first be created using create_discrete_from_pfrt')
            else:
                try:
                    candidate_info = self.pfrt_candidate_dict[candidate_num_peaks]
                    return candidate_info
                except KeyError:
                    raise ValueError(f'No candidate with {candidate_num_peaks} peaks exists')
        else:
            raise ValueError(f"Invalid candidate_type {candidate_type}. Options: 'continuous', 'discrete'")

    def get_best_candidate_id(self, candidate_type, criterion=None):
        candidate_types = ['discrete', 'continuous']
        if candidate_type not in candidate_types:
            raise ValueError(f'Invalid candidate_type {candidate_type}. Options: {candidate_types}')

        criterion_directions = {
            'bic': -1,
            'lml': 1
        }

        if criterion is not None:
            if criterion_directions.get(criterion, None) is None:
                raise ValueError(f'Invalid criterion {criterion}. Options: {criterion_directions.keys()}')

        if candidate_type == 'discrete':
            if criterion is None:
                criterion = 'bic'
            model_df = self.discrete_candidate_df
        else:
            if criterion is None:
                criterion = 'bic'
            model_df = self.best_candidate_df

        crit_direction = criterion_directions[criterion]
        best_index = np.argmax(crit_direction * model_df[criterion].values)
        best_id = model_df.loc[model_df.index[best_index], 'model_id']

        return best_id

    def predict_pdrt(self, tau=None, ppd=20, criterion='bic', criterion_factor=1):
        if tau is None:
            tau = self.get_tau_eval(ppd)

        spread_func = get_similarity_function('gaussian')

        pdrt = np.zeros(len(tau))

        for cand_id, cand_info in self.discrete_candidate_dict.items():
            peak_tau = cand_info['model'].get_peak_tau(find_peaks_kw={'height': 0})  # TODO: replace with true_peak_tau
            # peak_tau = cand_info['model'].get_time_constants()
            if criterion == 'bic':
                peak_prob = np.exp(-0.5 * criterion_factor * cand_info['rel_bic'])
            elif criterion == 'lml':
                peak_prob = np.exp(criterion_factor * cand_info['rel_lml'])
            elif criterion is None:
                peak_prob = 1
            else:
                raise ValueError(f"Invalid criterion {criterion}. Options: 'bic', 'lml'")

            print(peak_prob)

            cand_pdf = [peak_prob * spread_func(np.log(tau / pt), 1, self.tau_epsilon) for pt in peak_tau]
            pdrt += np.sum(cand_pdf, axis=0)

        pdrt /= np.max(pdrt)

        return pdrt

    def plot_pdrt(self, tau=None, ppd=20, criterion='bic', criterion_factor=1, ax=None, log_scale=False, **kw):
        if tau is None:
            tau = self.get_tau_eval(ppd)

        pdrt = self.predict_pdrt(tau, ppd, criterion, criterion_factor)

        if log_scale:
            y = np.log(pdrt)
        else:
            y = pdrt

        ax = plot_distribution(tau, y, ax, None, '', None, False, **kw)

        if log_scale:
            ax.set_ylabel(r'$\ln{p_{\gamma}}$')
            ax.set_ylim(-10, 0.05)
        else:
            ax.set_ylabel(r'$p_{\gamma}$')

        fig = ax.get_figure()
        fig.tight_layout()

        return ax

    # --------------------------
    # PFRT
    # --------------------------
    def generate_pfrt(self, frequencies, z, factors=None, max_iter_per_step=5,
                      max_init_iter=20, xtol=1e-2, **kw):
        ppd = pp.get_ppd(frequencies)
        ppd_eff = np.sqrt(2) * ppd

        data_factor = qphb.get_data_factor(np.sqrt(2) * len(frequencies), ppd_eff)
        if self.print_diagnostics:
            print('data factor:', data_factor)

        # Get default hyperparameters
        qphb_hypers = qphb.get_default_hypers(data_factor, True)

        if factors is None:
            factors = np.logspace(-1, 1, 11)
            # factors = np.concatenate((np.logspace(-0.5, 0, 6), np.logspace(0.2, 1, 5)))

        def prep_step_hypers(step_factor):
            # if step_factor < 1:
            #     new_hypers = {'weight_factor': step_factor}
            # else:
            s_0 = qphb_hypers['s_0'] * step_factor
            l2_lambda_0 = qphb_hypers['l2_lambda_0'] / step_factor
            new_hypers = {'s_0': s_0, 'l2_lambda_0': l2_lambda_0}
            return new_hypers

        # Initialize fit at first factor
        factor = factors[0]
        update_hypers = prep_step_hypers(factor)
        self.qphb_fit_eis(frequencies, z, max_iter=max_init_iter, xtol=xtol, **update_hypers, **kw)

        # Initialize history and insert initial fit
        pfrt_history = []
        step_x = []
        step_llh = []
        step_hypers = []
        # hyper_history = []
        step_p_mat = []

        def step_update(old_history, new_history, new_hypers):
            current_history = old_history + new_history
            # hyper_history += [new_hypers] * len(new_history)
            step_hypers.append(new_hypers)
            step_x.append(new_history[-1]['x'])

            # Get weights estimate based only on current x for llh calculation
            weights, _ = qphb.estimate_weights(new_history[-1]['x'], self.qphb_params['rv'], self.qphb_params['vmm'],
                                               self.qphb_params['rm'])
            step_llh.append(self.evaluate_llh(weights, x=step_x[-1], marginalize_weights=True))

            # Get P matrix
            p_matrix, _ = qphb.calculate_pq(self.qphb_params['rm'], self.qphb_params['rv'],
                                            self.qphb_params['penalty_matrices'], self.fit_kwargs['penalty_type'],
                                            self.qphb_params['hypers'], self.qphb_params['l1_lambda_vector'],
                                            new_history[-1]['rho_vector'], new_history[-1]['s_vectors'],
                                            weights)

            step_p_mat.append(p_matrix)

            return current_history

        pfrt_history = step_update(pfrt_history, self.qphb_history, update_hypers)

        # print(factor, len(pfrt_history))

        # Proceed through remaining factors
        for factor in factors[1:]:
            update_hypers = prep_step_hypers(factor)

            x_in = pfrt_history[-1]['x'].copy()
            rho_in = pfrt_history[-1]['rho_vector'].copy()
            s_in = pfrt_history[-1]['s_vectors'].copy()
            weights_in = pfrt_history[-1]['weights'].copy()
            outlier_t_in = pfrt_history[-1]['outlier_t'].copy()

            hist = self._continue_from_init(self.qphb_params['hypers'], x_in,
                                            self.qphb_params['rv'], self.qphb_params['rm'], self.qphb_params['vmm'],
                                            rho_in, s_in, outlier_t_in,
                                            self.qphb_params['penalty_matrices'], self.qphb_params['xmx_norms'],
                                            self.qphb_params['est_weights'], weights_in,
                                            self.qphb_params['l1_lambda_vector'],
                                            nonneg=self.fit_kwargs['nonneg'], update_scale=False,
                                            penalty_type=self.fit_kwargs['penalty_type'],
                                            eff_hp=self.fit_kwargs['eff_hp'], xtol=xtol, max_iter=max_iter_per_step,
                                            **update_hypers, **kw
                                            )

            pfrt_history = step_update(pfrt_history, hist, update_hypers)
            # print(factor, len(hist))

        self.pfrt_result = {
            'factors': factors,
            'history': pfrt_history,
            'step_x': step_x,
            'step_llh': step_llh,
            'step_p_mat': step_p_mat,
            'step_hypers': step_hypers
        }

    def predict_pfrt(self, tau=None, tau_pfrt=None, prior_mu=-4, prior_sigma=0.5, find_peaks_kw=None,
                     n_eff_factor=0.5,
                     fxx_var_floor=1e-5, fxx_extrap_var_scale=2e-5,
                     smooth=True, smooth_kw=None, integrate=False, integrate_threshold=1e-6,
                     normalize=True):

        factors = self.pfrt_result['factors']
        step_llh = self.pfrt_result['step_llh']
        step_x = self.pfrt_result['step_x']
        step_p_mat = self.pfrt_result['step_p_mat']

        # Calculate posterior
        log_prior = stats.log_pdf_normal(np.log(factors), prior_mu, prior_sigma)
        # log_prior = np.empty(len(factors))
        # log_prior[factors < 1] = stats.log_pdf_normal(3 * np.log(factors[factors < 1]), prior_mu, prior_sigma)
        # log_prior[factors >= 1] = stats.log_pdf_normal(np.log(factors[factors >= 1]), prior_mu, prior_sigma)
        log_post = log_prior + np.array(step_llh)
        # Normalize for better precision
        log_post_norm = log_post - np.max(log_post)
        # Apply effective factor
        log_post_eff = log_post_norm * n_eff_factor
        # Normalize to posterior area
        post_area = np.trapz(np.exp(log_post_eff), x=np.log(factors))
        post_prob_eff = np.exp(log_post_eff) / post_area

        # Identify peaks
        if find_peaks_kw is None:
            find_peaks_kw = {'height': 1e-3, 'prominence': 5e-3}
        for k in ['height', 'prominence']:
            # Ensure that height and prominence are included so that they will be returned in peak_info
            find_peaks_kw[k] = find_peaks_kw.get(k, 0)

        if tau_pfrt is None:
            tau_pfrt = self.get_tau_eval(10)  # used for peak finding
        if tau is None:
            tau = tau_pfrt  # used for evaluating pfrt at end. Only used if smooth=True

        tot_pfrt = np.zeros(len(tau_pfrt))
        step_pfrt = np.zeros((len(factors), len(tau_pfrt)))
        fxx_sigmas = []
        for i, x_raw in enumerate(step_x):
            x_drt = self.extract_qphb_parameters(x_raw)['x']
            fxx = self.predict_distribution(tau_pfrt, x=x_drt, order=2, normalize=True)

            # Get curvature std
            fxx_cov = self.estimate_distribution_cov(tau_pfrt, p_matrix=step_p_mat[i], order=2, normalize=True,
                                                     var_floor=fxx_var_floor,
                                                     extrapolation_var_scale=fxx_extrap_var_scale
                                                     )
            fxx_sigma = np.diag(fxx_cov) ** 0.5
            fxx_sigmas.append(fxx_sigma)

            # Find peaks
            peak_index, peak_info = signal.find_peaks(-fxx, **find_peaks_kw)
            # peak_index, peak_info = signal.find_peaks(-fxx / fxx_sigma, **find_peaks_kw)
            min_prom = np.minimum(peak_info['prominences'], peak_info['peak_heights'])

            # Evaluate peak confidence
            peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, fxx_sigma[peak_index])
            # peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, 1)
            # peak_prob=1

            step_pfrt[i, peak_index] = peak_prob

            tot_pfrt[peak_index] += post_prob_eff[i] * peak_prob

        self.pfrt_result['tau_pfrt'] = tau_pfrt
        self.pfrt_result['raw_pfrt'] = tot_pfrt.copy()
        self.pfrt_result['step_pfrt'] = step_pfrt

        if smooth:
            # Smooth to aggregate neighboring peak probs, which may arise due to
            # slight peak shifts with hyperparameter changes
            spread_func = get_similarity_function('gaussian')
            if smooth_kw is None:
                smooth_kw = {'order': 2, 'epsilon': 5}
            xx_basis, xx_eval = np.meshgrid(np.log(tau_pfrt), np.log(tau))
            basis_matrix = spread_func(xx_eval - xx_basis, **smooth_kw)
            tot_pfrt = basis_matrix @ tot_pfrt

        if integrate:
            peak_index, peak_prob = pfrt.integrate_peaks(tot_pfrt, integrate_threshold)
            tot_pfrt = np.zeros_like(tot_pfrt)
            tot_pfrt[peak_index] = peak_prob

        if normalize:
            tot_pfrt = tot_pfrt / np.max(tot_pfrt)

        return tot_pfrt, step_pfrt, post_prob_eff, step_x, step_p_mat, factors, fxx_sigmas

    def select_pfrt_candidates(self, start_thresh=0.99, end_thresh=0.01, peak_thresh=1e-6):
        target_peak_indices, step_indices = pfrt.select_candidates(
            self.pfrt_result['raw_pfrt'], self.pfrt_result['step_pfrt'], self.pfrt_result['step_llh'],
            start_thresh, end_thresh, peak_thresh
        )
        return target_peak_indices, step_indices

    def continuous_to_discrete(self, x, tau_find_peaks=None, peak_indices=None, model_init_kw=None, **fit_kw):
        start = time.time()

        if tau_find_peaks is None:
            tau_find_peaks = self.get_tau_eval(10)

        # Initialize discrete model from candidate DRT
        if model_init_kw is None:
            model_init_kw = {'estimate_peak_distributions': True}
        dem = elements.DiscreteElementModel.from_drt(self, x, tau_find_peaks, peak_indices, **model_init_kw)
        # dem.drt_estimates['eis_weights'] = utils.eis.complex_vector_to_concat(self.predict_sigma('eis')) ** -1

        if self.fit_type.find('eis') > -1:
            dem.fit_eis(self.get_fit_frequencies(), self.z_fit, from_drt=True, **fit_kw)
        else:
            pass

        if self.print_diagnostics:
            print('Candidate conversion time: {:.3f} s'.format(time.time() - start))

        return dem

    def create_discrete_from_pfrt(self, start_thresh=0.99, end_thresh=0.01, peak_thresh=1e-6, max_num_peaks=10,
                                  model_init_kw=None, llh_kw=None, lml_kw=None, **fit_kw):

        start = time.time()

        target_peak_indices, step_indices = self.select_pfrt_candidates(start_thresh, end_thresh, peak_thresh)

        if llh_kw is None:
            llh_kw = {}
        if lml_kw is None:
            lml_kw = {}

        # self.discrete_model_kwargs = {
        #     'model_init_kw': model_init_kw,
        #     'llh_kw': llh_kw,
        #     'lml_kw': lml_kw,
        #     'fit_kw': fit_kw
        # }

        # Generate discrete models
        self.pfrt_candidate_dict = {}
        num_candidates = len(step_indices)
        discrete_llh = np.empty(num_candidates)
        discrete_lml = np.empty(num_candidates)
        discrete_bic = np.empty(num_candidates)
        for i in range(num_candidates):
            cand_x = self.pfrt_result['step_x'][step_indices[i]]
            cand_peak_indices = target_peak_indices[i]
            cand_num_peaks = len(cand_peak_indices)
            if cand_num_peaks <= max_num_peaks:
                dem = self.continuous_to_discrete(cand_x, self.pfrt_result['tau_pfrt'], cand_peak_indices,
                                                  model_init_kw, **fit_kw)
                weights = dem.weights
                llh = dem.evaluate_llh(weights=weights, **llh_kw)
                lml = dem.estimate_lml(weights=weights, **lml_kw, **llh_kw)
                bic = dem.evaluate_bic(weights=weights, **llh_kw)
                discrete_llh[i] = llh
                discrete_lml[i] = lml
                discrete_bic[i] = bic
                self.pfrt_candidate_dict[cand_num_peaks] = {
                    'model': dem,
                    'llh': llh,
                    'bic': bic,
                    'lml': lml,
                    'peak_tau': dem.get_peak_tau(),
                    'time_constants': dem.get_time_constants()
                }

        # Get best metrics across models
        best_llh = np.max(discrete_llh)
        best_lml = np.max(discrete_lml)
        best_bic = np.min(discrete_bic)

        # Fill in metrics relative to best
        for i, candidate in enumerate(self.pfrt_candidate_dict.keys()):
            self.pfrt_candidate_dict[candidate]['rel_llh'] = discrete_llh[i] - best_llh
            self.pfrt_candidate_dict[candidate]['rel_bic'] = discrete_bic[i] - best_bic
            self.pfrt_candidate_dict[candidate]['rel_lml'] = discrete_lml[i] - best_lml

        candidates = list(self.pfrt_candidate_dict.keys())

        self.pfrt_candidate_df = pd.DataFrame(
            np.vstack([candidates, np.array(candidates).astype(int), discrete_llh, discrete_bic, discrete_lml,
                       discrete_llh - best_llh, discrete_bic - best_bic, discrete_lml - best_lml]).T,
            columns=['model_id', 'num_peaks', 'llh', 'bic', 'lml', 'rel_llh', 'rel_bic', 'rel_lml']
        )

        if self.print_diagnostics:
            print('Discrete models created from PFRT in {:.3f} s'.format(time.time() - start))

    # def plot_discrete_candidate_distribution(self, candidate_num_peaks, tau=None, ppd=20, **kw):
    #     # Get model
    #     candidate_info = self.get_candidate(candidate_num_peaks, 'discrete')
    #     dem = candidate_info['model']
    #
    #     if tau is None:
    #         tau = self.get_tau_eval(ppd)
    #
    #     ax = dem.plot_distribution(tau, **kw)
    #
    #     return ax
    #
    # def plot_discrete_candidate_eis_fit(self, candidate_id, **kw):
    #     # Get model
    #     candidate_info = self.get_candidate(candidate_id, 'discrete')
    #     dem = candidate_info['model']
    #
    #     ax = dem.plot_eis_fit(**kw)
    #
    #     return ax

    # # CHB fit
    # # ----------------------------
    # def fit(self, times, i_signal, v_signal, step_times=None, nonneg=True, scale_signal=True, offset_baseline=True,
    #         offset_steps=False,
    #         downsample=True, downsample_kw=None, smooth_inf_response=True,
    #         init_from_ridge=False, ridge_kw={},
    #         # step_times_sizes=None,
    #         model_name=None, add_stan_data={}, init_values=None,
    #         # optimization control
    #         max_iter=10000, random_seed=1234):
    #
    #     # Process data and calculate matrices for fit
    #     sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, frequencies=None, z=None,
    #                                                step_times=step_times, downsample=downsample,
    #                                                downsample_kw=downsample_kw, offset_steps=offset_steps,
    #                                                smooth_inf_response=smooth_inf_response, scale_data=scale_signal,
    #                                                rp_scale=7, penalty_type='discrete',
    #                                                derivative_weights=derivative_weights)
    #
    #     sample_times, sample_i, sample_v, response_baseline, _ = sample_data
    #     rm, induc_rv, inf_rv, _, _, penalty_matrices = matrices
    #
    #     # Prepare data for Stan model
    #     stan_data = {
    #         'N': len(sample_times),
    #         'K': len(self.basis_tau),
    #         # 'I': sample_i,
    #         'V': sample_v,
    #         'times': sample_times,
    #         'A': rm,
    #         'inf_rv': inf_rv,
    #         'L0': 1.5 * 0.24 * penalty_matrices['l0'],
    #         'L1': 1.5 * 0.16 * penalty_matrices['l1'],
    #         'L2': 1.5 * 0.08 * penalty_matrices['l2'],
    #         'inductance_response': induc_rv,
    #         'sigma_min': 1e-4,
    #         'ups_alpha': 0.05,
    #         'ups_beta': 0.1,
    #         'ups_scale': 0.15,
    #         'ds_alpha': 5,
    #         'ds_beta': 5,
    #         'sigma_res_scale': 1,  # 0.1 * np.sqrt(len(sample_times) / 100),
    #         # Ishwaran & Rao recommended variance inflation factor
    #         'R_inf_scale': 100,
    #         'inductance_scale': 1,
    #         'v_baseline_scale': 100
    #     }
    #
    #     stan_data.update(add_stan_data)
    #
    #     print('Finished prepping stan_data')
    #
    #     # load Stan model
    #     if model_name is None:
    #         # Construct model name from arguments
    #         if self.op_mode == 'galvanostatic':
    #             model_name = 'Galv'
    #         elif self.op_mode == 'potentiostatic':
    #             model_name = 'Pot'
    #         else:
    #             raise ValueError
    #
    #         model_name += 'SquareWaveDRT'
    #
    #         if nonneg:
    #             model_name += '_pos'
    #
    #         if not self.fit_inductance:
    #             model_name += '_noL'  # no inductance
    #
    #         model_name += '.stan'
    #     self.stan_model_name = model_name
    #
    #     # Get initial values from ridge fit
    #     if init_from_ridge:
    #         scaled_response_offset, ridge_values = self.get_init_from_ridge(times, i_signal, v_signal,
    #                                                                  frequencies=None, z=None, step_times=step_times,
    #                                                                  z_weight=None, fit_type='normal', nonneg=nonneg,
    #                                                                  downsample=downsample, downsample_kw=downsample_kw,
    #                                                                  scale_signal=scale_signal,
    #                                                                  offset_steps=offset_steps,
    #                                                                  smooth_inf_response=smooth_inf_response,
    #                                                                  **ridge_kw)
    #         ridge_values['R_inf_raw'] = ridge_values['R_inf'] / stan_data['R_inf_scale']
    #         if self.fit_inductance:
    #             ridge_values['inductance_raw'] = ridge_values['inductance'] / stan_data['inductance_scale']
    #
    #         # Update with user-supplied inits if provided
    #         if init_values is not None:
    #             ridge_values.update(init_values)
    #             init_values = ridge_values
    #         else:
    #             init_values = ridge_values
    #
    #         self._init_values = init_values
    #     else:
    #         # Offset voltage baseline
    #         if offset_baseline:
    #             scaled_response_offset = -response_baseline
    #         else:
    #             scaled_response_offset = 0
    #
    #     print('response offset:', scaled_response_offset * self.response_signal_scale)
    #
    #     sample_v += scaled_response_offset
    #     # print(sample_v)
    #
    #     self.stan_input = stan_data.copy()
    #
    #     # stan_model = load_pickle(os.path.join(module_dir, 'stan_model_files', model_name))
    #     stan_model = CmdStanModel(stan_file=os.path.join(module_dir, 'stan_model_files', model_name))
    #     print('Loaded stan model')
    #     # Fit model
    #     print('stan data check:', utils.array.check_equality(self.stan_input, stan_data))
    #     self.stan_mle = stan_model.optimize(stan_data, iter=max_iter, seed=random_seed, inits=init_values,
    #                                         algorithm='lbfgs')
    #     self.stan_result = self.stan_mle.stan_variables()
    #     # self.stan_result = stan_model.optimizing(stan_data, iter=max_iter, seed=random_seed, init=init_values)
    #     print('Optimized model')
    #     # Extract model parameters
    #     self.fit_parameters = {'x': self.stan_result['x'] * self.coefficient_scale,
    #                            'R_inf': self.stan_result['R_inf'] * self.coefficient_scale,
    #                            'v_baseline': (self.stan_result['v_baseline'] - scaled_response_offset) \
    #                                          * self.response_signal_scale,
    #                            'v_sigma_tot': self.stan_result['sigma_tot'] * self.response_signal_scale,
    #                            'v_sigma_res': self.stan_result['sigma_res'] * self.response_signal_scale}
    #
    #     if self.fit_inductance:
    #         self.fit_parameters['inductance'] = self.stan_result['inductance'] * self.coefficient_scale
    #     else:
    #         self.fit_parameters['inductance'] = 0
    #
    #     self.fit_type = 'chb'

    def qphb_fit_hybrid(self, times, i_signal, v_signal, frequencies, z, step_times=None, nonneg=True,
                        scale_data=True, offset_steps=True, offset_baseline=True, update_scale=False,
                        downsample=False, downsample_kw=None, smooth_inf_response=True,
                        # basic fit control
                        v_baseline_penalty=0, R_inf_penalty=0, inductance_penalty=0, inductance_scale=1e-5,
                        penalty_type='integral',
                        chrono_error_structure='uniform', eis_error_structure=None,
                        chrono_vmm_epsilon=0.1, eis_vmm_epsilon=0.25, eis_reim_cor=0.25,
                        vz_offset=False, vz_offset_scale=0.05,
                        # Prior hyperparameters
                        info_factor=1, eff_hp=True,
                        eis_weight_factor=None, chrono_weight_factor=None,
                        # rp_scale=14, derivative_weights=[1.5, 1.0, 0.5],
                        # l2_lambda_0=None, l1_lambda_0=0.0,
                        # chrono_iw_alpha=1.5, chrono_iw_beta=None,
                        # eis_iw_alpha=1.5, eis_iw_beta=None,
                        # s_alpha=[1.5, 2.5, 25], s_0=1,
                        # rho_alpha=[1.1, 1.15, 1.2], rho_0=1,
                        # w_alpha=None, w_beta=None,
                        # optimization control
                        xtol=1e-2, max_iter=50, **kw):

        utils.validation.check_penalty_type(penalty_type)
        utils.validation.check_error_structure(chrono_error_structure)
        utils.validation.check_error_structure(eis_error_structure)

        # Format list/scalar arguments into arrays
        # derivative_weights = np.array(derivative_weights)
        # k_range = len(derivative_weights)
        # if np.shape(s_alpha) == ():
        #     s_alpha = [s_alpha] * k_range
        #
        # if np.shape(rho_alpha) == ():
        #     rho_alpha = [rho_alpha] * k_range
        #
        # s_alpha = np.array(s_alpha)
        # rho_alpha = np.array(rho_alpha)

        # Define special parameters included in quadratic programming parameter vector
        self.special_qp_params = {
            'v_baseline': {'index': 0, 'nonneg': False},
            'R_inf': {'index': 1, 'nonneg': True},
        }

        if self.fit_inductance:
            self.special_qp_params['inductance'] = {'index': len(self.special_qp_params), 'nonneg': True}

        if vz_offset:
            self.special_qp_params['vz_offset'] = {'index': len(self.special_qp_params), 'nonneg': False}

        # If rp_scale is to be set automatically, pass a temporary value of 1 to prep_for_fit
        # rp_scale can only be determined after downsampling
        # if rp_scale is None:
        #     rp_scale_tmp = 1
        # else:
        #     rp_scale_tmp = rp_scale

        # Get preprocessing hyperparameters
        pp_hypers = qphb.get_default_hypers(1, eff_hp)
        pp_hypers.update(kw)

        # Process data and calculate matrices for fit
        sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, frequencies, z, step_times=step_times,
                                                   downsample=downsample, downsample_kw=downsample_kw,
                                                   offset_steps=offset_steps, smooth_inf_response=smooth_inf_response,
                                                   scale_data=scale_data, rp_scale=pp_hypers['rp_scale'],
                                                   penalty_type=penalty_type,
                                                   derivative_weights=pp_hypers['derivative_weights'])

        sample_times, sample_i, sample_v, response_baseline, z_scaled = sample_data
        rm_drt, induc_rv, inf_rv, zm_drt, induc_zv, drt_penalty_matrices = matrices  # partial matrices

        # Offset voltage baseline
        if offset_baseline:
            self.scaled_response_offset = -response_baseline
        else:
            self.scaled_response_offset = 0
        print('scaled_response_offset:', self.scaled_response_offset * self.response_signal_scale)
        rv = self.scaled_response_signal + self.scaled_response_offset

        # Set lambda_0 and iw_0 based on sample size and density
        # chrono_ppd = get_time_ppd(sample_times, self.step_times) * np.sqrt(2)
        # chrono_num = len(sample_times[sample_times >= self.step_times[0]])
        # eis_ppd = 2 * get_ppd(frequencies)
        # eis_num = 2 * len(frequencies)

        chrono_ppd = pp.get_time_ppd(sample_times, self.step_times) * info_factor
        chrono_num = len(sample_times[sample_times >= self.step_times[0]])
        eis_ppd = np.sqrt(2) * pp.get_ppd(frequencies)
        eis_num = np.sqrt(2) * len(frequencies)

        chrono_data_factor = qphb.get_data_factor(chrono_num, chrono_ppd)
        eis_data_factor = qphb.get_data_factor(eis_num, eis_ppd)

        num_decades = pp.get_num_decades(frequencies, sample_times, self.step_times)
        print('num_decades:', num_decades)
        tot_num = eis_num + chrono_num
        tot_ppd = (tot_num - 1) / num_decades
        tot_data_factor = qphb.get_data_factor(tot_num, tot_ppd)  # num_decades / np.sqrt(tot_num)
        # tot_data_factor = 0.5 * eis_data_factor + 0.5 * chrono_data_factor

        print('eis_ppd:', eis_ppd)
        print('chrono_ppd:', chrono_ppd)
        print('tot_ppd:', tot_ppd)

        print('eis_data_factor:', eis_data_factor)
        print('chrono_data_factor:', chrono_data_factor)
        print('tot_data_factor:', tot_data_factor)

        # Get default hyperparameters and update with user-specified values
        qphb_hypers = qphb.get_default_hypers(tot_data_factor, eff_hp)
        qphb_hypers.update(kw)

        # Store fit kwargs for reference (after prep_for_fit creates self.fit_kwargs)
        self.fit_kwargs.update(qphb_hypers)
        self.fit_kwargs['nonneg'] = nonneg
        # self.fit_kwargs.update(
        #     {
        #         'nonneg': nonneg,
        #         'derivative_weights': derivative_weights,
        #         'rho_alpha': rho_alpha,
        #         'rho_0': rho_0,
        #         's_alpha': s_alpha,
        #         's_0': s_0,
        #         'w_alpha': w_alpha,
        #         'w_beta': w_beta,
        #         'vz_offset': vz_offset,
        #         'vz_offset_scale': vz_offset_scale
        #     }
        # )

        # # Update rp_scale
        # if rp_scale is None:
        #     rp_scale = 14 * chrono_data_factor ** -1
        #     rv *= rp_scale
        #     self.scaled_response_offset *= rp_scale
        #     z_scaled *= rp_scale
        #     self.update_data_scale(rp_scale)
        # print('rp_scale:', rp_scale)

        # if l2_lambda_0 is None:
        #     # lambda_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     # l2_lambda_0 = 142 * (reduce_factor * chrono_data_factor) ** -1
        #     l2_lambda_0 = 142 * tot_data_factor ** -1
        print('l2_lambda_0:', qphb_hypers['l2_lambda_0'])

        # Scale vz_offset_scale to penalty strength
        vz_offset_scale = vz_offset_scale * qphb_hypers['l2_lambda_0']

        # if chrono_iw_beta is None:
        #     # iw_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     # chrono_iw_beta = 0.5 * (reduce_factor * chrono_data_factor) ** 2
        #     chrono_iw_beta = 0.5 * tot_data_factor ** 2
        #
        # if eis_iw_beta is None:
        #     # iw_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
        #     # eis_iw_beta = 0.5 * eis_data_factor ** 2
        #     eis_iw_beta = 0.5 * tot_data_factor ** 2
        eis_iw_beta = qphb_hypers.get('eis_iw_beta', qphb_hypers['iw_beta'])
        chrono_iw_beta = qphb_hypers.get('chrono_iw_beta', qphb_hypers['iw_beta'])

        print('eis_iw_beta:', eis_iw_beta)
        print('chrono_iw_beta:', chrono_iw_beta)

        # Format matrices for QP fit
        rm, zm, penalty_matrices = self._format_qp_matrices(rm_drt, inf_rv, induc_rv, zm_drt, induc_zv,
                                                            drt_penalty_matrices, v_baseline_penalty, R_inf_penalty,
                                                            inductance_penalty, vz_offset_scale, inductance_scale,
                                                            penalty_type, qphb_hypers['derivative_weights'])

        # Construct hybrid response-impedance matrix
        rzm = np.vstack((rm, zm))

        # Make a copy for vz_offset calculation
        rzm_vz = rzm.copy()
        # Remove v_baseline from rzm_vz - don't want to scale the baseline, only the delta
        rzm_vz[:, self.special_qp_params['v_baseline']['index']] = 0

        # self.rzm = rzm.copy()

        # Construct hybrid response-impedance vector
        zv = np.concatenate([z_scaled.real, z_scaled.imag])
        rzv = np.concatenate([rv, zv])

        # Construct lambda vectors
        l1_lambda_vector = np.zeros(rzm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        l1_lambda_vector[self.get_qp_mat_offset():] = qphb_hypers['l1_lambda_0']

        # Initialize s and rho vectors at prior mode
        k_range = len(qphb_hypers['derivative_weights'])
        rho_vector = np.ones(k_range) * qphb_hypers['rho_0']
        s_vectors = [np.ones(rzm.shape[1]) * qphb_hypers['s_0']] * k_range

        # Initialize x near zero
        x = np.zeros(rzm.shape[1]) + 1e-6

        # Construct matrices for variance estimation
        chrono_vmm = mat1d.construct_chrono_var_matrix(sample_times, self.step_times,
                                                       chrono_vmm_epsilon,
                                                       chrono_error_structure)
        eis_vmm = mat1d.construct_eis_var_matrix(frequencies, eis_vmm_epsilon, eis_reim_cor,
                                                 eis_error_structure)
        vmm = np.zeros((len(rzv), len(rzv)))
        vmm[:len(sample_times), :len(sample_times)] = chrono_vmm
        vmm[len(sample_times):, len(sample_times):] = eis_vmm
        # print(vmm[0])

        # Initialize data weight (IMPORTANT)
        # ----------------------------------
        # if chrono_iw_beta is None:
        #     chrono_iw_beta_tmp = 1
        # else:
        #     chrono_iw_beta_tmp = chrono_iw_beta
        #
        # if eis_iw_beta is None:
        #     eis_iw_beta_tmp = 1
        # else:
        #     eis_iw_beta_tmp = eis_iw_beta

        # Initialize chrono and eis weights separately
        chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = \
            qphb.initialize_weights(penalty_matrices, penalty_type, qphb_hypers['derivative_weights'], rho_vector,
                                    s_vectors, rv, rm, chrono_vmm, nonneg, self.special_qp_params,
                                    qphb_hypers.get('chrono_iw_alpha', qphb_hypers['iw_alpha']),
                                    qphb_hypers.get('chrono_iw_beta', qphb_hypers['iw_beta']), qphb_hypers['outlier_p'])

        eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = \
            qphb.initialize_weights(penalty_matrices, penalty_type, qphb_hypers['derivative_weights'], rho_vector,
                                    s_vectors, zv, zm, eis_vmm, nonneg, self.special_qp_params,
                                    qphb_hypers.get('eis_iw_alpha', qphb_hypers['iw_alpha']),
                                    qphb_hypers.get('eis_iw_beta', qphb_hypers['iw_beta']), qphb_hypers['outlier_p'])

        # est_weights, init_weights, x_overfit = qphb.initialize_weights(penalty_matrices, derivative_weights,
        #                                                                rho_vector,
        #                                                                s_vectors, rzv, rzm, vmm,
        #                                                                l1_lambda_vector,
        #                                                                nonneg, self.special_qp_params,
        #                                                                eis_iw_alpha,
        #                                                                eis_iw_beta)
        #
        # chrono_est_weights = est_weights[:len(rv)]
        # eis_est_weights = est_weights[len(rv):]
        # chrono_init_weights = init_weights[:len(rv)]
        # eis_init_weights = init_weights[len(rv):]

        # eis_est_weights = eis_est_weights_raw * eis_weight_factor
        # eis_init_weights = eis_init_weights_raw * eis_weight_factor

        eis_weight_scale = np.mean(eis_est_weights ** -2) ** -0.5
        chrono_weight_scale = np.mean(chrono_est_weights ** -2) ** -0.5

        print('w_eis / w_ci:', eis_weight_scale / chrono_weight_scale)

        if eis_weight_factor is None:
            # eis_weight_factor = (eis_data_factor / tot_data_factor) ** 0.5
            eis_weight_factor = (chrono_weight_scale / eis_weight_scale) ** 0.25
            # if eis_weight_scale > chrono_weight_scale:
            #     eis_weight_factor = (chrono_weight_scale / eis_weight_scale) ** 0.5
            # else:
            #     eis_weight_factor = 1

        # else:
        #     chrono_weight_factor = 1 #eis_weight_factor ** -1
        if chrono_weight_factor is None:
            # chrono_weight_factor = (chrono_data_factor / tot_data_factor) ** 0.5
            chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.25
            # if chrono_weight_scale > eis_weight_scale:
            #     chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.5
            # else:
            #     chrono_weight_factor = 1

        print('eis weight factor:', eis_weight_factor)
        print('chrono weight factor:', chrono_weight_factor)

        est_weights = np.concatenate([chrono_est_weights, eis_est_weights])
        init_weights = np.concatenate([chrono_init_weights, eis_init_weights])
        # if qphb_hypers.get('outlier_p', None) is not None:
        outlier_t = np.concatenate([chrono_outlier_t, eis_outlier_t])

        weights = init_weights

        print('Initial chrono weight:', np.mean(chrono_init_weights))
        print('Initial EIS weight:', np.mean(eis_init_weights))
        # print('Initial Rp:', np.sum(x_overfit[3:]) * (np.pi ** 0.5 / self.tau_epsilon) * self.coefficient_scale)
        # print('Initial R_inf:', x_overfit[2] * self.coefficient_scale)

        # Initialize xmx_norms at 1
        xmx_norms = [1] * 3

        self.qphb_history = []
        it = 0

        while it < max_iter:

            x_in = x.copy()

            # current_eis_ws = np.mean(weights[len(rv):] ** -2) ** -0.5
            # current_chrono_ws = np.mean(weights[:len(rv)] ** -2) ** -0.5
            # print(it, current_chrono_ws / chrono_weight_scale, current_eis_ws / eis_weight_scale)

            # Apply weight factors
            weights[:len(rv)] *= chrono_weight_factor
            weights[len(rv):] *= eis_weight_factor

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and scale_data and update_scale:
                # Get scale factor
                rp = np.sum(x[self.get_qp_mat_offset():]) * np.pi ** 0.5 / self.tau_epsilon
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                print('scale factor:', scale_factor)
                # Update data and qphb parameters to reflect new scale
                x_in *= scale_factor
                rzv *= scale_factor
                xmx_norms *= scale_factor  # shouldn't this be scale_factor ** 2?
                est_weights /= scale_factor
                init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                x_overfit_chrono *= scale_factor
                x_overfit_eis *= scale_factor
                # x_overfit *= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            x, s_vectors, rho_vector, weights, outlier_t, cvx_result, converged = qphb.iterate_qphb(x_in, s_vectors,
                                                                                                    rho_vector, rzv,
                                                                                                    weights,
                                                                                                    est_weights,
                                                                                                    outlier_t, rzm, vmm,
                                                                                                    penalty_matrices,
                                                                                                    penalty_type,
                                                                                                    l1_lambda_vector,
                                                                                                    qphb_hypers, eff_hp,
                                                                                                    xmx_norms, None,
                                                                                                    None, None, nonneg,
                                                                                                    self.special_qp_params,
                                                                                                    xtol, 1,
                                                                                                    self.qphb_history)

            # Normalize to ordinary ridge solution
            if it == 0:
                # Only include DRT penalty in XMX norms
                x_drt = x[self.get_qp_mat_offset():]
                xmx_norms = np.array([x_drt.T @ drt_penalty_matrices[f'm{n}'] @ x_drt for n in range(k_range)])
                print('xmx', xmx_norms)

            if vz_offset:
                # Update the response matrix with the current predicted y vector
                # vz_offset offsets chrono and eis predictions
                y_hat = rzm_vz @ x
                vz_sep = y_hat.copy()
                vz_sep[len(rv):] *= -1  # vz_offset > 0 means EIS Rp smaller chrono Rp
                rzm[:, self.special_qp_params['vz_offset']['index']] = vz_sep

            if converged:
                break
            elif it == max_iter - 1:
                warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

            it += 1

        # Store the scaled weights for MAP sampling and reference
        scaled_weights = weights.copy()
        scaled_weights[:len(rv)] *= chrono_weight_factor
        scaled_weights[len(rv):] *= eis_weight_factor

        # Store QPHB diagnostic parameters
        p_matrix, q_vector = qphb.calculate_pq(rzm, rzv, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                               rho_vector, s_vectors, scaled_weights)

        post_lp = qphb.evaluate_posterior_lp(x, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                             rho_vector, s_vectors, weights, rzm, rzv, xmx_norms)

        self.qphb_params = {'est_weights': est_weights.copy(),
                            'init_weights': init_weights.copy(),
                            'weights': scaled_weights.copy(),  # scaled weights
                            'true_weights': weights.copy(),  # unscaled weights
                            'xmx_norms': xmx_norms.copy(),
                            'x_overfit_chrono': x_overfit_chrono,
                            'x_overfit_eis': x_overfit_eis,
                            # 'x_overfit': x_overfit,
                            'p_matrix': p_matrix,
                            'q_vector': q_vector,
                            'rho_vector': rho_vector,
                            's_vectors': s_vectors,
                            'outlier_t': outlier_t,
                            'vmm': vmm,
                            'l1_lambda_vector': l1_lambda_vector,
                            'posterior_lp': post_lp,
                            'rm': rzm,
                            'rv': rzv,
                            'penalty_matrices': penalty_matrices,
                            }

        # Store final cvxopt result
        self.cvx_result = cvx_result

        # Get sigma vector from weights
        sigma_vec = (weights ** -1)
        v_sigma = sigma_vec[:len(rv)] * self.response_signal_scale
        z_sigma = sigma_vec[len(rv):len(rv) + len(z)] \
                  + 1j * sigma_vec[len(rv) + len(z):]
        z_sigma *= self.impedance_scale

        # Extract model parameters
        x_out = np.array(list(cvx_result['x']))
        self.fit_parameters = self.extract_qphb_parameters(x_out)
        self.fit_parameters['v_sigma_tot'] = v_sigma
        self.fit_parameters['v_sigma_res'] = None
        self.fit_parameters['z_sigma_tot'] = z_sigma

        self.fit_type = 'qphb_hybrid'

    # def hybrid_fit(self, times, i_signal, v_signal, frequencies, z, step_times=None, nonneg=True, scale_signal=True,
    #                z_weight=None, v_weight=None, offset_baseline=True, offset_steps=False, downsample=True,
    #                downsample_kw=None,
    #                smooth_inf_response=True,
    #                init_from_ridge=False, ridge_kw={},
    #                adjust_vz_factor=False, log_vz_factor_scale=0.05,
    #                model_name=None, add_stan_data={}, init_values=None,
    #                # optimization control
    #                max_iter=10000, random_seed=1234, stan_kw={}):
    #     # Process data and calculate matrices for fit
    #     sample_data, matrices = self._prep_for_hybrid_fit(times, i_signal, v_signal, frequencies, z, step_times,
    #                                                       downsample, downsample_kw, scale_signal, offset_steps,
    #                                                       'discrete', smooth_inf_response)
    #
    #     sample_times, sample_i, sample_v, response_baseline, z_scaled = sample_data
    #     rm, zm, induc_rv, inf_rv, induc_zv, penalty_matrices = matrices
    #
    #     # If no EIS weights provided, set weights to account for dataset size ratio
    #     if z_weight is None:
    #         z_weight = len(sample_times) / len(z)
    #         print('z_weight:', z_weight)
    #
    #     if v_weight is None:
    #         v_weight = 1
    #
    #     # Get initial values from ridge fit
    #     # TODO: sequence correctly such that R_inf_scale and inductance_scale can be updated by add_stan_data
    #     if init_from_ridge:
    #         scaled_response_offset, ridge_values = self.get_init_from_ridge(times, i_signal, v_signal, frequencies, z,
    #                                                                  step_times, z_weight, 'hybrid', nonneg=nonneg,
    #                                                                  downsample=downsample,
    #                                                                  downsample_kw=downsample_kw,
    #                                                                  scale_signal=scale_signal,
    #                                                                  offset_steps=offset_steps,
    #                                                                  smooth_inf_response=smooth_inf_response,
    #                                                                  adjust_vz_factor=adjust_vz_factor,
    #                                                                  vz_factor_scale=log_vz_factor_scale,
    #                                                                  **ridge_kw)
    #     else:
    #         # Offset voltage baseline
    #         if offset_baseline:
    #             scaled_response_offset = -response_baseline
    #         else:
    #             scaled_response_offset = 0
    #
    #     sample_v += scaled_response_offset
    #
    #     # Prepare data for Stan model
    #     stan_data = {
    #         # Dimensions
    #         'N_t': len(sample_times),
    #         'N_f': len(frequencies),
    #         'K': len(self.basis_tau),
    #
    #         # Data
    #         'V': sample_v * v_weight,
    #         'Z': np.concatenate([z_scaled.real, z_scaled.imag]) * z_weight,
    #         # 'times': sample_times,
    #
    #         # Response matrices
    #         'A_t': rm * v_weight,
    #         'A_f': np.vstack([zm.real, zm.imag]) * z_weight,
    #
    #         # Response vectors
    #         'induc_rv': induc_rv * v_weight,
    #         'inf_rv': inf_rv * v_weight,
    #         'induc_zv': np.concatenate([induc_zv.real, induc_zv.imag]) * z_weight,
    #         'inf_zv': np.concatenate([np.ones(len(z)), np.zeros(len(z))]) * z_weight,
    #         'L0': 1.5 * 0.24 * penalty_matrices['l0'],
    #         'L1': 1.5 * 0.16 * penalty_matrices['l1'],
    #         'L2': 1.5 * 0.08 * penalty_matrices['l2'],
    #
    #         # Error scale
    #         'v_sigma_min': 1e-4,
    #         'v_sigma_res_scale': 1 * v_weight,  # 0.1 * np.sqrt(len(sample_times) / 100),
    #         'z_sigma_min': 1e-4,
    #         'z_sigma_scale': 0.01 * z_weight / self.coefficient_scale,  # z_sigma_scale must scale with z_weight
    #         'z_alpha_scale': 0.005,  # z_alpha_scale gets multiplied by Z_hat - doesn't scale with z_weight
    #
    #         # Complexity hyperparameters
    #         'ups_alpha': 0.05,
    #         'ups_beta': 0.1,
    #         'ups_scale': 0.15,
    #         'ds_alpha': 5,
    #         'ds_beta': 5,
    #
    #         # Offset scales
    #         'R_inf_scale': 100,
    #         'inductance_scale': 1,
    #         'v_baseline_scale': 100 * v_weight,
    #
    #         # z factor scale
    #         'log_Z_factor_scale': log_vz_factor_scale,
    #         'log_VZ_factor_scale': log_vz_factor_scale,
    #     }
    #
    #     stan_data.update(add_stan_data)
    #
    #     self.stan_input = stan_data.copy()
    #
    #     print('Finished prepping stan_data')
    #
    #     # Update init values with offset scales
    #     ridge_values['R_inf_raw'] = ridge_values['R_inf'] / stan_data['R_inf_scale']
    #     if self.fit_inductance:
    #         ridge_values['inductance_raw'] = ridge_values['inductance'] / stan_data['inductance_scale']
    #
    #     # Update with user-supplied inits if provided
    #     if init_values is not None:
    #         ridge_values.update(init_values)
    #         init_values = ridge_values
    #     else:
    #         init_values = ridge_values
    #
    #     self._init_values = init_values
    #
    #     # load Stan model
    #     if model_name is None:
    #         # Construct model name from arguments
    #         if self.op_mode == 'galvanostatic':
    #             model_name = 'Hybrid_Galv'
    #         elif self.op_mode == 'potentiostatic':
    #             model_name = 'Hybrid_Pot'
    #         else:
    #             raise ValueError
    #
    #         model_name += 'SquareWaveDRT'
    #
    #         if nonneg:
    #             model_name += '_pos'
    #
    #         if not self.fit_inductance:
    #             model_name += '_noL'  # no inductance
    #
    #         if adjust_vz_factor:
    #             model_name += '_VZFactor'
    #
    #         model_name += '.stan'
    #     self.stan_model_name = model_name
    #
    #     # stan_model = load_pickle(os.path.join(module_dir, 'stan_model_files', model_name))
    #     stan_model = CmdStanModel(stan_file=os.path.join(module_dir, 'stan_model_files', model_name))
    #     print('Loaded stan model')
    #     # Fit model
    #     print('stan data check:', utils.array.check_equality(self.stan_input, stan_data))
    #     self.stan_mle = stan_model.optimize(stan_data, iter=max_iter, seed=random_seed, inits=init_values,
    #                                         algorithm='lbfgs', **stan_kw)
    #     self.stan_result = self.stan_mle.stan_variables()
    #     # self.stan_result = stan_model.optimizing(stan_data, iter=max_iter, seed=random_seed, init=init_values)
    #     print('Optimized model')
    #     # Extract model parameters
    #     self.fit_parameters = {'x': self.stan_result['x'] * self.coefficient_scale,
    #                            'R_inf': self.stan_result['R_inf'] * self.coefficient_scale,
    #                            'v_baseline': (
    #                                                  self.stan_result['v_baseline'] / v_weight - scaled_response_offset
    #                                          ) * self.response_signal_scale
    #                            }
    #
    #     for param in ['v_sigma_tot', 'v_sigma_res']:
    #         self.fit_parameters[param] = self.stan_result[param] * self.response_signal_scale / v_weight
    #     for param in ['z_sigma_tot', 'z_sigma_res', 'z_alpha_prop', 'z_alpha_re', 'z_alpha_im']:
    #         self.fit_parameters[param] = self.stan_result[param] * self.impedance_scale
    #
    #     # Convert z_sigma_tot to complex
    #     self.fit_parameters['z_sigma_tot'] = self.fit_parameters['z_sigma_tot'][:len(z)] \
    #                                          + 1j * self.fit_parameters['z_sigma_tot'][len(z):]
    #
    #     if adjust_vz_factor:
    #         self.fit_parameters['vz_factor'] = self.stan_result['vz_factor']
    #
    #     if self.fit_inductance:
    #         self.fit_parameters['inductance'] = self.stan_result['inductance'] * self.coefficient_scale
    #     else:
    #         self.fit_parameters['inductance'] = 0
    #
    #     self.fit_type = 'hybrid_chb'

    # # Initialization
    # # -------------------------------------------
    # def get_init_from_ridge(self, times, i_signal, v_signal, frequencies, z, step_times, z_weight, fit_type,
    #                         **ridge_kw):
    #     # User over-penalized ridge to initialize CHB fit
    #     # This avoids trapping the CHB fit in a deep local minimum
    #     ridge_defaults = dict(l2_lambda_0=10, hyper_l2_lambda=True, hl_l2_beta=100, l1_lambda_0=1)
    #     ridge_defaults.update(ridge_kw)
    #
    #     if fit_type == 'normal':
    #         self.ridge_fit(times, i_signal, v_signal, step_times, **ridge_defaults)
    #     elif fit_type == 'hybrid':
    #         self.hybrid_ridge_fit(times, i_signal, v_signal, frequencies, z, step_times, eis_weights=z_weight,
    #                               **ridge_defaults)
    #
    #     init_values = {'x': self.fit_parameters['x'] / self.coefficient_scale,
    #                    'R_inf': self.fit_parameters['R_inf'] / self.coefficient_scale,
    #                    'inductance': self.fit_parameters['inductance'] / self.coefficient_scale,
    #                    }
    #     response_offset = -self.fit_parameters['v_baseline'] / self.response_signal_scale
    #
    #     if ridge_defaults.get('adjust_vz_factor', False):
    #         vz_factor_scale = ridge_defaults.get('vz_factor_scale', 0.1)
    #         # init_values['log_z_factor_raw'] = np.log(self.fit_parameters['z_factor']) / z_factor_scale
    #         init_values['log_vz_factor_raw'] = np.log(self.fit_parameters['vz_factor']) / vz_factor_scale
    #
    #     return response_offset, init_values

    # def get_hybrid_init_from_ridge(self, times, i_signal, v_signal, frequencies, z, **ridge_kw):
    #	  # User over-penalized ridge to initialize CHB fit
    #	  # This avoids trapping the CHB fit in a deep local minimum
    #	  ridge_defaults = dict(l2_lambda_0=10, hyper_l2_lambda=True, hl_l2_beta=100, l1_lambda_0=1)
    #	  ridge_defaults.update(ridge_kw)
    #	  self.hybrid_ridge_fit(times, i_signal, v_signal, frequencies, z, **ridge_defaults)
    #
    #	  init_values = {'x': self.fit_parameters['x'] / self.coefficient_scale,
    #					 'R_inf': self.fit_parameters['R_inf'] / self.coefficient_scale,
    #					 'inductance': self.fit_parameters['inductance'] / self.coefficient_scale,
    #					 }
    #	  scaled_response_offset = -self.fit_parameters['v_baseline'] / self.response_signal_scale
    #
    #	  return scaled_response_offset, init_values

    # Prediction
    # --------------------------------------------
    def get_drt_params(self, x):
        if x is not None:
            if len(x) > len(self.basis_tau):
                x = self.extract_qphb_parameters(x)['x']
        else:
            x = self.fit_parameters['x']

        return x

    def predict_distribution(self, tau=None, ppd=20, x=None, order=0, normalize=False):
        """
        Predict distribution as function of tau
        :param ndarray tau: tau values at which to evaluate the distribution
        :return: array of distribution density values
        """
        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            tau = self.get_tau_eval(ppd)

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), np.log(tau),
                                                        self.tau_basis_type, epsilon=self.tau_epsilon,
                                                        order=order, zga_params=self.zga_params)

        x = self.get_drt_params(x)

        if normalize:
            normalize_by = self.predict_r_p()
        else:
            normalize_by = 1

        return basis_matrix @ x / normalize_by

    def estimate_distribution_cov(self, tau=None, ppd=20, p_matrix=None, order=0, normalize=False, var_floor=0.0,
                                  tau_data_limits=None, extrapolation_var_scale=0.0):
        """
        Predict distribution as function of tau
        :param ndarray tau: tau values at which to evaluate the distribution
        :return: array of distribution density values
        """
        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            tau = self.get_tau_eval(ppd)

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), np.log(tau),
                                                        self.tau_basis_type, epsilon=self.tau_epsilon,
                                                        order=order, zga_params=self.zga_params)

        if normalize:
            normalize_by = self.predict_r_p() ** 2
        else:
            normalize_by = 1

        # Get parameter covariance
        x_cov = self.estimate_param_cov(p_matrix)

        if x_cov is not None:
            # Limit to DRT parameters
            x_cov = x_cov[self.get_qp_mat_offset():, self.get_qp_mat_offset():]

            # Distribution covariance given by matrix product
            dist_cov = basis_matrix @ x_cov @ basis_matrix.T

            # Normalize
            dist_cov = dist_cov / normalize_by

            # Add variance beyond measurement bounds
            if extrapolation_var_scale > 0:
                add_var = np.zeros(len(tau))
                if tau_data_limits is None:
                    tau_data_limits = pp.get_tau_lim(self.get_fit_frequencies(True), self.get_fit_times(True),
                                                     self.step_times)
                t_start, t_end = tau_data_limits
                add_var[tau > t_end] = np.log(tau[tau > t_end] / t_end) ** 2
                add_var[tau < t_start] = (-np.log(tau[tau < t_start] / t_start)) ** 2
                add_var *= extrapolation_var_scale
                dist_cov += np.diag(add_var)

            # Set variance floor
            if var_floor > 0:
                dist_var = np.diag(dist_cov).copy()
                dist_var[dist_var < var_floor] = var_floor
                np.fill_diagonal(dist_cov, dist_var)

            return dist_cov
        else:
            # Error in precision matrix inversion
            return None

    def predict_distribution_ci(self, tau=None, ppd=20, x=None, x_cov=None, order=0, normalize=False,
                                quantiles=[0.025, 0.975]):
        # Get distribution std
        dist_cov = self.estimate_distribution_cov(tau, ppd, x_cov, order, normalize)
        if dist_cov is not None:
            dist_sigma = np.diag(dist_cov) ** 0.5

            # Distribution mean
            dist_mu = self.predict_distribution(tau, ppd, x, order, normalize)

            # Determine number of std devs to obtain quantiles
            s_lo, s_hi = stats.std_normal_quantile(quantiles)

            dist_lo = dist_mu + s_lo * dist_sigma
            dist_hi = dist_mu + s_hi * dist_sigma

            return dist_lo, dist_hi
        else:
            # Error in covariance calculation
            return None, None

    def predict_response(self, times=None, input_signal=None, op_mode=None, offset_steps=None,
                         smooth_inf_response=None, x=None, include_vz_offset=True):
        # If op_mode is not provided, use fitted op_mode
        if op_mode is None:
            op_mode = self.op_mode
        utils.validation.check_op_mode(op_mode)

        # If times is not provided, use self.t_fit
        if times is None:
            times = self.get_fit_times()

        # If kwargs not provided, use same values used in fitting
        if offset_steps is None:
            offset_steps = self.fit_kwargs['offset_steps']
        if smooth_inf_response is None:
            smooth_inf_response = self.fit_kwargs['smooth_inf_response']

        # Get prediction matrix and vectors
        rm, induc_rv, inf_rv = self._prep_chrono_prediction_matrix(times, input_signal, op_mode, offset_steps,
                                                                   smooth_inf_response)
        # Response matrices from _prep_response_prediction_matrix will be scaled. Rescale to data scale
        # rm *= self.input_signal_scale
        # induc_rv *= self.input_signal_scale
        # inf_rv *= self.input_signal_scale

        # Get parameters
        if x is not None:
            fit_parameters = self.extract_qphb_parameters(x)
        else:
            fit_parameters = self.fit_parameters

        x = fit_parameters['x']
        r_inf = fit_parameters['R_inf']
        induc = fit_parameters['inductance']
        v_baseline = fit_parameters['v_baseline']

        response = rm @ x + inf_rv * r_inf + induc * induc_rv

        if include_vz_offset:
            # # Need to back out the offset to get the fit parameter, apply offset to fit parameter by offset
            # v_baseline_param = (v_baseline + self.scaled_response_offset * self.response_signal_scale)
            # v_baseline_offset = v_baseline_param * (1 + self.fit_parameters.get('vz_offset', 0)) \
            #                     - self.scaled_response_offset * self.response_signal_scale
            # Apply offset before adding baseline
            response *= (1 + self.fit_parameters.get('vz_offset', 0))

        response += v_baseline

        return response

    def predict_impedance(self, frequencies, include_vz_offset=True, x=None):
        # Get matrix
        zm = self._prep_impedance_prediction_matrix(frequencies)

        # Get parameters
        if x is not None:
            fit_parameters = self.extract_qphb_parameters(x)
        else:
            fit_parameters = self.fit_parameters

        x = fit_parameters['x']
        r_inf = fit_parameters['R_inf']
        induc = fit_parameters['inductance']

        z = zm @ x + r_inf + induc * 2j * np.pi * frequencies

        if include_vz_offset:
            z *= (1 - self.fit_parameters.get('vz_offset', 0))

        return z

    def predict_sigma(self, measurement):
        if measurement == 'chrono':
            key = 'v_sigma_tot'
        elif measurement == 'eis':
            key = 'z_sigma_tot'

        return self.fit_parameters.get(key, None)

    def predict_r_p(self, x=None):
        basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)
        x = self.get_drt_params(x)
        return np.sum(x) * basis_area

    def predict_r_tot(self):
        return self.fit_parameters['R_inf'] + self.predict_r_p()

    def integrate_distribution(self, tau_min, tau_max, ppd=10, **predict_kw):
        num_decades = np.log10(tau_max) - np.log10(tau_min)
        tau = np.logspace(np.log10(tau_min), np.log10(tau_max), int(num_decades * ppd) + 1)
        gamma = self.predict_distribution(tau, **predict_kw)
        return np.trapz(gamma, x=np.log(tau))

    def split_r_p(self, tau_splits, resolve_peaks=False, **predict_kw):
        tau_splits = sorted(tau_splits)
        if 'tau' not in predict_kw.keys():
            ppd = predict_kw.get('ppd', 20)
            tau = self.get_tau_eval(ppd)
        else:
            tau = predict_kw.pop('tau')

        gamma = self.predict_distribution(tau, **predict_kw)

        split_index = [utils.array.nearest_index(tau, ts) for ts in tau_splits]
        start_index = np.array([0] + split_index)
        end_index = np.array(split_index + [len(tau)]) + 1

        if resolve_peaks:
            # Find min curvature in each window
            fxx = self.predict_distribution(tau, order=2, **predict_kw)
            peak_index = [np.argmin(fxx[i:j]) + i for i, j in zip(start_index, end_index)]
            # peaks.estimate_peak_weight_distributions(tau, gamma, fxx, peak_index, self.basis_tau, )
            peak_coef = self.estimate_peak_coef(tau, peak_indices=peak_index)
            split_area = np.array([self.predict_r_p(x=pc) for pc in peak_coef])
        else:
            split_area = np.array([np.trapz(gamma[i:j], x=np.log(tau[i:j])) for i, j in zip(start_index, end_index)])

        return split_area

    # def evaluate_residuals(self):
    #     # Get fitted response
    #     y_hat = self.predict_response(**self.fit_kwargs)
    #
    #     return y_hat - self.raw_response_signal

    # ----------------------------------------------------
    # Peak finding
    # ----------------------------------------------------
    def find_peaks(self, tau=None, x=None, normalize=True, ppd=10, prominence=None, return_index=False, **kw):

        # If tau is not provided, go one decade beyond basis_tau with standard spacing
        # Finer spacing can cause many minor peaks to appear in 2nd derivative
        if tau is None:
            tau = self.get_tau_eval(ppd)

        f = self.predict_distribution(tau=tau, x=x, order=0)
        fx = self.predict_distribution(tau=tau, x=x, order=1)
        fxx = self.predict_distribution(tau=tau, x=x, order=2)

        if normalize:
            fx = fx / self.predict_r_p()
            fxx = fxx / self.predict_r_p()

        # peak_indices = peaks.find_peaks(fx, fxx, fx_kw, fxx_kw)
        # if fxx_kw is None:
        #     fxx_kw = {}

        # f_mix = 0.5 * (f - fxx)

        # if prominence is None:
        #     prominence = -np.min(fxx) * 2.5e-2
        #     print('prom:', prominence)

        if prominence is None:
            # prominence = np.median(np.abs(fxx)) + 0.05 * np.percentile(-fxx[fxx > -np.inf], 95)
            prominence = 0.05 * np.std(fxx[~np.isinf(fxx)]) + 5e-3
            # print(prominence)

        peak_indices = peaks.find_peaks_simple(fxx, order=2, height=0, prominence=prominence, **kw)
        # peak_indices = peaks.find_peaks_compound(fx, fxx, **kw)

        if return_index:
            return tau[peak_indices], tau, peak_indices
        else:
            return tau[peak_indices]

    def estimate_peak_coef(self, tau=None, peak_indices=None, x=None, epsilon_factor=1.25, max_epsilon=1.25,
                           epsilon_uniform=None,
                           **find_peaks_kw):
        if peak_indices is not None and tau is None:
            raise ValueError('If peak_indices are provided, the corresponding tau grid must also be provided')

        x = self.get_drt_params(x)

        if peak_indices is None:
            _, tau, peak_indices = self.find_peaks(x=x, return_index=True, **find_peaks_kw)

        f = self.predict_distribution(tau, x=x)
        fxx = self.predict_distribution(tau, x=x, order=2)
        peak_weights = peaks.estimate_peak_weight_distributions(tau, f, fxx, peak_indices, self.basis_tau,
                                                                epsilon_factor, max_epsilon, epsilon_uniform)

        x_peaks = x * peak_weights

        return x_peaks

    def estimate_peak_distributions(self, tau=None, ppd=10, tau_find_peaks=None, peak_indices=None, x=None,
                                    epsilon_factor=1.25,
                                    max_epsilon=1.25, epsilon_uniform=None, squeeze_factors=None, find_peaks_kw=None):
        """
        Estimate separate distributions for all identified peaks by applying local weighting functions to the total DRT.
        :param ndarray tau: tau grid over which to evaluate peak distributions
        :param int ppd: Points per decade to use for tau grid if tau is not specified
        :param ndarray tau_find_peaks: tau grid to use for peak finding. This may or may not be the same as tau;
        for example, 10 ppd is ideal for peak finding, but a finer density may be desirable for plotting.
        :param ndarray peak_indices: Indices of peaks for which to estimate distributions.
        If None, the find_peaks method will be applied to identify peaks
        :param ndarray x: DRT parameter values. If None, the fitted parameter values are used
        :param float epsilon_factor: Factor to use for automatic determination of width of weighting functions.
        Larger values will result in narrower weighting functions
        :param float max_epsilon: Maximum value of epsilon (inverse length scale) allowed when determining
        epsilon automatically
        :param float epsilon_uniform: Universal value of inverse length scale for weighting functions.
        If None, length scale will be determined separately for each peak
        :param find_peaks_kw: Keyword args to pass to find_peaks. Only used if peak_indices is None
        :return:
        """
        if tau is None:
            tau = self.get_tau_eval(ppd)

        if find_peaks_kw is None:
            find_peaks_kw = {}

        x_peaks = self.estimate_peak_coef(tau_find_peaks, peak_indices, x, epsilon_factor, max_epsilon, epsilon_uniform,
                                          **find_peaks_kw)

        if squeeze_factors is None:
            squeeze_factors = np.ones(len(x_peaks))

        peak_gammas = np.empty((x_peaks.shape[0], len(tau)))

        for i, x_peak in enumerate(x_peaks):
            squeeze_factor = squeeze_factors[i]
            if squeeze_factor != 1:
                x_peak = peaks.squeeze_peak_coef(x_peak, self.basis_tau, squeeze_factor)
            peak_gammas[i] = self.predict_distribution(tau, x=x_peak)

        return peak_gammas

    def mark_peaks(self, ax, x=None, peak_tau=None, find_peaks_kw=None, scale_prefix=None, area=None, normalize=False,
                   **plot_kw):
        if find_peaks_kw is None:
            find_peaks_kw = {}

        # Find peaks
        if peak_tau is None:
            peak_tau = self.find_peaks(x=x, **find_peaks_kw)

        gamma_peaks = self.predict_distribution(peak_tau, normalize=normalize, x=x)

        if area is not None:
            gamma_peaks *= area

        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(gamma_peaks)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        ax.scatter(peak_tau, gamma_peaks / scale_factor, **plot_kw)

    def plot_peak_distributions(self, ax=None, tau=None, ppd=10, peak_gammas=None, estimate_peak_distributions_kw=None,
                                scale_prefix=None, x=None, **plot_kw):

        if estimate_peak_distributions_kw is None:
            estimate_peak_distributions_kw = {}

        if tau is None:
            tau = self.get_tau_eval(ppd)

        if peak_gammas is None:
            peak_gammas = self.estimate_peak_distributions(tau=tau, x=x, **estimate_peak_distributions_kw)

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(np.sum(peak_gammas, axis=0))
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        for gamma in peak_gammas:
            ax.plot(tau, gamma / scale_factor, **plot_kw)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (s)')
        ax.set_ylabel(fr'$\gamma$ ({scale_prefix}$\Omega$)')

        fig.tight_layout()

        return ax

    def quantify_peaks(self, tau=None, ppd=10, **estimate_peak_distributions_kw):
        if tau is None:
            tau = self.get_tau_eval(ppd)

        # Get peak distributions
        peak_gammas = self.estimate_peak_distributions(tau=tau, **estimate_peak_distributions_kw)

        # Get peak magnitudes
        r_peaks = [np.trapz(gamma, x=np.log(tau)) for gamma in peak_gammas]

        return r_peaks

    # ----------------------------------------------------
    # Parameter variance and credible interval estimation
    # ----------------------------------------------------
    def estimate_param_cov(self, p_matrix=None):
        """
        Estimate parameter covariance matrix
        :return: covariance matrix
        """
        if p_matrix is None:
            p_matrix = self.qphb_params.get('p_matrix_cov', self.qphb_params['p_matrix'])

        if self.fit_type.find('qphb') > -1:
            try:
                p_inv = np.linalg.inv(p_matrix)
                return p_inv * self.coefficient_scale ** 2
            except np.linalg.LinAlgError:
                warnings.warn('Singular P matrix - could not obtain covariance estimate')
                return None
        else:
            raise Exception('Parameter covariance estimation is only available for qphb fits')

    def generate_map_samples(self, max_iter=2, shift_frac=0.05, shift_scale=1.5, random_seed=None):
        # Generate samples if no existing samples OR sampling arguments have changed
        # kwargs = {'max_iter': max_iter, 'x_interval': x_interval, 'delta_vals': delta_vals}
        kwargs = {'max_iter': max_iter, 'shift_frac': shift_frac,
                  'shift_scale': shift_scale, 'random_seed': random_seed}
        if self.map_samples is None or not utils.array.check_equality(kwargs, self.map_sample_kw):

            # Generate MAP samples
            # Estimate covariance matrix for coefficients
            cov = self.estimate_param_cov()
            sigma_x = np.diag(cov) ** 0.5
            sigma_x /= self.coefficient_scale

            # Get estimated coefficients from regular QPHB
            x_hat = self.qphb_history[-1]['x']

            # Get args for optimization func
            arg_list = inspect.getfullargspec(qphb.optimize_lp_semi_fixed).args

            # Set kwargs for optimize func
            kwargs = dict(fixed_prior=True, special_qp_params=self.special_qp_params,
                          xtol=1e-3, max_iter=max_iter, history=None)
            kwargs.update(self.fit_kwargs)
            kwargs.update(self.qphb_params)

            # Get ordered arg values
            arg_vals = [kwargs.get(arg, None) for arg in arg_list[2:]]

            # Set floor for parameter values
            if self.fit_kwargs['nonneg']:
                x_floor = np.zeros(len(x_hat))
                for v in self.special_qp_params.values():
                    if not v['nonneg']:
                        x_floor[v['index']] = -np.inf
            else:
                x_floor = np.ones(len(x_hat)) * -np.inf

            # Perform initial optimization with fixed prior and R_inf fixed - ensures that lp_hat is correct
            # (sometimes lp increases significantly with 1-2 iterations of optimize_lp_semi_fixed)
            fixed_x_index = [self.special_qp_params['R_inf']['index']]
            x_in = x_hat.copy()
            x_init, lp_hat = qphb.optimize_lp_semi_fixed(x_in, fixed_x_index, *arg_vals)
            # print('delta x:', x_init - x_hat)
            print('delta lp:', lp_hat - self.qphb_params['posterior_lp'])
            x_hat = x_init

            # Determine number of parameters to shift in each sample
            shift_num = int(len(x_hat) * shift_frac)
            print('shift_num:', shift_num)

            # Set up random state
            rng = np.random.default_rng(seed=random_seed)

            # lp_array = np.empty(num_samples)
            # x_array = np.empty((num_samples, len(x_hat)))

            all_indices = np.arange(0, len(x_hat), dtype=int)
            possible_indices = all_indices.copy()

            # Define selection probability to disfavor selection of adjacent indices
            prob_func = basis.get_basis_func('gaussian')
            prob_eps = 0.25 * self.tau_epsilon
            prob_tau = np.concatenate(
                (
                    np.ones(len(self.special_qp_params)) * self.basis_tau[0] * 1e-2,
                    self.basis_tau
                )
            )

            def rng_prob(index):
                prob = 1 - prob_func(np.log(prob_tau / prob_tau[index]), prob_eps)
                return prob

            # index_sep = 10

            print('Sampling...')
            # for i in range(num_samples):
            #     # print(i)
            #     if len(possible_indices) == 0:
            #         possible_indices = all_indices.copy()
            #     # Select parameters to shift
            #     primary_index = rng.choice(possible_indices, 1, replace=False)
            #     possible_indices = np.delete(possible_indices, np.where(possible_indices == primary_index))
            #     select_prob = rng_prob(primary_index)
            #     select_prob /= np.sum(select_prob)
            #     # print(select_prob)
            #
            #     shift_index = [primary_index[0]]
            #     while len(shift_index) < shift_num:
            #         new_index = rng.choice(all_indices, 1, replace=False, p=select_prob)
            #         shift_index.append(new_index[0])
            #         # print(new_index, shift_index)
            #         select_prob *= rng_prob(new_index)
            #         select_prob /= np.sum(select_prob)
            #         # print(select_prob)
            #
            #     shift_index = np.array(sorted(shift_index))
            #     # print(shift_index)
            #
            #     # Get shift sizes
            #     shift_sigma = sigma_x[shift_index]
            #     delta_x = rng.normal(0, shift_sigma * shift_scale)
            #
            #     #
            #     x_in = x_hat.copy()
            #     x_in[shift_index] += delta_x
            #     x_in = np.maximum(x_in, x_floor)
            #
            #
            #     # Optimize with shifted parameters fixed in place
            #     x_array[i], lp_array[i] = qphb.optimize_lp_semi_fixed(x_in, shift_index, *arg_vals)
            #
            #     print(shift_index, delta_x, lp_array[i])

            #     if i == int(num_samples / 4):
            #         print('Sampling 25% complete')
            #     elif i == int(num_samples / 2):
            #         print('Sampling 50% complete')
            #     elif i == int(num_samples * 0.75):
            #         print('Sampling 75% complete')

            lp_array = []
            x_array = []

            for primary_index in range(len(self.special_qp_params), len(x_hat)):
                # # Add a copy of the optimal solution rather than sampling 0
                # lp_array.append(lp_hat)
                # x_array.append(x_hat)

                for delta in [-3, -2, -1, 1, 2, 3]:
                    primary_delta = delta * sigma_x[primary_index] * shift_scale
                    # Check if shift is within bounds; if not, skip
                    if x_hat[primary_index] >= x_floor[primary_index]:
                        shift_index = []
                        if shift_num > 0:
                            # Select additional indices to shift
                            select_prob = rng_prob(primary_index)
                            select_prob /= np.sum(select_prob)
                            while len(shift_index) < shift_num:
                                new_index = rng.choice(all_indices, 1, replace=False, p=select_prob)
                                shift_index.append(new_index[0])
                                # print(new_index, shift_index)
                                select_prob *= rng_prob(new_index)
                                select_prob /= np.sum(select_prob)
                                # print(select_prob)

                            # shift_index = np.array(sorted(shift_index))
                            # print(shift_index)

                            # Get shift sizes
                            shift_sigma = sigma_x[shift_index]
                            delta_x = rng.normal(0, shift_sigma)

                            #
                            x_in = x_hat.copy()
                            x_in[shift_index] += delta_x

                        # Shift the primary index
                        x_in[primary_index] += primary_delta
                        x_in = np.maximum(x_in, x_floor)

                        # Optimize with shifted parameters fixed in place
                        shift_index.append(primary_index)
                        shift_index = sorted(np.array(shift_index))
                        x_out, lp_out = qphb.optimize_lp_semi_fixed(x_in, shift_index, *arg_vals)

                        x_array.append(x_out)
                        lp_array.append(lp_out)

                if primary_index == int(len(x_hat) / 4):
                    print('Sampling 25% complete')
                elif primary_index == int(len(x_hat) / 2):
                    print('Sampling 50% complete')
                elif primary_index == int(len(x_hat) * 0.75):
                    print('Sampling 75% complete')

            print('Sampling 100% complete')

            x_array = np.array(x_array)
            lp_array = np.array(lp_array)

            # delta_array = np.concatenate((-np.array(delta_vals), np.array(delta_vals)))

            # for fixed_x_index in range(len(self.special_qp_params), len(x_hat), x_interval):
            #     if fixed_x_index % 10 == 0:
            #         print(f'Sampling parameter {fixed_x_index}')
            #
            #     lp_out = np.zeros(len(delta_array) + 1)
            #     x_out = np.empty((len(delta_array) + 1, len(x_hat)))
            #
            #     for i, delta in enumerate(delta_array):
            #         # Fix the parameter at delta std devs from its optimal value
            #         x_in = x_hat.copy()
            #         x_in[fixed_x_index] += delta * sigma_x[fixed_x_index]
            #
            #         # Optimize with fixed parameter
            #         x_out[i], lp_out[i] = qphb.optimize_lp_semi_fixed(x_in, fixed_x_index, *arg_vals)
            #
            #     # Insert values for original solution
            #     lp_out[-1] = lp_hat
            #     x_out[-1] = x_hat.copy()
            #
            #     lp_array.append(lp_out)
            #     x_array.append(x_out)

            # Concatenate log-prob values and coef vectors
            # lp_array = np.concatenate(lp_array)
            # x_array = np.vstack(x_array)
            x_array *= self.coefficient_scale

            # Store samples
            self.map_samples = {'lp': lp_array, 'x': x_array}
            # self.map_sample_kw = {'max_iter': max_iter, 'x_interval': x_interval, 'delta_vals': delta_vals}
            self.map_sample_kw = {'max_iter': max_iter, 'shift_frac': shift_frac,
                                  'shift_scale': shift_scale, 'random_seed': random_seed}

    def estimate_ci(self, method, quantiles=[0.025, 0.975], **sample_kw):
        if self.fit_type.find('qphb') > -1:
            if method == 'cov':
                # Estimate covariance matrix for coefficients
                cov = self.estimate_param_cov()
                if cov is not None:
                    sigma_x = np.diag(cov) ** 0.5

                    x_hat = np.empty(len(sigma_x))
                    for key, info in self.special_qp_params.items():
                        x_hat[info['index']] = self.fit_parameters[key]
                    x_hat[len(self.special_qp_params):] = self.fit_parameters['x']

                    # Determine number of std devs to obtain quantiles
                    s_quant = stats.std_normal_quantile(quantiles)
                    s_lo = s_quant[0]
                    s_hi = s_quant[1]

                    x_lo = x_hat + s_lo * sigma_x
                    x_hi = x_hat + s_hi * sigma_x
                else:
                    x_lo, x_hi = None, None
            elif method == 'map_sample':
                # Use QPHB algorithm to sample MAP solutions with different parameters fixed
                self.generate_map_samples(**sample_kw)

                x_array = self.map_samples['x']
                lp_array = self.map_samples['lp']
                lp_hat = self.qphb_params['posterior_lp']
                x_lo, x_hi = utils.weighted_quantile_2d(x_array, quantiles, np.exp(lp_array - lp_hat), axis=0)

        return x_lo, x_hi

    def get_parameter_quantile(self, quantile):
        if self.map_samples is not None:
            x_array = self.map_samples['x']
            lp_array = self.map_samples['lp']
            lp_hat = self.qphb_params['posterior_lp']
            if np.shape(quantile) == ():
                quantile = [quantile]
            x_out = utils.weighted_quantile_2d(x_array, quantile, np.exp(lp_array - lp_hat), axis=0)
            return x_out
        else:
            raise Exception('Map samples must be generated before estimating parameter quantiles')

    def estimate_posterior_mean(self, **sample_kw):
        self.generate_map_samples(**sample_kw)

        x_array = self.map_samples['x']
        lp_array = self.map_samples['lp']
        lp_hat = self.qphb_params['posterior_lp']
        x_mean = np.average(x_array, axis=0, weights=np.exp(lp_array - lp_hat))

        return x_mean

    # def get_lml(self):
    #     """
    #     Calculate log marginal likelihood (evidence) assuming fixed hyperparameters (s, rho)
    #     :return:
    #     """
    #     cov = np.linalg.inv(self.qphb_params['p_matrix'])
    #     det_sign, log_det = np.linalg.slogdet(cov)

    def evaluate_rss(self, weights=None, x=None, normalize=False):
        if weights is None:
            weights = self.qphb_params['est_weights']

        if x is None:
            x = self.qphb_history[-1]['x']  # Get parameters in scaled space with special params included

        rss = qphb.evaluate_rss(x, self.qphb_params['rm'], self.qphb_params['rv'], weights)

        # Normalize to number of data points
        if normalize:
            rss /= self.num_data

        return rss

    def evaluate_llh(self, weights=None, x=None, marginalize_weights=True, alpha_0=2, beta_0=1):
        if weights is None:
            weights = self.qphb_params['est_weights']

        if x is None:
            x = self.qphb_history[-1]['x']  # Get parameters in scaled space with special params included

        llh = qphb.evaluate_llh(x, self.qphb_params['rm'],
                                self.qphb_params['rv'], weights,
                                marginalize_weights=marginalize_weights, alpha_0=alpha_0, beta_0=beta_0
                                )

        return llh

    def evaluate_bic(self, x=None, find_peaks_kw=None, **llh_kw):
        # Get log-likelihood
        llh = self.evaluate_llh(x=x, **llh_kw)

        # Find peaks
        if find_peaks_kw is None:
            find_peaks_kw = {}
        peak_tau = self.find_peaks(x=x, **find_peaks_kw)

        # Determine number of parameters
        # Assume each peak is fully described by 4 parameters (size, location, dispersion, symmetry)
        num_params = self.get_qp_mat_offset() + len(peak_tau) * 4

        return stats.bic(num_params, self.num_independent_data, llh)

    def evaluate_lml(self, history_entry=None, weights=None, update_hypers=None):
        # Extract basic info from fit
        penalty_matrices = self.qphb_params['penalty_matrices']
        penalty_type = self.fit_kwargs['penalty_type']
        qphb_hypers = self.qphb_params['hypers'].copy()
        l1_lambda_vector = self.qphb_params['l1_lambda_vector']
        rm = self.qphb_params['rm']
        rv = self.qphb_params['rv']

        # Update qphb_hypers
        if update_hypers is None:
            update_hypers = {}
        qphb_hypers.update(update_hypers)

        # If history entry not provided, use final state from optimization
        if history_entry is None:
            history_entry = self.qphb_history[-1]

        x_hat = history_entry['x']
        rho_vector = history_entry['rho_vector']
        s_vectors = history_entry['s_vectors']

        if weights is None:
            weights = self.qphb_params['est_weights']

        lml = qphb.evaluate_lml(x_hat, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector, rho_vector,
                                s_vectors, weights, rm, rv)

        return lml

    # Plotting
    # --------------------------------------------
    def plot_chrono_fit(self, ax=None, transform_time=True, linear_time_axis=True,
                        plot_data=True, data_kw=None, data_label='',
                        scale_prefix=None, predict_kw={}, c='k', **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # Get fit times
        times = self.get_fit_times()

        # Transform time to visualize each step on a log scale
        if transform_time:
            # Transforms are not defined for times outside input times.
            # Must pad times to prevent issues with transforms if making secondary x-axis
            x, trans_functions = get_transformed_plot_time(times, self.step_times, linear_time_axis)
        else:
            x = times

        # Add linear time axis
        if linear_time_axis and transform_time:
            axt = add_linear_time_axis(ax, self.step_times, trans_functions)

        # Get fitted response
        y_hat = self.predict_response(**predict_kw)

        # Get appropriate scale
        if scale_prefix is None:
            if plot_data:
                scale_prefix = utils.scale.get_common_scale_prefix([y_hat, self.raw_response_signal])
            else:
                scale_prefix = utils.scale.get_scale_prefix(y_hat)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot data
        if plot_data:
            ax.scatter(x, self.raw_response_signal / scale_factor, label=data_label, **data_kw)

        # Plot fitted response
        ax.plot(x, y_hat / scale_factor, c=c, **kw)

        # Labels
        if transform_time:
            ax.set_xlabel('$f(t)$')
        else:
            ax.set_xlabel('$t$ (s)')

        if self.op_mode == 'galvanostatic':
            ax.set_ylabel(f'$v$ ({scale_prefix}V)')
        elif self.op_mode == 'potentiostatic':
            ax.set_ylabel(f'$i$ ({scale_prefix}A)')

        fig.tight_layout()

        return ax

    def plot_chrono_residuals(self, plot_sigma=True, ax=None, x_axis='index', linear_time_axis=True, predict_kw={},
                              s=10, alpha=0.5, scale_prefix=None, **kw):

        # Check x_axis string
        x_axis_options = ['t', 'f(t)', 'index']
        if x_axis not in x_axis_options:
            raise ValueError(f'Invalid x_axis option {x_axis}. Options: {x_axis_options}')

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        # Get fit times
        times = self.get_fit_times()

        if x_axis == 'f(t)':
            # Transform time to visualize each step on a log scale
            x, trans_functions = get_transformed_plot_time(times, self.step_times, linear_time_axis)
            ax.set_xlabel('$f(t)$')
        elif x_axis == 'index':
            # Uniform spacing
            x = np.arange(len(times))
            ax.set_xlabel('Sample index')
        elif x_axis == 't':
            x = times
            ax.set_xlabel('$t$ (s)')

        # Add linear time axis
        if linear_time_axis and x_axis == 'f(t)':
            axt = add_linear_time_axis(ax, self.step_times, trans_functions)

        # Get model response
        y_hat = self.predict_response(**predict_kw)

        # Calculate residuals
        y_err = y_hat - self.raw_response_signal

        # Get appropriate scale
        if scale_prefix is None:
            scale_prefix, scale_factor = utils.scale.get_scale_prefix_and_factor(y_err)
        else:
            scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot residuals
        ax.scatter(x, y_err / scale_factor, s=s, alpha=alpha)

        # Indicate zero
        ax.axhline(0, c='k', lw=1, zorder=-10)

        # Show error structure estimated by CHB model
        if plot_sigma:
            sigma = self.predict_sigma('chrono') / scale_factor
            if sigma is not None:
                ax.fill_between(x, -3 * sigma, 3 * sigma, color='k', lw=0, alpha=0.15)

        # Labels
        if self.op_mode == 'galvanostatic':
            ax.set_ylabel(f'$\hat{{v}} - v$ ({scale_prefix}V)')
        elif self.op_mode == 'potentiostatic':
            ax.set_ylabel(f'$\hat{{i}} - i$ ({scale_prefix}A)')

        fig.tight_layout()

        return ax

    def plot_eis_fit(self, frequencies=None, axes=None, plot_type='nyquist', plot_data=True, data_kw=None,
                     bode_cols=['Zreal', 'Zimag'], data_label='', scale_prefix=None, area=None, normalize=False,
                     predict_kw={}, c='k', **kw):

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # Get data df if requested
        if plot_data:
            f_fit = self.get_fit_frequencies()
            data_df = utils.eis.construct_eis_df(f_fit, self.z_fit)

        # Get model impedance
        if frequencies is None:
            frequencies = self.get_fit_frequencies()
        z_hat = self.predict_impedance(frequencies, **predict_kw)
        df_hat = utils.eis.construct_eis_df(frequencies, z_hat)

        # Get rp for normalization
        if normalize:
            normalize_rp = self.predict_r_p()
            scale_prefix = ''
        else:
            normalize_rp = None

        # Get scale prefix
        if scale_prefix is None:
            if plot_data:
                z_data_concat = np.concatenate([data_df['Zreal'], data_df['Zimag']])
                z_hat_concat = np.concatenate([df_hat['Zreal'], df_hat['Zimag']])
                scale_prefix = utils.scale.get_common_scale_prefix([z_data_concat, z_hat_concat])
            else:
                z_hat_concat = np.concatenate([df_hat['Zreal'], df_hat['Zimag']])
                scale_prefix = utils.scale.get_scale_prefix(z_hat_concat)

        # Plot data if requested
        if plot_data:
            axes = plot_eis(data_df, plot_type, axes=axes, scale_prefix=scale_prefix, label=data_label,
                            bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                            **data_kw)

        # Plot fit
        axes = plot_eis(df_hat, plot_type, axes=axes, plot_func='plot', c=c, scale_prefix=scale_prefix,
                        bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                        **kw)

        fig = np.atleast_1d(axes)[0].get_figure()
        fig.tight_layout()

        return axes

    def plot_eis_residuals(self, plot_sigma=True, axes=None, scale_prefix=None, predict_kw={}, part='both',
                           s=10, alpha=0.5, **kw):

        if part == 'both':
            bode_cols = ['Zreal', 'Zimag']
        elif part == 'real':
            bode_cols = ['Zreal']
        elif part == 'imag':
            bode_cols = ['Zimag']

        if axes is None:
            fig, axes = plt.subplots(1, len(bode_cols), figsize=(3 * len(bode_cols), 2.75))
        else:
            fig = np.atleast_1d(axes)[0].get_figure()
        axes = np.atleast_1d(axes)

        # Get fit frequencies
        f_fit = self.get_fit_frequencies()

        # Get model impedance
        y_hat = self.predict_impedance(f_fit, **predict_kw)

        # Calculate residuals
        y_err = y_hat - self.z_fit

        # Construct dataframe from residuals
        err_df = utils.eis.construct_eis_df(f_fit, y_err)

        # Get scale prefix
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(np.concatenate([y_err.real, y_err.imag]))

        # Plot residuals
        plot_eis(err_df, axes=axes, plot_type='bode', bode_cols=bode_cols,
                 s=s, alpha=alpha, label='Residuals', scale_prefix=scale_prefix, **kw)

        # Indicate zero
        for ax in axes:
            ax.axhline(0, c='k', lw=1, zorder=-10)

        # Plot error structure
        if plot_sigma:
            sigma = self.predict_sigma('eis')
            if sigma is not None:
                scale_factor = scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)
                if 'Zreal' in bode_cols:
                    axes[bode_cols.index('Zreal')].fill_between(f_fit, -3 * sigma.real / scale_factor,
                                                                3 * sigma.real / scale_factor,
                                                                color='k', lw=0, alpha=0.15, zorder=-10,
                                                                label=r'$\pm 3 \sigma$')
                if 'Zimag' in bode_cols:
                    axes[bode_cols.index('Zimag')].fill_between(f_fit, -3 * sigma.imag / scale_factor,
                                                                3 * sigma.imag / scale_factor,
                                                                color='k', lw=0, alpha=0.15, zorder=-10,
                                                                label=r'$\pm 3 \sigma$')

            axes[-1].legend()

        # Update axis labels
        if 'Zreal' in bode_cols:
            axes[bode_cols.index('Zreal')].set_ylabel(fr'$\hat{{Z}}^{{\prime}}-Z^{{\prime}}$ ({scale_prefix}$\Omega$)')
        if 'Zimag' in bode_cols:
            axes[bode_cols.index('Zimag')].set_ylabel(
                fr'$-(\hat{{Z}}^{{\prime\prime}}-Z^{{\prime\prime}})$ ({scale_prefix}$\Omega$)')

        fig.tight_layout()

        return axes

    def plot_distribution(self, tau=None, ppd=20, x=None, ax=None, scale_prefix=None, plot_bounds=False,
                          normalize=False, line='mode',
                          plot_ci=False, ci_method='cov', ci_kw=None, ci_quantiles=[0.025, 0.975], sample_kw={},
                          area=None, order=0, mark_peaks=False, mark_peaks_kw=None, tight_layout=True,
                          return_line=False,
                          **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if tau is None:
            # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
            tau = self.get_tau_eval(ppd)

        if sample_kw == {} and self.map_sample_kw is not None:
            sample_kw = self.map_sample_kw

        if line == 'mode':
            # Calculate MAP distribution at evaluation points
            gamma = self.predict_distribution(tau, x=x, order=order)
        elif line == 'mean':
            # Estimate posterior mean from MAP samples
            x_mean = self.estimate_posterior_mean(**sample_kw)
            gamma = self.predict_distribution(tau, x=x_mean[len(self.special_qp_params):], order=order)

        # Normalize
        if normalize:
            r_p = self.predict_r_p(x=x)
            scale_prefix = ''
        else:
            r_p = None

        ax, info = plot_distribution(tau, gamma, ax, area, scale_prefix, normalize_by=r_p, return_info=True, **kw)
        line, scale_prefix, scale_factor = info

        if normalize:
            y_label = r'$\gamma \, / \, R_p$'
        else:
            if area is not None:
                y_units = r'$\Omega \cdot \mathrm{cm}^2$'
            else:
                y_units = r'$\Omega$'
            y_label = fr'$\gamma$ ({scale_prefix}{y_units})'
        ax.set_ylabel(y_label)

        # Indicate measurement range
        if plot_bounds:
            # Get min and max tau across all measurement types
            if self.fit_type.find('eis') > -1 or self.fit_type.find('hybrid') > -1:
                eis_tau_min = 1 / (2 * np.pi * np.max(self.get_fit_frequencies()))
                eis_tau_max = 1 / (2 * np.pi * np.min(self.get_fit_frequencies()))
            else:
                eis_tau_min = np.inf
                eis_tau_max = -np.inf

            if self.fit_type.find('chrono') > -1 or self.fit_type.find('hybrid') > -1:
                # TODO: fix this - need to get time after step time
                time_deltas = pp.get_time_since_step(self.get_fit_times(), self.step_times)
                chrono_tau_min = np.min(time_deltas)
                chrono_tau_max = np.max(time_deltas)
            else:
                chrono_tau_min = np.inf
                chrono_tau_max = -np.inf

            tau_min = min(eis_tau_min, chrono_tau_min)
            tau_max = max(eis_tau_max, chrono_tau_max)

            # Indicate bounds with vertical lines
            ax.axvline(tau_min, ls=':', c='k', alpha=0.6, lw=1.5, zorder=-10)
            ax.axvline(tau_max, ls=':', c='k', alpha=0.6, lw=1.5, zorder=-10)

        if plot_ci:
            if self.fit_type.find('qphb') > -1:
                gamma_lo, gamma_hi = self.predict_distribution_ci(tau, ppd, x=x, x_cov=None, order=order,
                                                                  normalize=normalize, quantiles=ci_quantiles)
                if area is not None:
                    for g in (gamma_lo, gamma_hi):
                        g *= area

                if gamma_lo is not None:
                    if self.fit_kwargs['nonneg'] and order == 0:
                        gamma_lo = np.maximum(gamma_lo, 0)

                    if ci_kw is None:
                        ci_kw = {}
                    ci_defaults = dict(color=line[0].get_color(), lw=0.5, alpha=0.2, zorder=-10)
                    ci_defaults.update(ci_kw)
                    ci_kw = ci_defaults

                    ax.fill_between(tau, gamma_lo / scale_factor, gamma_hi / scale_factor, **ci_kw)

        if mark_peaks:
            if mark_peaks_kw is None:
                mark_peaks_kw = {}
            self.mark_peaks(ax, x=x, scale_prefix=scale_prefix, area=area, normalize=normalize, **mark_peaks_kw)

        if tight_layout:
            fig.tight_layout()

        if return_line:
            return ax, line
        else:
            return ax

    def plot_results(self, axes=None, distribution_kw=None, eis_fit_kw=None, eis_residuals_kw=None,
                     chrono_fit_kw=None, chrono_residuals_kw=None):

        if distribution_kw is None:
            distribution_kw = {}

        # Define axes
        if self.fit_type.find('eis') > -1:
            if axes is None:
                fig, axes = plt.subplots(2, 2, figsize=(6, 5))
            else:
                fig = axes.ravel()[0].get_figure()
            drt_ax = axes[0, 0]
            eis_fit_ax = axes[0, 1]
            eis_resid_axes = axes[1]
            chrono_fit_ax = None
            chrono_resid_ax = None
        elif self.fit_type.find('chrono') > -1:
            if axes is None:
                fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
            else:
                fig = axes.ravel()[0].get_figure()
            drt_ax = axes[0]
            chrono_fit_ax = axes[1]
            chrono_resid_ax = axes[2]
            eis_fit_ax = None
            eis_resid_axes = None
        elif self.fit_type.find('hybrid') > -1:
            if axes is None:
                fig, axes = plt.subplots(2, 3, figsize=(9, 5.))
            else:
                fig = axes.ravel()[0].get_figure()
            drt_ax = axes[0, 0]
            eis_fit_ax = axes[0, 1]
            chrono_fit_ax = axes[0, 2]
            eis_resid_axes = axes[1, :2]
            chrono_resid_ax = axes[1, 2]

        # Plot distribution
        self.plot_distribution(ax=drt_ax, plot_ci=True, c='k', **distribution_kw)
        drt_ax.set_title('DRT')

        # Plot EIS fit and residuals
        if eis_fit_ax is not None:
            if eis_fit_kw is None:
                eis_fit_kw = {}
            if eis_residuals_kw is None:
                eis_residuals_kw = {}

            self.plot_eis_fit(axes=eis_fit_ax, plot_type='nyquist', **eis_fit_kw)
            eis_fit_ax.set_title('EIS Fit')

            self.plot_eis_residuals(axes=eis_resid_axes, **eis_residuals_kw)
            eis_resid_axes[0].set_title('$Z^\prime$ Residuals')
            eis_resid_axes[1].set_title('$Z^{\prime\prime}$ Residuals')
            eis_resid_axes[0].legend()
            eis_resid_axes[1].get_legend().remove()

        # Plot chrono fit and residuals
        if chrono_fit_ax is not None:
            if chrono_fit_kw is None:
                chrono_fit_kw = {'linear_time_axis': False}
            if chrono_residuals_kw is None:
                chrono_residuals_kw = {'linear_time_axis': False}

            self.plot_chrono_fit(ax=chrono_fit_ax, **chrono_fit_kw)
            chrono_fit_ax.set_title('Chrono Fit')

            self.plot_chrono_residuals(ax=chrono_resid_ax, **chrono_residuals_kw)
            chrono_resid_ax.set_title('Chrono Residuals')

        fig.tight_layout()

        return axes

    # Preprocessing (matrix calculations, scaling)
    # --------------------------------------------
    def _prep_for_fit(self,
                      # Chrono data
                      times, i_signal, v_signal,
                      # EIS data
                      frequencies, z,
                      # Chrono options
                      step_times, downsample, downsample_kw, offset_steps, smooth_inf_response,
                      # Scaling
                      scale_data, rp_scale,
                      penalty_type, derivative_weights):

        start_time = time.time()

        # Checks
        utils.validation.check_penalty_type(penalty_type)
        utils.validation.check_eis_data(frequencies, z)
        utils.validation.check_chrono_data(times, i_signal, v_signal)

        # Store fit kwargs that may be relevant for prediction and plotting
        self.fit_kwargs = {'smooth_inf_response': smooth_inf_response, 'offset_steps': offset_steps}

        # Clear map samples
        self.map_samples = None
        self.map_sample_kw = None

        # If chrono data provided, get input signal step information
        sample_times, sample_i, sample_v, step_times, step_sizes, tau_rise = self.process_chrono_signals(
            times, i_signal, v_signal, step_times, offset_steps, downsample, downsample_kw
        )

        # Set basis_tau - must have chrono step information
        if self.fixed_basis_tau is None:
            # Default: 10 ppd basis grid. Extend basis tau one decade beyond data on each end
            self.basis_tau = pp.get_basis_tau(frequencies, times, step_times)
        else:
            self.basis_tau = self.fixed_basis_tau

        # If epsilon is not set, apply default value
        if self.tau_epsilon is None:
            if self.tau_basis_type in ('gaussian', 'zga'):
                # Determine epsilon from basis_tau spacing
                dlntau = np.mean(np.diff(np.log(self.basis_tau)))
                self.tau_epsilon = 1 / dlntau
            elif self.tau_basis_type == 'Cole-Cole':
                # Default for Cole-Cole
                self.tau_epsilon = 0.95

        # If using ZGA basis function, set parameters
        if self.tau_basis_type == 'zga' and self.zga_params is None:
            self.set_zga_params()

        if sample_times is not None:
            # Get matrix and vectors for chrono fit
            rm, inf_rv, induc_rv = self._prep_chrono_fit_matrix(sample_times, step_times, step_sizes, tau_rise,
                                                                offset_steps, smooth_inf_response)
        else:
            # No chrono data
            rm = None
            inf_rv = None
            induc_rv = None

        if frequencies is not None:
            # Get matrix and vector for impedance fit
            zm, induc_zv = self._prep_impedance_fit_matrix(frequencies)
        else:
            # No EIS data
            zm = None
            induc_zv = None

        # Calculate penalty matrices
        penalty_matrices = self._prep_penalty_matrices(penalty_type, derivative_weights)

        # Perform scaling
        i_signal_scaled, v_signal_scaled, z_scaled = self.scale_data(sample_times, sample_i, sample_v, step_times,
                                                                     step_sizes, z, scale_data, rp_scale)
        if self.print_diagnostics:
            print('Finished signal scaling')

        # Estimate chrono baseline after scaling
        if sample_times is not None:
            if self.op_mode == 'galvanostatic':
                response_baseline = np.mean(v_signal_scaled[sample_times < step_times[0]])
            elif self.op_mode == 'potentiostatic':
                response_baseline = np.mean(i_signal_scaled[sample_times < step_times[0]])
        else:
            response_baseline = None

        # scale chrono response matrix/vectors to input_signal_scale
        if rm is not None:
            rm = rm / self.input_signal_scale
            induc_rv = induc_rv / self.input_signal_scale
            inf_rv = inf_rv / self.input_signal_scale

        if self.print_diagnostics:
            print('Finished prep_for_fit in {:.2f} s'.format(time.time() - start_time))

        return (sample_times, i_signal_scaled, v_signal_scaled, response_baseline, z_scaled), \
               (rm, induc_rv, inf_rv, zm, induc_zv, penalty_matrices)

    def _prep_chrono_fit_matrix(self, times, step_times, step_sizes, tau_rise, offset_steps, smooth_inf_response):
        # Recalculate matrices if necessary
        if self._recalc_chrono_fit_matrix:
            if self.print_diagnostics:
                print('Calculating chrono response matrix')
            rm, rm_layered = mat1d.construct_response_matrix(self.basis_tau, times, self.step_model, step_times,
                                                             step_sizes, basis_type=self.tau_basis_type,
                                                             epsilon=self.tau_epsilon, tau_rise=tau_rise,
                                                             op_mode=self.op_mode,
                                                             integrate_method=self.integrate_method,
                                                             zga_params=self.zga_params,
                                                             interpolate_grids=self.interpolate_lookups['response'])
            if self.print_diagnostics:
                print('Constructed chrono response matrix')
            self.fit_matrices['response'] = rm.copy()
            self.fit_matrices['rm_layered'] = rm_layered.copy()

            induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                  tau_rise, self.op_mode)

            self.fit_matrices['inductance_response'] = induc_rv.copy()

            # With all matrices calculated, set recalc flags to False
            self._recalc_chrono_fit_matrix = False
            self._recalc_chrono_prediction_matrix = False

        # Otherwise, reuse existing matrices as appropriate
        elif self._t_fit_subset_index is not None:
            # times is a subset of self.t_fit. Use sub-matrices of existing A array; do not overwrite
            rm = self.fit_matrices['response'][self._t_fit_subset_index, :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._t_fit_subset_index].copy()
        else:
            # All matrix parameters are the same. Use existing matrices
            rm = self.fit_matrices['response'].copy()
            induc_rv = self.fit_matrices['inductance_response'].copy()

        # Construct R_inf response vector. Fast - always calculate
        inf_rv = mat1d.construct_inf_response_vector(times, self.step_model, step_times, step_sizes, tau_rise,
                                                     self.raw_input_signal, smooth_inf_response, self.op_mode)
        self.fit_matrices['inf_response'] = inf_rv.copy()

        return rm, inf_rv, induc_rv

    def _prep_impedance_fit_matrix(self, frequencies):
        self.f_fit = frequencies
        if self._recalc_eis_fit_matrix:
            # Matrix calculation is required
            # Real matrix
            zmr = mat1d.construct_impedance_matrix(frequencies, 'real', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_real'])
            # Imaginary matrix
            zmi = mat1d.construct_impedance_matrix(frequencies, 'imag', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_imag'])
            # Complex matrix
            zm = zmr + 1j * zmi
            self.fit_matrices['impedance'] = zm.copy()

            # With matrices calculated, set recalc flags to False
            self._recalc_eis_fit_matrix = False
            self._recalc_eis_prediction_matrix = False
        elif self._f_fit_subset_index is not None:
            # frequencies is a subset of self.f_fit. Use sub-matrices of existing matrix; do not overwrite
            zm = self.fit_matrices['impedance'][self._f_fit_subset_index, :].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            zm = self.fit_matrices['impedance'].copy()

        # Construct inductance impedance vector
        induc_zv = mat1d.construct_inductance_impedance_vector(frequencies)

        return zm, induc_zv

    def _prep_penalty_matrices(self, penalty_type, derivative_weights):
        # Always calculate penalty (derivative) matrices - fast, avoids any gaps in recalc logic
        # if self._recalc_chrono_fit_matrix:
        penalty_matrices = {}
        for k in range(len(derivative_weights)):
            if penalty_type == 'discrete':
                dk = basis.construct_func_eval_matrix(np.log(self.basis_tau), None, self.tau_basis_type,
                                                      self.tau_epsilon, order=k,
                                                      zga_params=self.zga_params)

                penalty_matrices[f'l{k}'] = dk.copy()
                penalty_matrices[f'm{k}'] = dk.T @ dk
            elif penalty_type == 'integral':
                dk = mat1d.construct_integrated_derivative_matrix(np.log(self.basis_tau),
                                                                  basis_type=self.tau_basis_type,
                                                                  order=k, epsilon=self.tau_epsilon,
                                                                  zga_params=self.zga_params)
                penalty_matrices[f'm{k}'] = dk.copy()

        self.fit_matrices.update(penalty_matrices)

        if self.print_diagnostics:
            print('Constructed penalty matrices')
        return penalty_matrices

    def _format_qp_matrices(self, rm_drt, inf_rv, induc_rv, zm_drt, induc_zv, drt_penalty_matrices,
                            v_baseline_penalty, R_inf_penalty, inductance_penalty, vz_offset_scale,
                            inductance_scale, penalty_type, derivative_weights):
        """
        Format matrices for quadratic programming solution
        :param derivative_weights:
        :param v_baseline_scale:
        :param rm_drt:
        :param drt_penalty_matrices:
        :param R_inf_scale:
        :param inductance_scale:
        :param penalty_type:
        :return:
        """
        # Count number of special params for padding
        num_special = len(self.special_qp_params)

        # Extract indices for convenience
        special_indices = {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}

        # Store inductance scale for reference
        self.inductance_scale = inductance_scale

        # Add columns to rm for v_baseline, R_inf, and inductance
        if rm_drt is not None:
            rm = np.empty((rm_drt.shape[0], rm_drt.shape[1] + num_special))

            # Add entries for special parameters
            rm[:, special_indices['v_baseline']] = 1  # v_baseline
            if 'inductance' in special_indices.keys():
                rm[:, special_indices['inductance']] = induc_rv * inductance_scale  # inductance response
            rm[:, special_indices['R_inf']] = inf_rv  # R_inf response. only works for galvanostatic mode

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                rm[:, special_indices['vz_offset']] = 0

            # Insert main DRT matrix
            rm[:, num_special:] = rm_drt
        else:
            # No chrono matrix
            rm = None

        # Add columns to zm (impedance matrix) for v_baseline, R_inf, and inductance
        if zm_drt is not None:
            zm = np.empty((zm_drt.shape[0], zm_drt.shape[1] + num_special), dtype=complex)

            # Add entries for special parameters
            if 'v_baseline' in special_indices.keys():
                zm[:, special_indices['v_baseline']] = 0  # v_baseline has no effect on impedance

            if 'inductance' in special_indices.keys():
                zm[:, special_indices['inductance']] = induc_zv * inductance_scale  # inductance contribution
            zm[:, special_indices['R_inf']] = 1  # R_inf contribution to impedance

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                zm[:, special_indices['vz_offset']] = 0

            # Insert main DRT matrix
            zm[:, num_special:] = zm_drt  # DRT response

            # Stack real and imag matrices
            zm = np.vstack([zm.real, zm.imag])
        else:
            # No EIS matrix
            zm = None

        # Construct L2 penalty matrices
        penalty_matrices = {}
        if penalty_type == 'integral':
            for k in range(len(derivative_weights)):
                # Get penalty matrix for DRT coefficients
                m_drt = drt_penalty_matrices[f'm{k}']

                # Add rows/columns for v_baseline, inductance, and R_inf
                m = np.zeros((m_drt.shape[0] + num_special, m_drt.shape[1] + num_special))

                # Insert penalties for special parameters
                if 'v_baseline' in special_indices.keys():
                    m[special_indices['v_baseline'], special_indices['v_baseline']] = v_baseline_penalty
                if 'inductance' in special_indices.keys():
                    m[special_indices['inductance'], special_indices['inductance']] = inductance_penalty
                m[special_indices['R_inf'], special_indices['R_inf']] = R_inf_penalty

                # Add entry for vz_offset if applicable
                if 'vz_offset' in special_indices.keys():
                    m[special_indices['vz_offset'], special_indices['vz_offset']] = 1 / vz_offset_scale

                # Insert main DRT matrix
                m[num_special:, num_special:] = m_drt

                penalty_matrices[f'm{k}'] = m.copy()
        elif penalty_type == 'discrete':
            for k in range(len(derivative_weights)):
                # Get penalty matrix for DRT coefficients
                l_drt = drt_penalty_matrices[f'l{k}']

                # Add rows/columns for v_baseline, inductance, and R_inf
                l_k = np.zeros((l_drt.shape[0] + num_special, l_drt.shape[1] + num_special))

                # Insert penalties for special parameters
                if 'v_baseline' in special_indices.keys():
                    l_k[special_indices['v_baseline'], special_indices['v_baseline']] = v_baseline_penalty ** 0.5
                if 'inductance' in special_indices.keys():
                    l_k[special_indices['inductance'], special_indices['inductance']] = inductance_penalty ** 0.5
                l_k[special_indices['R_inf'], special_indices['R_inf']] = R_inf_penalty ** 0.5

                # Add entry for vz_offset if applicable
                if 'vz_offset' in special_indices.keys():
                    l_k[special_indices['vz_offset'], special_indices['vz_offset']] = 1 / vz_offset_scale ** 0.5

                # Insert main DRT matrix
                l_k[num_special:, num_special:] = l_drt

                penalty_matrices[f'l{k}'] = l_k.copy()

                # Calculate norm matrix
                penalty_matrices[f'm{k}'] = l_k.T @ l_k

        return rm, zm, penalty_matrices

    # def _prep_for_hybrid_fit(self, times, i_signal, v_signal, frequencies, z, step_times, downsample, downsample_kw,
    #                          scale_signal, offset_steps, penalty_type, smooth_inf_response):
    #
    #     # Perform preprocessing and matrix preparation for regular fit
    #     sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, None, None, step_times, downsample,
    #                                                downsample_kw, offset_steps, smooth_inf_response, scale_signal, 7,
    #                                                penalty_type, derivative_weights)
    #
    #     # Extract time response data from sample_data
    #     sample_times, i_signal_scaled, v_signal_scaled, response_baseline = sample_data
    #
    #     # Extract time response matrices and penalty matrics
    #     rm, induc_rv, inf_rv, penalty_matrices = matrices
    #
    #     # Scale measured impedance
    #     z_scaled = self.scale_impedance(z)
    #
    #     self.z_fit = z.copy()
    #     self.z_fit_scaled = z_scaled.copy()
    #
    #     # Construct impedance matrix
    #     zm = self._prep_impedance_fit_matrix(frequencies)
    #
    #     # Construct inductance impedance vector
    #     induc_zv = mat1d.construct_inductance_z_vector(frequencies)
    #
    #     return (sample_times, i_signal_scaled, v_signal_scaled, response_baseline, z_scaled), \
    #            (rm, zm, induc_rv, inf_rv, induc_zv, penalty_matrices)

    def _prep_chrono_prediction_matrix(self, times, input_signal, op_mode, offset_steps, smooth_inf_response):
        # If input signal is not provided, use self.raw_input_signal
        if input_signal is None:
            if self._t_fit_subset_index is not None:
                input_signal = self.raw_input_signal[self._t_fit_subset_index]
            else:
                input_signal = self.raw_input_signal
            use_fit_signal = True
        else:
            use_fit_signal = False

        self.t_predict = times
        self.raw_prediction_input_signal = input_signal.copy()
        self.op_mode_predict = op_mode

        if self.print_diagnostics:
            print('recalc_response_prediction_matrix:', self._recalc_chrono_prediction_matrix)

        if use_fit_signal:
            # Allow times to have a different length than input_signal if using fitted signal (input_signal = None)
            # This will break construct_inf_response_vector if smooth_inf_response==False
            step_times = self.step_times
            step_sizes = self.step_sizes
            tau_rise = self.tau_rise
        else:
            # Identify steps in applied signal
            step_times, step_sizes, tau_rise = pp.process_input_signal(times, input_signal, self.step_model,
                                                                       offset_steps)

        if self._recalc_chrono_prediction_matrix:
            # Matrix recalculation is required
            rm, rm_layered = mat1d.construct_response_matrix(self.basis_tau, times, self.step_model, step_times,
                                                             step_sizes, basis_type=self.tau_basis_type,
                                                             epsilon=self.tau_epsilon, tau_rise=tau_rise,
                                                             op_mode=op_mode,
                                                             integrate_method=self.integrate_method,
                                                             zga_params=self.zga_params,
                                                             interpolate_grids=self.interpolate_lookups['response'])
            if self.step_model == 'expdecay':
                # num_steps = len(step_times)
                # signal_fit = fit_signal_steps(times, input_signal)
                # # Get step time offsets
                # t_step_offset = signal_fit['x'][:num_steps] * 1e-6
                # step_times += t_step_offset
                # # Get rise times
                # tau_rise = np.exp(signal_fit['x'][num_steps:])
                induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times,
                                                                      step_sizes, tau_rise,
                                                                      op_mode)
            else:
                induc_rv = np.zeros(len(times))
            self.prediction_matrices = {'response': rm.copy(), 'inductance_response': induc_rv.copy()}

            # With prediction matrices calculated, set recalc flag to False
            self._recalc_chrono_prediction_matrix = False
            if self.print_diagnostics:
                print('Calculated response prediction matrices')
        elif self._t_predict_eq_t_fit:
            # times is the same as self.t_fit. Do not overwrite
            rm = self.fit_matrices['response'].copy()
            induc_rv = self.fit_matrices['inductance_response'].copy()
        elif self._t_predict_subset_index[0] == 'predict':
            # times is a subset of self.t_predict. Use sub-matrices of existing matrix; do not overwrite
            rm = self.prediction_matrices['response'][self._t_predict_subset_index[1], :].copy()
            induc_rv = self.prediction_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
        elif self._t_predict_subset_index[0] == 'fit':
            # times is a subset of self.t_fit. Use sub-matrices of existing matrix; do not overwrite
            rm = self.fit_matrices['response'][self._t_predict_subset_index[1], :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            rm = self.prediction_matrices['response'].copy()
            induc_rv = self.prediction_matrices['inductance_response'].copy()

        # Construct R_inf response vector
        inf_rv = mat1d.construct_inf_response_vector(times, self.step_model, step_times, step_sizes,
                                                     tau_rise, input_signal, smooth_inf_response,
                                                     op_mode)

        self.prediction_matrices['inf_response'] = inf_rv.copy()

        return rm, induc_rv, inf_rv

    def _prep_impedance_prediction_matrix(self, frequencies):
        self.f_predict = frequencies
        if self._recalc_eis_prediction_matrix:
            # Matrix calculation is required
            # Real matrix
            zmr = mat1d.construct_impedance_matrix(frequencies, 'real', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_real'])
            # Imaginary matrix
            zmi = mat1d.construct_impedance_matrix(frequencies, 'imag', tau=self.basis_tau,
                                                   basis_type=self.tau_basis_type, epsilon=self.tau_epsilon,
                                                   frequency_precision=self.frequency_precision,
                                                   zga_params=self.zga_params, integrate_method=self.integrate_method,
                                                   interpolate_grids=self.interpolate_lookups['z_imag'])
            # Complex matrix
            zm = zmr + 1j * zmi
            self.prediction_matrices['impedance'] = zm.copy()

            # With prediction matrix calculated, set recalc flag to False
            self._recalc_eis_prediction_matrix = False
        elif self._f_predict_eq_f_fit:
            # frequencies is same as self.f_fit. Use existing fit matrix; do not overwrite
            zm = self.fit_matrices['impedance'].copy()
        elif self._f_predict_subset_index[0] == 'fit':
            # frequencies is a subset of self.f_fit. Use sub-matrices of existing matrix; do not overwrite
            zm = self.fit_matrices['impedance'][self._f_predict_subset_index[1], :].copy()
        elif self._f_predict_subset_index[0] == 'predict':
            # frequencies is a subset of self.f_predict. Use sub-matrices of existing matrix; do not overwrite
            zm = self.prediction_matrices['impedance'][self._f_predict_subset_index[1], :].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            zm = self.prediction_matrices['impedance'].copy()

        return zm

    def extract_qphb_parameters(self, x):
        special_indices = {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}
        fit_parameters = {'x': x[len(self.special_qp_params):] * self.coefficient_scale,
                          'R_inf': x[special_indices['R_inf']] * self.coefficient_scale,
                          }

        if 'v_baseline' in special_indices.keys():
            fit_parameters['v_baseline'] = (x[special_indices['v_baseline']] - self.scaled_response_offset) \
                                           * self.response_signal_scale

        if 'vz_offset' in special_indices.keys():
            fit_parameters['vz_offset'] = x[special_indices['vz_offset']]

        if 'inductance' in special_indices.keys():
            fit_parameters['inductance'] = x[special_indices['inductance']] \
                                           * self.coefficient_scale * self.inductance_scale
        else:
            fit_parameters['inductance'] = 0

        return fit_parameters
