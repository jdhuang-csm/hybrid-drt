import time
import warnings
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt, colors

import hybdrt.matrices
import hybdrt.matrices.mat1d
from hybdrt import utils, preprocessing as pp
from hybdrt.matrices import mat2d as mat2d, mat1d as mat1d
from hybdrt.models import qphb
from hybdrt.models.drtbase import DRTBase, format_chrono_weights, format_eis_weights
from hybdrt.plotting import get_transformed_plot_time, add_linear_time_axis


class DRT2d(DRTBase):
    def __init__(self, basis_tau, tau_basis_type='gaussian', tau_epsilon=None,
                 basis_psi=None, psi_basis_type='gaussian', psi_epsilon=None,
                 step_model='ideal', op_mode='galvanostatic',
                 fit_inductance=False):
        self.basis_psi = basis_psi
        self.psi_basis_type = psi_basis_type
        self.psi_epsilon = psi_epsilon

        self.psi_scale = None
        self.basis_psi_scaled = None
        self.psi_epsilon_scaled = None

        self.psi_map_coef = None
        self.psi_hat = None

        self._fit_subset_index = []
        self._psi_fit_subset_index = []
        self._psi_predict_subset_index = ('', [])
        self._predict_subset_index = ('', [])

        self.psi_fit = []
        self.psi_predict = []

        self.basis_times = None

        super().__init__(basis_tau, tau_basis_type, tau_epsilon, step_model, op_mode, fit_inductance,
                         input_signal_precision=input_signal_precision)

    def ridge_fit(self, times, i_signal, v_signal, psi, independent_measurements, psi_static, psi_is_time, psi_is_i,
                  basis_psi=None, psi_epsilon=None, psi_basis_type='gaussian', scale_psi=True,
                  basis_times=None, time_epsilon=None, time_basis_type='step',
                  nonneg=True, scale_signal=True, offset_baseline=True, offset_steps=False, smooth_inf_response=False,
                  downsample=True, downsample_kw=None,
                  # basic fit control
                  l2_lambda_0=2, l1_lambda_0=0.1, weights=None, R_inf_scale=100, inductance_scale=1e-4,
                  penalty_type='integral', derivative_weights=[0, 0, 1],
                  # partial derivative penalties
                  fxx_penalty=1, fyy_penalty=1, fxy_penalty=1,
                  # hyper-lambda options
                  hyper_l2_lambda=True, hl_l2_beta=100,
                  hyper_l1_lambda=False, hl_l1_beta=2.5,
                  # optimization control
                  xtol=1e-3, max_iter=20):

        # Preprocess data and calculate matrices
        sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, psi, None, None, None, step_times,
                                                   downsample, downsample_kw, offset_steps, smooth_inf_response,
                                                   independent_measurements, psi_static, psi_is_time, psi_is_i,
                                                   basis_psi, psi_epsilon, psi_basis_type, scale_psi, basis_times,
                                                   time_epsilon, time_basis_type, penalty_type, derivative_weights,
                                                   scale_signal, rp_scale, fxx_penalty, fyy_penalty, fxy_penalty)

        sample_times, sample_i, sample_v, response_baseline = sample_data
        rm_drt, rm_inf, induc_rv, penalty_matrices = matrices
        print(response_baseline)

        # if penalty_type == 'integral':
        #     # Extract integrated derivative matrices (x.T @ m @ x = integral of squared derivative across all tau, psi)
        #     m0_p = penalty_matrices['m0']
        #     m1_p = penalty_matrices['m1']
        #     m2_p = penalty_matrices['m2']
        if penalty_type == 'discrete':
            # Extract differentiation matrices (l @ x = vector of derivatives at discrete tau, psi coordinates)
            l0 = penalty_matrices['l_drt']
            lx = penalty_matrices['lx_drt']
            ly = penalty_matrices['ly_drt']
            lxx = penalty_matrices['lxx_drt']
            lyy = penalty_matrices['lyy_drt']
            lxy = penalty_matrices['lxy_drt']

            # Construct squared derivative matrices (x.T @ M @ x = L2 norm of derivative vector)
            penalty_matrices['m0_drt'] = l0 @ l0
            penalty_matrices['m1_drt'] = fxx_penalty * lx @ lx + fyy_penalty * ly @ ly

            # TODO: investigate gaussian curvature vs. sum of squares
            # Can only get gaussian curvature, not its square, with x.T @ M @ x. It still seems to work decently...
            # m2_p = 0.5 * (fxx_penalty + fyy_penalty) * lxx @ lyy - fxy_penalty * lxy @ lxy
            m2_drt = fxx_penalty * lxx @ lxx + fyy_penalty * lyy @ lyy + fxy_penalty * lxy @ lxy
            penalty_matrices['m2_drt'] = m2_drt / m2_drt.shape[0]  # normalize by number of evaluation points

        print('m2_drt sum:', np.sum(penalty_matrices['m2_drt']))

        # Offset voltage baseline
        if offset_baseline:
            response_offset = -response_baseline
        else:
            response_offset = 0
        scaled_response_signal = self.scaled_response_signal + response_offset

        # Add columns to rm for v_baseline, inductance, and R_inf
        rm = np.empty((rm_drt.shape[0], rm_inf.shape[1] + rm_drt.shape[1] + 2))
        rm[:, 0] = 1  # v_baseline
        if self.fit_inductance:
            rm[:, 1] = induc_rv.copy() * inductance_scale  # inductance response
        else:
            rm[:, 1] = 0
        rm[:, 2: 2 + rm_inf.shape[1]] = rm_inf.copy()  # R_inf response
        rm[:, 2 + rm_inf.shape[1]:] = rm_drt.copy()  # DRT response

        # print('rm:', rm[0])
        # print('rank(rm):', np.linalg.matrix_rank(rm))

        # Construct L2 penalty matrices
        # l2_matrices = []
        # for m_p in [m0_p, m1_p, m2_p]:
        for order in range(0, 3):
            m_drt = penalty_matrices[f'm{order}_drt']
            m_inf = penalty_matrices[f'm{order}_inf']

            m = np.zeros((m_drt.shape[0] + m_inf.shape[0] + 2, m_drt.shape[1] + m_inf.shape[1] + 2))
            # No penalty applied to v_baseline (m[0, 0] = 0)
            # Insert penalties for R_inf and inductance
            m[1, 1] = 1  # inductance
            m[2: 2 + m_inf.shape[0], 2: 2 + m_inf.shape[1]] = m_inf.copy() / R_inf_scale  # R_inf penalty matrix
            m[2 + m_inf.shape[0]:, 2 + m_inf.shape[1]:] = m_drt.copy()  # DRT penalty matrix
            penalty_matrices[f'm{order}'] = m
            print(order, np.sum(m_drt) / m_drt.shape[1], np.sum(m_inf) / m_inf.shape[1])

        # Indices of special variables - always nonneg or always unbounded
        special_indices = {'nonneg': np.concatenate(([int(1)], np.arange(2, 2 + m_inf.shape[1], dtype=int))),
                           'unbnd': np.array([0], dtype=int)}

        # Construct lambda vectors
        l1_lambda_vector = np.zeros(rm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        # Remaining entries set to l1_lambda_0 - for DRT coefficients
        l1_lambda_vector[2 + rm_inf.shape[1]:] = l1_lambda_0
        l2_lv = np.ones(rm.shape[1])  # First 2 entries are one - for v_baseline and inductance
        # Remaining entries set to l2_lambda_0 - for R_inf and DRT coefficients
        l2_lv[2:] = l2_lambda_0
        l2_lambda_vectors = [l2_lv.copy()] * 3  # need one vector for each order of the derivative

        # print('lml:', lml)

        # Get weight vector
        wv = format_chrono_weights(scaled_response_signal, weights)
        wm = np.diag(wv)
        # print('wm:', wm)
        # Apply weights to rm and signal
        wrm = wm @ rm
        wrv = wm @ scaled_response_signal
        # print('wrv:', wrv)

        if hyper_l1_lambda or hyper_l2_lambda:
            self.ridge_iter_history = []
            it = 0

            x = np.zeros(rm.shape[1]) + 1e-6

            # P = wrm.T @ wrm + lml
            # q = (-wrm.T @ wrv + l1_lambda_vector)
            # cost = 0.5 * x.T @ P @ x + q.T @ x

            while it < max_iter:

                x_prev = x.copy()

                # Always pass penalty_type 'integral' since l matrices are used to construct m matrices
                # when penalty == 'discrete'
                x, cvx_result, converged = self._iterate_hyper_ridge(x_prev, 2 + rm_inf.shape[1], 1, nonneg,
                                                                     hyper_l1_lambda, hyper_l2_lambda, wrv, wrm,
                                                                     l1_lambda_vector, penalty_matrices,
                                                                     l2_lambda_vectors, derivative_weights, hl_l1_beta,
                                                                     l1_lambda_0, hl_l2_beta, l2_lambda_0, xtol, )

                if converged:
                    break
                elif it == max_iter - 1:
                    warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

                it += 1

        else:
            # Ordinary ridge fit
            # # Make lml matrix for each derivative order
            # lml = np.zeros_like(l2_matrices[0])
            # for n, d_weight in enumerate(derivative_weights):
            #	  l2_lv = l2_lambda_vectors[n]
            #	  lm = np.diag(l2_lv ** 0.5)
            #	  m = l2_matrices[n]
            #	  lml += d_weight * lm @ m @ lm

            # Make lml matrix for each derivative order
            l2_matrices = [penalty_matrices[f'm{k}'] for k in range(0, 3)]
            lml = qphb.calculate_qp_l2_matrix(derivative_weights, rho_vector, l2_matrices, l2_lambda_vectors,
                                              l2_lambda_0, penalty_type)

            cvx_result = qphb.solve_convex_opt(wrv, wrm, lml, l1_lambda_vector, nonneg, special_indices)

        # Store final cvxopt result
        self.cvx_result = cvx_result

        # Extract model parameters
        x_out = np.array(list(cvx_result['x']))
        self.fit_parameters = {'x': x_out[2 + rm_inf.shape[1]:] * self.coefficient_scale,
                               'x_inf': x_out[2: 2 + rm_inf.shape[1]] * self.coefficient_scale,
                               'v_baseline': (x_out[0] - response_offset) * self.response_signal_scale,
                               'v_sigma_tot': None,
                               'v_sigma_res': None}

        if self.fit_inductance:
            self.fit_parameters['inductance'] = x_out[1] * self.coefficient_scale * inductance_scale
        else:
            self.fit_parameters['inductance'] = 0

        self.fit_type = 'ridge'

    # ===========================================
    # QPHB 2d fit
    # ===========================================
    def qphb_fit_chrono(self, times, i_signal, v_signal, psi,
                        independent_measurements, psi_static, psi_is_time, psi_is_i,
                  basis_psi=None, psi_epsilon=None, psi_basis_type='gaussian', scale_psi=True,
                  basis_times=None, time_epsilon=None, time_basis_type='step',
                  nonneg=True, scale_data=True, step_times=None, offset_baseline=True, offset_steps=True, smooth_inf_response=True,
                  downsample=True, downsample_kw=None,
                  # basic fit control
                l2_lambda_0=None, l1_lambda_0=0.0,
                v_baseline_penalty=0, R_inf_penalty=0, inductance_penalty=0, inductance_scale=1e-4,
                penalty_type='integral', derivative_weights=[1.5, 1.0, 0.5],
                error_structure='uniform', vmm_epsilon=0.1,
                # Prior hyperparameters
                rp_scale=14,
                iw_alpha=1.5, iw_beta=None,
                s_alpha=[1.5, 2.5, 25], s_0=1,
                rho_alpha=[1.1, 1.15, 1.2], rho_0=1,
                w_alpha=None, w_beta=None,
                # optimization control
                xtol=1e-2, max_iter=50,
                  # partial derivative penalties
                  fxx_penalty=1, fyy_penalty=1, fxy_penalty=1,
    ):

        # Checks
        utils.check_error_structure(error_structure)

        # Format list/scalar arguments into arrays
        derivative_weights = np.array(derivative_weights)
        k_range = len(derivative_weights)
        if np.shape(s_alpha) == ():
            s_alpha = [s_alpha] * k_range

        if np.shape(rho_alpha) == ():
            rho_alpha = [rho_alpha] * k_range

        s_alpha = np.array(s_alpha)
        rho_alpha = np.array(rho_alpha)

        # Preprocess data and calculate matrices
        sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, psi, None, None, None, step_times,
                                                   downsample, downsample_kw, offset_steps, smooth_inf_response,
                                                   independent_measurements, psi_static, psi_is_time, psi_is_i,
                                                   basis_psi, psi_epsilon, psi_basis_type, scale_psi, basis_times,
                                                   time_epsilon, time_basis_type, penalty_type, derivative_weights,
                                                   scale_data, rp_scale, fxx_penalty, fyy_penalty, fxy_penalty)

        sample_times, sample_i, sample_v, response_baseline, _ = sample_data
        rm_drt, rm_inf, induc_rv, _, _, _, drt_penalty_matrices = matrices
        print(response_baseline)

        # Define special parameters included in quadratic programming parameter vector
        # Must happen after basis_psi is set
        if independent_measurements:
            # v_baseline defined independently for each measurement
            v_baseline_size = len(self.basis_psi)
        else:
            # Single v_baseline for all measurements
            v_baseline_size = 1

        self.special_qp_params = {
            'v_baseline': {'index': 0, 'nonneg': False, 'size': v_baseline_size},
            'R_inf': {'index': 1, 'nonneg': True, 'size': len(self.basis_psi)}
        }

        if self.fit_inductance:
            self.special_qp_params['inductance'] = {'index': self.get_qp_mat_offset(), 'nonneg': True, 'size': 1}

        # if penalty_type == 'integral':
        #     # Extract integrated derivative matrices (x.T @ m @ x = integral of squared derivative across all tau, psi)
        #     m0_p = penalty_matrices['m0']
        #     m1_p = penalty_matrices['m1']
        #     m2_p = penalty_matrices['m2']
        if penalty_type == 'discrete':
            # Extract differentiation matrices (l @ x = vector of derivatives at discrete tau, psi coordinates)
            l0 = drt_penalty_matrices['l_drt']
            lx = drt_penalty_matrices['lx_drt']
            ly = drt_penalty_matrices['ly_drt']
            lxx = drt_penalty_matrices['lxx_drt']
            lyy = drt_penalty_matrices['lyy_drt']
            lxy = drt_penalty_matrices['lxy_drt']

            # Construct squared derivative matrices (x.T @ M @ x = L2 norm of derivative vector)
            drt_penalty_matrices['m0_drt'] = l0 @ l0
            drt_penalty_matrices['m1_drt'] = fxx_penalty * lx @ lx + fyy_penalty * ly @ ly

            # TODO: investigate gaussian curvature vs. sum of squares
            # Can only get gaussian curvature, not its square, with x.T @ M @ x. It still seems to work decently...
            # m2_p = 0.5 * (fxx_penalty + fyy_penalty) * lxx @ lyy - fxy_penalty * lxy @ lxy
            m2_drt = fxx_penalty * lxx @ lxx + fyy_penalty * lyy @ lyy + fxy_penalty * lxy @ lxy
            drt_penalty_matrices['m2_drt'] = m2_drt / m2_drt.shape[0]  # normalize by number of evaluation points

        print('m2_drt sum:', np.sum(drt_penalty_matrices['m2_drt']))

        # Offset voltage baseline
        if offset_baseline:
            self.response_offset = -response_baseline
        else:
            self.response_offset = 0
        rv = self.scaled_response_signal + self.response_offset

        # Add columns to rm for v_baseline, inductance, and R_inf
        rm, _, penalty_matrices = self._format_qp_matrices(rm_drt, rm_inf, induc_rv, None, None, drt_penalty_matrices, v_baseline_penalty,
                                 R_inf_penalty, inductance_penalty, 1, inductance_scale, penalty_type,
                                 derivative_weights)

        # Set lambda_0 and iw_0 based on sample size and density
        ppd = pp.get_time_ppd(sample_times, self.step_times)
        num_post_step = len(sample_times[sample_times >= self.step_times[0]])
        print('ppd:', ppd)

        data_factor = qphb.get_data_factor(num_post_step, ppd)
        print('data_factor:', data_factor)

        if l2_lambda_0 is None:
            # lambda_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
            # l2_lambda_0 = 142 * np.sqrt(len(frequencies) / 71) * (10 / ppd)
            l2_lambda_0 = 142 * data_factor ** -1  # * np.sqrt(n_eff / 142) * (20 / ppd_eff)  #

        if iw_beta is None:
            # iw_0 decreases with increasing ppd (linear), increases with increasing num frequencies (sqrt)
            # iw_beta = (4 ** 2 * 0.05) * (np.sqrt(len(frequencies) / 71) * (10 / ppd)) ** 2
            iw_beta = 0.5 * data_factor ** 2  # * (np.sqrt(n_eff / 142) * (20 / ppd_eff)) ** 2 #
        print('lambda_0, iw_beta:', l2_lambda_0, iw_beta)

        # Construct l1 lambda vectors
        l1_lambda_vector = np.zeros(rm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        l1_lambda_vector[self.get_qp_mat_offset():] = l1_lambda_0  # Remaining entries set to l1_lambda_0

        # Initialize s and rho vectors at prior mode
        rho_vector = np.ones(k_range) * rho_0
        s_vectors = [np.ones(rm.shape[1]) * s_0] * k_range  # need one vector for each order of the derivative

        # Initialize x near zero
        x = np.zeros(rm.shape[1]) + 1e-6

        # Construct matrix for variance estimation
        vmm = hybdrt.matrices.mat1d.construct_chrono_var_matrix(sample_times, self.step_times, vmm_epsilon, error_structure)
        # print(vmm[0])

        # Initialize data weight (IMPORTANT)
        # ----------------------------------
        est_weights, init_weights, x_overfit = qphb.initialize_weights(penalty_matrices, penalty_type,
                                                                       derivative_weights, rho_vector, s_vectors, rv,
                                                                       rm, vmm, nonneg, self.special_qp_params,
                                                                       iw_alpha, iw_beta, outlier_lambda)

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

            # print(it, weights[0], weight_factor)
            x, s_vectors, rho_vector, weights, cvx_result, converged = qphb.iterate_qphb(x_in, s_vectors, rho_vector,
                                                                                         rv, weights, est_weights,
                                                                                         outlier_variance, rm, vmm,
                                                                                         penalty_matrices, penalty_type,
                                                                                         l1_lambda_vector, l2_lambda_0,
                                                                                         derivative_weights, rho_alpha,
                                                                                         rho_0, s_alpha, None, s_0,
                                                                                         xmx_norms, None, None, nonneg)

            # Normalize to ordinary ridge solution
            if it == 0:
                # Only include DRT penalty in XMX norms
                x_drt = x[self.get_qp_mat_offset():]
                xmx_norms = np.array([x_drt.T @ drt_penalty_matrices[f'm{k}_drt'] @ x_drt for k in range(k_range)])
                print('xmx', xmx_norms)
                self.xmx_norms = xmx_norms

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 0 and scale_data:
                # Get scale factor
                rp = np.sum(x[self.get_qp_mat_offset():]) * np.pi ** 0.5 / (self.tau_epsilon * len(self.basis_psi))
                scale_factor = rp_scale / rp
                # print('scale factor:', scale_factor)
                # Update data and qphb parameters to reflect new scale
                self.response_offset *= scale_factor
                rv *= scale_factor
                xmx_norms *= scale_factor
                est_weights /= scale_factor
                init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            if converged:
                break
            elif it == max_iter - 1:
                warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

            it += 1

        # Store QPHB diagnostic parameters
        p_matrix, q_vector = qphb.calculate_pq(rm, rv, penalty_matrices, penalty_type, derivative_weights, l2_lambda_0,
                                               l1_lambda_vector, rho_vector, s_vectors)

        # l2_matrices = [penalty_matrices[f'm{n}'] for n in range(3)]
        # sms = qphb.calculate_sms(np.array(derivative_weights) * rho_vector, l2_matrices, s_vectors)
        # sms *= l2_lambda_0
        #
        # wm = np.diag(weights)
        # wrm = wm @ rm
        # wrv = wm @ rv
        #
        # p_matrix = 2 * sms + wrm.T @ wrm
        # q_vector = -wrm.T @ wrv + l1_lambda_vector

        post_lp = qphb.evaluate_posterior_lp(x, derivative_weights, penalty_type, penalty_matrices, l2_lambda_0,
                                             l1_lambda_vector, rho_vector, s_vectors, weights, rm, rv)

        self.qphb_params = {'est_weights': est_weights.copy(),
                            'init_weights': init_weights.copy(),
                            'weights': weights.copy(),
                            'xmx_norms': xmx_norms.copy(),
                            'x_overfit': x_overfit,
                            'p_matrix': p_matrix,
                            'q_vector': q_vector,
                            'rho_vector': rho_vector,
                            's_vectors': s_vectors,
                            'vmm': vmm,
                            'l2_lambda_0': l2_lambda_0,
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


        # x_out = np.array(list(cvx_result['x']))
        # self.fit_parameters = {'x': x_out[self.get_qp_mat_offset():] * self.coefficient_scale,
        #                        'x_inf': x_out[special_indices['R_inf']] * self.coefficient_scale,
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

        # ---------------------------------------
        #
        # # Construct lambda vectors
        # l1_lambda_vector = np.zeros(rm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        # # Remaining entries set to l1_lambda_0 - for DRT coefficients
        # l1_lambda_vector[self.get_qp_mat_offset():] = l1_lambda_0
        #
        # l2_lv = np.ones(rm.shape[1])  # First 2 entries are one - for v_baseline and inductance
        # # Remaining entries set to l2_lambda_0 - for R_inf and DRT coefficients
        # l2_lv[2:] = l2_lambda_0
        # l2_lambda_vectors = [l2_lv.copy()] * 3  # need one vector for each order of the derivative
        #
        # # print('lml:', lml)
        #
        # # Get weight vector
        # wv = format_chrono_weights(scaled_response_signal, weights)
        # wm = np.diag(wv)
        # # print('wm:', wm)
        # # Apply weights to rm and signal
        # wrm = wm @ rm
        # wrv = wm @ scaled_response_signal
        # # print('wrv:', wrv)
        #
        # if hyper_l1_lambda or hyper_l2_lambda:
        #     self.ridge_iter_history = []
        #     it = 0
        #
        #     x = np.zeros(rm.shape[1]) + 1e-6
        #
        #     # P = wrm.T @ wrm + lml
        #     # q = (-wrm.T @ wrv + l1_lambda_vector)
        #     # cost = 0.5 * x.T @ P @ x + q.T @ x
        #
        #     while it < max_iter:
        #
        #         x_prev = x.copy()
        #
        #         # Always pass penalty_type 'integral' since l matrices are used to construct m matrices
        #         # when penalty == 'discrete'
        #         x, cvx_result, converged = self._iterate_hyper_ridge(x_prev, 2 + rm_inf.shape[1], 1, nonneg,
        #                                                              hyper_l1_lambda, hyper_l2_lambda, wrv, wrm,
        #                                                              l1_lambda_vector, penalty_matrices,
        #                                                              l2_lambda_vectors, derivative_weights, hl_l1_beta,
        #                                                              l1_lambda_0, hl_l2_beta, l2_lambda_0, xtol, )
        #
        #         if converged:
        #             break
        #         elif it == max_iter - 1:
        #             warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')
        #
        #         it += 1
        #
        # else:
        #     # Ordinary ridge fit
        #     # # Make lml matrix for each derivative order
        #     # lml = np.zeros_like(l2_matrices[0])
        #     # for n, d_weight in enumerate(derivative_weights):
        #     #	  l2_lv = l2_lambda_vectors[n]
        #     #	  lm = np.diag(l2_lv ** 0.5)
        #     #	  m = l2_matrices[n]
        #     #	  lml += d_weight * lm @ m @ lm
        #
        #     # Make lml matrix for each derivative order
        #     l2_matrices = [penalty_matrices[f'm{n}'] for m in range(0, 3)]
        #     lml = qphb.calculate_sms(derivative_weights, l2_matrices, l2_lambda_vectors)
        #
        #     cvx_result = qphb.solve_convex_opt(wrv, wrm, lml, l1_lambda_vector, nonneg, special_indices)
        #
        # # Store final cvxopt result
        # self.cvx_result = cvx_result
        #
        # # Extract model parameters
        # x_out = np.array(list(cvx_result['x']))
        # self.fit_parameters = {'x': x_out[2 + rm_inf.shape[1]:] * self.coefficient_scale,
        #                        'x_inf': x_out[2: 2 + rm_inf.shape[1]] * self.coefficient_scale,
        #                        'v_baseline': (x_out[0] - scaled_response_offset) * self.response_signal_scale,
        #                        'v_sigma_tot': None,
        #                        'v_sigma_res': None}
        #
        # if self.fit_inductance:
        #     self.fit_parameters['inductance'] = x_out[1] * self.coefficient_scale * inductance_scale
        # else:
        #     self.fit_parameters['inductance'] = 0
        #
        # self.fit_type = 'ridge'

    # Hybrid fit
    # --------------------------------------------
    def hybrid_ridge_fit(self, times, i_signal, v_signal, psi_t, frequencies, z, psi_f,
                         independent_measurements, psi_static, psi_is_time, psi_is_i,
                         basis_psi=None, psi_epsilon=None, psi_basis_type='gaussian', scale_psi=True,
                         basis_times=None, time_epsilon=None, time_basis_type='step',
                         nonneg=True, scale_signal=True, offset_baseline=True, offset_steps=False,
                         smooth_inf_response=False,
                         downsample=True, downsample_kw=None,
                         chrono_weights=None, eis_weights=None,
                         # basic fit control
                         l2_lambda_0=2, l1_lambda_0=0.1, R_inf_scale=100, inductance_scale=1e-4,
                         penalty_type='integral', derivative_weights=[0, 0, 1],
                         # partial derivative penalties
                         fxx_penalty=1, fyy_penalty=1, fxy_penalty=1,
                         # hyper-lambda options
                         hyper_l2_lambda=True, hl_l2_beta=100,
                         hyper_l1_lambda=False, hl_l1_beta=2.5,
                         # optimization control
                         xtol=1e-3, max_iter=20):

        sample_data, matrices = self._prep_for_hybrid_fit(times, i_signal, v_signal, psi_t, frequencies, z, psi_f,
                                                          independent_measurements, psi_static, psi_is_time, psi_is_i,
                                                          basis_psi, psi_epsilon, psi_basis_type, scale_psi,
                                                          basis_times, time_epsilon, time_basis_type, downsample,
                                                          downsample_kw, scale_signal, offset_steps,
                                                          smooth_inf_response, penalty_type,
                                                          fxx_penalty, fyy_penalty, fxy_penalty)

        sample_times, sample_i, sample_v, response_baseline, z_scaled = sample_data
        rm_drt, zm_drt, penalty_matrices, rm_inf, induc_rv, zm_inf, induc_zv = matrices

        # Offset voltage baseline
        if offset_baseline:
            response_offset = -response_baseline
        else:
            response_offset = 0
        scaled_response_signal = self.scaled_response_signal + response_offset

        # Add columns to rm (time response matrix) for v_baseline, R_inf, and inductance
        rm = np.empty((rm_drt.shape[0], rm_inf.shape[1] + rm_drt.shape[1] + 2))
        rm[:, 0] = 1  # v_baseline
        if self.fit_inductance:
            rm[:, 1] = induc_rv.copy() * inductance_scale  # inductance response
        else:
            rm[:, 1] = 0
        rm[:, 2: 2 + rm_inf.shape[1]] = rm_inf.copy()  # R_inf response
        rm[:, 2 + rm_inf.shape[1]:] = rm_drt.copy()  # DRT response

        # Add columns to zm (impedance matrix) for v_baseline, R_inf, and inductance
        zm = np.empty((zm_drt.shape[0], zm_inf.shape[1] + zm_drt.shape[1] + 2), dtype=complex)
        zm[:, 0] = 0  # v_baseline
        if self.fit_inductance:
            zm[:, 1] = induc_zv.copy() * inductance_scale  # inductance response
        else:
            zm[:, 1] = 0
        zm[:, 2: 2 + zm_inf.shape[1]] = zm_inf.copy()  # R_inf response
        zm[:, 2 + zm_inf.shape[1]:] = zm_drt.copy()  # DRT response

        # Construct hybrid response-impedance matrix
        rzm = np.vstack((rm, zm.real, zm.imag))

        # Construct effective m matrices for discrete penalty
        if penalty_type == 'discrete':
            # Extract differentiation matrices (l @ x = vector of derivatives at discrete tau, psi coordinates)
            l0 = penalty_matrices['l_drt']
            lx = penalty_matrices['lx_drt']
            ly = penalty_matrices['ly_drt']
            lxx = penalty_matrices['lxx_drt']
            lyy = penalty_matrices['lyy_drt']
            lxy = penalty_matrices['lxy_drt']

            # Construct squared derivative matrices (x.T @ M @ x = L2 norm of derivative vector)
            penalty_matrices['m0_drt'] = l0 @ l0
            penalty_matrices['m1_drt'] = fxx_penalty * lx @ lx + fyy_penalty * ly @ ly

            # TODO: investigate gaussian curvature vs. sum of squares
            # Can only get gaussian curvature, not its square, with x.T @ M @ x. It still seems to work decently...
            # m2_p = 0.5 * (fxx_penalty + fyy_penalty) * lxx @ lyy - fxy_penalty * lxy @ lxy
            m2_drt = fxx_penalty * lxx @ lxx + fyy_penalty * lyy @ lyy + fxy_penalty * lxy @ lxy
            penalty_matrices['m2_drt'] = m2_drt / m2_drt.shape[0]  # normalize by number of evaluation points

        # Construct L2 penalty matrices
        # l2_matrices = []
        # for m_p in [m0_p, m1_p, m2_p]:
        for order in [0, 1, 2]:
            m_drt = penalty_matrices[f'm{order}_drt']
            m_inf = penalty_matrices[f'm{order}_inf']

            m = np.zeros((m_drt.shape[0] + m_inf.shape[0] + 2, m_drt.shape[1] + m_inf.shape[1] + 2))
            # No penalty applied to v_baseline (m[0, 0] = 0)
            # Insert penalties for R_inf and inductance
            m[1, 1] = 1  # inductance
            m[2: 2 + m_inf.shape[0], 2: 2 + m_inf.shape[1]] = m_inf.copy() / R_inf_scale  # R_inf penalty matrix
            m[2 + m_inf.shape[0]:, 2 + m_inf.shape[1]:] = m_drt.copy()  # DRT penalty matrix
            # l2_matrices.append(m)
            penalty_matrices[f'm{order}'] = m
            print(order, np.sum(m_drt) / m_drt.shape[1], np.sum(m_inf) / m_inf.shape[1])

        # Indices of special variables - always nonneg or always unbounded
        special_indices = {'nonneg': np.concatenate(([int(1)], np.arange(2, 2 + m_inf.shape[1], dtype=int))),
                           'unbnd': np.array([0], dtype=int)}

        # Construct lambda vectors
        l1_lambda_vector = np.zeros(rm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        # Remaining entries set to l1_lambda_0 - for DRT coefficients
        l1_lambda_vector[2 + rm_inf.shape[1]:] = l1_lambda_0
        l2_lv = np.ones(rm.shape[1])  # First 2 entries are one - for v_baseline and inductance
        # Remaining entries set to l2_lambda_0 - for R_inf and DRT coefficients
        l2_lv[2:] = l2_lambda_0
        l2_lambda_vectors = [l2_lv.copy()] * 3  # need one vector for each order of the derivative

        # Construct hybrid response-impedance vector
        rzv = np.concatenate([scaled_response_signal, z_scaled.real, z_scaled.imag])

        # Get weight vector
        chrono_wv = format_chrono_weights(scaled_response_signal, chrono_weights)
        if eis_weights is None:
            # If no EIS weights provided, set weights to account for dataset size ratio
            eis_weights = len(sample_times) / len(z)
            print('eis weights:', eis_weights)
        eis_wv = format_eis_weights(frequencies, z, eis_weights, 'both')

        # Concatenate chrono and eis weights
        wv = np.concatenate([chrono_wv, eis_wv.real, eis_wv.imag])

        # Construct weighting matrix
        wm = np.diag(wv)

        # Apply weights to rzm and rzv (hybrid matrix and vector)
        wrm = wm @ rzm
        wrv = wm @ rzv

        # Perform ridge fit
        if hyper_l1_lambda or hyper_l2_lambda:
            # Hyper-ridge fit
            self.ridge_iter_history = []
            it = 0

            x = np.zeros(rm.shape[1]) + 1e-6

            # P = wrm.T @ wrm + lml
            # q = (-wrm.T @ wrv + l1_lambda_vector)
            # cost = 0.5 * x.T @ P @ x + q.T @ x

            while it < max_iter:

                x_prev = x.copy()

                x, cvx_result, converged = self._iterate_hyper_ridge(x_prev, 2 + rm_inf.shape[1], 1, nonneg,
                                                                     hyper_l1_lambda, hyper_l2_lambda, wrv, wrm,
                                                                     l1_lambda_vector, penalty_matrices,
                                                                     l2_lambda_vectors, derivative_weights, hl_l1_beta,
                                                                     l1_lambda_0, hl_l2_beta, l2_lambda_0, xtol, )

                if converged:
                    break
                elif it == max_iter - 1:
                    warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

                it += 1

        else:
            # Ordinary ridge fit

            # Make lml matrix for each derivative order
            l2_matrices = [penalty_matrices[f'm{k}'] for k in range(0, 3)]
            lml = qphb.calculate_qp_l2_matrix(derivative_weights, rho_vector, l2_matrices, l2_lambda_vectors,
                                              l2_lambda_0, penalty_type)

            cvx_result = qphb.solve_convex_opt(wrv, wrm, lml, l1_lambda_vector, nonneg, special_indices)

        # Store final cvxopt result
        self.cvx_result = cvx_result

        # Extract model parameters
        x_out = np.array(list(cvx_result['x']))
        self.fit_parameters = {'x': x_out[2 + rm_inf.shape[1]:] * self.coefficient_scale,
                               'x_inf': x_out[2: 2 + rm_inf.shape[1]] * self.coefficient_scale,
                               'v_baseline': (x_out[0] - response_offset) * self.response_signal_scale,
                               'v_sigma_tot': None,
                               'v_sigma_res': None}

        if self.fit_inductance:
            self.fit_parameters['inductance'] = x_out[1] * self.coefficient_scale * inductance_scale
        else:
            self.fit_parameters['inductance'] = 0

        self.fit_type = 'hybrid_ridge'

    # Prediction
    # --------------------------------------------
    def predict_distribution(self, tau=None, psi=None):
        """
		Predict distribution as function of tau
		:param ndarray tau: tau values at which to evaluate the distribution
		:return: array of distribution density values
		"""
        if tau is None:
            # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
            ltmin = np.min(np.log10(self.basis_tau))
            ltmax = np.max(np.log10(self.basis_tau))
            tau = np.logspace(ltmin - 1, ltmax + 1, 100)
        elif np.shape(tau) == ():
            # If single value provided, convert to list
            tau = [tau]

        if psi is None:
            # If psi is not provided, go 3 length parameters beyond self.basis_psi with finer spacing
            pmin = np.min(self.basis_psi)
            pmax = np.max(self.basis_psi)
            psi = np.linspace(pmin - 3 / self.psi_epsilon, pmax + - 3 / self.psi_epsilon, 100)
        elif np.shape(psi) == ():
            # If single value provided, convert to list
            psi = [psi]

        # Construct evaluation matrix
        em = mat2d.construct_2d_func_eval_matrix(
            self.basis_tau, self.basis_psi, tau, psi, self.tau_basis_type, self.psi_basis_type,
            self.tau_epsilon, self.psi_epsilon, 'f'
        )

        # Calculate (flattened) distribution array
        gamma = em @ self.fit_parameters['x']

        # Reshape to tau and psi dimensions
        gamma = gamma.reshape((len(psi), len(tau)))

        return gamma

    def predict_R_p(self, psi):
        # Evaluate distribution across tau grid spanning 3 length scales beyond basis_tau range
        ltmin = np.min(np.log10(self.basis_tau))
        ltmax = np.max(np.log10(self.basis_tau))
        tau = np.logspace(ltmin - 3 / self.tau_epsilon, ltmax + 3 / self.tau_epsilon, 200)

        # Get distribution at all psi values
        gamma = self.predict_distribution(tau, psi)

        # Integrate over tau to get polarization resistance
        R_p = np.trapz(gamma, axis=1, x=np.log(tau))

        return R_p

    def predict_R_inf(self, psi):
        # Construct R_inf evaluation matrix
        em = hybdrt.matrices.basis.construct_func_eval_matrix(self.basis_psi, psi, self.psi_basis_type, self.psi_epsilon, 0)

        # Get R_inf at all psi values
        R_inf = em @ self.fit_parameters['x_inf']

        return R_inf

    def predict_R_tot(self, psi):
        R_inf = self.predict_R_inf(psi)
        R_p = self.predict_R_p(psi)
        return R_inf + R_p

    def predict_response(self, times=None, psi=None, input_signal=None, offset_steps=None, smooth_inf_response=None,
                         op_mode=None,
                         independent_measurements=None, psi_static=None, psi_is_time=None, psi_is_i=None,
                         basis_times=None, time_basis_type=None, time_epsilon=None):
        # If times is not provided, use self.t_fit
        if times is None:
            if len(self._fit_subset_index) > 0:
                times = self.t_fit[self._fit_subset_index]
            else:
                times = self.t_fit

        # If psi is not provided, use self.psi_fit
        if psi is None:
            if len(self._psi_fit_subset_index) > 0:
                psi = self.psi_fit[self._psi_fit_subset_index]
            else:
                psi = self.psi_fit

        # If op_mode is not provided, use fitted op_mode
        if op_mode is None:
            op_mode = self.op_mode
        utils.check_op_mode(op_mode)

        # TODO: add getters and setters for basis_times etc.
        # If basis_times etc. not provided, use those used in fit
        if basis_times is None:
            basis_times = self.basis_times
        if time_basis_type is None:
            time_basis_type = self.time_basis_type
        if time_epsilon is None:
            time_epsilon = self.time_epsilon

        # Use fit kwargs if not provided
        if offset_steps is None:
            offset_steps = self.fit_kwargs['offset_steps']
        if smooth_inf_response is None:
            smooth_inf_response = self.fit_kwargs['smooth_inf_response']
        if independent_measurements is None:
            independent_measurements = self.fit_kwargs['independent_measurements']
        if psi_static is None:
            psi_static = self.fit_kwargs['psi_static']
        if psi_is_time is None:
            psi_is_time = self.fit_kwargs['psi_is_time']
        if psi_is_i is None:
            psi_is_i = self.fit_kwargs['psi_is_i']

        # Get prediction matrix
        if input_signal is None:
            # If input_signal is not provided, use fitted input signal
            rm, rm_inf, induc_rv = self._prep_response_prediction_matrix(times, psi, self.raw_input_signal,
                                                                    offset_steps, smooth_inf_response,
                                                                    op_mode,
                                                                    independent_measurements, psi_static, psi_is_time,
                                                                    psi_is_i,
                                                                    basis_times, time_basis_type, time_epsilon)
        else:
            rm, rm_inf, induc_rv = self._prep_response_prediction_matrix(times, psi, input_signal,
                                                                    # / self.input_signal_scale,
                                                                    offset_steps, smooth_inf_response,
                                                                    op_mode,
                                                                    independent_measurements, psi_static, psi_is_time,
                                                                    psi_is_i,
                                                                    basis_times, time_basis_type, time_epsilon)
        # rm, rm_inf, and induc_rv are scaled. Rescale to data scale
        # rm *= self.input_signal_scale
        # rm_inf *= self.input_signal_scale
        # induc_rv *= self.input_signal_scale

        response = rm @ self.fit_parameters['x'] \
                   + self.fit_parameters['inductance'] * induc_rv \
                   + self.fit_parameters['v_baseline'] \
                   + rm_inf @ self.fit_parameters['x_inf']

        return response

    # def predict_impedance(self, frequencies):
    #	  # Get matrix
    #	  zm = self._prep_impedance_prediction_matrix(frequencies)
    #
    #	  impedance = zm @ self.fit_parameters['x'] + self.fit_parameters['R_inf'] + \
    #				  self.fit_parameters['inductance'] * 2j * np.pi * frequencies
    #
    #	  return impedance

    # Plotting
    # -------------------------------------------------
    def plot_fit(self, ax=None, step_index=None, transform_time=False, linear_time_axis=False,
                 plot_data=True, data_kw=None,
                 data_label='', predict_kw={}, c='k', **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # Get times to plot
        if len(self._fit_subset_index) > 0:
            # If times is subset of previously fitted dataset, get subset
            times = self.t_fit[self._fit_subset_index]
        else:
            times = self.t_fit

        # If step_index provided, get times corresponding to those steps
        if step_index is not None:
            if np.shape(step_index) == ():
                step_index = [step_index]

            time_index = []
            for index in step_index:
                start_time = self.step_times[index]
                if index < len(self.step_times) - 1:
                    end_time = self.step_times[index + 1]
                else:
                    end_time = np.inf
                # Identify indices corresponding to time after step and before next step
                t_index = np.where((times >= start_time) & (times < end_time))
                # Include 2 data points prior to step
                t_index = np.concatenate((t_index[0][0:2] - 2, t_index[0]))
                time_index.append(t_index)
            time_index = np.unique(np.concatenate(time_index))
            times = times[time_index]
        else:
            time_index = np.arange(len(times), dtype=int)

        # Transform time to visualize each step on a log scale
        if transform_time:
            x, trans_functions = get_transformed_plot_time(times, self.step_times, linear_time_axis)
        else:
            x = times

        # Add linear time axis
        if linear_time_axis and transform_time:
            axt = add_linear_time_axis(ax, self.step_times, trans_functions)

        # Plot data
        if plot_data:
            ax.scatter(x, self.raw_response_signal[time_index], label=data_label, **data_kw)

        # Plot fitted response
        print('predicting response')
        start = time.time()
        y_hat = self.predict_response(times=times, input_signal=self.raw_input_signal[time_index],
                                      psi=self.psi_fit[time_index],
                                      **predict_kw)
        print('prediction time: {:.2f}'.format(time.time() - start))
        ax.plot(x, y_hat, c=c, **kw)

        # Labels
        if transform_time:
            ax.set_xlabel('$f(t)$')
        else:
            ax.set_xlabel('$t$ (s)')

        if self.op_mode == 'galvanostatic':
            ax.set_ylabel('$v$ (V)')
        elif self.op_mode == 'potentiostatic':
            ax.set_ylabel('$i$ (A)')

        fig.tight_layout()

        return ax

    # def plot_impedance_fit(self, frequencies=None, psi=None, axes=None, plot_type='nyquist',
    #						 plot_data=True, data_kw=None, data_label='', predict_kw={}, c='k', **kw):
    #
    #	  # Set default data plotting kwargs if not provided
    #	  if data_kw is None:
    #		  data_kw = dict(s=10, alpha=0.5)
    #
    #	  # Plot data if requested
    #	  if plot_data:
    #		  if self._f_fit_subset_index is not None:
    #			  f_fit = self.f_fit[self._f_fit_subset_index]
    #		  else:
    #			  f_fit = self.f_fit
    #		  data_df = construct_eis_df(f_fit, self.z_fit)
    #
    #	  # Get model impedance
    #	  if frequencies is None:
    #		  frequencies = f_fit
    #	  z_hat = self.predict_impedance(frequencies, **predict_kw)
    #	  df_hat = construct_eis_df(frequencies, z_hat)
    #
    #	  if plot_data:
    #		  z_data_concat = np.concatenate([data_df['Zreal'], data_df['Zimag']])
    #		  z_hat_concat = np.concatenate([df_hat['Zreal'], df_hat['Zimag']])
    #		  scale_prefix = get_common_scale_prefix([z_data_concat, z_hat_concat])
    #	  else:
    #		  scale_prefix = get_scale_prefix(df_hat)
    #
    #	  if plot_data:
    #		  axes = plot_eis(data_df, plot_type, axes=axes, scale_prefix=scale_prefix, label=data_label, **data_kw)
    #
    #	  axes = plot_eis(df_hat, plot_type, axes=axes, plot_func='plot', c=c, scale_prefix=scale_prefix,
    #					  **kw)
    #
    #	  fig = np.atleast_1d(axes)[0].get_figure()
    #	  fig.tight_layout()
    #
    #	  return axes

    def plot_residuals(self, ax=None, step_index=None, x_axis='f(t)', linear_time_axis=False, predict_kw={},
                       s=10, alpha=0.5, **kw):
        # Check x_axis string
        x_axis_options = ['t', 'f(t)', 'index']
        if x_axis not in x_axis_options:
            raise ValueError(f'Invalid x_axis option {x_axis}. Options: {x_axis_options}')

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        # Get times to plot
        if len(self._fit_subset_index) > 0:
            # If times is subset of previously fitted dataset, get subset
            times = self.t_fit[self._fit_subset_index]
        else:
            times = self.t_fit

        # If step_index provided, get times corresponding to those steps
        if step_index is not None:
            if np.shape(step_index) == ():
                step_index = [step_index]

            time_index = []
            for index in step_index:
                start_time = self.step_times[index]
                if index < len(self.step_times) - 1:
                    end_time = self.step_times[index] + 1
                else:
                    end_time = np.inf
                # Identify indices corresponding to time after step and before next step
                t_index = np.where((times >= start_time) & (times < end_time))
                # Include 2 data points prior to step
                t_index = np.concatenate((t_index[0][0:2] - 2, t_index[0]))
                time_index.append(t_index)
            time_index = np.unique(np.concatenate(time_index))
            times = times[time_index]
        else:
            time_index = np.arange(len(times), dtype=int)

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
        y_hat = self.predict_response(times=times, input_signal=self.raw_input_signal[time_index],
                                      psi=self.psi_fit[time_index],
                                      **predict_kw)

        # Calculate residuals
        y_err = y_hat - self.raw_response_signal[time_index]

        # Plot residuals
        ax.scatter(x, y_err, s=s, alpha=alpha, **kw)

        # Indicate zero
        ax.axhline(0, c='k', lw=1, zorder=-10)

        # Labels
        if self.op_mode == 'galvanostatic':
            ax.set_ylabel('$\hat{v} - v$ (V)')
        elif self.op_mode == 'potentiostatic':
            ax.set_ylabel('$\hat{i} - i$ (A)')

        fig.tight_layout()

        return ax

    def plot_distribution(self, tau=None, psi=None, ax=None, log_scale=False, colorbar=True, **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if tau is None:
            # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
            ltmin = np.min(np.log10(self.basis_tau))
            ltmax = np.max(np.log10(self.basis_tau))
            tau = np.logspace(ltmin - 1, ltmax + 1, 100)

        if psi is None:
            # If psi is not provided, go 2 scale parameters beyond self.basis_psi with finer spacing
            pmin = np.min(self.basis_psi)
            pmax = np.max(self.basis_psi)
            psi = np.linspace(pmin - 2 / self.psi_epsilon, pmax + 2 / self.psi_epsilon, 100)

        # Calculate distribution at evaluation points
        gamma = self.predict_distribution(tau, psi)

        if np.shape(tau) == ():
            # 1d plot along psi
            ax.plot(psi, gamma.flatten(), **kw)
            ax.set_xlabel(r'$\psi$')
            ax.set_ylabel(r'$\gamma$ ($\Omega$)')
        elif np.shape(psi) == ():
            # 1d plot along tau
            ax.plot(tau, gamma.flatten(), **kw)
            ax.set_xscale('log')
            ax.set_xlabel(r'$\tau$ (s)')
            ax.set_ylabel(r'$\gamma$ ($\Omega$)')
        else:
            # 2d colormap
            tt, pp = np.meshgrid(tau, psi)

            if log_scale:
                norm = colors.LogNorm(vmin=kw.get('vmin', None), vmax=kw.get('vmax', None))
                for key in ['vmin', 'vmax']:
                    if key in kw.keys():
                        del kw[key]
                cm = ax.pcolormesh(tt, pp, gamma, norm=norm, **kw)
            else:
                cm = ax.pcolormesh(tt, pp, gamma, **kw)

            ax.set_xscale('log')
            ax.set_xlabel(r'$\tau$ (s)')
            ax.set_ylabel(r'$\psi$')

            if colorbar:
                fig.colorbar(cm, ax=ax, label=r'$\gamma$ ($\Omega$)')

        fig.tight_layout()

        return ax

    # Preprocessing
    # -------------------------------------------------
    def _prep_for_fit(self, times, i_signal, v_signal, psi_chrono, frequencies, z, psi_eis,
                      step_times, downsample, downsample_kw,  offset_steps, smooth_inf_response,
                      independent_measurements, psi_static, psi_is_time, psi_is_i,
                      basis_psi, psi_epsilon, psi_basis_type, scale_psi,
                      basis_times, time_epsilon, time_basis_type,
                      penalty_type, derivative_weights, scale_data, rp_scale,
                      fxx_penalty, fyy_penalty, fxy_penalty):

        start_time = time.time()

        # Checks
        utils.check_penalty_type(penalty_type)
        utils.check_basis_type(psi_basis_type)
        utils.check_basis_type(time_basis_type)
        utils.check_eis_data(frequencies, z)
        utils.check_chrono_data(times, i_signal, v_signal)

        # Clear map samples
        self.map_samples = None
        self.map_sample_kw = None

        # Store kwargs relevant to matrix calculation - this is used to replicate fit parameters
        # for default prediction
        self.fit_kwargs = {'offset_steps': offset_steps,
                           'smooth_inf_response': smooth_inf_response,
                           'independent_measurements': independent_measurements,
                           'psi_static': psi_static,
                           'psi_is_time': psi_is_time,
                           'psi_is_i': psi_is_i}

        # If chrono data provided, get input signal step information
        sample_times, sample_i, sample_v, step_times, step_sizes, tau_rise = self.process_chrono_signals(
            times, i_signal, v_signal, step_times, offset_steps, downsample, downsample_kw
        )

        # Set basis_tau if not provided - must have chrono step information
        if self.basis_tau is None:
            # Default: 10 ppd basis grid. Extend basis tau one decade beyond data on each end
            self.basis_tau = pp.get_basis_tau(frequencies, times, step_times)

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

        # Set basis_times at step times if not provided
        if basis_times is None:
            basis_times = step_times.copy()
        print('basis times:', basis_times)

        if time_epsilon is None and time_basis_type != 'step':
            dt = np.abs(np.mean(np.diff(basis_times)))
            time_epsilon = 1 / dt

        # Create basis_psi if not provided
        if basis_psi is None:
            # step_indices =
            basis_psi = np.linspace(np.min(psi_chrono), np.max(psi_chrono), 5)

        if psi_epsilon is None:
            # Determine psi_epsilon from basis_psi spacing
            dpsi = np.abs(np.mean(np.diff(basis_psi)))
            if psi_basis_type == 'gaussian':
                psi_epsilon = 1 / dpsi  # 1.5 / dpsi
            elif psi_basis_type == 'pw_linear':
                psi_epsilon = 1 / dpsi

        # Write basis parameters to attributes
        self.basis_psi = basis_psi
        self.psi_epsilon = psi_epsilon
        self.psi_basis_type = psi_basis_type

        self.basis_times = basis_times
        self.time_epsilon = time_epsilon
        self.time_basis_type = time_basis_type

        if sample_times is not None:
            # # Perform scaling
            # i_signal_scaled, v_signal_scaled = self.scale_signal(sample_times, sample_i, sample_v, step_times,
            #                                                      step_sizes,
            #                                                      apply_scaling=scale_signal)
            # # step_sizes /= self.input_signal_scale	 # steps must be determined prior to scaling
            # print('Finished signal scaling')
            #
            # # Estimate baseline
            # if self.op_mode == 'galvanostatic':
            #     response_baseline = np.mean(v_signal_scaled[sample_times < step_times[0]])
            # elif self.op_mode == 'potentiostatic':
            #     response_baseline = np.mean(i_signal_scaled[sample_times < step_times[0]])

            # Get matrices
            rm, rm_inf, induc_rv = self._prep_chrono_fit_matrices(sample_times, step_times,
                                                                                    step_sizes, psi_chrono, tau_rise,
                                                                                    smooth_inf_response, penalty_type,
                                                                                    basis_psi, psi_basis_type,
                                                                                    psi_epsilon, scale_psi, basis_times,
                                                                                    time_basis_type, time_epsilon,
                                                                                    independent_measurements,
                                                                                    psi_static, psi_is_time, psi_is_i,
                                                                                    fxx_penalty, fyy_penalty,
                                                                                    fxy_penalty)

            # # Scale response matrices to input_signal_scale
            # rm = rm / self.input_signal_scale
            # rm_inf = rm_inf / self.input_signal_scale
            # induc_rv = induc_rv / self.input_signal_scale
        else:
            rm = None
            rm_inf = None
            induc_rv = None

        if frequencies is not None:
            zm, zm_inf, induc_zv = self._prep_impedance_fit_matrices(frequencies, psi_eis, independent_measurements,
                                                                     psi_static, psi_is_time)
        else:
            zm = None
            zm_inf = None
            induc_zv = None

        # Calculate penalty matrices
        penalty_matrices = self._prep_penalty_matrices(penalty_type, derivative_weights, fxx_penalty, fyy_penalty,
                                                       fxy_penalty)

        # Perform scaling
        i_signal_scaled, v_signal_scaled, z_scaled = self.scale_data(sample_times, sample_i, sample_v, step_times,
                                                                     step_sizes, z, scale_data, rp_scale)
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
            rm_inf = rm_inf / self.input_signal_scale
            induc_rv = induc_rv / self.input_signal_scale

        print('Finished prep_for_fit in {:.2f} s'.format(time.time() - start_time))

        return (sample_times, i_signal_scaled, v_signal_scaled, response_baseline, z_scaled), \
               (rm, rm_inf, induc_rv, zm, zm_inf, induc_zv, penalty_matrices)

    # def _prep_for_hybrid_fit(self, times, i_signal, v_signal, psi_t, frequencies, z, psi_f,
    #                          independent_measurements, psi_static, psi_is_time, psi_is_i,
    #                          basis_psi, psi_epsilon, psi_basis_type, scale_psi,
    #                          basis_times, time_epsilon, time_basis_type,
    #                          downsample, downsample_kw, scale_signal, offset_steps, smooth_inf_response, penalty_type,
    #                          fxx_penalty, fyy_penalty, fxy_penalty):
    #
    #     # Perform preprocessing and matrix preparation for regular fit
    #     sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, psi_t, None, None, None,
    #                                                independent_measurements, psi_static, psi_is_time, psi_is_i,
    #                                                basis_psi, psi_epsilon, psi_basis_type, scale_psi, basis_times,
    #                                                time_epsilon, time_basis_type, downsample, downsample_kw,
    #                                                scale_signal, offset_steps, smooth_inf_response, penalty_type,
    #                                                fxx_penalty, fyy_penalty, fxy_penalty)
    #
    #     # Extract time response data from sample_data
    #     sample_times, i_signal_scaled, v_signal_scaled, response_baseline = sample_data
    #
    #     # Extract time response matrices and penalty matrics
    #     rm, rm_inf, induc_rv, penalty_matrices = matrices
    #
    #     # Scale measured impedance
    #     z_scaled = self.scale_impedance(z)
    #
    #     self.z_fit = z.copy()
    #     self.z_fit_scaled = z_scaled.copy()
    #
    #     # Construct impedance matrix
    #     zm = self._prep_impedance_fit_matrices(frequencies, psi_f, independent_measurements, psi_static, psi_is_time)
    #
    #     # Construct R_inf impedance matrix
    #     zm_inf = mat1d.construct_func_eval_matrix(self.basis_psi, psi_f, self.psi_basis_type, self.psi_epsilon, order=0)
    #
    #     # Construct inductance impedance vector
    #     induc_zv = mat1d.construct_inductance_z_vector(frequencies)
    #
    #     return (sample_times, i_signal_scaled, v_signal_scaled, response_baseline, z_scaled), \
    #            (rm, zm, penalty_matrices, rm_inf, induc_rv, zm_inf, induc_zv)

    def _prep_chrono_fit_matrices(self, times, step_times, step_sizes, psi, tau_rise, smooth_inf_response, penalty_type,
                           basis_psi, psi_basis_type, psi_epsilon, scale_psi,
                           basis_times, time_basis_type, time_epsilon,
                           independent_measurements, psi_static, psi_is_time, psi_is_i,
                           fxx_penalty, fyy_penalty, fxy_penalty):
        utils.check_penalty_type(penalty_type)
        utils.check_basis_type(psi_basis_type)
        utils.check_basis_type(time_basis_type)

        # Set t_fit and psi_fit
        self.set_t_psi_fit(times, psi)

        # Scale psi such that psi_epsilon matches tau_epsilon
        # psi_scaled is used only for penalty matrices
        psi_scaled, basis_psi_scaled, psi_epsilon_scaled = self.scale_psi(psi, basis_psi, psi_epsilon, self.tau_epsilon,
                                                                          scale_psi)

        # Create psi to time mapping if needed
        if not psi_is_time and not independent_measurements:
            psi_map_coef = hybdrt.matrices.basis.fit_basis_functions(times, psi, basis_times, time_basis_type, time_epsilon)
            # TODO: determine whether psi_hat or psi_exp should be used for response matrix calculation
            # For current implementation, psi is actually not used (psi mapping takes its place)
            # For other implementations, this may matter.
            # This also currently matters for R_inf response
            psi_hat = hybdrt.matrices.basis.evaluate_basis_fit(psi_map_coef, times, basis_times, time_basis_type, time_epsilon)
            psi_hat_scaled = psi_hat / self.psi_scale

            # Overwrite psi with psi_hat
            psi = psi_hat
            psi_scaled = psi_hat_scaled
        else:
            psi_map_coef = None
            psi_hat = None
            psi_hat_scaled = None

        self.psi_map_coef = psi_map_coef
        self.psi_hat = psi_hat
        self.psi_hat_scaled = psi_hat_scaled
        print(psi_map_coef)

        # Recalculate matrices if necessary
        if self._recalc_chrono_fit_matrix:
            print('Calculating matrices')
            rm = mat2d.construct_2d_response_matrix(self.basis_tau, times, psi, basis_psi,
                                                    independent_measurements,
                                                    psi_static, psi_is_time, psi_is_i,
                                                    self.step_model, step_times, step_sizes,
                                                    self.tau_basis_type, self.tau_epsilon,
                                                    psi_basis_type, psi_epsilon,
                                                    time_basis_type, basis_times, time_epsilon, psi_map_coef,
                                                    tau_rise, self.op_mode)
            rm, rm_layered = rm

            print('Constructed response matrix')
            self.fit_matrices['response'] = rm.copy()
            self.fit_matrices['rm_layered'] = rm_layered.copy()

            if self.step_model == 'expdecay':
                induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times, step_sizes, tau_rise,
                                                                      self.op_mode)
            else:
                induc_rv = np.zeros(len(times))
            self.fit_matrices['inductance_response'] = induc_rv.copy()

        # Otherwise, reuse existing matrices as appropriate
        elif self._t_fit_subset_index is not None:
            # times is a subset of self.t_fit. Use sub-matrices of existing A array; do not overwrite
            rm = self.fit_matrices['response'][self._t_fit_subset_index, :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._t_fit_subset_index].copy()
        else:
            # All matrix parameters are the same. Use existing matrices
            rm = self.fit_matrices['response'].copy()
            induc_rv = self.fit_matrices['inductance_response'].copy()

        # With all matrices calculated, set recalc flag to False
        self._recalc_chrono_fit_matrix = False
        self._recalc_chrono_prediction_matrix = False

        # Calculate R_inf response matrix - always recalculate since this is fast and depends on input_signal
        rm_inf = mat2d.construct_2d_inf_response_matrix(self.basis_psi, psi, self.psi_basis_type,
                                                        psi_epsilon, times, self.raw_input_signal, step_times,
                                                        step_sizes, tau_rise, self.step_model, smooth_inf_response)
        self.fit_matrices['inf_response'] = rm_inf.copy()

        return rm, rm_inf, induc_rv

    def _prep_impedance_fit_matrices(self, frequencies, psi, independent_measurements, psi_static, psi_is_time):
        self.f_fit = frequencies
        if self._recalc_eis_fit_matrix:
            # Matrix calculation is required
            # Real matrix
            zmr = mat2d.construct_2d_impedance_matrix(frequencies, psi, 'real', self.basis_tau, self.basis_psi,
                                                      independent_measurements, psi_static, psi_is_time,
                                                      self.tau_basis_type, self.tau_epsilon,
                                                      self.psi_basis_type, self.psi_epsilon,
                                                      integrate_method='trapz', zga_params=self.zga_params)
            # Imaginary matrix
            zmi = mat2d.construct_2d_impedance_matrix(frequencies, psi, 'imag', self.basis_tau, self.basis_psi,
                                                      independent_measurements, psi_static, psi_is_time,
                                                      self.tau_basis_type, self.tau_epsilon,
                                                      self.psi_basis_type, self.psi_epsilon,
                                                      integrate_method='trapz', zga_params=self.zga_params)
            # Complex matrix
            zm = zmr + 1j * zmi
            self.fit_matrices['impedance'] = zm.copy()

        # TODO: handle intersection of freq and psi_exp for _recalc_eis_fit_matrix
        elif self._f_fit_subset_index is not None:
            # frequencies is a subset of self.f_fit. Use sub-matrices of existing matrix; do not overwrite
            zm = self.fit_matrices['impedance'][self._f_fit_subset_index, :].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            zm = self.fit_matrices['impedance'].copy()

        # Construct R_inf impedance matrix
        zm_inf = hybdrt.matrices.basis.construct_func_eval_matrix(self.basis_psi, psi, self.psi_basis_type, self.psi_epsilon,
                                                                  order=0)
        self.fit_matrices['inf_impedance'] = zm_inf.copy()

        # Construct inductance impedance vector
        induc_zv = mat1d.construct_inductance_impedance_vector(frequencies)

        return zm, zm_inf, induc_zv

    def _prep_penalty_matrices(self, penalty_type, derivative_weights, fxx_penalty, fyy_penalty, fxy_penalty):
        # Handle penalty (derivative) matrices separately - depend only on self.basis_tau, self.op_mode, and penalty
        # Always recalculate since these depend on some args passed to fit, rather than attributes
        # (and are quick to calculate)
        penalty_matrices = {}
        if penalty_type == 'discrete':
            for func_string in ['', 'x', 'y', 'xx', 'yy', 'xy']:
                dk = mat2d.construct_2d_func_eval_matrix(self.basis_tau, self.basis_psi_scaled, None, None,
                                                         self.tau_basis_type, self.psi_basis_type,
                                                         self.tau_epsilon, self.psi_epsilon_scaled,
                                                   f'f{func_string}')
                penalty_matrices[f'l{func_string}_drt'] = dk.copy()
            # for order in range(len(derivative_weights)):
            #     dk_inf = mat1d.construct_func_eval_matrix(self.basis_psi_scaled, None, self.psi_basis_type,
            #                                             self.psi_epsilon_scaled, order, None)
            #     penalty_matrices[f'l{order}']
        elif penalty_type == 'integral':
            for order in range(len(derivative_weights)):
                # DRT penalty matrix
                dk = mat2d.construct_2d_integrated_derivative_matrix(self.basis_tau, self.basis_psi_scaled, self.tau_basis_type,
                                                                     self.psi_basis_type, self.tau_epsilon, self.psi_epsilon_scaled,
                                                                     fxx_penalty, fyy_penalty, fxy_penalty, order)
                penalty_matrices[f'm{order}_drt'] = dk.copy()

                # R_inf penalty matrix
                # dk_inf = construct_func_eval_matrix(basis_psi_scaled, None, psi_basis_type, psi_epsilon_scaled, order)
                dk_inf = mat1d.construct_integrated_derivative_matrix(self.basis_psi_scaled, self.psi_basis_type, order,
                                                                      self.psi_epsilon_scaled)
                dk_inf /= len(self.basis_psi)  # normalize to number of psi basis points
                penalty_matrices[f'm{order}_inf'] = dk_inf.copy()

        self.fit_matrices.update(penalty_matrices)

        return penalty_matrices

    def _format_qp_matrices(self, rm_drt, rm_inf, induc_rv, zm_drt, zm_inf, penalty_matrices,
                            v_baseline_penalty, R_inf_penalty, inductance_penalty, vz_offset_scale,
                            inductance_scale, penalty_type, derivative_weights):
        """
        Format matrices for quadratic programming solution
        :param derivative_weights:
        :param v_baseline_scale:
        :param rm_drt:
        :param penalty_matrices:
        :param R_inf_scale:
        :param inductance_scale:
        :param penalty_type:
        :return:
        """
        # Count number of special params for padding
        num_special = self.get_qp_mat_offset()

        # Extract indices for convenience
        special_indices = {
            k: (
                self.special_qp_params[k]['index'],
                self.special_qp_params[k]['index'] + self.special_qp_params[k]['size']
            )
            for k in self.special_qp_params.keys()
        }

        # Store inductance scale for reference
        self.inductance_scale = inductance_scale

        # Add columns to rm for v_baseline, R_inf, and inductance
        if rm_drt is not None:
            rm = np.empty((rm_drt.shape[0], rm_drt.shape[1] + num_special))

            # Add entries for special parameters
            rm[:, special_indices['v_baseline'][0]: special_indices['v_baseline'][1]] = 1
            rm[:, special_indices['R_inf'][0]: special_indices['R_inf'][1]] = rm_inf  # galvanostatic mode only

            if 'inductance' in special_indices.keys():
                rm[:, special_indices['inductance'][0]: special_indices['inductance'][1]] = induc_rv * inductance_scale

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                rm[:, special_indices['vz_offset'][0]: special_indices['vz_offset'][1]] = 0

            # Insert main DRT matrix
            rm[:, num_special:] = rm_drt
        else:
            # No chrono matrix
            rm = None

        # Add columns to zm (impedance matrix) for v_baseline, R_inf, and inductance
        if zm_drt is not None:
            zm = np.empty((zm_drt.shape[0], zm_drt.shape[1] + num_special), dtype=complex)

            # Add entries for special parameters
            zm[:, special_indices['v_baseline'][0]: special_indices['v_baseline'][1]] = 0 # v_baseline has no effect on impedance
            zm[:, special_indices['R_inf'][0]: special_indices['R_inf'][1]] = zm_inf

            if 'inductance' in special_indices.keys():
                zm[:, special_indices['inductance'][0]: special_indices['inductance'][1]] = induc_zv * inductance_scale

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                zm[:, special_indices['vz_offset'][0]: special_indices['vz_offset'][1]] = 0

            # Insert main DRT matrix
            zm[:, num_special:] = zm_drt  # DRT response

            # Stack real and imag matrices
            zm = np.vstack([zm.real, zm.imag])
        else:
            # No EIS matrix
            zm = None

        # Construct L2 penalty matrices
        penalty_matrices_out = {}
        if penalty_type == 'integral':
            for n in range(len(derivative_weights)):
                # Get penalty matrix for DRT coefficients
                m_drt = penalty_matrices[f'm{n}_drt']
                m_inf = penalty_matrices[f'm{n}_inf']

                # Add rows/columns for v_baseline, inductance, and R_inf
                m = np.zeros((m_drt.shape[0] + num_special, m_drt.shape[1] + num_special))

                # Insert penalties for special parameters
                if 'v_baseline' in special_indices.keys():
                    m[special_indices['v_baseline'][0]: special_indices['v_baseline'][1],
                    special_indices['v_baseline'][0]: special_indices['v_baseline'][1]
                    ] = v_baseline_penalty * v_baseline_penalty

                if 'inductance' in special_indices.keys():
                    m[special_indices['inductance'][0]: special_indices['inductance'][1],
                    special_indices['inductance'][0]: special_indices['inductance'][1]
                    ] = inductance_penalty * inductance_penalty

                if 'vz_offset' in special_indices.keys():
                    m[special_indices['vz_offset'][0]: special_indices['vz_offset'][1],
                    special_indices['vz_offset'][0]: special_indices['vz_offset'][1]
                    ] = 1 / vz_offset_scale

                m[special_indices['R_inf'][0]: special_indices['R_inf'][1],
                special_indices['R_inf'][0]: special_indices['R_inf'][1]
                ] = m_inf * R_inf_penalty

                # Insert main DRT matrix
                m[num_special:, num_special:] = m_drt

                penalty_matrices_out[f'm{n}'] = m.copy()
        # elif penalty_type == 'discrete':
        #     # TODO: return to this later if it will be used. Need to figure out how to handle l, lx, ly, etc for
        #     # special parameters
        #     for n in range(len(derivative_weights)):
        #         # Get penalty matrix for DRT coefficients
        #         l_drt = penalty_matrices[f'l{n}']
        #
        #         # Add rows/columns for v_baseline, inductance, and R_inf
        #         l = np.zeros((l_drt.shape[0] + num_special, l_drt.shape[1] + num_special))
        #
        #         # Insert penalties for special parameters
        #         if 'v_baseline' in special_indices.keys():
        #             l[special_indices['v_baseline'], special_indices['v_baseline']] = v_baseline_penalty ** 0.5
        #         if 'inductance' in special_indices.keys():
        #             l[special_indices['inductance'], special_indices['inductance']] = inductance_penalty ** 0.5
        #         l[special_indices['R_inf'], special_indices['R_inf']] = R_inf_penalty ** 0.5
        #
        #         # Add entry for vz_offset if applicable
        #         if 'vz_offset' in special_indices.keys():
        #             m[special_indices['vz_offset'], special_indices['vz_offset']] = 1 / vz_offset_scale ** 0.5
        #
        #         # Insert main DRT matrix
        #         l[num_special:, num_special:] = l_drt
        #
        #         penalty_matrices_out[f'l{n}'] = l.copy()
        #
        #         # Calculate norm matrix
        #         penalty_matrices_out[f'm{n}'] = l.T @ l


        return rm, zm, penalty_matrices_out

    def extract_qphb_parameters(self, x):
        special_indices = {
            k: (
                self.special_qp_params[k]['index'],
                self.special_qp_params[k]['index'] + self.special_qp_params[k]['size']
            )
            for k in self.special_qp_params.keys()
        }

        fit_parameters = {'x': x[self.get_qp_mat_offset():] * self.coefficient_scale,
                          'x_inf': x[special_indices['R_inf'][0]: special_indices['R_inf'][1]] * self.coefficient_scale,
                          'v_baseline': (x[special_indices['v_baseline'][0]] - self.response_offset) \
                                         * self.response_signal_scale
                          }

        if 'vz_offset' in special_indices.keys():
            fit_parameters['vz_offset'] = x[special_indices['vz_offset'][0]]

        if 'inductance' in special_indices.keys():
            fit_parameters['inductance'] = x[special_indices['inductance'][0]] \
                                                * self.coefficient_scale * self.inductance_scale
        else:
            fit_parameters['inductance'] = 0

        return fit_parameters


    # def _prep_impedance_prediction_matrix(self, frequencies, psi, independent_measurements, psi_static, psi_is_time, psi_is_i,):
    #	  self.f_predict = frequencies
    #	  if self._recalc_eis_prediction_matrix:
    #		  # Matrix calculation is required
    #		  # Real matrix
    #		  zmr = construct_impedance_matrix(frequencies, 'real', tau=self.basis_tau, basis_type=self.tau_basis_type,
    #										   epsilon=self.tau_epsilon, zga_params=self.zga_params)
    #		  # Imaginary matrix
    #		  zmi = construct_impedance_matrix(frequencies, 'imag', tau=self.basis_tau, basis_type=self.tau_basis_type,
    #										   epsilon=self.tau_epsilon, zga_params=self.zga_params)
    #		  # Complex matrix
    #		  zm = zmr + 1j * zmi
    #		  self.prediction_matrices['impedance'] = zm.copy()
    #	  elif len(self._f_predict_subset_index) > 0:
    #		  # frequencies is a subset of self.f_predict. Use sub-matrices of existing matrix; do not overwrite
    #		  zm = self.prediction_matrices['impedance'][self._f_predict_subset_index, :].copy()
    #	  else:
    #		  # All matrix parameters are the same. Use existing matrix
    #		  zm = self.prediction_matrices['impedance'].copy()
    #
    #	  return zm

    def scale_psi(self, psi, basis_psi, psi_epsilon, tau_epsilon, apply_scaling):
        if apply_scaling:
            self.psi_scale = tau_epsilon / psi_epsilon
        else:
            self.psi_scale = 1
        psi_scaled = psi / self.psi_scale
        basis_psi_scaled = basis_psi / self.psi_scale
        psi_epsilon_scaled = psi_epsilon * self.psi_scale

        self.basis_psi_scaled = basis_psi_scaled.copy()
        self.psi_epsilon_scaled = psi_epsilon_scaled

        return psi_scaled, basis_psi_scaled, psi_epsilon_scaled

    def _prep_response_prediction_matrix(self, times, psi, input_signal, offset_steps, smooth_inf_response, op_mode,
                                         independent_measurements, psi_static, psi_is_time, psi_is_i,
                                         basis_times, time_basis_type, time_epsilon):
        print('starting _prep_response_prediction_matrix')
        # Set t_predict and psi_predict and update recalc flag and subset index
        # self.t_predict = times
        # self.psi_predict = psi
        start = time.time()
        self.set_predict_recalc_status(times, psi, input_signal)
        print('setter time: {:.2f}'.format(time.time() - start))
        print('ran setters')

        # # Identify steps in applied signal
        # step_times, step_sizes, tau_rise = process_input_signal(times, input_signal, self.step_model, offset_steps)

        if self._recalc_chrono_prediction_matrix:
            step_times, step_sizes, tau_rise = pp.process_input_signal(times, input_signal, self.step_model, offset_steps)
        else:
            step_times = self.step_times
            step_sizes = self.step_sizes  # / self.input_signal_scale
            tau_rise = self.tau_rise
        # step_times = self.predict_step_times
        # step_sizes = self.predict_step_sizes
        # tau_rise = self.predict_tau_rise

        print('_recalc_chrono_prediction_matrix:', self._recalc_chrono_prediction_matrix)

        # Scale psi
        # psi_scaled = psi / self.psi_scale

        # Create psi to time mapping if needed
        if not psi_is_time and not independent_measurements:
            if self._recalc_chrono_prediction_matrix:
                psi_map_coef = hybdrt.matrices.basis.fit_basis_functions(times, psi, basis_times, time_basis_type, time_epsilon)
            else:
                psi_map_coef = self.psi_map_coef
            psi_hat = hybdrt.matrices.basis.evaluate_basis_fit(psi_map_coef, times, basis_times, time_basis_type, time_epsilon)
            psi = psi_hat
        # print('psi_hat', psi_hat)
        else:
            psi_map_coef = None

        if self._recalc_chrono_prediction_matrix:
            # Matrix calculation is required
            rm = mat2d.construct_2d_response_matrix(self.basis_tau, times, psi, self.basis_psi,
                                                    independent_measurements,
                                                    psi_static, psi_is_time, psi_is_i,
                                                    self.step_model, step_times, step_sizes,
                                                    self.tau_basis_type, self.tau_epsilon,
                                                    self.psi_basis_type, self.psi_epsilon,
                                                    time_basis_type, basis_times, time_epsilon, psi_map_coef,
                                                    tau_rise, self.op_mode)
            rm, rm_layered = rm
            if self.step_model == 'expdecay':
                induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times, step_sizes, tau_rise,
                                                                      op_mode)
            else:
                induc_rv = np.zeros(len(times))
            self.prediction_matrices = {'response': rm.copy(), 'inductance_response': induc_rv.copy()}

            # With prediction matrices calculated, set recalc flag to False
            self._recalc_chrono_prediction_matrix = False
            print('Calculated response prediction matrices')
        elif self._predict_subset_index[0] == 'predict':
            # times and psi are a subset of self.t_predict. Use sub-matrices of existing matrix; do not overwrite
            rm = self.prediction_matrices['response'][self._predict_subset_index[1], :].copy()
            induc_rv = self.prediction_matrices['inductance_response'][self._predict_subset_index[1]].copy()
        elif self._predict_subset_index[0] == 'fit':
            # times is a subset of self.t_fit. Use sub-matrices of existing matrix; do not overwrite
            rm = self.fit_matrices['response'][self._predict_subset_index[1], :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._predict_subset_index[1]].copy()
        # else:
        #	  # All matrix parameters are the same. Use existing matrix
        #	  rm = self.prediction_matrices['response'].copy()
        #	  induc_rv = self.fit_matrices['inductance_response'].copy()

        # Calculate R_inf response matrix - always recalculate since this is fast and depends on input_signal
        rm_inf = mat2d.construct_2d_inf_response_matrix(self.basis_psi, psi, self.psi_basis_type,
                                                        self.psi_epsilon, times, input_signal, step_times, step_sizes,
                                                        tau_rise, self.step_model, smooth_inf_response)
        self.prediction_matrices['inf_response'] = rm_inf.copy()

        return rm, rm_inf, induc_rv

    # Getters and setters to control matrix calculation
    # -------------------------------------------------
    def get_basis_psi(self):
        return self._basis_psi

    def set_basis_psi(self, basis_psi):
        if hasattr(self, 'basis_psi'):
            if not utils.check_equality(basis_psi, self.basis_psi):
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
                self._recalc_eis_prediction_matrix = True
        self._basis_psi = basis_psi

    basis_psi = property(get_basis_psi, set_basis_psi)

    def get_psi_basis_type(self):
        return self._psi_basis_type

    def set_psi_basis_type(self, psi_basis_type):
        utils.check_basis_type(psi_basis_type)
        if hasattr(self, 'psi_basis_type'):
            if psi_basis_type != self.psi_basis_type:
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
                self._recalc_eis_prediction_matrix = True
        self._psi_basis_type = psi_basis_type

    psi_basis_type = property(get_psi_basis_type, set_psi_basis_type)

    def get_psi_epsilon(self):
        return self._psi_epsilon

    def set_psi_epsilon(self, psi_epsilon):
        if hasattr(self, 'psi_epsilon'):
            if psi_epsilon != self.psi_epsilon:
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
                self._recalc_eis_prediction_matrix = True
        self._psi_epsilon = psi_epsilon

    psi_epsilon = property(get_psi_epsilon, set_psi_epsilon)

    def get_psi_fit(self):
        return self._psi_fit

    def set_psi_fit(self, psi):
        if hasattr(self, 'psi_fit'):
            self._psi_fit_subset_index = []
            # Check if psi is the same as self.psi_fit
            if utils.check_equality(utils.rel_round(self._psi_fit, 10), utils.rel_round(psi, 10)):
                self._psi_fit = psi
            # self._recalc_chrono_fit_matrix = False
            # Check if psi is a subset of self.psi_fit
            else:
                # If psi is a subset of self.psi_fit, we can use sub-matrices of the existing matrices
                # In this case, we should not update self._psi_fit
                if utils.is_subset(psi, self.psi_fit):
                    self._psi_fit_subset_index = utils.get_subset_index(psi, self.psi_fit)
                # self._recalc_chrono_fit_matrix = False
                else:
                    # if psi is not a subset of self.psi__fit, must recalculate matrices
                    self._psi_fit = psi
                    self._recalc_chrono_fit_matrix = True
        else:
            self._psi_fit = psi
            self._recalc_chrono_fit_matrix = True

    psi_fit = property(get_psi_fit, set_psi_fit)

    def set_t_psi_fit(self, times, psi):
        # First update individual statuses
        self.t_fit = times
        self.psi_fit = psi
        self._fit_subset_index = []
        # Next, compare statuses
        if self._t_fit_subset_index is not None or len(self._psi_fit_subset_index) > 0:
            # If using a subset of previous fit data, subset indices for times and psi must match
            if utils.check_equality(self._t_fit_subset_index, self._psi_fit_subset_index):
                # Indices match - set fit_subset_index to match one
                self._fit_subset_index = self._t_fit_subset_index.copy()
            else:
                # Indices don't match - must recalculate matrices
                self._recalc_chrono_fit_matrix = True
                # Set t_fit and psi_fit directly
                self._t_fit = times
                self._psi_fit = psi
                # Overwrite subset indices
                self._t_fit_subset_index = []
                self._psi_fit_subset_index = []

    # Otherwise: either recalc flag is already True, or psi==psi_fit and times==t_fit

    def get_psi_predict(self):
        return self._psi_predict

    def set_psi_predict(self, psi):
        if hasattr(self, 'psi_predict'):
            self._psi_predict_subset_index = ('', [])
            self._psi_predict_eq_psi_fit = False
            # Check if psi is the same as self.psi_fit
            if utils.check_equality(utils.rel_round(self._psi_fit, 10), utils.rel_round(psi, 10)):
                # self._psi_predict = psi
                self._psi_predict_eq_psi_fit = True
                # don't update recalc status here - another attribute change may have set this to True
                # self._recalc_chrono_prediction_matrix = False
                print('a')
            # Check if psi is the same as self.psi_predict
            elif utils.check_equality(utils.rel_round(self._psi_predict, 10), utils.rel_round(psi, 10)):
                self._psi_predict = psi
                # self._recalc_chrono_prediction_matrix = False
                print('b')
            # Check if psi is a subset of self.psi_fit or self.psi_predict
            else:
                # If psi is a subset of self.psi_fit or self.psi_predict,
                # we can use sub-matrices of the existing matrices
                if utils.is_subset(psi, self._psi_predict):
                    # psi is a subset of psi_predict
                    self._psi_predict_subset_index = (
                        'predict',
                        utils.get_subset_index(psi, self.psi_predict)
                    )
                    # self._recalc_chrono_prediction_matrix = False
                    print('c')
                # In this case, we should not update self._psi_predict
                elif utils.is_subset(psi, self._psi_fit):
                    # psi is a subset of psi_fit
                    self._psi_predict_subset_index = (
                        'fit',
                        utils.get_subset_index(psi, self.psi_fit)
                    )
                    # self._recalc_chrono_prediction_matrix = False
                    print('d')
                # In this case, we should not update self._psi_predict
                else:
                    # if psi is not a subset of self.psi_fit or self.psi_predict, must calculate matrices
                    self._psi_predict = psi
                    self._recalc_chrono_prediction_matrix = True
                    print('e')
        else:
            self._psi_predict = psi
            self._recalc_chrono_prediction_matrix = True
            print('f')

    psi_predict = property(get_psi_predict, set_psi_predict)

    def set_predict_recalc_status(self, times, psi, input_signal):
        # First update individual statuses
        self.t_predict = times
        self.psi_predict = psi
        # self.predict_step_times, self.predict_step_sizes, self.predict_tau_rise = \
        #	  process_input_signal(times, input_signal, self.step_model, offset_steps)
        self.raw_prediction_input_signal = input_signal
        self._predict_subset_index = ('', [])
        if not self._recalc_chrono_prediction_matrix:
            # Next, compare statuses
            # Both t and psi must be available from same source - either fit or predict.
            # Can't pull t from one and psi from the other since they interact
            if self._psi_predict_eq_psi_fit and self._t_predict_eq_t_fit:
                # times and psi match fitted data. Set subset index to full index
                self._predict_subset_index = ('fit', np.arange(len(times), dtype=int))
            elif self._psi_predict_subset_index[0] == self._t_predict_subset_index[0] \
                    and self._t_predict_subset_index[0] != '':
                # times and psi are subsets of the same dataset. Need to check if they match up with each other
                if utils.check_equality(self._psi_predict_subset_index[1], self._psi_predict_subset_index[1]):
                    # Subset indices match for times and psi. Use the indices
                    self._predict_subset_index = deepcopy(self._t_predict_subset_index)
                else:
                    # Subset indices for times and psi do not match - must recalculate prediction matrix
                    self._recalc_chrono_prediction_matrix = True
                    # Set _t_predict and _psi_predict directly
                    self._t_predict = times
                    self._psi_predict = psi
                    # Overwrite subset indices
                    self._t_predict_subset_index = ('', [])
                    self._psi_predict_subset_index = ('', [])