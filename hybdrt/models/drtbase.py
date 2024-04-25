import numpy as np
import os
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt
# from cmdstanpy import CmdStanModel
from ..matrices import basis
from hybdrt import preprocessing as pp
from hybdrt import utils

module_dir = os.path.dirname(os.path.realpath(__file__))


class DRTBase:
    def __init__(self, fixed_basis_tau=None, tau_supergrid=None, tau_basis_type='gaussian', tau_epsilon=None,
                 basis_tau_ppd=10,
                 step_model='ideal', chrono_mode='galv', interpolate_integrals=True, chrono_tau_rise=None,
                 fixed_basis_nu=None, nu_basis_type='gaussian', nu_epsilon=None, fit_dop=False, normalize_dop=True,
                 fit_inductance=True, fit_ohmic=True, fit_capacitance=False,
                 time_precision=10, input_signal_precision=10, frequency_precision=10,
                 print_diagnostics=False, warn=True):

        if fixed_basis_tau is not None and tau_supergrid is not None:
            warnings.warn('If fixed_basis_tau is provided, tau_supergrid will be ignored')

        self._recalc_chrono_fit_matrix = True
        self._recalc_chrono_prediction_matrix = True
        self._recalc_eis_fit_matrix = True
        self._recalc_eis_prediction_matrix = True
        self.fit_matrices = {}
        self.prediction_matrices = {}
        self.fixed_basis_tau = fixed_basis_tau
        self.tau_supergrid = tau_supergrid
        self.basis_tau = None
        self.tau_basis_type = tau_basis_type
        self.tau_epsilon = tau_epsilon
        self.step_model = step_model
        self.chrono_mode = chrono_mode
        self.chrono_mode_predict = chrono_mode
        self.frequency_precision = frequency_precision
        self.time_precision = time_precision
        self.input_signal_precision = input_signal_precision
        # self.tau_zga_epsilon = None
        self._zga_params = None
        self.fit_inductance = fit_inductance
        self.fit_ohmic = fit_ohmic
        self.fit_capacitance = fit_capacitance
        self.sample_index = None
        self._t_predict_subset_index = ('', [])
        self._f_predict_subset_index = ('', [])
        self._t_predict_eq_t_fit = False
        self._f_predict_eq_f_fit = False
        self.t_fit = []
        self.f_fit = []
        self.t_predict = []
        self.f_predict = []
        self._t_fit_subset_index = None
        self._f_fit_subset_index = None

        # Distribution of phasances
        self.fixed_basis_nu = fixed_basis_nu
        self.basis_nu = None
        self.nu_epsilon = nu_epsilon
        self.nu_basis_type = nu_basis_type
        self.fit_dop = fit_dop
        self.normalize_dop = normalize_dop
        self.dop_scale_vector = None

        self.step_times = None
        self.step_sizes = None
        self.tau_rise = None
        self.nonconsec_step_times = None
        self.raw_input_signal = None
        self.raw_response_signal = None
        self.raw_response_background = None
        self.z_fit = None
        self.z_fit_scaled = None
        self.raw_prediction_input_signal = None
        self.scaled_input_signal = None
        self.scaled_response_signal = None
        self.scaled_response_offset = None
        self.scaled_response_background = None
        self.input_signal_scale = 1.0
        self.response_signal_scale = 1.0
        self.coefficient_scale = 1.0
        self.impedance_scale = 1.0
        self.inductance_scale = None

        self.cvx_result = None
        self.ridge_iter_history = None
        self.qphb_history = None
        self.special_qp_params = {}
        self.qphb_params = None
        self.map_samples = None
        self.map_sample_kw = None

        self.fit_kwargs = None
        self.series_neg = None

        # Outliers
        self.eis_outlier_index = None
        self.eis_outliers = None
        self.chrono_outlier_index = None
        self.chrono_outliers = None

        self._init_values = None
        self.stan_model_name = None
        self.stan_input = None
        self.stan_mle = None
        self.stan_result = None
        self.fit_parameters = None
        self.fit_type = None

        self.print_diagnostics = print_diagnostics
        self.warn = warn

        # Set tau_epsilon if not provided
        if self.tau_epsilon is None:
            if self.fixed_basis_tau is not None:
                dlntau = np.mean(np.diff(np.log(self.fixed_basis_tau)))
                self.tau_epsilon = 1 / dlntau
            elif self.tau_supergrid is not None:
                dlntau = np.mean(np.diff(np.log(self.tau_supergrid)))
                self.tau_epsilon = 1 / dlntau
            elif basis_tau_ppd is not None:
                self.tau_epsilon = pp.get_epsilon_from_ppd(basis_tau_ppd)

        # Generate integral lookups for interpolation
        if interpolate_integrals:
            if self.step_model != 'ideal' and chrono_tau_rise is None:
                raise ValueError('A constant chrono_tau_rise value must be provided to use interpolation for'
                                 'integral evaluation')
            # elif basis_tau is None:
            #     raise ValueError('basis_tau must be specified at instantiation to use interpolation for integral '
            #                      'evaluation')

            print('Generating impedance integral lookups...')
            zre_lookup, zim_lookup = basis.generate_impedance_lookup(self.tau_basis_type, self.tau_epsilon, 2000,
                                                                     zga_params=self.zga_params)

            print('Generating response integral lookups...')
            response_lookup = basis.generate_response_lookup(self.tau_basis_type, self.chrono_mode, self.step_model,
                                                             self.tau_epsilon, 2000, chrono_tau_rise, self.zga_params)

            self.interpolate_lookups = {'z_real': zre_lookup, 'z_imag': zim_lookup, 'response': response_lookup}
            self.integrate_method = 'interp'
            print('Integral lookups ready')
        else:
            self.interpolate_lookups = {'z_real': None, 'z_imag': None, 'response': None}
            self.integrate_method = 'trapz'

    # def _iterate_hyper_ridge(self, x0, num_offsets, inductance_index, nonneg, penalty_type,
    #                          hyper_l1_lambda, hyper_l2_lambda,
    #                          wrv, wrm, l1_lambda_vector, penalty_matrices, l2_lambda_vectors, derivative_weights,
    #                          hl_l1_beta, l1_lambda_0, hl_l2_beta, l2_lambda_0, xtol):
    #     if hyper_l1_lambda:
    #         # Solve for l1 lambdas. Exclude first num_offsets entries - these should remain at zero
    #         l1_lambda_vector[num_offsets:] = qphb.solve_hyper_l1_lambda(x0[num_offsets:], hl_l1_beta, l1_lambda_0)
    #
    #     if hyper_l2_lambda:
    #         # solve for l2 lambdas
    #         # print('iter',iter)
    #         for n, d_weight in enumerate(derivative_weights):
    #             if d_weight > 0:
    #                 if penalty_type == 'integral':
    #                     m = penalty_matrices[f'm{n}']
    #                     lv_in = l2_lambda_vectors[n]
    #                     lv_out = qphb.solve_hyper_l2_lambda(m, x0, lv_in, hl_beta=hl_l2_beta,
    #                                                         lambda_0=l2_lambda_0)
    #                 elif penalty_type == 'discrete':
    #                     l = penalty_matrices[f'l{n}']
    #                     lv_out = qphb.solve_hyper_l2_lambda_discrete(l, x0, hl_l2_beta, l2_lambda_0)
    #
    #                 # Handle numerical instabilities that may arise for large lambda_0 and small hl_beta
    #                 lv_out[lv_out <= 0] = 1e-15
    #                 l2_lambda_vectors[n] = lv_out
    #
    #         # print(n,lam_vectors[n])
    #
    #         # Make lml matrix for each derivative order
    #         l2_matrices = [penalty_matrices[f'm{n}'] for n in range(3)]
    #         lml = qphb.calculate_qp_l2_matrix(derivative_weights, rho_vector, l2_matrices, l2_lambda_vectors,
    #                                           l2_lambda_0, penalty_type)
    #
    #     # optimize x
    #     cvx_result = qphb.solve_convex_opt(wrv, wrm, lml, l1_lambda_vector, nonneg, self.special_qp_params)
    #     x = np.array(list(cvx_result['x']))
    #
    #     # Calculate cost for diagnostics
    #     P = wrm.T @ wrm + lml
    #     q = (-wrm.T @ wrv + l1_lambda_vector)
    #     cost = 0.5 * x.T @ P @ x + q.T @ x
    #     # for frac,lam_vec,ha,hb in zip(reg_ord,lam_vectors,hyper_as,hyper_bs):
    #     # cost += frac*np.sum((hb*lam_vec - (ha-1)*np.log(lam_vec)))
    #     # print('cost after x optimization:',cost)
    #
    #     self.ridge_iter_history.append(
    #         {'x': x.copy(),
    #          'l1_lambda_vector': l1_lambda_vector.copy(),
    #          'l2_lambda_vectors': l2_lambda_vectors.copy(),
    #          'fun': cvx_result['primal objective'],
    #          'cost': cost,
    #          'cvx_result': cvx_result,
    #          }
    #     )
    #
    #     # check for convergence
    #     x_delta = (x - x0) / x0
    #     # If inductance not fitted, set inductance delta to zero (inductance goes to random number)
    #     if not self.fit_inductance:
    #         x_delta[inductance_index] = 0
    #     # print(np.mean(np.abs(coef_delta)))
    #     if np.mean(np.abs(x_delta)) < xtol:
    #         converged = True
    #     else:
    #         converged = False
    #
    #     return x, l2_lambda_vectors, cvx_result, converged

    # def get_tau_from_times(self, times, step_times, ppd=10):
    #     # Determine min and max post-step times
    #     # Should be obsolete
    #     min_times = np.empty(len(step_times))
    #     max_times = np.empty(len(step_times))
    #     for i, t_start in enumerate(step_times):
    #         if i < len(step_times) - 1:
    #             t_end = step_times[i + 1]
    #         else:
    #             t_end = np.inf
    #
    #         times_i = times[(times > t_start) & (times < t_end)]
    #         min_times[i] = np.min(times_i - t_start)
    #         max_times[i] = np.max(times_i - t_start)
    #
    #     # Set minimum and maximum tau based on min and max post-step times
    #     log_tau_min = np.log10(np.mean(min_times) / 2)
    #     # TODO: re-test equilibration check with new log_tau_max (extended by 1 decade)
    #     log_tau_max = np.log10(np.mean(max_times)) + 1
    #
    #     # Create tau grid with specified ppd
    #     num_points = int((log_tau_max - log_tau_min) * ppd) + 1
    #     tau = np.logspace(log_tau_min, log_tau_max, num_points)
    #
    #     return tau
    
    @property
    def tau_basis_area(self):
        return basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)
    
    @property
    def nu_basis_area(self):
        return basis.get_basis_func_area(self.nu_basis_type, self.nu_epsilon)

    def get_tau_eval(self, ppd):
        """
        Get tau grid for DRT evaluation and plotting
        :param ppd:
        :return:
        """
        # Go one decade beyond self.basis_tau in each direction
        if self.fixed_basis_tau is not None:
            basis_tau = self.fixed_basis_tau
        else:
            basis_tau = self.basis_tau

        if basis_tau is None:
            raise ValueError('basis_tau must be set, either by specifying fixed_basis_tau or by fitting the '
                             'DRT instanceto data, before using get_tau_eval')

        log_tau_min = np.min(np.log10(basis_tau)) - 1
        log_tau_max = np.max(np.log10(basis_tau)) + 1
        tau = np.logspace(log_tau_min, log_tau_max, int((log_tau_max - log_tau_min) * ppd) + 1)

        return tau

    def process_chrono_signals(self, times, i_signal, v_signal, step_times, offset_steps, downsample, downsample_kw):
        # TODO: move this to DRT1d
        # If chrono data provided, get input signal step information
        if times is not None:
            # Prepare to fit chrono data
            # Determine input signal from chrono_mode
            if self.chrono_mode == 'galv':
                input_signal = i_signal
            else:
                input_signal = v_signal

            # Determine step times and sizes in input signal
            if step_times is None:
                # Step times not provided - determine from input signal
                step_times, step_sizes, tau_rise = pp.process_input_signal(times, input_signal, self.step_model,
                                                                           offset_steps)
            else:
                # Step times provided - only need to get step sizes
                step_sizes = pp.get_step_sizes(times, input_signal, step_times)
                tau_rise = None

            # Get non-consecutive steps for plotting functions
            if len(step_times) > 1:
                step_diff = np.diff(step_times)
                t_sample = np.min(np.diff(times))
                nonconsec_step_times = step_times[1:][step_diff > 1.1 * t_sample]
                self.nonconsec_step_times = np.insert(nonconsec_step_times, 0, step_times[0])
            else:
                self.nonconsec_step_times = step_times

            if self.print_diagnostics:
                print('Step data:', step_times, step_sizes)
                print('Got step data')

            # Downsample data
            if downsample:
                if downsample_kw is None:
                    downsample_kw = {'prestep_samples': 10,
                                     'target_times': None}  # np.concatenate(([0], np.logspace(-5, 0, 201)))}

                sample_times, sample_i, sample_v, sample_index = \
                    pp.downsample_data(times, i_signal, v_signal, stepwise_sample_times=True,
                                       step_times=self.nonconsec_step_times, op_mode=self.chrono_mode, **downsample_kw)
                # Record sample_index for reference
                self.sample_index = sample_index
                if self.print_diagnostics:
                    print('Downsampled size:', len(sample_times))
            else:
                self.sample_index = np.arange(0, len(times), 1, dtype=int)
                sample_times = times.copy()
                sample_i = i_signal.copy()
                sample_v = v_signal.copy()

            # Set t_fit - must be done before setting raw_input_signal
            self.t_fit = sample_times

            # # Set input and response signals based on control mode
            input_signal, response_signal = utils.chrono.get_input_and_response(sample_i, sample_v, self.chrono_mode)
            self.raw_input_signal = input_signal.copy()
            self.raw_response_signal = response_signal.copy()
            # if self.chrono_mode == 'galv':
            #     self.raw_input_signal = sample_i.copy()
            #     self.raw_response_signal = sample_v.copy()
            # elif self.chrono_mode == 'pot':
            #     self.raw_input_signal = sample_v.copy()
            #     self.raw_response_signal = sample_i.copy()
        else:
            input_signal = None
            step_times = None
            step_sizes = None
            tau_rise = None
            sample_times = None
            sample_i = None
            sample_v = None

        # Store step data
        self.step_times = deepcopy(step_times)
        self.step_sizes = deepcopy(step_sizes)
        self.tau_rise = deepcopy(tau_rise)

        return sample_times, sample_i, sample_v, step_times, step_sizes, tau_rise

    # def scale_signal(self, times, i_signal, v_signal, step_times, step_sizes, apply_scaling, rp_scale):
    #     """
    #     *OBSOLETE - replaced by scale_data*
    #     Scale the signal to a normalized scale for fitting
    #     :param ndarray times: measurement times
    #     :param ndarray i_signal: current signal values
    #     :param ndarray v_signal: voltage signal values
    #     :param ndarray step_times: array of step times
    #     :param ndarray step_sizes: array of input signal step sizes
    #     :param apply_scaling: if True, scale input and response signals to ensure consistent data and coefficient
    #     scales. If False, set signal scales to 1 and return raw signals without applying scaling.
    #     :return: array of scaled values
    #     """
    #     if self.chrono_mode == 'galv':
    #         self.raw_input_signal = i_signal.copy()
    #         self.raw_response_signal = v_signal.copy()
    #     elif self.chrono_mode == 'pot':
    #         self.raw_input_signal = v_signal.copy()
    #         self.raw_response_signal = i_signal.copy()
    #
    #     if apply_scaling:
    #         input_signal_scale, response_signal_scale = pp.get_signal_scales(times, step_times, step_sizes,
    #                                                                          self.raw_response_signal, self.step_model)
    #
    #         # Scale input signal such that mean step size is 1
    #         self.input_signal_scale = input_signal_scale
    #
    #         # Scale response signal to achieve desired Rp scale
    #         self.response_signal_scale = response_signal_scale / rp_scale
    #     else:
    #         # No scaling
    #         self.input_signal_scale = 1.0
    #         self.response_signal_scale = 1.0
    #
    #     # Determine coefficient scale
    #     # x_scaled = x / s_x; s_x = s_v / s_i
    #     if self.chrono_mode == 'galv':
    #         self.coefficient_scale = self.response_signal_scale / self.input_signal_scale
    #     elif self.chrono_mode == 'pot':
    #         self.coefficient_scale = self.input_signal_scale / self.response_signal_scale
    #
    #     self.scaled_input_signal = self.raw_input_signal / self.input_signal_scale
    #     self.scaled_response_signal = self.raw_response_signal / self.response_signal_scale
    #
    #     if self.chrono_mode == 'galv':
    #         scaled_i_signal = self.scaled_input_signal
    #         scaled_v_signal = self.scaled_response_signal
    #     elif self.chrono_mode == 'pot':
    #         scaled_i_signal = self.scaled_response_signal
    #         scaled_v_signal = self.scaled_input_signal
    #
    #     return scaled_i_signal, scaled_v_signal

    # def scale_impedance(self, z, rp_scale):
    #     # Should be replaced by scale_data
    #     if rp_scale is None:
    #         # Match impedance scaling to time response scaling
    #         self.impedance_scale = deepcopy(self.coefficient_scale)
    #     else:
    #         rp_est = np.max(z.real) - np.min(z.real)
    #         self.impedance_scale = rp_est / rp_scale
    #
    #     return z / self.impedance_scale

    def scale_data(self, times, i_signal, v_signal, step_times, step_sizes, z, apply_scaling, rp_scale):
        if apply_scaling:
            input_signal, response_signal = utils.chrono.get_input_and_response(i_signal, v_signal, self.chrono_mode)
            rp_est = pp.estimate_rp(times, step_times, step_sizes, response_signal, self.step_model, z)
            if self.print_diagnostics:
                print('Estimated Rp: {:.4f}'.format(rp_est))
            # if self.chrono_mode == 'galv':
            self.coefficient_scale = rp_est / rp_scale
        else:
            rp_est = 1.0
            self.coefficient_scale = 1.0

        if self.print_diagnostics:
            print('Initial coefficient scale:', self.coefficient_scale)

        # Scale chrono data
        if times is not None:
            if apply_scaling:
                # Scale input signal such that mean step size is 1
                # self.input_signal_scale = pp.get_input_signal_scale(times, step_times, step_sizes, self.step_model)
                # Scale to max step size
                self.input_signal_scale = np.max(np.abs(step_sizes))

                # Scale response signal to achieve desired Rp scale
                self.response_signal_scale = self.input_signal_scale * rp_est / rp_scale

                if self.print_diagnostics:
                    print('Chrono input and response scales: {:.5f}, {:5f}'.format(
                        self.input_signal_scale, self.response_signal_scale)
                    )

            else:
                # No scaling
                self.input_signal_scale = 1.0
                self.response_signal_scale = 1.0

            # Set input and response signals based on control mode
            # input_signal, response_signal = utils.chrono.get_input_and_response(i_signal, v_signal, self.chrono_mode)
            # self.raw_input_signal = input_signal.copy()
            # self.raw_response_signal = response_signal.copy()

            self.scaled_input_signal = self.raw_input_signal / self.input_signal_scale
            self.scaled_response_signal = self.raw_response_signal / self.response_signal_scale

            if self.chrono_mode == 'galv':
                scaled_i_signal = self.scaled_input_signal.copy()
                scaled_v_signal = self.scaled_response_signal.copy()
            else:
                scaled_i_signal = self.scaled_response_signal.copy()
                scaled_v_signal = self.scaled_input_signal.copy()
        else:
            # No chrono data provided
            scaled_i_signal = None
            scaled_v_signal = None
            self.input_signal_scale = None
            self.response_signal_scale = None
            self.raw_input_signal = None
            self.raw_response_signal = None
            self.scaled_input_signal = None
            self.scaled_response_signal = None

        # Scale EIS data
        if z is not None:
            # Impedance scale is coefficient scale - already set whether apply_scaling True or False
            self.impedance_scale = deepcopy(self.coefficient_scale)
            z_scaled = z / self.impedance_scale

            self.z_fit = z.copy()
            self.z_fit_scaled = z_scaled.copy()
        else:
            # No EIS data provided
            z_scaled = None
            self.z_fit = None
            self.z_fit_scaled = None

        return scaled_i_signal, scaled_v_signal, z_scaled

    def update_data_scale(self, factor):
        """
        Update response signal scale by factor
        :param float factor: factor by which to multiply data to obtain desired scale
        :return:
        """
        if self.scaled_response_signal is not None:
            self.response_signal_scale /= factor
            self.scaled_response_offset *= factor
            self.scaled_response_signal *= factor

        if self.z_fit_scaled is not None:
            self.z_fit_scaled *= factor

        if self.chrono_mode == 'galv':
            self.coefficient_scale /= factor
            self.impedance_scale /= factor
        elif self.chrono_mode == 'pot':
            # unclear how this would actually change...
            self.coefficient_scale *= factor
            self.impedance_scale /= factor

    def _add_special_qp_param(self, name, nonneg, size=1):
        options = ['R_inf', 'v_baseline', 'inductance', 'C_inv', 'vz_offset', 'background_scale', 'x_dop']
        if name not in options:
            raise ValueError('Invalid special QP parameter {name}. Options: {options}')

        self.special_qp_params[name] = {'index': self.get_qp_mat_offset(), 'nonneg': nonneg, 'size': size}

    def get_qp_mat_offset(self):
        """Get matrix offset for special QP params"""
        return int(np.sum([v.get('size', 1) for v in self.special_qp_params.values()]))

    def plot_zga_approximation(self, ax=None):
        if self.zga_params is None:
            raise ValueError('ZGA parameters have not been set')
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(4, 3))

            # Get exact and approximate basis functions
            phi_rbf = basis.get_basis_func('gaussian')
            phi_zga = basis.get_basis_func('zga', self.zga_params)

            # Make x grid
            x = np.linspace(-10 / self.tau_epsilon, 10 / self.tau_epsilon, 1000)

            # Plot both functions
            ax.plot(x, phi_rbf(x, self.tau_epsilon), label='RBF')
            ax.plot(x, phi_zga(x, None), label='ZGA', c='k', lw=1.5)

            # Labels
            ax.set_xlabel('$y$')
            ax.set_ylabel('$\phi(y)$')

            ax.legend()

    def get_fit_times(self, return_none=True):
        """Get times that were fitted"""
        if self._t_fit_subset_index is not None:
            # If times is subset of previously fitted dataset, get subset
            times = self.t_fit[self._t_fit_subset_index]
        else:
            times = self.t_fit

        if len(times) == 0 and return_none:
            times = None

        return times

    def get_fit_frequencies(self, return_none=True):
        """Get frequencies that were fitted"""
        if self._f_fit_subset_index is not None:
            frequencies = self.f_fit[self._f_fit_subset_index]
        else:
            frequencies = self.f_fit

        if len(frequencies) == 0 and return_none:
            frequencies = None

        return frequencies

    @property
    def num_chrono(self):
        """Number of chrono data points"""
        if self._t_fit_subset_index is not None:
            # If times is subset of previously fitted dataset, get subset
            return len(self._t_fit_subset_index)
        else:
            return len(self.t_fit)

    @property
    def num_eis(self):
        """Number of EIS data points"""
        if self._f_fit_subset_index is not None:
            return len(self._f_fit_subset_index)
        else:
            return len(self.f_fit)

    # Getters and setters to control matrix calculation
    # -------------------------------------------------
    # def get_recalc_chrono_fit_matrix(self):
    #     return self._recalc_chrono_fit_matrix
    #
    # def set_recalc_chrono_fit_matrix(self, status):
    #     self._recalc_chrono_fit_matrix = status
    #     if status:
    #         # If matrices must be recalculated, overwrite t_fit
    #         self.t_fit = self.get_fit_times()
    #
    # def get_recalc_chrono_prediction_matrix(self):
    #     return self._recalc_chrono_prediction_matrix

    def get_basis_tau(self):
        return self._basis_tau

    def set_basis_tau(self, basis_tau):
        if hasattr(self, 'basis_tau'):
            if not utils.array.check_equality(basis_tau, self.basis_tau):
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
                self._recalc_eis_fit_matrix = True
                self._recalc_eis_prediction_matrix = True
        self._basis_tau = basis_tau

    basis_tau = property(get_basis_tau, set_basis_tau)

    def get_tau_basis_type(self):
        return self._tau_basis_type

    def set_tau_basis_type(self, tau_basis_type):
        utils.validation.check_basis_type(tau_basis_type)
        if hasattr(self, 'tau_basis_type'):
            if tau_basis_type != self.nu_basis_type:
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
                self._recalc_eis_fit_matrix = True
                self._recalc_eis_prediction_matrix = True
        self._tau_basis_type = tau_basis_type

    tau_basis_type = property(get_tau_basis_type, set_tau_basis_type)

    def get_nu_basis_type(self):
        return self._nu_basis_type

    def set_nu_basis_type(self, nu_basis_type):
        utils.validation.check_basis_type(nu_basis_type)
        self._nu_basis_type = nu_basis_type

    nu_basis_type = property(get_nu_basis_type, set_nu_basis_type)

    def get_tau_epsilon(self):
        return self._tau_epsilon

    def set_tau_epsilon(self, tau_epsilon):
        if hasattr(self, 'epsilon'):
            if tau_epsilon != self.tau_epsilon:
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
                self._recalc_eis_fit_matrix = True
                self._recalc_eis_prediction_matrix = True
        self._tau_epsilon = tau_epsilon

    tau_epsilon = property(get_tau_epsilon, set_tau_epsilon)

    def get_zga_params(self):
        return self._zga_params

    def set_zga_params(self, approx_func_epsilon=None, num_bases=7, basis_extent=2, curvature_penalty=None,
                       nonneg=False):
        if self.tau_epsilon is None:
            # Determine epsilon from basis_tau spacing
            dlntau = np.mean(np.diff(np.log(self.basis_tau)))
            self.tau_epsilon = 1 / dlntau

        # Optimize parameters for best RBF approximation
        x_basis, coef, eps = basis.get_basis_approx_params('gaussian', 'Cole-Cole', self.tau_epsilon,
                                                           approx_func_epsilon, num_bases, basis_extent,
                                                           curvature_penalty, nonneg)
        # Store ZGA parameters
        self._zga_params = (x_basis, coef, eps)
        self._recalc_chrono_fit_matrix = True
        self._recalc_chrono_prediction_matrix = True
        self._recalc_eis_fit_matrix = True
        self._recalc_eis_prediction_matrix = True

    zga_params = property(get_zga_params, set_zga_params)

    # def get_tau_zga_epsilon(self):
    #	  return self._tau_zga_epsilon
    #
    # def set_tau_zga_epsilon(self, epsilon):
    #	  self._tau_zga_epsilon = epsilon
    #	  self._recalc_chrono_fit_matrix = True
    #	  self._recalc_chrono_prediction_matrix = True
    #	  self._recalc_eis_fit_matrix = True
    #	  self._recalc_eis_prediction_matrix = True
    #
    # tau_zga_epsilon = property(get_tau_zga_epsilon, set_tau_zga_epsilon)

    def get_t_fit(self):
        return self._t_fit

    def set_t_fit(self, times):
        if times is None:
            # EIS fit is being performed. We don't need to update anything
            return 
        if hasattr(self, 't_fit') and not self._recalc_chrono_fit_matrix:
            self._t_fit_subset_index = None
            # Check if times is the same as self.t_fit
            if not utils.array.check_equality(utils.array.rel_round(self._t_fit, self.time_precision),
                                              utils.array.rel_round(times, self.time_precision)):
                # Check if times is a subset of self.t_fit
                # If times is a subset of self.t_fit, we can use sub-matrices of the existing matrices
                # In this case, we should not update self._t_fit
                if utils.array.is_subset(times, self.t_fit, self.time_precision):
                    self._t_fit_subset_index = utils.array.get_subset_index(times, self.t_fit, self.time_precision)
                # self._recalc_chrono_fit_matrix = False
                else:
                    # if times is not a subset of self.t_fit, must recalculate matrices
                    self._t_fit = times
                    self._t_fit_subset_index = None
                    self._recalc_chrono_fit_matrix = True
        else:
            self._t_fit = times
            self._t_fit_subset_index = None
            self._recalc_chrono_fit_matrix = True

    t_fit = property(get_t_fit, set_t_fit)

    def get_raw_input_signal(self):
        return self._raw_input_signal

    def set_raw_input_signal(self, input_signal):
        # Must be run after set_t_fit
        # Check if the input signal matches the corresponding t_fit subset
        if self._t_fit_subset_index is not None:
            if not utils.array.check_equality(
                    utils.array.rel_round(input_signal, self.input_signal_precision),
                    utils.array.rel_round(self.raw_input_signal[self._t_fit_subset_index], self.input_signal_precision)
            ):
                # Input signal does not match stored input signal. Must recalculate matrices
                self._raw_input_signal = deepcopy(input_signal)
                self._t_fit = self.t_fit[self._t_fit_subset_index]
                self._t_fit_subset_index = None
                self._recalc_chrono_fit_matrix = True
        else:
            # Check if input_signal matches self.raw_input_signal
            if hasattr(self, 'raw_input_signal'):
                if not utils.array.check_equality(
                        utils.array.rel_round(input_signal, self.input_signal_precision),
                        utils.array.rel_round(self.raw_input_signal, self.input_signal_precision)
                ):
                    self._raw_input_signal = deepcopy(input_signal)
                    self._recalc_chrono_fit_matrix = True
            else:
                # No raw_input_signal set yet
                self._raw_input_signal = deepcopy(input_signal)
                self._recalc_chrono_fit_matrix = True

    raw_input_signal = property(get_raw_input_signal, set_raw_input_signal)

    # def get_fit_step_times(self):
    #	  return self._fit_step_times
    #
    # def set_fit_step_times(self, step_times):
    #	  if hasattr(self, 'fit_step_times'):
    #		  if not check_equality(rel_round(step_times, self.time_precision),
    #							rel_round(self.fit_step_times, self.time_precision)):
    #			  self._fit_step_times = step_times
    #			  self._recalc_chrono_fit_matrix = True
    #	  else:
    #		  self._fit_step_times = step_times
    #		  self._recalc_chrono_fit_matrix = True
    #
    # fit_step_times = property(get_fit_step_times, set_fit_step_times)
    #
    # def get_fit_step_sizes(self):
    #	  return self._fit_step_sizes
    #
    # def set_fit_step_sizes(self, step_sizes):
    #	  """
    #	  Set fit_step_sizes. Should be called with scaled step sizes
    #	  :param step_sizes:
    #	  :return:
    #	  """
    #	  if hasattr(self, 'fit_step_sizes'):
    #		  if not check_equality(rel_round(step_sizes, 10),
    #								rel_round(self.fit_step_sizes, 10)):
    #			  self._fit_step_sizes = step_sizes
    #			  self._recalc_chrono_fit_matrix = True
    #	  else:
    #		  self._fit_step_sizes = step_sizes
    #		  self._recalc_chrono_fit_matrix = True
    #
    # fit_step_sizes = property(get_fit_step_sizes, set_fit_step_sizes)
    #
    # def get_fit_tau_rise(self):
    #	  return self._fit_tau_rise
    #
    # def set_fit_tau_rise(self, tau_rise):
    #	  if hasattr(self, 'fit_tau_rise'):
    #		  if not check_equality(rel_round(tau_rise, 10),
    #								rel_round(self.fit_tau_rise, 10)):
    #			  self._fit_tau_rise = tau_rise
    #			  self._recalc_chrono_fit_matrix = True
    #	  else:
    #		  self._fit_tau_rise = tau_rise
    #		  self._recalc_chrono_fit_matrix = True
    #
    # fit_tau_rise = property(get_fit_tau_rise, set_fit_tau_rise)

    def get_t_predict(self):
        return self._t_predict

    def set_t_predict(self, times):
        if hasattr(self, 't_predict') and not self._recalc_chrono_prediction_matrix:
            self._t_predict_subset_index = ('', [])
            self._t_predict_eq_t_fit = False
            # Check if times is the same as self.t_fit
            if utils.array.check_equality(utils.array.rel_round(self._t_fit, self.time_precision),
                                          utils.array.rel_round(times, self.time_precision)):
                # self._t_predict = times
                self._t_predict_eq_t_fit = True
                # don't update recalc status here - another attribute change may have set this to True
                # print('a')
            # Check if times is the same as self.t_predict
            elif utils.array.check_equality(utils.array.rel_round(self._t_predict, self.time_precision),
                                            utils.array.rel_round(times, self.time_precision)):
                self._t_predict = times
                # print('b')
            # Check if times is a subset of self.t_fit or self.t_predict
            else:
                # If times is a subset of self.t_fit or self.t_predict, we can use sub-matrices of the existing matrices
                if utils.array.is_subset(times, self.t_predict, self.time_precision):
                    # times is a subset of t_predict
                    self._t_predict_subset_index = (
                        'predict',
                        utils.array.get_subset_index(times, self.t_predict, self.time_precision)
                    )
                    # print('c')
                # In this case, we should not update self._t_predict
                elif utils.array.is_subset(times, self.t_fit, self.time_precision):
                    # times is a subset of t_fit
                    self._t_predict_subset_index = (
                        'fit',
                        utils.array.get_subset_index(times, self.t_fit, self.time_precision)
                    )
                    # print('d')
                # In this case, we should not update self._t_predict
                else:
                    # if times is not a subset of self.t_fit or self.t_predict, must calculate matrices
                    self._t_predict = times
                    self._recalc_chrono_prediction_matrix = True
                    # print('e')
        else:
            self._t_predict = times
            self._recalc_chrono_prediction_matrix = True
            # print('f')

    t_predict = property(get_t_predict, set_t_predict)

    # def get_predict_step_times(self):
    #	  return self._predict_step_times
    #
    # def set_predict_step_times(self, step_times):
    #	  if self._t_predict_eq_t_fit or self._t_predict_subset_index[0] == 'fit':
    #		  if not check_equality(rel_round(step_times, self.time_precision),
    #							rel_round(self._fit_step_times, self.time_precision)):
    #			  self._predict_step_times = step_times
    #			  self._recalc_chrono_prediction_matrix = True
    #	  elif hasattr(self, 'predict_step_times'):
    #		  if not check_equality(rel_round(step_times, self.time_precision),
    #							rel_round(self._predict_step_times, self.time_precision)):
    #			  self._predict_step_times = step_times
    #			  self._recalc_chrono_prediction_matrix = True
    #	  else:
    #		  self._predict_step_times = step_times
    #		  self._recalc_chrono_prediction_matrix = True
    #
    # predict_step_times = property(get_predict_step_times, set_predict_step_times)
    #
    # def get_predict_step_sizes(self):
    #	  return self._predict_step_sizes
    #
    # def set_predict_step_sizes(self, step_sizes):
    #	  """
    #	  Set predict_step_sizes. Should be called with scaled step sizes
    #	  :param step_sizes:
    #	  :return:
    #	  """
    #	  if self._t_predict_eq_t_fit or self._t_predict_subset_index[0] == 'fit':
    #		  if not check_equality(rel_round(step_sizes, 10),
    #								rel_round(self.fit_step_sizes, 10)):
    #			  self._predict_step_sizes = step_sizes
    #			  self._recalc_chrono_prediction_matrix = True
    #	  elif hasattr(self, 'predict_step_sizes'):
    #		  if not check_equality(rel_round(step_sizes, 10),
    #								rel_round(self.predict_step_sizes, 10)):
    #			  self._predict_step_sizes = step_sizes
    #			  self._recalc_chrono_prediction_matrix = True
    #	  else:
    #		  self._predict_step_sizes = step_sizes
    #		  self._recalc_chrono_prediction_matrix = True
    #
    # predict_step_sizes = property(get_predict_step_sizes, set_predict_step_sizes)
    #
    # def get_predict_tau_rise(self):
    #	  return self._predict_tau_rise
    #
    # def set_predict_tau_rise(self, tau_rise):
    #	  if self._t_predict_eq_t_fit or self._t_predict_subset_index[0] == 'fit':
    #		  if not check_equality(rel_round(tau_rise, 10),
    #								rel_round(self.fit_tau_rise, 10)):
    #			  self._predict_tau_rise = deepcopy(tau_rise)
    #			  self._recalc_chrono_prediction_matrix = True
    #	  elif hasattr(self, 'predict_tau_rise'):
    #		  if not check_equality(rel_round(tau_rise, 10),
    #								rel_round(self.predict_tau_rise, 10)):
    #			  self._predict_tau_rise = deepcopy(tau_rise)
    #			  self._recalc_chrono_prediction_matrix = True
    #	  else:
    #		  self._predict_tau_rise = deepcopy(tau_rise)
    #		  self._recalc_chrono_prediction_matrix = True
    #
    # predict_tau_rise = property(get_predict_tau_rise, set_predict_tau_rise)

    def get_raw_prediction_input_signal(self):
        return self._raw_prediction_input_signal

    def set_raw_prediction_input_signal(self, input_signal):
        # Must be run after set_t_predict
        # Check if the input signal matches the corresponding t_predict subset
        if self._t_predict_eq_t_fit:
            if not utils.array.check_equality(utils.array.rel_round(input_signal, self.input_signal_precision),
                                              utils.array.rel_round(self.raw_input_signal,
                                                                    self.input_signal_precision)):
                self._raw_prediction_input_signal = deepcopy(input_signal)
                self._recalc_chrono_prediction_matrix = True
        elif self._t_predict_subset_index[0] == 'fit':
            if not utils.array.check_equality(utils.array.rel_round(input_signal, self.input_signal_precision),
                                              utils.array.rel_round(
                                                  self.raw_input_signal[self._t_predict_subset_index[1]],
                                                  self.input_signal_precision)
                                              ):
                self._raw_prediction_input_signal = deepcopy(input_signal)
                self._t_predict_subset_index = ('', [])
                self._recalc_chrono_prediction_matrix = True
        elif self._t_predict_subset_index[0] == 'predict':
            # TODO: This logic is somewhat wonky for cases in which input_signal represents a different array of times
            #  than the prediction times. Handled below by checking for length matches.
            #  Long-term, best solution would be to check step_times and step_sizes
            #  rather than the input signal vector itself

            # Verify that input_signal matches length of t_predict
            len_match = len(input_signal) == len(self._t_predict_subset_index[1])
            # Verify that t_predict subset is a valid subset of raw_prediction_input_signal
            index_match = self._t_predict_subset_index[1][-1] < len(self.raw_prediction_input_signal)
            # Verify that signal values match
            if len_match and index_match:
                val_match = utils.array.check_equality(utils.array.rel_round(input_signal, self.input_signal_precision),
                                                       utils.array.rel_round(
                                                           self.raw_prediction_input_signal[
                                                               self._t_predict_subset_index[1]],
                                                           self.input_signal_precision)
                                                       )
            else:
                val_match = False

            if not (len_match and index_match and val_match):
                self._raw_prediction_input_signal = deepcopy(input_signal)
                self._t_predict_subset_index = ('', [])
                self._recalc_chrono_prediction_matrix = True
        else:
            # Check if input_signal matches self.raw_prediction_input_signal
            if hasattr(self, 'raw_prediction_input_signal'):
                if not utils.array.check_equality(utils.array.rel_round(input_signal, self.input_signal_precision),
                                                  utils.array.rel_round(self.raw_prediction_input_signal,
                                                                        self.input_signal_precision)
                                                  ):
                    self._raw_prediction_input_signal = deepcopy(input_signal)
                    self._recalc_chrono_prediction_matrix = True
            else:
                # No raw_prediction_input_signal set yet
                self._raw_prediction_input_signal = deepcopy(input_signal)
                self._recalc_chrono_prediction_matrix = True

    raw_prediction_input_signal = property(get_raw_prediction_input_signal, set_raw_prediction_input_signal)

    def get_f_fit(self):
        return self._f_fit

    def set_f_fit(self, frequencies):
        if frequencies is None:
            # Chrono fit is being performed. We don't need to update anything
            return
        
        if hasattr(self, 'f_fit') and not self._recalc_eis_fit_matrix:
            self._f_fit_subset_index = None
            # Check if frequencies is the same as self.f_fit
            if not utils.array.check_equality(utils.array.rel_round(self._f_fit, self.frequency_precision),
                                              utils.array.rel_round(frequencies, self.frequency_precision)):
                # Check if frequencies is a subset of self.f_fit
                # If frequencies is a subset of self.f_fit, we can use sub-matrices of the existing matrices
                # In this case, we should not update self._f_fit
                if utils.array.is_subset(frequencies, self.f_fit):
                    self._f_fit_subset_index = utils.array.get_subset_index(frequencies, self.f_fit,
                                                                            self.frequency_precision)
                else:
                    # if frequencies is not a subset of self.f_fit, must recalculate matrices
                    self._f_fit = frequencies
                    self._f_fit_subset_index = None
                    self._recalc_eis_fit_matrix = True
        else:
            self._f_fit = frequencies
            self._f_fit_subset_index = None
            self._recalc_eis_fit_matrix = True

    f_fit = property(get_f_fit, set_f_fit)

    def get_f_predict(self):
        return self._f_predict

    def set_f_predict(self, frequencies):
        if hasattr(self, 'f_predict') and not self._recalc_eis_prediction_matrix:
            self._f_predict_subset_index = ('', [])
            self._f_predict_eq_f_fit = False
            # Check if frequencies is the same as self.f_fit
            if utils.array.check_equality(utils.array.rel_round(self._f_fit, self.frequency_precision),
                                          utils.array.rel_round(frequencies, self.frequency_precision)):
                # self._f_predict = frequencies
                self._f_predict_eq_f_fit = True
                # don't update recalc status here - another attribute change may have set this to True
                # print('a')
            # Check if frequencies is the same as self.f_predict
            elif utils.array.check_equality(utils.array.rel_round(self._f_predict, self.frequency_precision),
                                            utils.array.rel_round(frequencies, self.frequency_precision)):
                self._f_predict = frequencies
                # print('b')
            # Check if frequencies is a subset of self.f_fit or self.f_predict
            else:
                # If frequencies is a subset of self.f_fit or self.f_predict,
                # we can use sub-matrices of the existing matrices
                if utils.array.is_subset(frequencies, self.f_predict):
                    # frequencies is a subset of f_predict
                    self._f_predict_subset_index = (
                        'predict',
                        utils.array.get_subset_index(frequencies, self.f_predict, self.frequency_precision)
                    )
                    # print('c')
                # In this case, we should not update self._f_predict
                elif utils.array.is_subset(frequencies, self.f_fit):
                    # frequencies is a subset of f_fit
                    self._f_predict_subset_index = (
                        'fit',
                        utils.array.get_subset_index(frequencies, self.f_fit, self.frequency_precision)
                    )
                    # print('d')
                # In this case, we should not update self._f_predict
                else:
                    # if frequencies is not a subset of self.f_fit or self.f_predict, must calculate matrices
                    self._f_predict = frequencies
                    self._recalc_eis_prediction_matrix = True
                    # print('e')
        else:
            self._f_predict = frequencies
            self._recalc_eis_prediction_matrix = True
            # print('f')

    f_predict = property(get_f_predict, set_f_predict)

    def get_chrono_mode(self):
        return self._chrono_mode

    def set_chrono_mode(self, chrono_mode):
        utils.validation.check_ctrl_mode(chrono_mode)
        if hasattr(self, 'chrono_mode'):
            if chrono_mode != self.chrono_mode:
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
                self._recalc_eis_fit_matrix = True
                self._recalc_eis_prediction_matrix = True
        self._chrono_mode = chrono_mode

    chrono_mode = property(get_chrono_mode, set_chrono_mode)

    def get_chrono_mode_predict(self):
        return self._chrono_mode_predict

    def set_chrono_mode_predict(self, chrono_mode):
        utils.validation.check_ctrl_mode(chrono_mode)
        if hasattr(self, 'chrono_mode'):
            if chrono_mode != self.chrono_mode:
                raise ValueError('Use of different operation modes for fitting and predicting is not supported')
                # self._recalc_chrono_prediction_matrix = True
        self._chrono_mode_predict = chrono_mode

    chrono_mode_predict = property(get_chrono_mode_predict, set_chrono_mode_predict)

    def get_step_model(self):
        return self._step_model

    def set_step_model(self, step_model):
        utils.validation.check_step_model(step_model)
        if hasattr(self, 'step_model'):
            if step_model != self.step_model:
                self._recalc_chrono_fit_matrix = True
                self._recalc_chrono_prediction_matrix = True
        # if hasattr(self, 'fit_inductance'):
        #     if self.fit_inductance and step_model == 'ideal':
        #         raise ValueError('step_model cannot be set to ideal when fit_inductance==True.')
        self._step_model = step_model

    step_model = property(get_step_model, set_step_model)

    def get_fit_inductance(self):
        return self._fit_inductance

    def set_fit_inductance(self, fit_inductance):
        # if fit_inductance and self.step_model == 'ideal':
        #     warnings.warn('When using ideal step model, inductance can ONLY be fitted with hybrid fit')

        self._fit_inductance = fit_inductance

    fit_inductance = property(get_fit_inductance, set_fit_inductance)
