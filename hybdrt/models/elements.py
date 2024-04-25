import re
import numpy as np
import warnings
from scipy.optimize import least_squares
from scipy import special
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

from mitlef.pade_approx import create_approx_func, ml_pade_approx

from .. import utils
from . import peaks
from .. import preprocessing as pp
from ..matrices import mat1d
from hybdrt.plotting import plot_distribution, plot_eis


class DiscreteElementModel:
    def __init__(self, model_string, chrono_step_model='ideal', chrono_mode='galv'):
        self.model_string = model_string

        utils.validation.check_ctrl_mode(chrono_mode)
        utils.validation.check_step_model(chrono_step_model)
        self.chrono_mode = chrono_mode
        self.chrono_step_model = chrono_step_model

        # Parse model string
        el_names, el_types, param_types, param_names, param_bounds, param_indices = parse_model_string(model_string)

        self.element_names = el_names
        self.element_types = el_types
        self.parameter_types = param_types
        self.parameter_names = param_names
        self.parameter_bounds = param_bounds
        self.scaled_bounds = None
        self.parameter_indices = param_indices

        self.z_function = model_impedance_function(model_string)
        self.v_function = model_voltage_function(model_string, chrono_step_model)
        self.gamma_function = model_distribution_function(model_string)
        self.mass_function = model_mass_function(model_string)

        # Fit parameters
        self.drt_estimates = None
        self.init_values = None
        self.raw_parameter_values = None
        self.scaled_parameter_values = None
        self.parameter_values = None
        self.fit_result = None

        # Chrono data
        self.t_fit = None
        self.step_times = None
        self.step_sizes = None
        self.tau_rise = None
        self.raw_input_signal = None
        self.raw_response_signal = None
        self.scaled_input_signal = None
        self.scaled_response_signal = None
        self.scaled_response_offset = None

        # EIS data
        self.f_fit = None
        self.z_fit = None
        self.z_fit_scaled = None

        # Weights
        self.scaled_weights = None
        self.weights = None

        # Scales
        self.input_signal_scale = 1.0
        self.response_signal_scale = 1.0
        # self.impedance_scale = 1.0
        self.rp_scale = 1.0

        # Prior
        self.prior_params = None

    # -----------------------
    # Initialization from DRT
    # -----------------------
    @classmethod
    def from_drt(cls, drt, x_raw=None, tau=None, peak_indices=None, estimate_peak_distributions=True,
                 estimate_peak_distributions_kw=None,
                 model_string='R0-L0-{DRT}', drt_element='HN',
                 set_bounds=True, parameter_limits=None, **find_peaks_kw):

        # Get parameter vector
        if x_raw is None:
            x_raw = drt.qphb_history[-1]['x']

        # Get tau vector for peak finding
        if tau is None:
            tau = drt.get_tau_eval(10)

        # Define parameter limits by type for bound setting
        if parameter_limits is None:
            parameter_limits = {'R': ('multiply', 0.25, 4), 'lntau': ('add', -1, 1), 'lnL': ('add', -1, 1)}

        # Convert raw parameter vector (in scaled space) to data-scale parameters
        fit_parameters = drt.extract_qphb_parameters(x_raw)

        # Find peaks
        if peak_indices is None:
            _, peak_indices = drt.find_peaks_compound(**find_peaks_kw)

        # Estimate separate peak distributions
        if estimate_peak_distributions:
            if estimate_peak_distributions_kw is None:
                estimate_peak_distributions_kw = {}
            peak_gammas = drt.estimate_peak_distributions(tau=tau, tau_find_peaks=tau, peak_indices=peak_indices,
                                                          x=fit_parameters['x'],
                                                          **estimate_peak_distributions_kw)
        else:
            peak_gammas = None

        # Construct model string
        drt_index = model_string.find('-{DRT}')
        if drt_index >= 0 and drt_element is None:
            raise ValueError('An element type must be specified in drt_element if using automatic model construction')
        elif drt_index >= 0:
            # Parse offset model
            offset_model_string = model_string[:drt_index]
            element_names, element_types, _, _, _, _ = parse_model_string(offset_model_string)

            # Ensure that offset model contains only offset (non-DRT) elements
            if np.max([element_has_distribution(element_type) for element_type in element_types]):
                raise ValueError('If using automatic model construction, model_string must not contain any elements '
                                 'that contribute to the DRT')

            if type(drt_element) != str:
                raise ValueError('If using automatic model construction, drt_element must be a string corresponding to '
                                 'the desired element type for DRT approximation')

            # Construct model string
            _, last_id = parse_element_string(element_names[-1])  # Find last index in non-DRT elements
            start_id = last_id + 1  # Starting index for DRT elements to ensure unique element names
            drt_model_string = '-'.join([f'{drt_element}{i}' for i in range(start_id, start_id + len(peak_indices))])
            model_string = f'{offset_model_string}-{drt_model_string}'
        else:
            # Fixed model specified. Determine elements that define the DRT
            element_names, element_types, _, _, _, _ = parse_model_string(model_string)
            num_elements = len(element_names)
            is_drt = [element_has_distribution(element_type) for element_type in element_types]

            # Get list of DRT elements for parameter estimation
            drt_element = [element_types[i] for i in range(num_elements) if is_drt[i]]

            # Get offset and DRT model strings
            offset_element_names = [element_names[i] for i in range(num_elements) if not is_drt[i]]
            offset_model_string = '-'.join(offset_element_names)
            drt_element_names = [element_names[i] for i in range(num_elements) if is_drt[i]]
            drt_model_string = '-'.join(drt_element_names)

            # Construct full model string (offsets first, then DRT - may be different order than original model_string)
            model_string = offset_model_string + drt_model_string

        # Initialize model with constructed model_string
        model = cls(model_string)

        # Estimate DRT element parameters from peaks
        if estimate_peak_distributions:
            drt_params = peaks.estimate_peak_params(tau, drt_element, peak_indices=peak_indices, f_peaks=peak_gammas)
        else:
            f = drt.predict_distribution(tau, x=fit_parameters['x'])
            fxx = drt.predict_distribution(tau, x=fit_parameters['x'], order=2)
            trough_indices = peaks.find_troughs(f, fxx, peak_indices)
            drt_params = peaks.estimate_peak_params(tau, drt_element,
                                                    f=f, peak_indices=peak_indices, trough_indices=trough_indices)

        # Get estimates for offset elements
        offset_names, offset_types, _, _, _, _ = parse_model_string(offset_model_string)
        offset_params = []
        for i in range(len(offset_names)):
            if offset_types[i] == 'R':
                params = [fit_parameters['R_inf']]
            elif offset_types[i] == 'L':
                params = [np.log(fit_parameters['inductance'])]
            else:
                params = [np.nan]
            offset_params += params

        # Get full list of initial values
        init_params = np.array(offset_params + sum(drt_params, []))

        # Get parameter bounds
        element_names, element_types, param_types, param_names, param_bounds, param_indices = \
            parse_model_string(model_string)

        # Check for negative resistances and invert bounds
        for i in range(len(param_names)):
            if param_types[i] == 'R' and init_params[i] < 0:
                lb, ub = param_bounds[i]
                param_bounds[i] = (-ub, -lb)

        if set_bounds:
            # Set bounds based on estimated values
            # element_names, element_types, param_types, param_names, param_bounds, param_indices = parse_model_string(
            #     model_string)
            new_bounds = param_bounds.copy()  # Default bounds
            # for i in range(len(element_names)):
            #     # param_types, _ = element_parameters(element_types[i])
            #     for j in range(len(param_types)):
            #         # If limits are provided for the parameter type, set new bounds
            #         # Otherwise, keep default bounds
            #         print(param_types[j])
            #         limits = parameter_limits.get(param_types[j], None)
            #         init_value = init_params[i][j]
            #         if limits is not None and init_value is not None:
            #             # Get default bounds. Don't allow new bounds to exceed default bounds
            #             lb, ub = new_bounds[param_indices[i][0] + j]
            #             if limits[0] == 'add':
            #                 new_bounds[param_indices[i][0] + j] = (max(lb, init_value + limits[1]),
            #                                                        min(ub, init_value + limits[2])
            #                                                        )
            #             elif limits[0] == 'multiply':
            #                 new_bounds[param_indices[i][0] + j] = (max(lb, init_value * limits[1]),
            #                                                        min(ub, init_value * limits[2])
            #                                                        )
            #             else:
            #                 raise ValueError(f'Invalid limit type {limits[0]} for parameter type {param_types[j]}')
            for i in range(len(param_names)):
                # If limits are provided for the parameter type, set new bounds
                # Otherwise, keep default bounds
                limits = parameter_limits.get(param_types[i], None)
                init_value = init_params[i]

                if limits is not None and not np.isnan(init_value):
                    # Get default bounds. Don't allow new bounds to exceed default bounds
                    lb, ub = new_bounds[i]
                    if limits[0] == 'add':
                        new_bounds[i] = (max(lb, init_value + limits[1]), min(ub, init_value + limits[2]))
                    elif limits[0] == 'multiply':
                        # Handle negative values
                        if init_value < 0:
                            nlb = init_value * limits[2]
                            nub = init_value * limits[1]
                        else:
                            nlb = init_value * limits[1]
                            nub = init_value * limits[2]
                        new_bounds[i] = (max(lb, nlb), min(ub, nub))
                    else:
                        raise ValueError(f'Invalid limit type {limits[0]} for parameter type {param_types[i]}')

            # print('new bounds:', new_bounds)
            model.set_bounds(new_bounds)
        else:
            # Set to default bounds based on allowed values for param type
            model.set_bounds(param_bounds)

        # Estimate weights
        # eis_sigma = drt.predict_sigma('eis')
        # if eis_sigma is not None:
        #     eis_weights = np.concatenate([1 / eis_sigma.real, 1 / eis_sigma.imag])
        # else:
        #     eis_weights = None

        # eis_weights = drt.qphb_params['est_weights'] / drt.impedance_scale
        eis_weights = utils.eis.complex_vector_to_concat(drt.predict_sigma('eis')) ** -1

        chrono_sigma = drt.predict_sigma('chrono')
        if chrono_sigma is not None:
            chrono_weights = 1 / chrono_sigma
        else:
            chrono_weights = None

        model.drt_estimates = {
            'init_values': init_params,
            'eis_weights': eis_weights,
            'chrono_weights': chrono_weights,
            'rss': drt.evaluate_rss(x=x_raw, normalize=True)
        }

        return model

    # ---------------------
    # Utility methods
    # ---------------------
    def get_parameter_values(self):
        return self._parameter_values

    def set_parameter_values(self, values):
        if values is None:
            self._parameter_values = values
        else:
            values = np.array(values)
            if len(values) != self.num_parameters:
                raise ValueError('Expected {} parameter values, but received {} values'.format(
                    self.num_parameters, len(values)
                ))
            else:
                self._parameter_values = values

    parameter_values = property(get_parameter_values, set_parameter_values)

    @property
    def num_elements(self):
        return len(self.element_types)

    @property
    def num_parameters(self):
        return len(self.parameter_bounds)

    @property
    def parameter_dict(self):
        return dict(zip(self.parameter_names, self.parameter_values))

    @property
    def drt_elements(self):
        elements = [self.element_names[i] for i in range(self.num_elements)
                    if element_has_distribution(self.element_types[i])]
        return elements

    def get_element_parameter_values(self, element_name, x=None):
        if x is None:
            x = self.parameter_values

        element_index = self.element_names.index(element_name)
        param_indices = self.parameter_indices[element_index]
        return x[param_indices[0]:param_indices[1]]

    def get_element_parameter_types(self, element_name):
        element_index = self.element_names.index(element_name)
        param_indices = self.parameter_indices[element_index]
        return self.parameter_types[param_indices[0]:param_indices[1]]

    def transform_parameters(self, x, inverse):
        """
        Transform parameters from bounded to unbounded space (or back)
        :param x: parameter list or array
        :param inverse: if True, transform from unbounded space to bounded (real) space.
        If False, transform from bounded (real) space to unbounded space
        :return:
        """
        if len(x) != self.num_parameters:
            raise ValueError(f'Expected {self.num_parameters} parameters, received {len(x)} parameter values')
        else:
            return np.array([
                constraint_transform(x[i], self.scaled_bounds[i], inverse) for i in range(len(x))
            ])

    def scale_parameters_to_data(self, x, inverse, apply_scaling):
        if len(x) != self.num_parameters:
            raise ValueError(f'Expected {self.num_parameters} parameters, received {len(x)} parameter values')
        else:
            if apply_scaling:
                return np.array([
                    scale_parameter_to_data(x[i], self.parameter_types[i], self.rp_scale, inverse)
                    for i in range(len(x))
                ])
            else:
                return x

    def scale_bounds_to_data(self, bounds, inverse, apply_scaling):
        if len(bounds) != self.num_parameters:
            raise ValueError(f'Expected {self.num_parameters} bounds, received {len(bounds)} bounds')
        else:
            if apply_scaling:
                return [
                    (
                        scale_parameter_to_data(bound[0], self.parameter_types[i], self.rp_scale, inverse),
                        scale_parameter_to_data(bound[1], self.parameter_types[i], self.rp_scale, inverse)
                    )
                    for i, bound in enumerate(bounds)
                ]
            else:
                return bounds

    def get_parameter_scales(self, parameter_values):
        """
        Get scaling factors for parameters to ensure consistent magnitude. For use in optimization, Hessian scaling,
        and setting prior strength
        :param parameter_values:
        :return:
        """
        param_types_array = np.array(self.parameter_types)
        parameter_scale = np.abs(parameter_values)
        parameter_scale[np.where(param_types_array == 'alpha')] = 2  # 1
        parameter_scale[np.where(param_types_array == 'beta')] = 1  # 0.5
        parameter_scale[np.where(param_types_array == 'lntau')] = 1
        parameter_scale[np.where(param_types_array == 'lnL')] = 1
        parameter_scale[np.where(param_types_array == 'R')] *= 2

        return parameter_scale

    def set_bounds(self, bounds, element_name=None):
        """
        Set bounds on parameter values, either for a single element specified by element_name,
        or all elements in model if element_name is None.
        :param list bounds: list of bound tuples (lower, upper)
        :param str element_name: element for which to set bounds. If None, set bounds for all elements in model
        :return:
        """
        if element_name is None:
            # Set bounds for all elements
            if len(bounds) != len(self.parameter_bounds):
                raise ValueError(f'Length of provided bounds ({len(bounds)}) does not match '
                                 f'number of parameters ({self.num_parameters})')
            else:
                self.parameter_bounds = bounds
        else:
            # Set bounds for specified element only
            try:
                element_index = self.element_names.index(element_name)
                param_start_index = self.parameter_indices[element_index][0]
                for i, bound in enumerate(bounds):
                    self.parameter_bounds[param_start_index + i] = bound
            except ValueError:
                raise ValueError(f'No element named {element_name} in model')

    def get_element_bounds(self, element_name):
        """
        Get bounds on parameter values for the element specified by element_name.
        :param str element_name: Element for which to get bounds.
        :return:
        """
        try:
            element_index = self.element_names.index(element_name)
            param_start_index = self.parameter_indices[element_index][0]
            param_end_index = self.parameter_indices[element_index][1]
            return [self.parameter_bounds[i] for i in range(param_start_index, param_end_index)]
        except ValueError:
            raise ValueError(f'No element named {element_name} in model')

    def get_time_constants(self, sort=False, x=None):
        if x is None:
            x = self.parameter_values

        ln_tau = [pv for pt, pv in zip(self.parameter_types, x) if pt == 'lntau']
        tau = np.exp(np.array(ln_tau))

        # Sort time constants
        if sort:
            tau = np.sort(tau)

        return tau

    def get_peak_tau(self, tau_grid=None, normalize=True, find_peaks_kw=None, x=None):
        if tau_grid is None:
            # Use fine spacing to avoid missing any peaks
            # tau_c = self.get_time_constants()
            # tau_grid = pp.get_basis_tau(None, tau_c, [0], 50, 1)
            tau_grid = pp.get_basis_tau(self.f_fit, self.t_fit, self.step_times, 50, 2)
            # print(tau_grid)

        gamma = self.predict_distribution(tau_grid, x=x)
        if normalize:
            gamma /= self.predict_r_p()

        # Find peaks using numerical derivatives
        fx = np.diff(gamma) / np.diff(np.log(tau_grid))
        fxx = np.diff(fx) / np.diff(np.log(tau_grid[1:]))
        # peak_indices = peaks.find_peaks_compound(fx[1:], fxx, **find_peaks_kw) + 1
        if find_peaks_kw is None:
            find_peaks_kw = {'height': 0}
        peak_indices = peaks.find_peaks_simple(fxx, 2, **find_peaks_kw)

        if len(peak_indices) > 0:
            peak_tau = tau_grid[peak_indices + 1]
        else:
            peak_tau = np.array([])

        # Check for any singular elements whose peaks may have been missed
        if self.is_singular:
            sing_tau = np.array([si[1] for si in self.get_singularity_info(x)])

            # Check if each singularity was already captured by find_peaks
            # If it was, it should be within dx of the true location
            dx = np.mean(np.abs(np.diff(np.log(tau_grid))))
            add_peak_index = peaks.find_new_peaks(np.log(sing_tau), np.log(peak_tau), dx)

            peak_tau = np.sort(np.concatenate((peak_tau, sing_tau[add_peak_index])))

        return peak_tau

    def get_element_singular_status_list(self):
        """
        Get list of booleans indicating whether each element in the model contains a singularity in its distribution
        """
        return [
            element_distribution_is_singular(
                self.element_types[i], *self.parameter_values[self.parameter_indices[i][0]:self.parameter_indices[i][1]]
            )
            for i in range(self.num_elements)
        ]

    @property
    def is_singular(self):
        if self.get_element_singular_status_list().count(True) > 0:
            return True
        else:
            return False

    def get_singularity_info(self, x=None):
        """
        Get list of locations (tau) of singularities in the model distribution
        :return:
        """
        if x is None:
            x = self.parameter_values

        stat_info = [
            element_distribution_is_singular(
                self.element_types[i],
                *x[self.parameter_indices[i][0]:self.parameter_indices[i][1]],
                return_info=True
            )
            for i in range(self.num_elements)
        ]
        info = [sl[1] for sl in stat_info if sl[0]]

        return info

    @property
    def singularity_info(self):
        return self.get_singularity_info()

    # ------------------
    # Preprocessing
    # ------------------
    def scale_data(self, times, i_signal, v_signal, chrono_weights, step_times, step_sizes, z, eis_weights,
                   apply_scaling):
        if apply_scaling:
            # Estimate Rp with provided data (chrono, EIS, or both)
            input_signal, response_signal = utils.chrono.get_input_and_response(i_signal, v_signal, self.chrono_mode)
            self.rp_scale = pp.estimate_rp(times, step_times, step_sizes, response_signal, self.chrono_step_model, z)
            # self.rp_scale *= 1000
        else:
            self.rp_scale = 1

        # self.rp_scale = rp_est

        # Scale chrono data
        if times is not None:
            if apply_scaling:
                # Scale input signal such that mean step size is 1
                self.input_signal_scale = pp.get_input_signal_scale(times, step_times, step_sizes,
                                                                    self.chrono_step_model)

                # Scale response signal to achieve desired Rp scale
                self.response_signal_scale = self.input_signal_scale * self.rp_scale  # / rp_scale

            else:
                # No scaling
                self.input_signal_scale = 1.0
                self.response_signal_scale = 1.0

            # Set input and response signals based on control mode
            input_signal, response_signal = utils.chrono.get_input_and_response(i_signal, v_signal, self.chrono_mode)
            self.raw_input_signal = input_signal.copy()
            self.raw_response_signal = response_signal.copy()

            self.scaled_input_signal = self.raw_input_signal / self.input_signal_scale
            self.scaled_response_signal = self.raw_response_signal / self.response_signal_scale

            scaled_chrono_weights = chrono_weights * self.response_signal_scale

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
            scaled_chrono_weights = None

        # Scale EIS data
        if z is not None:
            # Impedance scale is rp_scale - already set whether apply_scaling True or False
            z_scaled = z / self.rp_scale

            self.z_fit = z.copy()
            self.z_fit_scaled = z_scaled.copy()

            scaled_eis_weights = eis_weights * self.rp_scale
        else:
            # No EIS data provided
            z_scaled = None
            scaled_eis_weights = None

        # Scale parameter bounds
        self.scaled_bounds = self.scale_bounds_to_data(self.parameter_bounds, False, apply_scaling)

        return scaled_i_signal, scaled_v_signal, scaled_chrono_weights, z_scaled, scaled_eis_weights

    # ------------------------
    # Fitting methods
    # ------------------------
    def fit_eis(self, freq, z, from_drt=False, weights=None, scale_data=True, init_values=None, fit_unbounded=False,
                fast_transform=True, jac=None, prior=False, prior_strength=None,
                seed=123, method=None):

        # Get DRT estimates
        if from_drt:
            if self.drt_estimates is None:
                raise ValueError('Model must be initialized using from_drt to use DRT estimates for fit')
            else:
                init_values = self.drt_estimates['init_values']
                weights = self.drt_estimates['eis_weights']

        # Get data weights
        if weights is None:
            weights = np.ones(2 * len(z))

        # Scale data
        self.f_fit = freq.copy()
        _, _, _, z_scaled, scaled_weights = self.scale_data(None, None, None, None, None, None, z, weights, scale_data)

        self.scaled_weights = scaled_weights.copy()
        self.weights = weights.copy()

        # Get transforms
        if fast_transform:
            transform, inv_transform = get_fast_constraint_transforms(self.scaled_bounds)
        else:
            def transform(x):
                return self.transform_parameters(x, False)

            def inv_transform(x):
                return self.transform_parameters(x, True)

        # Initialize parameters
        if init_values is None:
            # Initialize unbounded variables randomly
            rng = np.random.default_rng(seed=seed)
            x0_unbnd = rng.uniform(-2, 2, self.num_parameters)

            # Transform to desired space for fitting
            if not fit_unbounded:
                # Transform to bounded space
                x0 = inv_transform(x0_unbnd)
            else:
                # Leave in unbounded space
                x0 = x0_unbnd
        else:
            if len(init_values) != self.num_parameters:
                raise ValueError(f'Length of init ({len(init_values)}) does not match number of '
                                 f'model parameters ({self.num_parameters})')

            init_values = np.array(init_values)

            # Randomly initialize any parameters not specified
            rng = np.random.default_rng(seed=seed)
            x0_rand_unbnd = rng.uniform(-2, 2, self.num_parameters)
            x0_rand = inv_transform(x0_rand_unbnd)
            init_scaled = self.scale_parameters_to_data(init_values, False, scale_data)
            rand_index = np.isnan(init_values)
            init_scaled[rand_index] = x0_rand[rand_index]

            if fit_unbounded:
                # Transform raw initial values to unbounded space
                x0 = transform(init_scaled)
            else:
                x0 = init_scaled

        if prior:
            if init_values is None:
                prior = False
                self.prior_params = None
            else:
                if prior_strength is None:
                    if from_drt:
                        rss_factor = np.exp(1 - self.drt_estimates['rss'])
                        print('rss factor:', rss_factor)
                    else:
                        rss_factor = 1

                    prior_strength = rss_factor * (np.mean(scaled_weights) * 0.05) ** 0.5
                    print('prior_strength:', prior_strength)
                # TODO: set prior strength correctly for lnL
                # Scale the prior strength to the estimated parameter values
                raw_prior_weights = prior_strength / self.get_parameter_scales(x0)  # np.ones(self.num_parameters)
                # Don't apply a prior to any values that were initialized randomly
                raw_prior_weights[np.isnan(init_values)] = 0

                raw_prior_mu = x0.copy()  # Scaled and transformed
                prior_mu = init_values.copy()  # Real space
                raw_prior_mu[np.isnan(init_values)] = 0
                prior_mu[np.isnan(init_values)] = 0
                self.prior_params = {
                    'mu': prior_mu,
                    'raw_mu': raw_prior_mu,
                    'raw_weights': raw_prior_weights,
                }
                if fit_unbounded:
                    # self.prior_params['mu'] = self.scale_parameters_to_data(inv_transform(prior_mu), True, scale_data)
                    # self.prior_params['weights'] = self.
                    pass
                    # TODO: figure out how weights should be transformed
                else:
                    self.prior_params['weights'] = self.scale_parameters_to_data(raw_prior_weights ** -1, True,
                                                                                 scale_data) ** -1
                    # self.prior_params['weights'] = raw_prior_weights * np.mean(weights) / np.mean(scaled_weights)
        else:
            self.prior_params = None

        # Store initial values for reference. Store values in real (bounded, unscaled) space for easy interpretation
        if fit_unbounded:
            self.init_values = self.scale_parameters_to_data(inv_transform(x0), True, scale_data)
        else:
            self.init_values = self.scale_parameters_to_data(x0, True, scale_data)

        # Define residual function
        z_flat = utils.eis.complex_vector_to_concat(z_scaled)

        if fit_unbounded:
            def z_func(x):
                return self.z_function(freq, *inv_transform(x))
        else:
            def z_func(x):
                return self.z_function(freq, *x)

        if prior:
            def residual_func(x):
                z_hat = z_func(x)

                # Flatten complex to real
                z_hat = utils.eis.complex_vector_to_concat(z_hat)

                z_err = scaled_weights * (z_hat - z_flat)

                prior_resid = raw_prior_weights * (x - x0)

                return np.concatenate((z_err, prior_resid))
        else:
            def residual_func(x):
                z_hat = z_func(x)

                # Flatten complex to real
                z_hat = utils.eis.complex_vector_to_concat(z_hat)

                z_err = scaled_weights * (z_hat - z_flat)

                return z_err

        # Specify Jacobian method
        if jac is None:
            if fit_unbounded:
                # Can't use analytical Jacobian due to constraint transforms
                jac = '2-point'
            else:
                # Get analytical Jacobian function
                f_jac = model_f_jacobian(self.parameter_indices, self.element_types)
                weight_matrix = np.diag(scaled_weights)

                if prior:
                    def jac(x):
                        jac_z_err = weight_matrix @ f_jac(freq, *x)
                        jac_prior = np.diag(raw_prior_weights)
                        jac_tot = np.vstack((jac_z_err, jac_prior))
                        return jac_tot
                else:
                    def jac(x):
                        return weight_matrix @ f_jac(freq, *x)

        # print('x0', x0)
        # print('bounds', self.scaled_bounds)

        if fit_unbounded:
            if method is None:
                method = 'lm'
            self.fit_result = least_squares(residual_func, x0, method=method, jac=jac)
            self.raw_parameter_values = self.fit_result['x'].copy()
            self.scaled_parameter_values = inv_transform(self.fit_result['x'])
            self.parameter_values = self.scale_parameters_to_data(inv_transform(self.fit_result['x']), True, scale_data)
        else:
            if method is None:
                method = 'trf'
            self.fit_result = least_squares(residual_func, x0, bounds=flatten_bounds(self.scaled_bounds),
                                            method=method, jac=jac)
            self.raw_parameter_values = self.fit_result['x'].copy()
            self.scaled_parameter_values = self.fit_result['x'].copy()
            self.parameter_values = self.scale_parameters_to_data(self.fit_result['x'], True, scale_data)

    # -------------------------
    # Prediction and evaluation
    # -------------------------
    def predict_z(self, freq, x=None):
        if x is None:
            x = self.parameter_values
        return self.z_function(freq, *x)

    def predict_v(self, times, step_times, step_sizes, x=None):
        if x is None:
            x = self.parameter_values

        if len(step_times) != len(step_sizes):
            raise ValueError('step_times and step_sizes must have same length')

        # Get response to each step
        v_steps = np.zeros((len(step_times), len(times)))
        for i, step_time in enumerate(step_times):
            step_size = step_sizes[i]
            time_delta = times - step_time
            v_steps[i] = step_size * self.v_function(time_delta, *x)

        return np.sum(v_steps, axis=0)

    def predict_r_tot(self):
        """
        Estimate total resistance
        :return:
        """
        # Get indices of resistances
        r_index = np.where(np.array(self.parameter_types) == 'R')
        return np.sum(self.parameter_values[r_index])

    def predict_r_p(self):
        """
        Estimate polarization resistance
        :return:
        """
        rp = 0

        # Sum resistances of elements that contribute to the DRT
        for i, element_type in enumerate(self.element_types):
            if element_has_distribution(element_type):
                param_indices = self.parameter_indices[i]
                r_index = param_indices[0] + self.parameter_types[param_indices[0]:param_indices[1]].index('R')
                rp += self.parameter_values[r_index]

        return rp

    def predict_distribution(self, tau, x=None):
        if x is None:
            x = self.parameter_values
        return self.gamma_function(tau, *x)

    def predict_element_distribution(self, tau, element, x=None):
        if type(element) == int:
            element_index = element
            element_name = self.element_names[element]
        elif type(element) == str:
            element_index = self.element_names.index(element)
            element_name = element
        else:
            raise ValueError('element must be either an index or element name')

        element_type = self.element_types[element_index]
        element_params = self.get_element_parameter_values(element_name, x=x)

        gamma_func = element_distribution_function(element_type)

        return gamma_func(tau, *element_params)

    def predict_mass(self, tau, x=None):
        if x is None:
            x = self.parameter_values
        return self.mass_function(tau, *x)

    def evaluate_eis_residuals(self, x=None):
        z_hat = self.predict_z(self.f_fit, x=x)
        z_err = z_hat - self.z_fit
        return z_err

    def predict_sigma(self, epsilon=0.25, reim_cor=0.25, error_structure=None):
        vmm = mat1d.construct_eis_var_matrix(self.f_fit, epsilon, reim_cor, error_structure)
        z_err = self.evaluate_eis_residuals()
        z_err_flat = utils.eis.complex_vector_to_concat(z_err)
        s_hat = vmm @ z_err_flat ** 2
        sigma_flat = np.sqrt(s_hat)
        sigma_complex = utils.eis.concat_vector_to_complex(sigma_flat)
        return sigma_complex

    def estimate_eis_weights(self, epsilon=0.25, reim_cor=0.25, error_structure=None):
        sigma_complex = self.predict_sigma(epsilon, reim_cor, error_structure)
        sigma_flat = utils.eis.complex_vector_to_concat(sigma_complex)
        weights = sigma_flat ** -1
        return weights

    def evaluate_llh(self, weights=None, marginalize_weights=True, alpha_0=1, beta_0=1, include_constants=False,
                     x=None):
        """
        Evaluate log-likelihood
        :param weights:
        :param include_constants:
        :return:
        """
        z_err = self.evaluate_eis_residuals(x=x)
        z_err_flat = utils.eis.complex_vector_to_concat(z_err)

        if weights is None:
            weights = self.weights
            # weights = self.estimate_eis_weights()
            # print('mean weight:', np.mean(weights))

        rss = np.sum((weights * z_err_flat) ** 2)

        if marginalize_weights:
            alpha_n = alpha_0 - 1 + len(z_err_flat) / 2
            beta_n = beta_0 + 0.5 * rss
            llh = alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n) + special.loggamma(alpha_n) - special.loggamma(alpha_0)
        else:
            llh = -0.5 * rss

        # Add sum of log weights
        llh += np.sum(np.log(weights))

        if include_constants:
            llh -= 0.5 * len(weights) * np.log(2 * np.pi)

        return llh

    def evaluate_bic(self, **llh_kw):
        num_data = len(self.z_fit)
        llh = self.evaluate_llh(**llh_kw)
        return utils.stats.bic(self.num_parameters, num_data, llh)

    def evaluate_hessian(self, weights=None, include_prior=False):
        if weights is None:
            weights = self.weights

        # Get Hessian
        hess_func = model_llh_hessian(self.parameter_indices, self.element_types, self.z_function)
        hess = hess_func(self.f_fit, self.z_fit, weights, *self.parameter_values)

        if include_prior and self.prior_params is not None:
            prior_precision = np.diag(self.prior_params['weights'] ** 2)
            hess -= prior_precision

        return hess

    def estimate_lml(self, weights=None, scale_hessian=False, include_prior=True, **kw):
        if weights is None:
            weights = self.weights

        # Get log-likelihood
        llh = self.evaluate_llh(weights=weights, **kw)

        # Get Hessian
        hess = self.evaluate_hessian(weights, include_prior)

        # Scale hessian
        if scale_hessian:
            parameter_scale = self.get_parameter_scales(self.parameter_values)
            sm = np.diag(parameter_scale)
            hess = sm @ hess @ sm

        # Posterior precision matrix
        prec_matrix = -hess

        # Incorporate prior terms
        if include_prior and self.prior_params is not None:
            # TODO: check this expression AND the scaling of the prior!!
            prior_weights = self.prior_params['weights']
            # prior_prec = np.diag(prior_weights ** 2)
            prior_mu = self.prior_params['mu']

            # # Add prior to precision matrix
            # prec_matrix += prior_prec  # This is already done in evaluate_hessian!

            # # Adjust prior_prec for params with uniform priors - these should not contribute to lp_theta
            # prior_weights_adj = prior_weights.copy()
            # prior_weights_adj[prior_weights == 0] = 1
            # prior_prec_adj = np.diag(prior_weights_adj ** 2)
            # det_sign, log_det_prior = np.linalg.slogdet(2 * np.pi * prior_prec_adj)
            #
            # # Get llh gradient wrt parameter vector
            # grad_func = model_llh_grad(self.parameter_indices, self.element_types, self.z_function)
            # grad_theta = grad_func(self.f_fit, self.z_fit, weights, *self.parameter_values)

            # lp_theta = -0.5 * np.sum((prior_weights * (self.parameter_values - prior_mu)) ** 2) \
            #            + 0.5 * log_det_prior - grad_theta @ self.parameter_values
            # print(lp_theta)
            lp_theta = -0.5 * np.sum((prior_weights * (self.parameter_values - prior_mu)) ** 2)
        else:
            lp_theta = 0

        # TODO: add warning for zeros in diagonal of precision - indicates junk parameters
        prec_matrix[prec_matrix == 0] = 1e-15  # Can't have zeros in precision matrix

        # Calculate determinant of Hessian
        det_sign, log_det = np.linalg.slogdet(2 * np.pi * prec_matrix)
        # print('log_det_prec:', log_det)
        # print('det_sign, log_det:', det_sign, log_det)
        if det_sign < 0:
            warnings.warn('Determinant of precision matrix is negative. This may indicate a suboptimal '
                          'solution. Check the fit and LML')

        lml = llh + lp_theta - 0.5 * log_det

        return lml

    # TODO: use Hessian to estimate parameter covariance
    def estimate_param_cov(self, weights=None, rel_jitter=0, **hess_kw):
        hess = self.evaluate_hessian(weights, **hess_kw)
        jitter = self.parameter_values * rel_jitter
        hess -= np.diag(jitter)
        cov = np.linalg.inv(-hess)

        return cov

    # ---------------------------
    # Plotting
    # ---------------------------
    def plot_distribution(self, tau, x=None, ax=None, area=None, scale_prefix=None,
                          mark_peaks=False, limit_peak_heights=True,
                          mark_peaks_kw=None, normalize=False, show_singularities=True, singularity_scale=None,
                          return_line=False, y_offset=0, **kw):

        gamma = self.predict_distribution(tau, x)

        # max_val = np.max(gamma[gamma < np.inf])
        # min_val = np.min(gamma[gamma > -np.inf])
        # gamma[gamma == np.inf] = max_val * 1.1
        # gamma[gamma == -np.inf] = min_val * 1.1

        if normalize:
            scale_prefix = ''
            normalize_by = self.predict_r_p()
        else:
            normalize_by = None

        ax, info = plot_distribution(tau, gamma + y_offset, ax, area, scale_prefix, normalize_by, return_info=True, **kw)
        line, scale_prefix, scale_factor = info

        # Update the scale factor for area and/or R_p
        if area is not None:
            scale_factor /= area

        if normalize_by is not None:
            scale_factor *= normalize_by

        if normalize:
            y_label = fr'$\gamma \, / \, R_p$'
        else:
            if area is not None:
                y_units = r'$\Omega \cdot \mathrm{cm}^2$'
            else:
                y_units = r'$\Omega$'
            y_label = fr'$\gamma$ ({scale_prefix}{y_units})'

        ax.set_ylabel(y_label)

        # Indicate peak positions
        if mark_peaks:
            if mark_peaks_kw is None:
                mark_peaks_kw = {}

            # Get exact peak positions and heights
            peak_tau = self.get_time_constants()  # self.get_peak_tau(tau, find_peaks_kw={'prominence': 0})
            # Evaluate just to the left of the peak to avoid infinite values in case of singularities
            peak_gamma = self.predict_distribution(peak_tau * (1 - 1e-5), x=x)

            if limit_peak_heights:
                # Select tau locations from provided tau. Use height at selected location to mark peak
                def get_peak_plot_tau(pt, pg):
                    t1 = tau[utils.array.nearest_index(tau, pt, constraint=-1)]
                    t2 = tau[utils.array.nearest_index(tau, pt, constraint=1)]
                    g1 = self.predict_distribution(t1, x=x)
                    g2 = self.predict_distribution(t2, x=x)
                    if abs(pg - g1) < abs(pg - g2):
                        return t1
                    else:
                        return t2

                plot_peak_tau = np.array(
                    [get_peak_plot_tau(pt, pg) for pt, pg in zip(peak_tau, peak_gamma)]
                )
                plot_peak_gamma = self.predict_distribution(plot_peak_tau, x=x)
            else:
                # Use the exact location and height to mark peaks
                # May yield undesirable results for peaks with dispersion parameter (beta) close to 1
                plot_peak_tau = peak_tau
                plot_peak_gamma = peak_gamma

            ax.scatter(plot_peak_tau, (plot_peak_gamma + y_offset) / scale_factor, **mark_peaks_kw)

        fig = ax.get_figure()
        fig.tight_layout()

        if show_singularities:
            # Match color to distribution
            if kw.get('c', kw.get('color', None)) is None:
                kw['c'] = line[0].get_color()

            # Remove label from kw to avoid duplicate legend entries
            kw['label'] = ''

            # Fix y limit to ensure singularity spans entire axis
            # ax.set_ylim(ax.get_ylim())

            self.plot_singularities(ax, scale_factor, singularity_scale, x, **kw)

        if return_line:
            return ax, line
        else:
            return ax

    def plot_element_distributions(self, tau, x=None, ax=None, area=None, scale_prefix=None, normalize=False,
                                   show_singularities=True, singularity_scale=None, return_lines=False,
                                   y_offset=0, kw_list=None, mark_peaks=False, mark_peaks_kw=None, **common_kw):
        plot_elements = [el_name for el_name, el_type in zip(self.element_names, self.element_types)
                         if element_has_distribution(el_type)]

        if kw_list is None:
            kw_list = [{} for i in range(len(plot_elements))]  # need unique dicts

        if len(kw_list) != len(plot_elements):
            raise ValueError(f'Length of kw_list ({len(kw_list)}) must match number of elements ({len(plot_elements)})')

        # Get normalization factor
        if normalize:
            scale_prefix = ''
            normalize_by = self.predict_r_p()
        else:
            normalize_by = None

        # Get element distributions
        el_gammas = [self.predict_element_distribution(tau, el_name, x=x) for el_name in plot_elements]

        # Set singularity scale
        if singularity_scale is None:
            all_gamma = np.concatenate(el_gammas)
            singularity_scale = np.max(all_gamma[~np.isinf(all_gamma)])

        # Get common scale factor
        if scale_prefix is None:
            scale_prefix = utils.scale.get_common_scale_prefix(el_gammas)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # # Update the scale factor for area and/or R_p
        # if area is not None:
        #     scale_factor /= area
        #
        # if normalize_by is not None:
        #     scale_factor *= normalize_by

        # Plot each element distribution
        lines = []
        for i, el_name in enumerate(plot_elements):
            el_index = self.element_names.index(el_name)
            el_is_singular, sing_info = element_distribution_is_singular(
                self.element_types[el_index], *self.get_element_parameter_values(el_name, x=x), return_info=True
            )

            el_gamma = el_gammas[i]

            ax, info = plot_distribution(tau, el_gamma + y_offset, ax, area, scale_prefix, normalize_by,
                                         return_info=True, **kw_list[i], **common_kw)
            line, _, _ = info
            lines.append(line)

            if el_is_singular and show_singularities:
                # Format array for vertical line
                r_sing, tau_sing = sing_info

                y_end = np.sign(r_sing) * singularity_scale

                sing_tau = np.array([tau_sing, tau_sing])
                sing_gamma = np.array([0, y_end])

                # Match color of existing line
                if kw_list[i].get('c', kw_list[i].get('color', None)) is None:
                    kw_list[i]['color'] = line[0].get_color()

                plot_distribution(sing_tau, sing_gamma + y_offset, ax, area, scale_prefix, normalize_by, **kw_list[i],
                                  **common_kw)

            # Mark peaks
            if mark_peaks:
                peak_index = np.argmax(np.abs(el_gamma))
                peak_tau = tau[peak_index]
                peak_gamma = el_gamma[peak_index]
                # Get correct scaling factor
                if area is not None:
                    factor = area / scale_factor
                else:
                    factor = 1 / scale_factor
                # Get kwargs
                if mark_peaks_kw is None:
                    mark_peaks_kw_i = dict(alpha=0.6, s=20)
                    # If color provided in kw_list, use that color
                    color = kw_list[i].get('c', kw_list[i].get('color', None))
                    if color is not None:
                        mark_peaks_kw_i['color'] = color
                else:
                    mark_peaks_kw_i = mark_peaks_kw
                ax.scatter([peak_tau], [(peak_gamma + y_offset) * factor], **mark_peaks_kw_i)

        if normalize:
            y_label = fr'$\gamma \, / \, R_p$'
        else:
            if area is not None:
                y_units = r'$\Omega \cdot \mathrm{cm}^2$'
            else:
                y_units = r'$\Omega$'
            y_label = fr'$\gamma$ ({scale_prefix}{y_units})'

        ax.set_ylabel(y_label)
        fig = ax.get_figure()
        fig.tight_layout()

        if return_lines:
            return ax, lines
        else:
            return ax

    def plot_singularities(self, ax, scale_factor=1, scale=None, x=None, y_offset=0, **kw):
        """
        Plot singularities in the model distribution as vertical lines
        :param ax: Axes on which to plat
        :param float scale_factor: Scale factor to apply to y start value to ensure it matches up with the non-singular
        component of the distribution
        :param float scale: If provided, the maximum y value to which to extend the singularity line.
        If None, the line will extend to the axis limit
        :param kw: Keyword args to pass to plt.plot
        :return:
        """
        for si in self.get_singularity_info(x):
            r, tau = si
            # Start at distribution value just past singularity
            y_start = (self.predict_distribution(tau * (1 + 1e-3)) + y_offset) / scale_factor

            # End at axis limit
            if scale is not None:
                y_end = (np.sign(r) * scale + y_offset) / scale_factor
            elif abs(r) > 0:
                y_end = ax.get_ylim()[int(0.5 * (1 + np.sign(r)))]
            else:
                y_end = y_start

            ax.plot([tau, tau], [y_start, y_end], **kw)

    def plot_mass(self, tau, x=None, ax=None, area=None, scale_prefix=None, normalize=False, **kw):
        mass = self.predict_mass(tau, x)

        if normalize:
            scale_prefix = ''
            normalize_by = self.predict_r_p()
        else:
            normalize_by = None

        ax, info = plot_distribution(tau, mass, ax, area, scale_prefix, normalize_by, return_info=True, **kw)
        line, scale_prefix, scale_factor = info

        if normalize:
            y_label = fr'$p \, / \, R_p$ (a.u.)'
        else:
            if area is not None:
                y_units = r'$\Omega \cdot \mathrm{cm}^2$'
            else:
                y_units = r'$\Omega$'
            y_label = fr'$p$ ({scale_prefix}{y_units})'

        ax.set_ylabel(y_label)

        fig = ax.get_figure()
        fig.tight_layout()

    def plot_eis_fit(self, frequencies=None, axes=None, plot_type='nyquist', plot_data=True, data_kw=None,
                     bode_cols=['Zreal', 'Zimag'], data_label='', scale_prefix=None, area=None,
                     predict_kw=None, c='k', **kw):

        # Set default data plotting kwargs if not provided
        if predict_kw is None:
            predict_kw = {}
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # # Get data df if requested
        # if plot_data:
        #     f_fit = self.f_fit
        #     data_df = utils.eis.construct_eis_df(f_fit, self.z_fit)
        # else:
        #     data_df = None

        # Get model impedance
        if frequencies is None:
            frequencies = self.f_fit
        z_hat = self.predict_z(frequencies, **predict_kw)
        # df_hat = utils.eis.construct_eis_df(frequencies, z_hat)

        # Get scale prefix
        if scale_prefix is None:
            if area is None:
                area_mult = 1
            else:
                area_mult = area
            z_hat_concat = utils.eis.complex_vector_to_concat(z_hat) * area_mult
            if plot_data:
                z_data_concat = utils.eis.complex_vector_to_concat(self.z_fit) * area_mult
                scale_prefix = utils.scale.get_common_scale_prefix([z_data_concat, z_hat_concat])
            else:
                scale_prefix = utils.scale.get_scale_prefix(z_hat_concat)

            # if plot_data:
            #     z_data_concat = np.concatenate([data_df['Zreal'], data_df['Zimag']])
            #     z_hat_concat = np.concatenate([df_hat['Zreal'], df_hat['Zimag']])
            #     scale_prefix = utils.scale.get_common_scale_prefix([z_data_concat, z_hat_concat])
            # else:
            #     z_hat_concat = np.concatenate([df_hat['Zreal'], df_hat['Zimag']])
            #     scale_prefix = utils.scale.get_scale_prefix(z_hat_concat)

        # Plot data if requested
        if plot_data:
            axes = plot_eis((self.f_fit, self.z_fit), plot_type, axes=axes, scale_prefix=scale_prefix, label=data_label,
                            bode_cols=bode_cols, area=area, **data_kw)

        # Plot fit
        axes = plot_eis((frequencies, z_hat), plot_type, axes=axes, plot_func='plot', c=c, scale_prefix=scale_prefix,
                        bode_cols=bode_cols, area=area, **kw)

        fig = np.atleast_1d(axes)[0].get_figure()
        fig.tight_layout()

        return axes

    def plot_eis_residuals(self, plot_sigma=True, axes=None, scale_prefix=None, predict_kw=None, part='both',
                           s=10, alpha=0.5, **kw):

        if part == 'both':
            bode_cols = ['Zreal', 'Zimag']
        elif part == 'real':
            bode_cols = ['Zreal']
        elif part == 'imag':
            bode_cols = ['Zimag']
        else:
            raise ValueError(f"Invalid part {part}. Options: 'both', 'real', 'imag'")

        if axes is None:
            fig, axes = plt.subplots(1, len(bode_cols), figsize=(3 * len(bode_cols), 2.75))
        else:
            fig = np.atleast_1d(axes)[0].get_figure()
        axes = np.atleast_1d(axes)

        # Get model impedance
        if predict_kw is None:
            predict_kw = {}
        y_hat = self.predict_z(self.f_fit, **predict_kw)

        # Calculate residuals
        y_err = self.z_fit - y_hat

        # Get scale prefix
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(np.concatenate([y_err.real, y_err.imag]))

        # Plot residuals
        plot_eis((self.f_fit, y_err), axes=axes, plot_type='bode', bode_cols=bode_cols,
                 s=s, alpha=alpha, label='Residuals', scale_prefix=scale_prefix, **kw)

        # Indicate zero
        for ax in axes:
            ax.axhline(0, c='k', lw=1, zorder=-10)

        # Plot error structure
        if plot_sigma:
            sigma = self.predict_sigma()
            if sigma is not None:
                scale_factor = scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)
                if 'Zreal' in bode_cols:
                    axes[bode_cols.index('Zreal')].fill_between(self.f_fit, -3 * sigma.real / scale_factor,
                                                                3 * sigma.real / scale_factor,
                                                                color='k', lw=0, alpha=0.15, zorder=-10,
                                                                label=r'$\pm 3 \sigma$')
                if 'Zimag' in bode_cols:
                    axes[bode_cols.index('Zimag')].fill_between(self.f_fit, -3 * sigma.imag / scale_factor,
                                                                3 * sigma.imag / scale_factor,
                                                                color='k', lw=0, alpha=0.15, zorder=-10,
                                                                label=r'$\pm 3 \sigma$')

            axes[-1].legend()

        # Update axis labels
        if 'Zreal' in bode_cols:
            axes[bode_cols.index('Zreal')].set_ylabel(fr'$Z^{{\prime}} - \hat{{Z}}^{{\prime}}$ ({scale_prefix}$\Omega$)')
        if 'Zimag' in bode_cols:
            axes[bode_cols.index('Zimag')].set_ylabel(
                fr'$-(Z^{{\prime\prime}} - \hat{{Z}}^{{\prime\prime}})$ ({scale_prefix}$\Omega$)')

        fig.tight_layout()

        return axes


# ===============================
# Element functions
# ===============================
def element_has_distribution(element_type):
    if element_type in ('HN', 'RQ', 'RC'):
        return True
    else:
        return False


def element_parameters(element_type):
    if element_type == 'HN':
        parameter_types = ['R', 'lntau', 'alpha', 'beta']
        parameter_bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, 1), (0, 1)]
    elif element_type == 'RQ':
        parameter_types = ['R', 'lntau', 'beta']
        parameter_bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, 1)]
    elif element_type == 'RC':
        parameter_types = ['R', 'lntau']
        parameter_bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif element_type == 'L':
        parameter_types = ['lnL']
        parameter_bounds = [(-np.inf, np.inf)]
    elif element_type == 'R':
        parameter_types = ['R']
        parameter_bounds = [(-np.inf, np.inf)]
    elif element_type == 'C':
        parameter_types = ['Cinv']
        parameter_bounds = [(0, np.inf)]
    elif element_type == 'P':
        parameter_types = ['P', 'nu']
        parameter_bounds = [(0, np.inf), (-1, 1)]
    else:
        raise ValueError(f'Invalid element {element_type}')

    return parameter_types, parameter_bounds


def element_distribution_function(element_type):
    if element_type == 'HN':
        def gamma(tau, r, ln_tau, alpha, beta):
            t0 = np.exp(ln_tau)
            theta = np.arctan2(np.sin(np.pi * beta), ((tau / t0) ** beta + np.cos(np.pi * beta)))
            nume = r * (tau / t0) ** (beta * alpha) * np.sin(alpha * theta)
            deno = np.pi * (1 + 2 * np.cos(np.pi * beta) * (tau / t0) ** beta + (tau / t0) ** (2 * beta)) ** (alpha / 2)
            return nume / deno
    elif element_type == 'RQ':
        def gamma(tau, r, ln_tau, beta):
            # theta = np.arctan2(np.sin(np.pi * beta), ((tau / t0) ** beta + np.cos(np.pi * beta)))
            # g = (1 / np.pi) * (tau / t0) ** beta * np.sin(theta) / (
            #         1 + 2 * np.cos(np.pi * beta) * (tau / t0) ** beta + (tau / t0) ** (2 * beta)) ** (1 / 2)
            nume = r * np.sin((1 - beta) * np.pi)
            deno = 2 * np.pi * (np.cosh(beta * (np.log(tau) - ln_tau)) - np.cos((1 - beta) * np.pi))
            return nume / deno
    elif element_type == 'RC':
        def gamma(tau, r, ln_tau):
            if np.isscalar(tau):
                if np.log(tau) == ln_tau:
                    return np.inf * np.sign(r)
                else:
                    return 0
            else:
                out = np.zeros(len(tau))
                out[np.log(tau) == ln_tau] = np.inf * np.sign(r)
                return out
    elif element_type in ('R', 'L', 'C', 'P'):
        def gamma(tau, *args):
            if np.isscalar(tau):
                return 0
            else:
                return np.zeros(len(tau))
    else:
        raise ValueError(f'Invalid element {element_type}')

    return gamma


def element_distribution_is_singular(element_type, *args, return_info=False):
    if element_type == 'HN':
        r, ln_tau, alpha, beta = args
        if beta >= 1 - 1e-5:
            status = True
            info = (r, np.exp(ln_tau))
        else:
            status = False
            info = None
    elif element_type == 'RQ':
        r, ln_tau, beta = args
        if beta >= 1 - 1e-5:
            status = True
            info = (r, np.exp(ln_tau))
        else:
            status = False
            info = None
    elif element_type == 'RC':
        r, ln_tau = args
        status = True
        info = (r, np.exp(ln_tau))
    else:
        status = False
        info = None

    if return_info:
        return status, info
    else:
        return status


def element_distribution_integral_function(element_type):
    """
    Get function for evaluating integral of element distribution function
    :param str element_type: Element type
    :return:
    """
    def default_integral(el_type, tau, *args):
        """Numerically evaluate integral"""
        # Check for missing analytical functions
        if element_distribution_is_singular(el_type, *args):
            raise ValueError(f'Element type {el_type} with arguments {args} is singular! Need to define analytical '
                             f'integral function')

        gamma_func = element_distribution_function(el_type)
        gamma = gamma_func(tau, *args)

        # Get cumulative integral
        cum_mass = cumtrapz(gamma, x=np.log(tau), initial=0)

        # Fix numerical errors for nearly singular distributions
        r = args[0]
        cum_mass[cum_mass > r] = r

        return cum_mass

    if element_type == 'HN':
        def integral(tau, r, ln_tau, alpha, beta):
            if element_distribution_is_singular('HN', r, ln_tau, alpha, beta):
                y = np.log(tau) - ln_tau
                y_array = np.atleast_1d(y)
                out = np.empty(y_array.shape)
                y_prestep = y_array[y_array < 0]

                factor = -r * (np.sin(np.pi * alpha) / (np.pi * alpha))
                out[y_array < 0] = (
                        factor * (np.exp(y_prestep) - 1) * np.exp(alpha * y_prestep)
                        * np.abs(np.exp(y_prestep) - 1) ** (-alpha)
                        * special.hyp2f1(1, 1, alpha + 1, np.exp(y_prestep))
                )
                out[y_array >= 0] = r

                if np.isscalar(y):
                    out = out[0]
            else:
                out = default_integral('HN', tau, r, ln_tau, alpha, beta)

            return out
    elif element_type == 'RQ':
        def integral(tau, r, ln_tau, beta):
            if element_distribution_is_singular('RQ', r, ln_tau, beta):
                # RQ reduces to RC element
                int_func = element_distribution_integral_function('RC')
                out = int_func(tau, r, ln_tau)
            else:
                out = default_integral('RQ', tau, r, ln_tau, beta)

            return out
    elif element_type == 'RC':
        def integral(tau, r, ln_tau):
            """Integral of RC distribution. y=ln(tau/tau_0)"""
            y = np.log(tau) - ln_tau
            y_array = np.atleast_1d(y)
            out = np.zeros(y_array.shape)

            out[y_array >= 0] = r

            if np.isscalar(y):
                out = out[0]

            return out
    else:
        def integral(tau, *args):
            return default_integral(element_type, tau, *args)

    return integral


def element_relaxation_mass_function(element_type):
    # Get integral function
    integral_func = element_distribution_integral_function(element_type)

    def mass(tau, *args):
        # if element_distribution_is_singular('HN', r, ln_tau, alpha, beta):
        #     # Distribution contains singularity. Evaluate analytically
        #     cum_mass = hn_integral(np.log(tau) - ln_tau, r, alpha)
        # else:
        #     # Distribution is continuous. Just integrate all intervals numerically
        #     cum_mass = num_integral('HN', tau, r, ln_tau, alpha, beta)
        cum_mass = integral_func(tau, *args)

        # Get incremental mass in each interval
        inc_mass = np.diff(cum_mass)

        # Pad with zero to maintain array size
        return np.concatenate(([0], inc_mass))

    return mass
    # elif element_type == 'RQ':
    #     def mass(tau, r, ln_tau, beta):
    #         if element_distribution_is_singular('RQ', r, ln_tau, beta):
    #             # Distribution contains singularity. Evaluate analytically
    #             # RQ reduces to RC when beta=1
    #             cum_mass = rc_integral(np.log(tau) - ln_tau, r)
    #         else:
    #             # Distribution is continuous. Just integrate all intervals numerically
    #             cum_mass = num_integral('RQ', tau, r, ln_tau, beta)


def element_impedance_function(element_type):
    if element_type == 'HN':
        def z_func(freq, r, ln_tau, alpha, beta):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            return r / (1 + (1j * omega * t0) ** beta) ** alpha
    elif element_type == 'RQ':
        def z_func(freq, r, ln_tau, beta):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            return r / (1 + (1j * omega * t0) ** beta)
    elif element_type == 'RC':
        def z_func(freq, r, ln_tau):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            return r / (1 + 1j * omega * t0)
    elif element_type == 'L':
        def z_func(freq, ln_induc):
            omega = freq * 2 * np.pi
            return 1j * omega * np.exp(ln_induc)
    elif element_type == 'R':
        def z_func(freq, r):
            if np.isscalar(freq):
                return r + 0j
            else:
                return r * np.ones(len(freq), dtype=complex)
    elif element_type == 'C':
        def z_func(freq, c_inv):
            omega = freq * 2 * np.pi
            return 1j * c_inv / omega
    elif element_type == 'P':
        def z_func(freq, p, nu):
            omega = freq * 2 * np.pi
            return p * (1j * omega) ** nu
    else:
        raise ValueError(f'Invalid element {element_type}')

    return z_func


def element_voltage_function(element_type, step_model='ideal'):
    if step_model != 'ideal':
        raise ValueError('Element voltage responses not implemented for non-ideal current steps')

    if element_type == 'HN':
        def v_func(times,  r, ln_tau, alpha, beta):
            raise ValueError('Voltage response not implemented for HN elements')
        # def v_func(times, r, ln_tau, alpha, beta):
        #     t0 = np.exp(ln_tau)
        #     omega = freq * 2 * np.pi
        #     return r / (1 + (1j * omega * t0) ** beta) ** alpha
    elif element_type == 'RQ':
        def v_func(times, r, ln_tau, beta):
            ml_func = create_approx_func(beta, beta + 1)
            t0 = np.exp(ln_tau)
            if np.isscalar(times):
                if times > 0:
                    t_ratio = times / t0
                    return r * t_ratio ** beta * ml_func(-t_ratio ** beta)
                else:
                    return 0
            else:
                v_out = np.zeros(len(times))
                t_ratio = times[times > 0] / t0
                v_out[times > 0] = r * t_ratio ** beta * ml_func(-t_ratio ** beta)
                return v_out
    elif element_type == 'RC':
        def v_func(times, r, ln_tau):
            t0 = np.exp(ln_tau)
            if np.isscalar(times):
                if times > 0:
                    t_ratio = times / t0
                    return r * (1 - np.exp(-t_ratio))
                else:
                    return 0
            else:
                v_out = np.zeros(len(times))
                t_ratio = times[times > 0] / t0
                v_out[times > 0] = r * (1 - np.exp(-t_ratio))
                return v_out
    elif element_type == 'L':
        def v_func(times, ln_induc):
            if np.isscalar(times):
                if times == 0:
                    return 0  # np.inf
                else:
                    return 0
            else:
                v_out = np.zeros(len(times))
                # v_out[times == 0] = np.inf
                return v_out
    elif element_type == 'R':
        def v_func(times, r):
            if np.isscalar(times):
                if times > 0:
                    return r
                else:
                    return 0
            else:
                v_out = np.zeros(len(times))
                v_out[times > 0] = r
                return v_out
    elif element_type == 'C':
        def v_func(times, c_inv):
            if np.isscalar(times):
                if times > 0:
                    return c_inv * times
                else:
                    return 0
            else:
                v_out = np.zeros(len(times))
                v_out[times > 0] = c_inv * times[times > 0]
                return v_out
    elif element_type == 'P':
        def v_func(times, p, nu):
            if np.isscalar(times):
                if times > 0:
                    return p * times ** -nu / special.gamma(-nu + 1)
                else:
                    return 0
            else:
                v_out = np.zeros(len(times))
                v_out[times > 0] = p * times[times > 0] ** -nu / special.gamma(-nu + 1)
                return v_out

    else:
        raise ValueError(f'Invalid element {element_type}')

    return v_func


def parse_element_string(element_string):
    id_match = re.search('\d', element_string)
    if id_match is not None:
        id_start = id_match.start()
        element_type = element_string[:id_start]
        element_id = int(element_string[id_start:])
        return element_type, element_id
    else:
        raise ValueError(f'No ID in element string {element_string}')


def parse_model_string(model_string):
    element_names = model_string.split('-')
    if len(element_names) > len(set(element_names)):
        raise ValueError('Model contains duplicate elements')
    else:
        parameter_types = []
        parameter_names = []
        parameter_bounds = []
        element_types = []
        parameter_indices = []
        start_index = 0
        for element_string in element_names:
            element_type, element_id = parse_element_string(element_string)
            param_types, bounds = element_parameters(element_type)
            parameter_types += param_types
            parameter_names += [f'{param_type}_{element_string}' for param_type in param_types]
            parameter_bounds += bounds
            element_types.append(element_type)
            parameter_indices.append((start_index, start_index + len(param_types)))
            start_index += len(param_types)

        return element_names, element_types, parameter_types, parameter_names, parameter_bounds, parameter_indices


def model_impedance_function(model_string):
    el_names, el_types, param_types, param_names, param_bounds, param_indices = parse_model_string(model_string)
    z_functions = [element_impedance_function(element) for element in el_types]

    def z_model(freq, *args):
        z_vectors = np.array([
            z_func(freq, *args[param_indices[i][0]:param_indices[i][1]]) for i, z_func in enumerate(z_functions)
        ])
        return np.sum(z_vectors, axis=0)

    return z_model


def model_voltage_function(model_string, step_model='ideal'):
    el_names, el_types, param_types, param_names, param_bounds, param_indices = parse_model_string(model_string)
    v_functions = [element_voltage_function(element, step_model) for element in el_types]

    def v_model(times, *args):
        v_vectors = np.array([
            v_func(times, *args[param_indices[i][0]:param_indices[i][1]]) for i, v_func in enumerate(v_functions)
        ])
        return np.sum(v_vectors, axis=0)

    return v_model


def model_distribution_function(model_string):
    el_names, el_types, param_types, param_names, param_bounds, param_indices = parse_model_string(model_string)
    gamma_functions = [element_distribution_function(element) for element in el_types]

    def gamma_model(freq, *args):
        gamma_vectors = np.array([
            gamma_func(freq, *args[param_indices[i][0]:param_indices[i][1]])
            for i, gamma_func in enumerate(gamma_functions)
        ])
        return np.sum(gamma_vectors, axis=0)

    return gamma_model


def model_mass_function(model_string):
    el_names, el_types, param_types, param_names, param_bounds, param_indices = parse_model_string(model_string)
    mass_functions = [element_relaxation_mass_function(element) for element in el_types]

    def mass_model(freq, *args):
        mass_vectors = np.array([
            mass_func(freq, *args[param_indices[i][0]:param_indices[i][1]])
            for i, mass_func in enumerate(mass_functions)
        ])
        return np.sum(mass_vectors, axis=0)

    return mass_model


def flatten_bounds(bounds):
    """Format bounds for scipy optimization function - requires arrays of lower and upper bounds"""
    return [bound[0] for bound in bounds], [bound[1] for bound in bounds]


def pair_bounds(lb, ub):
    """
    Format arrays of bounds into list of (lower, upper) tuples
    :param lb:
    :param ub:
    :return:
    """
    return [(lb[i], ub[i]) for i in range(len(lb))]


def constraint_transform(x, bounds, inverse=False):
    """
    Transform a bounded variable to an unbounded space (or the inverse) using the logit transform
    :param x: Variable(s) to transform. May be a float or an array
    :param bounds: Bounds for variables. If x is a float, bounds must be a 2-tuple of bounds (lower, upper).
    If x is an array, bounds must be a 2-tuple of arrays
    :param inverse: If True, transform the variable from the unbounded space to the real space. If False, transform
    the variable from the real (bounded) space to the unbounded space
    :return:
    """
    if bounds[0] > -np.inf and bounds[1] < np.inf:
        a, b = bounds
        if inverse:
            y = (b * np.exp(x) + a) / (1 + np.exp(x))
        else:
            y = np.log((x - a) / (b - x))
    elif bounds[0] > -np.inf:
        a = bounds[0]
        if inverse:
            y = a + np.exp(x)
        else:
            y = np.log(x - a)
    elif bounds[1] < np.inf:
        b = bounds[1]
        if inverse:
            y = b - np.exp(-x)
        else:
            y = np.log(1 / (b - x))
    else:
        # Both bounds are infinite - variable is unbounded
        y = x

    return y


def get_fast_constraint_transforms(bounds, max_bound=1e6):
    """
    Get vectorized constraint transform functions using finite bounds for all parameters
    :param list bounds: Bounds formatted as list of (lower, upper) tuples, one tuple for each parameter
    :param float max_bound: Maximum absolute bound. Any bounds larger in magnitude than max_bound will be replaced
    by max_bound
    :return:
    """
    # Replace infinite bounds with large finite bounds
    lb = np.array([max(bound[0], -max_bound) for bound in bounds])
    ub = np.array([min(bound[1], max_bound) for bound in bounds])

    def transform(x):
        return np.log((x - lb) / (ub - x))

    def inverse_transform(x):
        return (ub * np.exp(x) + lb) / (1 + np.exp(x))

    return transform, inverse_transform


def scale_parameter_to_data(x, parameter_type, rp_scale, inverse):
    if parameter_type == 'R':
        if inverse:
            return x * rp_scale
        else:
            return x / rp_scale
    elif parameter_type == 'lnL':
        if inverse:
            return x + np.log(rp_scale)
        else:
            return x - np.log(rp_scale)
    else:
        # No scaling
        return x


# ================================
# Gradient, Jacobian, and Hessian
# ================================
def element_f_jacobian(element_type):
    """
    Get function for Jacobian of model function (z) wrt parameters of a single element (theta)
    :param str element_type: element type
    :return:
    """
    # TODO: fill in jacobian for RQ and RC elements
    if element_type == 'HN':
        def jac(freq, r, ln_tau, alpha, beta):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            jwt = 1j * omega * t0
            jwt_b = jwt ** beta
            out = np.zeros((2 * len(freq), 4))
            out[:, 0] = utils.eis.complex_vector_to_concat(
                (1 + jwt_b) ** -alpha
            )
            out[:, 1] = utils.eis.complex_vector_to_concat(
                -r * alpha * (1 + jwt_b) ** -(alpha + 1) * beta * jwt_b
            )
            out[:, 2] = utils.eis.complex_vector_to_concat(
                -r * np.log(1 + jwt_b) * (1 + jwt_b) ** -alpha
            )
            out[:, 3] = utils.eis.complex_vector_to_concat(
                -r * alpha * (1 + jwt_b) ** -(alpha + 1) * np.log(jwt) * jwt_b
            )
            return out
    elif element_type == 'RQ':
        def z_func(freq, r, ln_tau, beta):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            return r / (1 + (1j * omega * t0) ** beta)
    elif element_type == 'RC':
        def z_func(freq, r, ln_tau):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            return r / (1 + 1j * omega * t0)
    elif element_type == 'L':
        def jac(freq, ln_induc):
            omega = freq * 2 * np.pi
            out = utils.eis.complex_vector_to_concat(
                1j * omega * np.exp(ln_induc)
            )
            return out.reshape((2 * len(freq), 1))
    elif element_type == 'R':
        def jac(freq, r):
            out = utils.eis.complex_vector_to_concat(
                np.ones(len(freq))
            )
            return out.reshape((2 * len(freq), 1))
    else:
        raise ValueError(f'Invalid element {element_type}')

    return jac


def element_f_hessian(element_type):
    """
    Get function for Hessian of model function (z) wrt parameters of a single element (theta)
    :param str element_type: element type
    :return:
    """
    # TODO: fill in hessian for RQ and RC elements
    if element_type == 'HN':
        def hess(freq, r, ln_tau, alpha, beta):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            jwt = 1j * omega * t0
            jwt_b = jwt ** beta
            out = np.zeros((2 * len(freq), 4, 4))
            out[:, 1, 1] = utils.eis.complex_vector_to_concat(
                r * alpha * beta ** 2 * jwt_b * (
                        (alpha + 1) * (1 + jwt_b) ** -(alpha + 2) - beta * (1 + jwt_b) ** -(alpha + 1)
                )
            )
            out[:, 2, 2] = utils.eis.complex_vector_to_concat(
                r * (np.log(1 + jwt_b)) ** 2 * (1 + jwt_b) ** -alpha
            )
            out[:, 3, 3] = utils.eis.complex_vector_to_concat(
                -r * alpha * (np.log(jwt)) ** 2 * (
                        (1 + jwt_b) ** -(alpha + 1) * jwt_b
                        - (alpha + 1) * (1 + jwt_b) ** -(alpha + 2) * jwt ** (2 * beta)
                )
            )
            return out
    elif element_type == 'RQ':
        def z_func(freq, r, ln_tau, beta):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            return r / (1 + (1j * omega * t0) ** beta)
    elif element_type == 'RC':
        def z_func(freq, r, ln_tau):
            t0 = np.exp(ln_tau)
            omega = freq * 2 * np.pi
            return r / (1 + 1j * omega * t0)
    elif element_type == 'L':
        def hess(freq, ln_induc):
            omega = freq * 2 * np.pi
            out = utils.eis.complex_vector_to_concat(
                1j * omega * np.exp(ln_induc)
            )
            return out.reshape((2 * len(freq), 1, 1))
    elif element_type == 'R':
        def hess(freq, r):
            return np.zeros((2 * len(freq), 1, 1))
    else:
        raise ValueError(f'Invalid element {element_type}')

    return hess


def evaluate_dllh_df(y, f, weights):
    """
    Evaluate derivative of log-likelihood wrt model function (z)
    :param ndarray y: measured values
    :param ndarray f: model values
    :param ndarray weights: measurement weights
    :return:
    """
    pm = np.diag(weights ** 2)
    return (-pm @ f + pm @ y).reshape(1, len(y))


def element_dllh_df_dtheta(element_type):
    """
    Get function for mixed partial derivative of log-likelihood wrt model function (z)
    and parameters of a single element (theta)
    :param str element_type: element type
    :return:
    """
    jac = element_f_jacobian(element_type)

    def dllh_df_dtheta(freq, weights, *args):
        pm = np.diag(weights ** 2)
        return -pm @ jac(freq, *args)

    return dllh_df_dtheta


def model_llh_hessian(parameter_indices, element_types, z_func):
    """
    Get function for Hessian of log-likelihood wrt model parameter vector (theta)
    :param list parameter_indices: list of tuples indicating start and end indices for parameters of each element
    :param list element_types: list of element types
    :param z_func: function to evaluate model impedance
    :return:
    """
    hess_f = model_f_hessian(parameter_indices, element_types)
    jac_f = model_f_jacobian(parameter_indices, element_types)
    dllh_df_dtheta = model_dllh_df_dtheta(parameter_indices, element_types)

    def hess(freq, y, weights, *args):
        f_hat = z_func(freq, *args)
        f_hat = utils.eis.complex_vector_to_concat(f_hat)
        y = utils.eis.complex_vector_to_concat(y)
        dl_df = evaluate_dllh_df(y, f_hat, weights)
        d2f_dtheta2 = hess_f(freq, *args)
        d2f_dtheta2 = np.swapaxes(d2f_dtheta2, 0, 1)
        dl_df_dtheta = dllh_df_dtheta(freq, weights, *args)
        df_dtheta = jac_f(freq, *args)
        # print(dl_df.shape, d2f_dtheta2.shape, dl_df_dtheta.shape, df_dtheta.shape)

        a = dl_df @ d2f_dtheta2
        a = a[:, 0, :]
        b = dl_df_dtheta.T @ df_dtheta

        # print('slogdet(a):', np.linalg.slogdet(a))
        # print('slogdet(b):', np.linalg.slogdet(b))
        # print('a eig:', np.linalg.eigvals(-a))
        # print('b eig:', np.linalg.eigvals(-b))

        return a + b

    return hess


def model_llh_grad(parameter_indices, element_types, z_func):
    """
    Get function for gradient of log-likelihood wrt model parameter vector (theta)
    :param list parameter_indices: list of tuples indicating start and end indices for parameters of each element
    :param list element_types: list of element types
    :param z_func: function to evaluate model impedance
    :return:
    """
    jac_f = model_f_jacobian(parameter_indices, element_types)

    def grad(freq, y, weights, *args):
        f_hat = z_func(freq, *args)
        f_hat = utils.eis.complex_vector_to_concat(f_hat)
        y = utils.eis.complex_vector_to_concat(y)
        dl_df = evaluate_dllh_df(y, f_hat, weights)
        df_dtheta = jac_f(freq, *args)

        return dl_df @ df_dtheta

    return grad


def model_f_jacobian(parameter_indices, element_types):
    """
    Get function for Jacobian of model function (z) wrt model parameter vector (theta)
    :param list parameter_indices: list of tuples indicating start and end indices for parameters of each element
    :param list element_types: list of element types
    :return:
    """

    def jac(freq, *args):
        out = np.zeros((2 * len(freq), parameter_indices[-1][1]))
        for i, element_type in enumerate(element_types):
            start_index = parameter_indices[i][0]
            end_index = parameter_indices[i][1]
            out[:, start_index:end_index] = element_f_jacobian(element_type)(freq, *args[start_index:end_index])

        return out

    return jac


def model_dllh_df_dtheta(parameter_indices, element_types):
    """
    Get function for mixed partial derivative of log-likelihood wrt model function (z)
    and model parameter vector (theta)
    :param list parameter_indices: list of tuples indicating start and end indices for parameters of each element
    :param list element_types: list of element types
    :return:
    """

    def dllh_df_dtheta(freq, weights, *args):
        out = np.zeros((2 * len(freq), parameter_indices[-1][1]))
        for i, element_type in enumerate(element_types):
            start_index = parameter_indices[i][0]
            end_index = parameter_indices[i][1]
            func = element_dllh_df_dtheta(element_type)
            out[:, start_index:end_index] = func(freq, weights, *args[start_index:end_index])

        return out

    return dllh_df_dtheta


def model_f_hessian(parameter_indices, element_types):
    """
    Get function for Hessian of model function (z) wrt model parameter vector (theta)
    :param list parameter_indices: list of tuples indicating start and end indices for parameters of each element
    :param list element_types: list of element types
    :return:
    """

    def hess(freq, *args):
        out = np.zeros((2 * len(freq), parameter_indices[-1][1], parameter_indices[-1][1]))
        for i, element_type in enumerate(element_types):
            start_index = parameter_indices[i][0]
            end_index = parameter_indices[i][1]
            element_hess = element_f_hessian(element_type)
            out[:, start_index:end_index, start_index:end_index] = element_hess(freq, *args[start_index:end_index])

        return out

    return hess
