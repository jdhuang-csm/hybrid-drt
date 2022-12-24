import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy import ndimage
from scipy import signal

from .. import utils
from .. import fileload as fl
from ..matrices import basis
from .drt1d import DRT


class DRTMD(object):
    def __init__(self, tau_supergrid, psi_dim_names=None, store_attr_categories=None,
                 tau_basis_type='gaussian', tau_epsilon=None,
                 step_model='ideal', chrono_mode='galv',
                 fixed_basis_nu=None, fit_dop=False, normalize_dop=True, nu_basis_type='delta',
                 fit_inductance=True, time_precision=10, input_signal_precision=10, frequency_precision=10,
                 chrono_reader=None, eis_reader=None,
                 fit_kw=None, fit_type='drt', pfrt_factors=None,
                 print_diagnostics=False, print_progress=True, warn=False):

        # Initialize workhorse DRT instance. Set up integral interpolation using tau_supergrid
        self.drt1d = DRT(interpolate_integrals=True, tau_supergrid=tau_supergrid, tau_epsilon=tau_epsilon,
                         tau_basis_type=tau_basis_type)

        self.psi_dim_names = psi_dim_names

        # Attribute categories to store from drt1d fits
        if store_attr_categories is None:
            store_attr_categories = ['config', 'fit_core']
        self.store_attr_categories = store_attr_categories

        # DRT config
        self.tau_supergrid = tau_supergrid
        self.tau_basis_type = tau_basis_type
        self.tau_epsilon = self.drt1d.tau_epsilon
        self.fit_inductance = fit_inductance

        # Distribution of phasances
        self.fixed_basis_nu = fixed_basis_nu
        self.nu_basis_type = nu_basis_type
        self.fit_dop = fit_dop
        self.normalize_dop = normalize_dop

        # Chrono settings
        self.step_model = step_model
        self.chrono_mode = chrono_mode

        # File loading
        if chrono_reader is None:
            def chrono_reader(file):
                return fl.get_chrono_tuple(fl.read_chrono(file))
        self.chrono_reader = chrono_reader

        if eis_reader is None:
            def eis_reader(file):
                return fl.get_eis_tuple(fl.read_eis(file))
        self.eis_reader = eis_reader

        # Fit configuration
        self.fit_type = fit_type
        self.fit_kw = fit_kw
        if pfrt_factors is None:
            pfrt_factors = np.logspace(-0.7, 0.7, 11)
        self.pfrt_factors = pfrt_factors

        # Initialize lists to hold observations and results
        if self.psi_dim_names is not None:
            self.obs_psi = np.zeros((0, len(self.psi_dim_names)))
        else:
            self.obs_psi = None
        self.obs_fit_attr = []
        self.obs_fit_status = np.zeros(0, dtype=bool)
        self.obs_data = []
        self.obs_tau_indices = []
        # self.obs_data_types = []

        # Fit parameters
        self.obs_x = np.zeros((0, *self.drt_param_shape()))
        self.obs_special = None

        # Precision
        self.frequency_precision = frequency_precision
        self.time_precision = time_precision
        self.input_signal_precision = input_signal_precision

        # Diagnostic printing
        self.print_diagnostics = print_diagnostics
        self.warn = warn
        self.print_progress = print_progress

    # --------------------------------------------------------------------
    # Data addition and fitting
    # --------------------------------------------------------------------
    def add_observation(self, psi, chrono_data, eis_data, fit=False):
        psi = np.atleast_1d(psi).flatten()

        # Initialize obs_psi
        if self.obs_psi is None:
            self.obs_psi = np.zeros((0, len(psi)))

        # Validate data
        self.validate_psi(psi)
        # if chrono_data is not None:
        #     utils.validation.check_chrono_data(*chrono_data)
        # if eis_data is not None:
        #     utils.validation.check_eis_data(*eis_data)

        # Append to observation lists
        self.obs_psi = np.insert(self.obs_psi, self.num_obs, psi, axis=0)
        self.obs_data.append((chrono_data, eis_data))
        self.obs_fit_status = np.insert(self.obs_fit_status, len(self.obs_fit_status), False)
        self.obs_fit_attr.append(None)
        self.obs_tau_indices.append(None)
        self.obs_x = np.insert(self.obs_x, len(self.obs_x), np.zeros(self.drt_param_shape()), axis=0)

        if self.obs_special is not None:
            for key in self.drt1d.special_qp_params.keys():
                key_shape = self.special_param_shape(key)
                self.obs_special[key] = np.insert(self.obs_special[key], self.num_obs - 1, np.zeros(key_shape), axis=0)

        if fit:
            obs_index = self.num_obs - 1
            self.fit_observation(obs_index)

    def fit_observation(self, obs_index):
        # Get data
        chrono_data, eis_data = self.get_obs_data(obs_index)

        # Fit data
        getattr(self.drt1d, self._fit_func_name)(*chrono_data, *eis_data, **self.fit_kw)

        # Store result
        self.obs_fit_attr[obs_index] = self.drt1d.get_attributes(which=self.store_attr_categories)

        # Determine tau indices used for fit
        left_index = utils.array.nearest_index(self.tau_supergrid, self.drt1d.basis_tau[0])
        right_index = utils.array.nearest_index(self.tau_supergrid, self.drt1d.basis_tau[-1]) + 1
        # print(self.drt1d.basis_tau[0], self.drt1d.basis_tau[-1])
        # print(self.tau_supergrid[left_index], self.tau_supergrid[right_index - 1])
        self.obs_tau_indices[obs_index] = (left_index, right_index)

        # drt1d.special_qp_params will be set once first fit is complete. Initialize obs_special after fitting
        if self.obs_special is None:
            self.initialize_obs_special()

        # Format parameters and insert into arrays
        x_drt, x_special = self.format_1d_params(self.drt1d, left_index, right_index)
        self.obs_x[obs_index] = x_drt

        # print(obs_index)
        # print(self.obs_special, x_special)
        for key in self.drt1d.special_qp_params.keys():
            self.obs_special[key][obs_index] = x_special[key]

        # Update fit status
        self.obs_fit_status[obs_index] = True

    def fit_all(self, refit=False):
        if refit:
            # Fit all observations regardless of fit status
            fit_index = np.arange(self.num_obs)
        else:
            # Only fit unfitted observations
            fit_index = np.where(~np.array(self.obs_fit_status))[0]

        num_to_fit = len(fit_index)
        print_interval = int(np.ceil(num_to_fit / 10))
        if self.print_progress:
            print(f'Found {num_to_fit} observations to fit')

        for i, index in enumerate(fit_index):
            self.fit_observation(index)
            if self.print_progress and ((i + 1) % print_interval == 0 or i == num_to_fit - 1):
                print(f'{i + 1} / {num_to_fit}')

    # def update_fit_config(self):
    #     for name in ['tau_epsilon', '']

    def get_obs_data(self, obs_index):
        chrono_data, eis_data = self.obs_data[obs_index]

        # Load chrono data
        if type(chrono_data) in (str, Path):
            chrono_data = self.chrono_reader(chrono_data)
        elif chrono_data is None:
            chrono_data = (None, None, None)

        # Load EIS data
        if type(eis_data) in (str, Path):
            eis_data = self.eis_reader(eis_data)
        elif eis_data is None:
            eis_data = (None, None)

        return chrono_data, eis_data

    def get_fit(self, obs_index):
        if not self.obs_fit_status[obs_index]:
            raise ValueError(f'Observation {obs_index} has not been fitted')

        drt = DRT(interpolate_integrals=False)
        drt.set_attributes(self.obs_fit_attr[obs_index])

        # Copy integral lookups from working instance
        drt.interpolate_lookups = self.drt1d.interpolate_lookups
        drt.integrate_method = 'interp'

        return drt

    def clear_fits(self):
        """
        Clear all stored fit information and reset fit status for all observations.
        Does not clear stored observations or data
        :return:
        """
        self.obs_fit_attr = [None] * self.num_obs
        self.obs_fit_status = np.zeros(self.num_obs, dtype=bool)
        self.obs_tau_indices = [None] * self.num_obs
        self.obs_x = np.zeros((self.num_obs, len(self.tau_supergrid)))
        self.obs_special = None

    # --------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------
    def predict_r_p(self, psi=None, x=None, factor_index=None, **kw):
        if x is None:
            x = self.predict_x(psi, factor_index=factor_index, **kw)

        return np.sum(x, axis=-1) * self.tau_basis_area

    def predict_x(self, psi, factor_index=None, percentile=None, normalize=False, ndfilter=False, filter_func=None,
                  filter_kw=None):
        self.validate_psi(psi)

        psi_index = self.get_psi_index(psi)
        if np.min(psi_index) > -1:
            x = self.obs_x[psi_index]
        else:
            x = None

        if normalize:
            rp = self.predict_r_p(x=x)
            x = x / rp[:, None]

        # Select specified factor level for PFRT fit
        if self.fit_type == 'pfrt' and factor_index is not None:
            x = x[:, factor_index, :]

        # Get percentile if requested
        if percentile is not None:
            x_cov = self.predict_x_cov(psi_index, factor_index)
            x_sigma = np.array([np.diag(cov) ** 0.5 for cov in x_cov])
            num_std = utils.stats.std_normal_quantile(percentile / 100)
            print(num_std)
            x = x + num_std * x_sigma

        # Apply filter
        if ndfilter:
            x = apply_filter(x, filter_func, filter_kw)

        return x

    def predict_drt(self, psi, tau=None, x=None, order=0, factor_index=None, normalize=False, **kw):
        if x is None:
            x = self.predict_x(psi, factor_index=factor_index, normalize=False, **kw)

        if normalize:
            rp = self.predict_r_p(psi=psi, x=x, factor_index=factor_index, normalize=False, **kw)
            x = x / rp[:, None]

        if tau is None:
            tau = self.get_tau_eval(20)

        basis_mat = basis.construct_func_eval_matrix(np.log(self.tau_supergrid), np.log(tau), self.tau_basis_type,
                                                     self.tau_epsilon, order=order)

        return x @ basis_mat.T

    def predict_param_cov(self, obs_index, factor_index=None):
        cov_matrices = []
        obs_index = np.atleast_1d(obs_index)
        for index in obs_index:
            drt = self.get_fit(index)
            if self.fit_type == 'pfrt':
                if factor_index is not None:
                    p_mat = drt.pfrt_result['step_p_mat'][factor_index]
                    cov = drt.estimate_param_cov(p_matrix=p_mat)
                else:
                    # Calculate for all factors
                    cov = np.array([drt.estimate_param_cov(p_matrix=p_mat) for p_mat in drt.pfrt_result['step_p_mat']])
            else:
                cov = drt.estimate_param_cov()

            cov_matrices.append(cov)

        # if len(obs_index) == 1:
        #     return cov_matrices[0]
        # else:
        return cov_matrices

    def predict_x_cov(self, obs_index, factor_index=None):
        obs_index = np.atleast_1d(obs_index)
        cov = self.predict_param_cov(obs_index, factor_index)
        x_cov = np.zeros((len(cov), *self.drt_param_shape(factor_index), len(self.tau_supergrid)))
        # print(x_cov.shape)
        for i, index in enumerate(obs_index):
            left_index, right_index = self.obs_tau_indices[index]
            if self.fit_type == 'pfrt' and factor_index is None:
                x_cov[i, :, left_index:right_index, left_index:right_index] = \
                    cov[i][:, self.drt1d.get_qp_mat_offset():, self.drt1d.get_qp_mat_offset():]
            else:
                x_cov[i, left_index:right_index, left_index:right_index] = \
                    cov[i][self.drt1d.get_qp_mat_offset():, self.drt1d.get_qp_mat_offset():]

        return x_cov

    def predict_drt_cov(self, obs_index, tau=None, x_cov=None, order=0, factor_index=None, extend_var=False):
        obs_index = np.atleast_1d(obs_index)
        if x_cov is None:
            x_cov = self.predict_x_cov(obs_index, factor_index=factor_index)

        if tau is None:
            tau = self.get_tau_eval(20)

        basis_mat = basis.construct_func_eval_matrix(np.log(self.tau_supergrid), np.log(tau), self.tau_basis_type,
                                                     self.tau_epsilon, order=order)

        drt_cov = basis_mat @ x_cov @ basis_mat.T

        # Extend variance beyond measurement bounds
        if extend_var:
            for i in range(len(obs_index)):
                # Get tau data limits (basis goes one decade beyond data)
                tau_indices = self.obs_tau_indices[obs_index[i]]
                t_left = self.tau_supergrid[tau_indices[0]] * 10
                t_right = self.tau_supergrid[tau_indices[1]] / 10

                left_index = utils.array.nearest_index(tau, t_left) + 1
                right_index = utils.array.nearest_index(tau, t_right)
                var = np.diag(drt_cov[i]).copy()
                # Set the variance outside the measurement bounds to the variance at the corresponding bound
                var[:left_index] = np.maximum(var[:left_index], var[left_index])
                var[right_index:] = np.maximum(var[right_index:], var[right_index])
                drt_cov[i, np.diag_indices(drt_cov[i].shape[0])] = var

        return drt_cov

    def predict_drt_var(self, obs_index, tau=None, x_cov=None, order=0, factor_index=None, extend_var=False,
                        ndfilter=False, filter_func=None, filter_kw=None):
        drt_cov = self.predict_drt_cov(obs_index, tau, x_cov, order, factor_index, extend_var)
        drt_var = np.array([np.diag(cov) for cov in drt_cov])

        # Apply filter
        if ndfilter:
            drt_var = apply_filter(drt_var, filter_func, filter_kw)

        return drt_var

    def predict_peak_prob(self, psi, x=None, f_var=None, fxx_var=None, tau=None, factor_index=None, extend_var=False,
                          prominence=5e-3, height=1e-3, peak_spread_sigma=None,
                          ndfilter=False, filter_func=None, filter_kw=None, sign=1):

        if tau is None:
            tau = self.get_tau_eval(10)

        if x is None:
            x = self.predict_x(psi, factor_index=factor_index, normalize=True, ndfilter=ndfilter,
                               filter_func=filter_func, filter_kw=filter_kw)

        f = self.predict_drt(psi, tau=tau, x=x, order=0, factor_index=factor_index)
        fxx = self.predict_drt(psi, tau=tau, x=x, order=2, factor_index=factor_index)

        # Get DRT std
        if f_var is None:
            f_var = self.predict_drt_var(self.get_psi_index(psi), tau=tau, order=0, factor_index=factor_index,
                                         extend_var=extend_var, ndfilter=ndfilter, filter_func=filter_func,
                                         filter_kw=filter_kw)
        f_sigma = f_var ** 0.5

        # Get curvature std
        if fxx_var is None:
            fxx_var = self.predict_drt_var(self.get_psi_index(psi), tau=tau, order=2, factor_index=factor_index,
                                           extend_var=extend_var, ndfilter=ndfilter, filter_func=filter_func,
                                           filter_kw=filter_kw)
        fxx_sigma = fxx_var ** 0.5

        # Evaluate peak confidence
        peak_prob = utils.array.apply_along_axis_multi(peak_prob_1d, -1, [f, fxx, f_sigma, fxx_sigma],
                                                       self.fit_kw['nonneg'], sign, height, prominence)

        if peak_spread_sigma is not None:
            # Assume the peak probability has some spread in the tau dimension
            sigma = np.zeros(np.ndim(peak_prob))
            sigma[-1] = peak_spread_sigma
            factor = 1 / utils.stats.pdf_normal(0, 0, peak_spread_sigma)
            peak_prob = ndimage.gaussian_filter(peak_prob, sigma=sigma) * factor

        return peak_prob

    # def get_drt_params(self, obs_index=None, return_psi=False):
    #     if obs_index is None:
    #         obs_index = self.fitted_obs_index
    #
    #     x = np.array(self.obs_x_list[i] for i in obs_index)
    #
    #     if return_psi:
    #         psi = np.array(self.obs_psi)[obs_index]
    #         return x, psi
    #     else:
    #         return x
    #
    # def get_special_params(self, key, obs_index=None, return_psi=False):
    #     if obs_index is None:
    #         obs_index = self.fitted_obs_index
    #
    #     spec_list = self.obs_special[key]
    #     x = np.array(spec_list[i] for i in obs_index)
    #
    #     if return_psi:
    #         psi = np.array(self.obs_psi)[obs_index]
    #         return x, psi
    #     else:
    #         return x

    # --------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------
    def format_1d_params(self, drt1d, left_index, right_index):
        if self.fit_type == 'drt':
            x_drt = np.zeros(self.drt_param_shape())
            x_drt[left_index:right_index] = drt1d.fit_parameters['x'].copy()

            x_special = {}
            for key, info in drt1d.special_qp_params.items():
                x_special[key] = deepcopy(drt1d.fit_parameters[key])
        else:
            # PFRT fit has one solution for each factor
            # Extract parameters from raw x vectors
            fit_params = [drt1d.extract_qphb_parameters(x_raw) for x_raw in drt1d.pfrt_result['step_x']]

            # Make x_drt array
            x_drt = np.zeros(self.drt_param_shape())
            x_drt[:, left_index:right_index] = np.array([fp['x'] for fp in fit_params])

            # Make special parameter arrays
            x_special = {}
            for key, info in drt1d.special_qp_params.items():
                x_special[key] = np.array([fp[key] for fp in fit_params])

        return x_drt, x_special

    def initialize_obs_special(self):
        self.obs_special = {}
        for key in self.drt1d.special_qp_params.keys():
            self.obs_special[key] = np.zeros([self.num_obs, *self.special_param_shape(key)])
            # if info.get('size', 1) == 1:
            #     if self.fit_type == 'pfrt':
            #         self.obs_special[key] = np.zeros((self.num_obs, len(self.pfrt_factors)))
            #     else:
            #         self.obs_special[key] = np.zeros(self.num_obs)
            # else:
            #     if self.fit_type == 'pfrt':
            #         self.obs_special[key] = np.zeros((self.num_obs, len(self.pfrt_factors), info['size']))
            #     else:
            #         self.obs_special[key] = np.zeros((self.num_obs, info['size']))

    def validate_psi(self, psi):
        if self.psi_dim_names is not None:
            psi_len = len(self.psi_dim_names)
        elif self.obs_psi is not None:
            psi_len = self.obs_psi.shape[1]
        else:
            psi_len = None

        psi = np.atleast_2d(psi)

        if psi_len is not None:
            new_psi_len = np.shape(psi)[1]
            if not psi_len == new_psi_len:
                raise ValueError(f'Dimensions of provided psi ({new_psi_len}) '
                                 f'do not match shape of existing psi array ({self.obs_psi.shape})')

        return psi

    def get_psi_index(self, psi):
        psi = self.validate_psi(psi)
        return utils.array.row_match_index(self.obs_psi, psi, precision=8)

    def get_tau_eval(self, ppd, extend_decades=1):
        """
        Get tau grid for DRT evaluation and plotting
        :param ppd: points per decade
        :param extend_decades: number of decades to extend evaluation grid beyond basis limits
        :return:
        """
        log_tau_min = np.min(np.log10(self.tau_supergrid)) - extend_decades
        log_tau_max = np.max(np.log10(self.tau_supergrid)) + extend_decades
        tau = np.logspace(log_tau_min, log_tau_max, int((log_tau_max - log_tau_min) * ppd) + 1)

        return tau

    @property
    def tau_basis_area(self):
        return basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, None)

    @property
    def num_obs(self):
        return len(self.obs_psi)

    @property
    def fitted_obs_index(self):
        return np.where(np.array(self.obs_fit_status))[0]

    def drt_param_shape(self, factor_index=None):
        if self.fit_type == 'pfrt':
            if factor_index is None:
                return [len(self.pfrt_factors), len(self.tau_supergrid)]
            else:
                num_factors = len(np.atleast_1d(factor_index))
                if num_factors > 1:
                    return [num_factors, len(self.tau_supergrid)]
                else:
                    return [len(self.tau_supergrid)]
        else:
            return [len(self.tau_supergrid)]

    def special_param_shape(self, key):
        size = self.drt1d.special_qp_params[key].get('size', 1)
        if size == 1:
            if self.fit_type == 'pfrt':
                return [len(self.pfrt_factors)]
            else:
                return []
        else:
            if self.fit_type == 'pfrt':
                return [len(self.pfrt_factors), size]
            else:
                return [size]

    @property
    def _fit_func_name(self):
        if self.fit_type == 'drt':
            return '_qphb_fit_core'
        elif self.fit_type == 'pfrt':
            return '_pfrt_fit_core'

    # ----------------------
    # Getters and setters
    # ----------------------
    def _set_fit_type(self, fit_type):
        fit_types = ['drt', 'pfrt']
        if fit_type not in fit_types:
            raise ValueError(f"Invalid fit_type {fit_type}. Options: {fit_types}")
        self._fit_type = fit_type

    def _get_fit_type(self):
        return self._fit_type

    fit_type = property(fset=_set_fit_type, fget=_get_fit_type)

    def _set_fit_kw(self, fit_kw):
        if fit_kw is None:
            fit_kw = {}

        if self.fit_type == 'pfrt':
            # Ensure that factors are specified
            fit_kw = dict(factors=self.pfrt_factors, **fit_kw)

            # Write to pfrt_factors in case a different value was specified
            self.pfrt_factors = fit_kw['factors']

        self._fit_kw = fit_kw

    def _get_fit_kw(self):
        return self._fit_kw

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        # Propagate configuration items to workhorse DRT instance
        if name in [
            'tau_supergrid', 'tau_basis_type', 'tau_epsilon',
            'fixed_basis_nu', 'fit_dop', 'normalize_dop', 'nu_basis_type',
            'fit_inductance',
            'step_model', 'chrono_mode',
            'frequency_precision', 'time_precision', 'input_signal_precision',
            'print_diagnostics', 'warn'
        ]:
            setattr(self.drt1d, name, value)


def apply_filter(x_in, filter_func, filter_kw):
    if filter_kw is None:
        if filter_func is None:
            sigma = np.ones(np.ndim(x_in))
            sigma[-1] = 0
            filter_kw = {'sigma': sigma}
        else:
            filter_kw = None
    if filter_func is None:
        filter_func = ndimage.gaussian_filter

    return filter_func(x_in, **filter_kw)


def peak_prob_1d(arrays_1d, nonneg, sign, height, prominence):
    # Unpack arrays
    f, fxx, f_sigma, fxx_sigma = arrays_1d
    if nonneg and sign != 0:
        # This captures both the standard nonneg case and the series_neg case
        # peak_indices = peaks.find_peaks_simple(fxx, order=2, height=0, prominence=prominence, **kw)
        peak_indices, peak_info = signal.find_peaks(-sign * fxx, height=height, prominence=prominence)
        # peak_indices = peaks.find_peaks_compound(fx, fxx, **kw)
    else:
        # Find positive and negative peaks separately
        peak_index_list = []
        peak_info_list = []
        for peak_sign in [-1, 1]:
            peak_index, peak_info = signal.find_peaks(-peak_sign * fxx, height=height, prominence=prominence)
            # Limit to peaks that are positive in the direction of the current sign
            pos_index = peak_sign * f[peak_index] > 0
            # print(pos_index)
            peak_index = peak_index[pos_index]
            peak_info = {k: v[pos_index] for k, v in peak_info.items()}

            peak_index_list.append(peak_index)
            peak_info_list.append(peak_info)
        # Concatenate pos and neg peaks
        peak_indices = np.concatenate(peak_index_list)
        peak_info = {k: np.concatenate([pi[k] for pi in peak_info_list]) for k in peak_info.keys()}

        # Sort ascending
        sort_index = np.argsort(peak_indices)
        peak_indices = peak_indices[sort_index]
        peak_info = {k: v[sort_index] for k, v in peak_info.items()}

    # if prob_thresh is None:
    #     prob_thresh = 0.25

    # Use prominence or height, whichever is smaller
    min_prom = np.minimum(peak_info['prominences'], peak_info['peak_heights'])

    # Evaluate peak confidence
    curv_prob = 1 - utils.stats.cdf_normal(0, min_prom, fxx_sigma[peak_indices])
    f_prob = 1 - utils.stats.cdf_normal(0, np.sign(f[peak_indices]) * f[peak_indices], f_sigma[peak_indices])

    probs = np.minimum(curv_prob, f_prob)

    out = np.zeros(len(f))
    out[peak_indices] = probs

    return out
