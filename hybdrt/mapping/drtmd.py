import pickle
import numpy as np
from pathlib import Path, WindowsPath
from copy import deepcopy
from scipy import ndimage
import pandas as pd
import time

from .ndx import resample, assemble_ndx
from .curvature import peak_prob_1d
from .resolve import resolve_observations
from .nddata import impute_nans, flag_bad_obs, flag_outliers
from ..filters import apply_filter
from .. import utils
from .. import fileload as fl
from ..matrices import basis, mat1d, phasance
from ..models.drt1d import DRT


class DRTMD(object):
    def __init__(self, tau_supergrid, psi_dim_names=None, store_attr_categories=None,
                 tau_basis_type='gaussian', tau_epsilon=None,
                 step_model='ideal', chrono_mode='galv',
                 fit_inductance=True, fit_ohmic=True, fit_capacitance=False,
                 fixed_basis_nu=None, fit_dop=False, normalize_dop=True,
                 nu_basis_type='gaussian', nu_epsilon=None,
                 time_precision=10, input_signal_precision=10, frequency_precision=10,
                 chrono_reader=None, eis_reader=None,
                 fit_kw=None, fit_type='drt', pfrt_factors=None,
                 print_diagnostics=False, print_progress=True, warn=False):

        # Initialize workhorse DRT instance. Set up integral interpolation using tau_supergrid
        self.drt1d = DRT(interpolate_integrals=True,
                         tau_supergrid=tau_supergrid, tau_epsilon=tau_epsilon, tau_basis_type=tau_basis_type,
                         fixed_basis_nu=fixed_basis_nu, nu_epsilon=nu_epsilon, nu_basis_type=nu_basis_type
                         )

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
        self.fit_ohmic = fit_ohmic
        self.fit_capacitance = fit_capacitance

        # Distribution of phasances
        # TODO: create setter for fixed_basis_nu to update nu_epsilon?
        self.fixed_basis_nu = fixed_basis_nu
        self.nu_basis_type = nu_basis_type
        self.nu_epsilon = nu_epsilon
        self.fit_dop = fit_dop
        self.normalize_dop = normalize_dop

        # Set nu epsilon
        if self.nu_epsilon is None and self.nu_basis_type != 'delta' and self.fit_dop:
            dnu = np.median(self.drt1d.get_nu_basis_spacing())
            self.nu_epsilon = 1 / dnu

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
        if fit_kw is None:
            fit_kw = {}
        self.fit_kw = fit_kw
        if pfrt_factors is None:
            pfrt_factors = np.logspace(-0.7, 0.7, 11)
        self.pfrt_factors = pfrt_factors

        # Initialize lists to hold observations and results
        if self.psi_dim_names is not None:
            self.obs_psi = np.zeros((0, len(self.psi_dim_names)))
        else:
            self.obs_psi = None
        self.obs_data = []
        self.obs_group_id = []
        self.obs_data_badness = np.zeros(0)
        self.obs_ignore_flag = np.zeros(0, dtype=bool)
        # self.obs_data_types = []

        # Fit parameters
        self.obs_x = np.zeros((0, *self.drt_param_shape()))
        self.obs_drt_var = np.zeros((0, *self.drt_param_shape()))
        self.obs_special = None
        self.obs_fit_attr = []
        self.obs_fit_status = np.zeros(0, dtype=bool)
        self.obs_fit_errors = []
        self.obs_fit_badness = np.zeros(0)
        self.obs_tau_indices = []

        # Resolved parameters
        self.obs_resolve_status = np.zeros(0, dtype=bool)
        self.obs_x_resolved = np.zeros((0, *self.drt_param_shape()))
        self.obs_special_resolved = None

        # Precision
        self.frequency_precision = frequency_precision
        self.time_precision = time_precision
        self.input_signal_precision = input_signal_precision

        # Diagnostic printing
        self.print_diagnostics = print_diagnostics
        self.warn = warn
        self.print_progress = print_progress

    @classmethod
    def from_source(cls, source):
        if type(source) != dict:
            with open(source, 'rb') as f:
                att_dict = pickle.load(f)
        else:
            att_dict = source

        # Initialize with required/essential arguments
        config_keys = ['tau_supergrid', 'psi_dim_names', 'store_attr_categories',  
                       'tau_basis_type', 'tau_epsilon',
                       'fixed_basis_nu', 'nu_epsilon', 'nu_basis_type',
                       'fit_dop']
        init_keys = set.intersection(set(config_keys), set(att_dict.keys()))
        # Pass init args to __init__ and remove from att_dict to avoid 
        # overriding __init__ logic
        init_kw = {k: att_dict.pop(k) for k in init_keys}
        drtmd = cls(**init_kw)

        # Load all attributes
        # att_dict = {k: v for k, v in init_kw.items() if k not in none_keys}
        drtmd.set_attributes(att_dict)

        return drtmd

    # --------------------------------------------------------------------
    # Data addition and fitting
    # --------------------------------------------------------------------
    def add_observation(self, psi, chrono_data, eis_data, group_id=None, fit=False):
        """
        Add an observation with or without fitting it
        :param ndarray psi: psi array defining the observation conditions/coordinates
        :param tuple or path chrono_data: Either a path to a data file or a formatted tuple of chronopotentiometry data
        :param tuple or path eis_data: Either a path to a data file or a formatted tuple of EIS data
        :param str group_id: Optional group identifier for the observation
        :param bool fit: if True, fit the observation immediately. If False, just add the observation data; the
        observation can be fitted later with fit_observation or fit_all
        :return:
        """
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
        self.obs_data_badness = np.append(self.obs_data_badness, 0)
        self.obs_group_id.append(group_id)
        self.obs_ignore_flag = np.insert(self.obs_ignore_flag, len(self.obs_ignore_flag), False)
        self.obs_fit_status = np.insert(self.obs_fit_status, len(self.obs_fit_status), False)
        self.obs_fit_errors.append(None)
        self.obs_fit_badness = np.append(self.obs_fit_badness, 0)
        self.obs_fit_attr.append(None)
        self.obs_tau_indices.append(None)
        self.obs_x = np.insert(self.obs_x, len(self.obs_x), np.zeros(self.drt_param_shape()), axis=0)
        self.obs_drt_var = np.insert(self.obs_drt_var, len(self.obs_drt_var), np.zeros(self.drt_param_shape()), axis=0)
        self.obs_x_resolved = np.insert(self.obs_x_resolved, len(self.obs_x_resolved),
                                        np.zeros(self.drt_param_shape()), axis=0)
        self.obs_resolve_status = np.insert(self.obs_resolve_status, len(self.obs_resolve_status), False)

        if self.obs_special is not None:
            # Expand existing special arrays
            for key in list(self.obs_special.keys()):
                key_shape = self.special_param_shape(key)
                self.obs_special[key] = np.insert(self.obs_special[key], self.num_obs - 1, np.zeros(key_shape), axis=0)
                self.obs_special_resolved[key] = np.insert(self.obs_special_resolved[key], self.num_obs - 1,
                                                           np.zeros(key_shape), axis=0)

        if fit:
            obs_index = self.num_obs - 1
            self.fit_observation(obs_index)

    def fit_observation(self, obs_index, ignore_errors=False,
                        use_arg_data=False, chrono_data=None, eis_data=None):
        # Get data
        if not use_arg_data:
            chrono_data, eis_data = self.get_obs_data(obs_index)

        # Fit data
        try:
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

            # Get DRT variance
            drt_cov = self.drt1d.estimate_distribution_cov(tau=self.tau_supergrid, extend_var=True)
            self.obs_drt_var[obs_index] = np.diag(drt_cov)

            # print(obs_index)
            # print(self.obs_special, x_special)
            for key in self.drt1d.special_qp_params.keys():
                # Initialize array if key is new
                if key not in list(self.obs_special.keys()):
                    self.obs_special[key] = np.zeros((self.num_obs, *self.special_param_shape(key)))
                self.obs_special[key][obs_index] = x_special[key]

            # Update fit status
            self.obs_fit_status[obs_index] = True
            
        except Exception as err:
            if ignore_errors:
                print(f'Error encountered at obs_index {obs_index} (printed below). This observation will be ignored.'
                      f'\n{err}')
                self.obs_fit_status[obs_index] = False
                self.obs_ignore_flag[obs_index] = True
                self.obs_fit_errors[obs_index] = err
            else:
                raise err

    def fit_observations(self, obs_index, print_interval=None, ignore_errors=False):
        num_to_fit = len(obs_index)
        if print_interval is None:
            print_interval = int(np.ceil(num_to_fit / 10))
        if self.print_progress:
            print(f'Found {num_to_fit} observations to fit')

        start_time = time.time()
        for i, index in enumerate(obs_index):
            self.fit_observation(index, ignore_errors=ignore_errors)
            if self.print_progress and ((i + 1) % print_interval == 0 or i == num_to_fit - 1):
                print(f'{i + 1} / {num_to_fit}')

        elapsed = time.time() - start_time
        print('Fitted {} observations in {:.1f} minutes'.format(num_to_fit, elapsed / 60))
        print('{:.1f} seconds per observation'.format(elapsed / num_to_fit))

    def fit_all(self, refit=False, print_interval=None, ignore_errors=False):
        if refit:
            # Fit all observations regardless of fit status
            fit_index = np.arange(self.num_obs)
        else:
            # Only fit unfitted observations
            fit_index = np.where(~np.array(self.obs_fit_status) & ~np.array(self.obs_ignore_flag))[0]

        self.fit_observations(fit_index, print_interval, ignore_errors)

    def get_obs_data(self, obs_index):
        chrono_data, eis_data = self.obs_data[obs_index]

        # Load chrono data
        if isinstance(chrono_data, str) or isinstance(chrono_data, Path):
            chrono_data = self.chrono_reader(chrono_data)
        elif type(chrono_data) == pd.DataFrame:
            chrono_data = fl.get_chrono_tuple(chrono_data)
        elif chrono_data is None:
            chrono_data = (None, None, None)
        else:
            raise ValueError('Expected chrono data to be a path, DataFrame, or None. '
                             f'Received data of type {type(chrono_data)}')

        # Load EIS data
        if isinstance(eis_data, str) or isinstance(eis_data, Path):
            eis_data = self.eis_reader(eis_data)
        elif type(eis_data) == pd.DataFrame:
            eis_data = fl.get_eis_tuple(eis_data)
        elif eis_data is None:
            eis_data = (None, None)
        else:
            raise ValueError('Expected EIS data to be a path, DataFrame, or None. '
                             f'Received data of type {type(eis_data)}')

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
        self.obs_fit_errors = [None] * self.num_obs
        self.obs_fit_badness = np.zeros(self.num_obs)
        self.obs_tau_indices = [None] * self.num_obs
        self.obs_x = np.zeros((self.num_obs, *self.drt_param_shape()))
        self.obs_drt_var = np.zeros((self.num_obs, *self.drt_param_shape()))
        self.obs_special = None

        self.obs_resolve_status = np.zeros(self.num_obs, dtype=bool)
        self.obs_x_resolved = np.zeros((self.num_obs, *self.drt_param_shape()))
        self.obs_special_resolved = None

    def clear_obs(self):
        """
        Remove all observations and fit information
        """
        if self.psi_dim_names is not None:
            self.obs_psi = np.zeros((0, len(self.psi_dim_names)))
        else:
            self.obs_psi = None
        self.obs_data = []
        self.obs_data_badness = np.zeros(0)
        self.obs_group_id = []
        self.obs_ignore_flag = np.zeros(0, dtype=bool)
        self.obs_fit_attr = []
        self.obs_fit_status = np.zeros(0, dtype=bool)
        self.obs_fit_badness = np.zeros(0)
        self.obs_fit_errors = []
        self.obs_tau_indices = []
        self.obs_x = np.zeros((0, *self.drt_param_shape()))
        self.obs_drt_var = np.zeros((0, *self.drt_param_shape()))
        self.obs_special = None

        self.obs_fit_status = np.zeros(0, dtype=bool)
        self.obs_x_resolved = np.zeros((0, *self.drt_param_shape()))
        self.obs_special_resolved = None

    # ------------------------
    # Resolution
    # ------------------------
    def resolve_observations(self, obs_index, psi_sort_dims=None,
                             psi_distance_dims=None, truncate=False, sigma=1, lambda_psi=1,
                             tau_filter_sigma=0, special_filter_sigma=0):

        # Exclude unfitted observations
        include_index = self.obs_fit_status[obs_index] & ~self.obs_ignore_flag[obs_index]
        obs_index = obs_index[include_index]

        # Sort by psi
        if psi_sort_dims is not None:
            sort_vals = [self.obs_psi[obs_index, self.psi_dim_names.index(d)] for d in psi_sort_dims][::-1]
            obs_index = obs_index[np.lexsort(sort_vals)]

        # Get psi dims to determine obs-obs distance
        if psi_distance_dims is not None:
            obs_psi = self.obs_psi[obs_index, [self.psi_dim_names.index(d) for d in psi_distance_dims]]
        else:
            obs_psi = None

        # print(obs_psi, np.diff(obs_psi))

        # Perform fast coherent multi-observation fit using individual fit arrays
        obs_drt_list = [self.get_fit(i) for i in obs_index]
        obs_tau_indices = [self.obs_tau_indices[i] for i in obs_index]

        x_drt, x_special, tau_indices = resolve_observations(
            obs_drt_list, obs_tau_indices, self.fit_kw['nonneg'],
            obs_psi=obs_psi, truncate=truncate, sigma=sigma,
            lambda_psi=lambda_psi, unpack=True,
            tau_filter_sigma=tau_filter_sigma, special_filter_sigma=special_filter_sigma
        )

        # Insert resolved parameters
        self.obs_x_resolved[obs_index, tau_indices[0]:tau_indices[1]] = x_drt
        for key in x_special.keys():
            # Initialize array if key is new
            if key not in list(self.obs_special_resolved.keys()):
                self.obs_special_resolved[key] = np.zeros((self.num_obs, *self.special_param_shape(key)))
            self.obs_special_resolved[key][obs_index] = x_special[key]

        # Update status
        self.obs_resolve_status[obs_index] = 1

    def resolve_group(self, group_id, batch_size=7, overlap=2, psi_sort_dims=None,
                      psi_distance_dims=None, truncate=False, sigma=1, lambda_psi=1,
                      tau_filter_sigma=0, special_filter_sigma=0):

        obs_index = self.get_group_index(group_id)

        # Exclude unfitted observations
        include_index = self.obs_fit_status[obs_index] & ~self.obs_ignore_flag[obs_index]
        obs_index = obs_index[include_index]

        # Sort before selecting batches
        if psi_sort_dims is not None:
            sort_vals = [self.obs_psi[obs_index, self.psi_dim_names.index(d)] for d in psi_sort_dims][::-1]
        elif psi_distance_dims is not None:
            sort_vals = [self.obs_psi[obs_index, self.psi_dim_names.index(d)] for d in psi_distance_dims][::-1]
        else:
            sort_vals = None

        if sort_vals is not None:
            obs_index = obs_index[np.lexsort(sort_vals)]

        # Clear existing params
        self.obs_x_resolved[obs_index] = 0

        num_obs = len(obs_index)
        num_batches = 1 + int(np.ceil((num_obs - batch_size) / (batch_size - overlap)))
        # print(num_obs, num_batches)

        x_batch = np.zeros((num_batches, *self.obs_x_resolved[obs_index].shape))
        x_special = {k: np.zeros((num_batches, *v[obs_index].shape)) for k, v in self.obs_special_resolved.items()}
        batch_margins = np.empty((num_batches, num_obs))
        batch_margins.fill(-1)
        for i, start in enumerate(range(0, num_obs, batch_size - overlap)):
            # Ensure a full batch for the last batch
            if num_obs - start < batch_size:
                start = max(0, num_obs - batch_size)
            end = start + batch_size
            # print(i, start, end)

            batch_index = obs_index[start:end]
            self.resolve_observations(batch_index, psi_sort_dims, psi_distance_dims, truncate, sigma, lambda_psi,
                                      tau_filter_sigma, special_filter_sigma)

            x_batch[i, start:end] = self.obs_x_resolved[batch_index].copy()
            for key in self.obs_special_resolved.keys():
                x_special[key][i, start:end] = self.obs_special_resolved[key][batch_index].copy()
            batch_margins[i, start:end] = np.minimum(np.arange(batch_size), np.arange(batch_size)[::-1])

            if end >= len(obs_index):
                break

        # Replace last written values with values averaged across overlapping batches
        if overlap > 0:
            # Weight fits from overlapping batches by their distance to the batch edge
            # Add a small number (0.1) for observations at the edge of the batch,
            # since the observations at the edges of the group will always have zero margin
            batch_weights = batch_margins + 0.1
            batch_weights[batch_weights < 0] = 0  # obs not fitted in batch
            x_weights = np.moveaxis(np.tile(batch_weights, (x_batch.shape[-1], 1, 1)), 0, -1)
            # print(batch_weights.shape)
            # print(batch_weights, np.sum(batch_weights, axis=0))
            x_mean = np.average(x_batch, axis=0, weights=x_weights)
            self.obs_x_resolved[obs_index] = x_mean

            for key, val in x_special.items():
                if np.ndim(val) > 2:
                    key_weights = np.moveaxis(np.tile(batch_weights, (val.shape[-1], 1, 1)), 0, -1)
                else:
                    key_weights = batch_weights

                x_k = np.average(val, axis=0, weights=key_weights)
                self.obs_special_resolved[key][obs_index] = x_k

    # --------------------------------------------------------------------
    # Data/fit validation (badness)
    # --------------------------------------------------------------------
    def score_group_data_badness(self, group_id, psi_sort_dims, median_filter_size=(3, 1), std_size=(5, 3),
                                 ignore_outliers=True, impute=False):
        """
        Evaluate the badness of data for individual observations within a group
        Badness is evaluated by constructing a 2D data array and comparing the raw data for each observation
        to the local median-filtered values
        :param str group_id: group ID to score
        :param list psi_sort_dims: dimensions by which to sort observations to construct the 2D array
        :param tuple median_filter_size: size of median filter
        :param tuple std_size: size of std filter for calculation of weighted deviation from filtered values
        :param bool ignore_outliers: if True, don't count individual outlier data points towards observation badness
        :return:
        """

        obs_index = self.get_group_index(group_id, psi_sort_dims=psi_sort_dims)

        data_list = [self.get_obs_data(i) for i in obs_index]
        iv_data = [dl[0] for dl in data_list]
        z_data = [dl[1] for dl in data_list]

        # Assemble chrono data array
        # Expect all chrono measurements within group to have same length
        v_len = np.array([0 if tup[0] is None else len(tup[0]) for tup in iv_data])
        has_chrono = v_len > 0
        v_len = np.unique(v_len[v_len > 0])
        if len(v_len) > 1:
            raise ValueError(f'Found chrono data with different lengths: {v_len}')
        else:
            v_len = v_len[0]
        i_array = np.stack([np.empty(v_len) * np.nan if tup[1] is None else tup[1] for tup in iv_data], axis=0)
        v_array = np.stack([np.empty(v_len) * np.nan if tup[2] is None else tup[2] for tup in iv_data], axis=0)

        # Get v_diff for comparison
        v_hi = np.nanpercentile(v_array, 98, axis=1)
        v_lo = np.nanpercentile(v_array, 2, axis=1)
        v_mid = 0.5 * (v_hi + v_lo)
        i_range = np.nanpercentile(i_array, 98, axis=1) - np.nanpercentile(i_array, 2, axis=1)
        v_diff = (v_array - v_mid[:, None]) / i_range[:, None]

        # Assemble EIS data array
        # EIS data: truncate to shortest length (hybrid measurements)
        z_array = [tup[1] for tup in z_data]
        z_len = np.array([np.inf if z is None else len(z) for z in z_array])
        has_eis = z_len < np.inf
        z_len = np.min(z_len)
        z_array = np.stack(
            [np.empty(2 * z_len) * np.nan if z is None
             else utils.eis.complex_vector_to_concat(z[:z_len]) for z in z_array],
            axis=0
        )

        if ignore_outliers:
            # Flag individual outlier points
            v_out_flag = flag_outliers(v_diff, filter_size=(5, 1), thresh=0.7)
            z_out_flag = flag_outliers(z_array, filter_size=(5, 1), thresh=0.7)
            # print('z_out:', np.where(z_out_flag))

            # If fewer than 5% of data points in any observation are outliers,
            # set outliers to nan prior to checking for bad observations.
            v_out_count = np.sum(v_out_flag, axis=1)
            v_count_index = v_out_count < int(v_diff.shape[1] * 0.05)
            v_diff[v_count_index[:, None] & v_out_flag] = np.nan

            z_out_count = np.sum(z_out_flag, axis=1)
            z_count_index = z_out_count < int(z_array.shape[1] * 0.05)
            z_array[z_count_index[:, None] & z_out_flag] = np.nan

        # Get chrono data badness (normalized RSS from filtered values)
        if impute:
            v_diff_filt = impute_nans(v_diff)
        else:
            v_diff_filt = v_diff
        v_diff_filt = ndimage.median_filter(v_diff_filt, size=median_filter_size)
        _, v_rss = flag_bad_obs(v_diff, v_diff_filt, std_size=std_size, return_rss=True)

        # Get EIS data badness (normalized RSS from filtered values)
        if impute:
            z_filt = impute_nans(z_array)
        else:
            z_filt = z_array
        z_filt = ndimage.median_filter(z_filt, size=median_filter_size)
        _, z_rss = flag_bad_obs(z_array, z_filt, std_size=std_size, return_rss=True)

        # Get total RSS
        tot_rss = np.zeros(len(v_rss))
        hybrid_index = has_eis & has_chrono
        eis_index = has_eis & ~has_chrono
        chrono_index = has_chrono & ~has_eis
        tot_rss[hybrid_index] = 0.5 * (v_rss[hybrid_index] + z_rss[hybrid_index])
        tot_rss[eis_index] = z_rss[eis_index]
        tot_rss[chrono_index] = v_rss[chrono_index]

        # Store RSS
        self.obs_data_badness[obs_index] = tot_rss

    def score_group_fit_badness(self, group_id, psi_sort_dims, median_size=(3, 3), std_size=(5, 3),
                                include_special=False):

        obs_index = self.get_group_index(group_id, psi_sort_dims=psi_sort_dims)
        x_array = self.obs_x[obs_index].copy()
        ignore = self.obs_ignore_flag[obs_index] | ~self.obs_fit_status[obs_index]
        x_array[ignore] = np.nan

        x_filt = ndimage.median_filter(x_array, size=median_size)

        _, x_rss = flag_bad_obs(x_array, x_filt, std_size=std_size, return_rss=True)

        if include_special:
            num_drt = x_filt.shape[1]
            num_special = []
            special_rss = []
            for key in self.obs_special.keys():
                if key not in ['vz_offset', 'v_baseline']:
                    x_k = self.obs_special[key][obs_index].copy()
                    x_k[ignore] = np.nan

                    if key == 'x_dop':
                        # normalize
                        dop_scale_vector = phasance.phasor_scale_vector(self.fixed_basis_nu,
                                                                                        self.tau_supergrid)
                        x_k = x_k / dop_scale_vector[None, :]

                    # Filter
                    xk_filt = ndimage.median_filter(x_k, size=median_size)
                    if key == 'x_dop':
                        # DOP params tend to be sparse - smooth within each obs to obtain well-behaved std
                        xk_filt = ndimage.gaussian_filter(x_k, sigma=(0., 0.35))

                    _, xk_rss = flag_bad_obs(x_k, xk_filt, std_size=std_size, return_rss=True, robust_std=False)
                    special_rss.append(xk_rss)

                    if np.ndim(x_k) == 1:
                        num_special.append(1)
                    else:
                        num_special.append(x_k.shape[1])

            # Weighted average across DRT and special params
            tot_num = num_drt + np.sum(num_special)
            weights = np.array([num_drt] + num_special)
            x_rss = np.average(np.stack([x_rss] + special_rss, axis=0), axis=0, weights=weights)

        self.obs_fit_badness[obs_index] = x_rss

    # --------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------
    def predict_r_p(self, psi=None, x=None, factor_index=None, absolute=False, **kw):
        if x is None:
            x = self.predict_x(psi, factor_index=factor_index, **kw)

        if absolute:
            x = np.abs(x)

        return np.sum(x, axis=-1) * self.tau_basis_area

    def predict_x(self, psi, factor_index=None, percentile=None, normalize=False, ndfilter=False, filter_func=None,
                  resample_dims=None,
                  filter_kw=None):
        self.validate_psi(psi)

        psi_index = self.get_psi_index(psi)
        if np.min(psi_index) > -1:
            # All requested coordinates are contained in observations
            x = self.obs_x[psi_index].copy()
        else:
            # Some or all requested coordinates were not observed. Resample (interpolate) from observations
            # TODO: need to provide resample dimensions
            if resample_dims is None:
                resample_dims = self.psi_dim_names
            resample_dim_index = [self.psi_dim_names.index(d) for d in resample_dims]
            x = resample(psi[:, resample_dim_index], self.obs_psi[np.ix_(self.obs_fit_status, resample_dim_index)],
                         self.obs_x[self.obs_fit_status])

        if normalize:
            rp = self.predict_r_p(x=x, absolute=True)
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

    def predict_drt(self, psi=None, tau=None, x=None, order=0, factor_index=None, normalize=False, **kw):
        if x is None:
            x = self.predict_x(psi, factor_index=factor_index, normalize=False, **kw)

        if normalize:
            rp = self.predict_r_p(psi=psi, x=x, factor_index=factor_index, normalize=False, absolute=True, **kw)
            x = x / rp[:, None]

        if tau is None:
            tau = self.tau_supergrid  # self.get_tau_eval(20)

        basis_mat = basis.construct_func_eval_matrix(np.log(self.tau_supergrid), np.log(tau), self.tau_basis_type,
                                                     self.tau_epsilon, order=order)

        return x @ basis_mat.T

    def predict_dop(self, psi=None, x=None, nu=None, order=0, factor_index=None,
                    normalize=False, normalize_tau=None, normalize_quantiles=(0.25, 0.75),
                    delta_density=False, include_ohmic=False, x_ohmic=None,
                    **kw):
        # TODO: implement predict_x_dop
        # if x is None:
        #     x = self.predict_x(psi, factor_index=factor_index, normalize=False, **kw)

        if nu is None:
            nu = self.get_nu_eval(10)

        # Construct basis matrix
        basis_mat = basis.construct_func_eval_matrix(self.fixed_basis_nu, nu,
                                                     self.nu_basis_type, epsilon=self.nu_epsilon,
                                                     order=order, zga_params=None)

        if delta_density:
            dnu = self.drt1d.get_nu_basis_spacing()
            if self.nu_basis_type == 'delta':
                x = x / dnu

        dop = x @ basis_mat.T

        # TODO: take x_ohmic, x_induc, x_cap args?
        # # Add pure inductance, resistance, and capacitance
        # ohmic_index = np.where(nu == 0)
        # if len(ohmic_index) == 1:
        #     r_inf = self.obs_special['R_inf']
        #     if delta_density:
        #         r_inf = r_inf / dnu[utils.array.nearest_index(self.basis_nu, 0)]
        #     dop[:, ohmic_index] += r_inf
        #
        # induc_index = np.where(nu == 1)
        # if len(induc_index) == 1:
        #     induc = self.fit_parameters['inductance']
        #     if delta_density:
        #         induc = induc / dnu[utils.array.nearest_index(self.basis_nu, 1)]
        #     dop[induc_index] += induc
        #
        # cap_index = np.where(nu == -1)
        # if len(cap_index) == 1:
        #     c_inv = self.fit_parameters['C_inv']
        #     if delta_density:
        #         c_inv = c_inv / dnu[utils.array.nearest_index(self.basis_nu, -1)]
        #     dop[cap_index] += c_inv

        # TODO: revisit normalization
        # if normalize:
        #     if normalize_tau is None:
        #         # data_tau_lim = pp.get_tau_lim(self.get_fit_frequencies(), self.get_fit_times(), self.step_times)
        #         # normalize_tau = np.array(data_tau_lim)
        #         normalize_tau = self.tau_supergrid
        #     else:
        #         normalize_tau = np.array([np.min(normalize_tau), np.max(normalize_tau)])
        #     normalize_by = phasance.phasor_scale_vector(nu, normalize_tau, normalize_quantiles)
        # else:
        #     normalize_by = 1
        normalize_by = self.drt1d.get_dop_norm(nu, normalize, normalize_tau, 
                                                   normalize_quantiles=(0, 1))
        
        dop /= normalize_by
        
        # Add ohmic after normalization
        if include_ohmic and 0 in nu:
            if x_ohmic is not None:
                ohmic_index = np.where(nu == 0)[0][0]
                if normalize:
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                    x_ohmic = x_ohmic / (normalize_by[ohmic_index] * self.drt1d.nu_basis_area)
                    
                dop[..., ohmic_index] = x_ohmic.copy()

        return dop

    def predict_param_cov(self, obs_index, factor_index=None):
        cov_matrices = []
        obs_index = np.atleast_1d(obs_index)
        for index in obs_index:
            if self.obs_fit_status[index]:
                drt = self.get_fit(index)
                if self.fit_type == 'pfrt':
                    if factor_index is not None:
                        p_mat = drt.pfrt_result['step_p_mat'][factor_index]
                        cov = drt.estimate_param_cov(p_matrix=p_mat)
                    else:
                        # Calculate for all factors
                        cov = np.array(
                            [drt.estimate_param_cov(p_matrix=p_mat) for p_mat in drt.pfrt_result['step_p_mat']]
                        )
                else:
                    cov = drt.estimate_param_cov()
            else:
                # Observation not fitted
                cov = None

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
            if cov[i] is None:
                x_cov[i] = np.nan
            else:
                left_index, right_index = self.obs_tau_indices[index]
                drt = self.get_fit(index)
                if self.fit_type == 'pfrt' and factor_index is None:
                    x_cov[i, :, left_index:right_index, left_index:right_index] = \
                        cov[i][:, drt.get_qp_mat_offset():, drt.get_qp_mat_offset():]
                else:
                    x_cov[i, left_index:right_index, left_index:right_index] = \
                        cov[i][drt.get_qp_mat_offset():, drt.get_qp_mat_offset():]

        return x_cov

    def predict_x_var(self, obs_index, factor_index=None):
        x_cov = self.predict_x_cov(obs_index, factor_index)
        return np.array([np.diag(cov) for cov in x_cov])

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
                if not np.any(np.isnan(drt_cov[i])):
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

    def predict_drt_var(self, obs_index, tau=None, x_cov=None, order=0, factor_index=None, extend_var=True,
                        ndfilter=False, filter_func=None, filter_kw=None):
        drt_cov = self.predict_drt_cov(obs_index, tau, x_cov, order, factor_index, extend_var)
        drt_var = np.array([np.diag(cov) for cov in drt_cov])

        # Apply filter
        if ndfilter:
            drt_var = apply_filter(drt_var, filter_func, filter_kw)

        return drt_var

    def predict_peak_prob(self, psi, x=None, f_var=None, fxx_var=None, tau=None, factor_index=None, extend_var=True,
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
        # TODO: replace with std filter values?
        # TODO: consider fxx in both dimensions?
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

        return peak_prob * np.sign(f)

    def predict_curv_prob(self, psi=None, x=None, f_var=None, fxx_var=None, tau=None, factor_index=None,
                          extend_var=True,
                          ndfilter=False, filter_func=None, filter_kw=None):

        # Calculate probability of negative curvature and f > 0
        if psi is None and (x is None or f_var is None or fxx_var is None):
            raise ValueError('If psi is not provided, all of x, f_var, and fxx_var must be provided')

        if tau is None:
            tau = self.get_tau_eval(10)

        # Get drt and curvature
        f = self.predict_drt(psi=psi, tau=tau, order=0, x=x, factor_index=factor_index,
                             ndfilter=ndfilter, filter_func=filter_func, filter_kw=filter_kw)
        fxx = self.predict_drt(psi=psi, tau=tau, order=2, x=x, factor_index=factor_index,
                               ndfilter=ndfilter, filter_func=filter_func, filter_kw=filter_kw)

        # Get variance
        if psi is not None:
            obs_index = self.get_psi_index(psi)

            if f_var is None:
                f_var = self.predict_drt_var(obs_index, order=0, tau=tau, factor_index=factor_index,
                                             extend_var=extend_var,
                                             ndfilter=ndfilter, filter_func=filter_func, filter_kw=filter_kw)

            if fxx_var is None:
                fxx_var = self.predict_drt_var(obs_index, order=2, tau=tau,
                                               factor_index=factor_index, extend_var=extend_var,
                                               ndfilter=ndfilter, filter_func=filter_func, filter_kw=filter_kw)

        f_sigma = f_var ** 0.5
        fxx_sigma = fxx_var ** 0.5

        f_prob = 1 - utils.stats.cdf_normal(0, -np.sign(fxx) * f, f_sigma)
        curv_prob = 1 - utils.stats.cdf_normal(0, -np.sign(f) * fxx, fxx_sigma)
        f_prob = 2 * np.maximum(f_prob - 0.5, 0)
        curv_prob = 2 * np.maximum(curv_prob - 0.5, 0)
        fc_prob = np.minimum(f_prob, curv_prob) * np.sign(f)

        return fc_prob

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

        self.obs_special_resolved = deepcopy(self.obs_special)

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

    def get_group_index(self, group_id, psi_sort_dims=None, exclude_flagged=False):
        """
        Get observation indices of group or groups
        :param group_id: group_id or list of group_ids
        :param psi_sort_dims: names of psi dimensions by which to sort observations
        :return:
        """
        if type(group_id) == str:
            obs_index = np.where(np.array(self.obs_group_id) == group_id)[0]
        else:
            obs_index = np.where(np.isin(np.array(self.obs_group_id), group_id))[0]

        # Sort by specifieid psi dimensions
        if psi_sort_dims is not None:
            if type(group_id) == str:
                sort_vals = [self.obs_psi[obs_index, self.psi_dim_names.index(d)] for d in psi_sort_dims][::-1]
            else:
                # Sort within groups
                sort_vals = (
                                [np.array(self.obs_group_id)[obs_index]]
                                + [self.obs_psi[obs_index, self.psi_dim_names.index(d)] for d in psi_sort_dims]
                            )[::-1]
        else:
            sort_vals = None

        if sort_vals is not None:
            obs_index = obs_index[np.lexsort(sort_vals)]

        if exclude_flagged:
            obs_index = obs_index[~self.obs_ignore_flag[obs_index]]

        return obs_index

    def filter_psi(self, dim_eq=None, dim_gt=None, dim_lt=None, return_index=True,
                   exclude_flagged=False):
        if dim_eq is None:
            dim_eq = {}
        if dim_gt is None:
            dim_gt = {}
        if dim_lt is None:
            dim_lt = {}

        conditions = (
                [self.obs_psi[:, self.psi_dim_names.index(k)] == v for k, v in dim_eq.items()] +
                [self.obs_psi[:, self.psi_dim_names.index(k)] > v for k, v in dim_gt.items()] +
                [self.obs_psi[:, self.psi_dim_names.index(k)] < v for k, v in dim_lt.items()]
        )

        if exclude_flagged:
            conditions += [~self.obs_ignore_flag]

        psi_index = np.logical_and.reduce(conditions)

        if return_index:
            return np.where(psi_index)[0]
        else:
            return self.obs_psi[psi_index].copy()

    def get_tau_eval(self, ppd, extend_decades=0):
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

    def get_nu_eval(self, ppd=10):
        nu = np.linspace(-1, 1, 20 * ppd + 1)
        # Ensure basis_nu is included
        nu = np.unique(np.concatenate([self.fixed_basis_nu, nu]))
        # Ensure pure inductance, resistance, and capacitance are included
        nu = np.unique(np.concatenate([nu, np.array([-1, 0, 1])]))

        return nu

    @property
    def obs_dtype(self):
        def get_dtype(data):
            cp_data, eis_data = data
            if cp_data is None:
                return 'eis'
            elif eis_data is None:
                return 'chrono'
            else:
                return 'hybrid'

        return [get_dtype(od) for od in self.obs_data]

    @property
    def obs_psi_df(self):
        return pd.DataFrame(self.obs_psi, columns=self.psi_dim_names)

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
        if key in list(self.drt1d.special_qp_params.keys()):
            size = self.drt1d.special_qp_params[key].get('size', 1)
        else:
            arr = self.obs_special[key]
            if np.ndim(arr) == 1:
                size = 1
            else:
                size = arr.shape[-1]

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

    # ------------------------
    # Attribute management
    # ------------------------
    @property
    def attribute_categories(self):
        att = {
            'config': [
                'psi_dim_names', 'store_attr_categories',
                'tau_supergrid', 'tau_basis_type', 'tau_epsilon',
                'fit_inductance', 'fit_capacitance', 'fit_ohmic',
                # Distribution of phasances
                'fixed_basis_nu', 'nu_basis_type', 'nu_epsilon', 'fit_dop', 'normalize_dop',
                # Chrono settings
                'step_model', 'chrono_mode',
                # Data readers
                # 'chrono_reader', 'eis_reader',
                # Fit kwargs
                'fit_type', 'fit_kw', 'pfrt_factors',
                # Precision
                'frequency_precision', 'time_precision', 'input_signal_precision',
                # Diagnostics
                'print_diagnostics', 'warn', 'print_progress'
            ],
            'obs_data': [
                'obs_psi', 'obs_data', 'obs_group_id', 'obs_ignore_flag', 'obs_data_badness'
            ],
            'fit': [
                'obs_fit_status', 'obs_fit_errors', 'obs_fit_attr', 'obs_fit_badness',
                'obs_tau_indices', 'obs_x', 'obs_special',
                'obs_drt_var',
                'obs_resolve_status', 'obs_x_resolved', 'obs_special_resolved'
            ]
        }

        return att

    def get_attributes(self, which):
        """
        Get instance attributes by category
        :param str which: category of attributes to return. See attribute_categories for a dict of categories
        and corresponding attributes
        :return: dict of attribute names and values
        """
        att_names = None
        if type(which) == str:
            if which == 'all':
                att_names = sum(list(self.attribute_categories.values()), [])
            else:
                which = [which]

        if att_names is None:
            try:
                att_names = sum([self.attribute_categories[c] for c in which], [])
            except KeyError:
                raise ValueError('which argument contains an invalid attribute category. '
                                 'Valid categories: {}'.format(['all'] + list(self.attribute_categories.keys())))

        return {k: deepcopy(getattr(self, k)) for k in att_names}

    def set_attributes(self, att_dict):
        """
        Set instance attributes from dict
        :param dict att_dict: dict with attribute names and values
        :return:
        """
        for k, v in att_dict.items():
            setattr(self, k, deepcopy(v))

        # If observations are provided without fit data, clear fits
        if 'obs_psi' in att_dict.keys() and 'obs_x' not in att_dict.keys():
            self.clear_fits()

    def save_attributes(self, which, dest):
        """
        Save instance attributes to file using pickle
        :param str which: category of attributes to save. See attribute_categories for a dict of categories
        and corresponding attributes
        :param dest: destination file
        :return:
        """
        att_dict = self.get_attributes(which)
        with open(dest, 'wb') as f:
            pickle.dump(att_dict, f, pickle.DEFAULT_PROTOCOL)

    def load_attributes(self, source):
        """
        Set instance attributes from a file. Note that this will overwrite observations and fits if the corresponding
        attributes are contained in the source file!
        :param source: source file
        :return:
        """
        with open(source, 'rb') as f:
            att_dict = pickle.load(f)
        self.set_attributes(att_dict)

    def load_observations(self, source, append=True):
        """
        Load observations from a source file to the DRTMD instance. Does not overwrite configuration attributes.
        :param source: source file
        :param bool append: If True, append observations in source to existing instance observations.
        If False, overwrite instance observations with observations in source
        :return:
        """
        with open(source, 'rb') as f:
            att_dict = pickle.load(f)

        if append and self.num_obs > 0:
            # Append observations in source to existing observations
            for category in ['obs_data', 'fit']:
                for name in self.attribute_categories[category]:
                    existing = getattr(self, name)
                    new = att_dict[name]
                    if name in ['obs_special', 'obs_special_resolved']:
                        new_dict = {}
                        for k, v in existing.items():
                            new_dict[k] = np.concatenate([existing[k], new[k]])
                        setattr(self, name, new_dict)
                    elif type(existing) == list:
                        setattr(self, name, existing + new)
                    elif type(existing) == np.ndarray:
                        setattr(self, name, np.concatenate([existing, new]))
                    else:
                        raise ValueError(f'Attribute {name} has unexpected type {type(existing)}')
        else:
            # Overwrite existing observations with those from source
            names = sum([self.attribute_categories[k] for k in ['obs_data', 'fit']], [])
            obs_dict = {k: att_dict[k] for k in names}
            self.set_attributes(obs_dict)

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
            'fixed_basis_nu', 'fit_dop', 'normalize_dop', 'nu_basis_type', 'nu_epsilon',
            'fit_inductance', 'fit_ohmic', 'fit_capacitance',
            'step_model', 'chrono_mode',
            'frequency_precision', 'time_precision', 'input_signal_precision',
            'print_diagnostics', 'warn'
        ]:
            setattr(self.drt1d, name, deepcopy(value))
