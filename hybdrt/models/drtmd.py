import numpy as np
import warnings
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
import gpytorch
import torch

from .. import utils, preprocessing as pp
from hybdrt.matrices import basis, mat1d, matmd
from . import qphb
from . import gp
from .drtbase import DRTBase  # format_chrono_weights, format_eis_weights
from ..plotting import plot_eis, plot_chrono


# TODO:
# 1. Set up special_qp_params and extract_qp_params such that all parameters for each observation are contiguous
# 2. Consider: add cov matrix from QPHB to GP cov to "teach" GP about intra-observation, inter-task correlations
# 3. Consider: add precision matrix from GP to QPHB P matrix/q vector to "teach" QPHB about inter-observation,
#  inter-task correlations. May want to zero out diagonal to avoid over-penalizing, as well as give it a scale
# 4. Consider: multiply RBF kernel by diagonal variable-scale kernels to allow for local variations in covariance


class DRTMD(DRTBase):
    def __init__(self, basis_tau, tau_basis_type='gaussian', tau_epsilon=None,
                 step_model='ideal', op_mode='galvanostatic', fit_inductance=True,
                 downsample_chrono=True, downsample_chrono_kw=None, offset_chrono_baseline=True,
                 offset_chrono_steps=True, smooth_chrono_inf_response=True, chrono_tau_rise=None,
                 qphb_fit_kw=None, interpolate_integrals=True,
                 time_precision=10, input_signal_precision=10, frequency_precision=10):

        super().__init__(basis_tau, tau_basis_type, tau_epsilon, step_model, op_mode, interpolate_integrals,
                         chrono_tau_rise,
                         fit_inductance, time_precision, input_signal_precision, frequency_precision)

        # If epsilon is not set, apply default value
        if self.tau_epsilon is None:
            if self.tau_basis_type in ('gaussian', 'zga'):
                # Determine epsilon from basis_tau spacing
                delta_ln_tau = np.mean(np.diff(np.log(self.basis_tau)))
                self.tau_epsilon = 1 / delta_ln_tau
            elif self.tau_basis_type == 'Cole-Cole':
                # Default for Cole-Cole
                self.tau_epsilon = 0.95

        # If using ZGA basis function, set parameters
        if self.tau_basis_type == 'zga' and self.zga_params is None:
            self.set_zga_params()

        # # Generate integral lookups for interpolation
        # if interpolate_integrals:
        #     if self.step_model != 'ideal' and chrono_tau_rise is None:
        #         raise ValueError('A constant chrono_tau_rise value must be provided to use interpolation for'
        #                          'integral evaluation')
        #
        #     print('Generating impedance integral lookups...')
        #     zre_lookup, zim_lookup = basis.generate_impedance_lookup(self.tau_basis_type, self.tau_epsilon, 2000,
        #                                                              zga_params=self.zga_params)
        #
        #     print('Generating response integral lookups...')
        #     response_lookup = basis.generate_response_lookup(self.tau_basis_type, self.op_mode, self.step_model,
        #                                                      self.tau_epsilon, 2000, chrono_tau_rise, self.zga_params)
        #
        #     self.interpolate_lookups = {'z_real': zre_lookup, 'z_imag': zim_lookup, 'response': response_lookup}
        #     self.integrate_method = 'interp'
        #     print('Integral lookups ready')
        # else:
        #     self.interpolate_lookups = {'z_real': None, 'z_imag': None, 'response': None}
        #     self.integrate_method = 'trapz'

        # Set default QPHB fit kwargs
        qphb_defaults = dict(vz_offset=True, nonneg=True, scale_data=True, rp_scale=14, update_scale=True,
                             # basic fit control
                             l2_lambda_0=None, l1_lambda_0=0.0, inductance_scale=1e-5,
                             penalty_type='integral', derivative_weights=[1.5, 1.0, 0.5],
                             special_penalties={'R_inf': 0, 'inductance': 0, 'v_baseline': 0, 'vz_offset': 20 / 142},
                             # Prior hyperparameters
                             chrono_vmm_epsilon=0.25, chrono_error_structure='uniform',
                             eis_vmm_epsilon=0.25, eis_error_structure=None, eis_reim_cor=0.25,
                             chrono_weight_factor=None, eis_weight_factor=None, outlier_p=None,
                             iw_alpha=1.50, iw_beta=None,
                             s_alpha=[1.5, 2.5, 25], s_0=1,
                             rho_alpha=[0.1, 0.15, 0.2], rho_0=1)

        # Update with user-specified params
        if qphb_fit_kw is not None:
            qphb_defaults.update(qphb_fit_kw)

        qphb_fit_kw = qphb_defaults

        # Check string args
        utils.validation.check_error_structure(qphb_fit_kw['chrono_error_structure'])
        utils.validation.check_error_structure(qphb_fit_kw['eis_error_structure'])
        utils.validation.check_penalty_type(qphb_fit_kw['penalty_type'])

        # Update with default hypers
        default_hypers = qphb.get_default_hypers(1, True)
        for k, v in default_hypers.items():
            if qphb_fit_kw.get(k, None) is None and k not in ['l2_lambda_0', 'iw_beta']:
                qphb_fit_kw[k] = default_hypers[k]

        # Convert list and scalar arguments to arrays
        if np.shape(qphb_fit_kw['s_alpha']) == ():
            qphb_fit_kw['s_alpha'] = [qphb_fit_kw['s_alpha']] * len(qphb_fit_kw['derivative_weights'])
        if np.shape(qphb_fit_kw['rho_alpha']) == ():
            qphb_fit_kw['rho_alpha'] = [qphb_fit_kw['rho_alpha']] * len(qphb_fit_kw['derivative_weights'])
        for key in ['derivative_weights', 's_alpha', 'rho_alpha']:
            val = qphb_fit_kw[key]
            if type(val) == list:
                qphb_fit_kw[key] = np.array(val)

        # Store QPHB kwargs
        self._qphb_fit_kw = qphb_fit_kw

        # Store chrono data processing kwargs
        self.offset_chrono_baseline = offset_chrono_baseline
        self.downsample_chrono = downsample_chrono
        if downsample_chrono_kw is None:
            downsample_chrono_kw = {'prestep_samples': 10, 'ideal_times': None}
        self.downsample_chrono_kw = downsample_chrono_kw
        self.offset_chrono_steps = offset_chrono_steps
        self.smooth_chrono_inf_response = smooth_chrono_inf_response
        self.chrono_tau_rise = chrono_tau_rise

        # Initialize data arrays/lists
        self.obs_psi_array = None
        self.data_type = None
        self.chrono_data_list = []
        self.chrono_sample_index_list = []
        self.chrono_step_info_list = []
        self.eis_data_list = []

        # Observation fit status and data factors
        self.obs_fit_status = np.array([], dtype=bool)
        self.obs_data_factors = []

        # Initialize parameter and hyperparameter arrays/lists
        self.parameter_array = None
        self.parameter_variance = None
        self.scaled_parameter_array = None
        self.rho_array = None
        self.s_arrays = None
        self.xmx_norm_array = None
        self.chrono_est_weight_list = []
        self.chrono_init_weight_list = []
        self.chrono_init_outlier_t_list = []
        # self.chrono_outlier_t_list = []
        self.eis_est_weight_list = []
        self.eis_init_weight_list = []
        self.eis_init_outlier_t_list = []
        self.chrono_baseline_vector = []
        self.scaled_chrono_baseline_vector = []
        self.scaled_chrono_offset_vector = []

        # Initialize scale vectors
        self.coefficient_scale_vector = []
        self.input_signal_scale_vector = []
        self.response_signal_scale_vector = []
        self.impedance_scale_vector = []

        # GP
        self.gpr = None

    # --------------------------------------------
    # Fitting and related methods
    # --------------------------------------------
    def add_observations(self, psi_array, chrono_data_list, eis_data_list, chrono_step_time_list=None):

        utils.validation.check_md_data(psi_array, chrono_data_list, eis_data_list)

        # Expect psi to be of shape (num_new_obs, psi_dims). If psi is 1d, reshape to column vector
        if len(psi_array.shape) == 1:
            psi_array = psi_array[:, None]
        num_new_obs = psi_array.shape[0]

        # Append psi coordinates
        if self.obs_psi_array is None:
            self.obs_psi_array = psi_array.copy()
        else:
            self.obs_psi_array = np.vstack((self.obs_psi_array, psi_array.copy()))

        # Expand parameter and hyperparameter arrays
        # Create parameter array in expected shape. Initialize near zero
        new_x_array = np.zeros((num_new_obs, self.params_per_obs))  # + 1e-6

        # Initialize new hyperparameter arrays at mode
        new_rho_array = np.zeros((num_new_obs, self.num_derivatives)) + self.qphb_fit_kw['rho_0']
        new_s_array = np.zeros((num_new_obs, self.params_per_obs)) + self.qphb_fit_kw['s_0']

        # Initialize xmx_norms at 1 (value doesn't matter - will be overwritten at time of fit)
        new_xmx_norm_array = np.ones((num_new_obs, self.num_derivatives))

        # Initialize or append to array attributes
        if self.parameter_array is None:
            # Adding first observations - initialize arrays
            self.parameter_array = new_x_array.copy()
            self.scaled_parameter_array = new_x_array.copy()
            self.parameter_variance = new_x_array.copy()
            self.rho_array = new_rho_array
            self.s_arrays = [new_s_array] * self.num_derivatives
            self.xmx_norm_array = new_xmx_norm_array
        else:
            # Existing observations present - append to arrays
            self.parameter_array = np.vstack((self.parameter_array, new_x_array.copy()))
            self.scaled_parameter_array = np.vstack((self.scaled_parameter_array, new_x_array.copy()))
            self.parameter_variance = np.vstack((self.parameter_variance, new_x_array.copy()))
            self.rho_array = np.vstack((self.rho_array, new_rho_array))
            for k in range(self.num_derivatives):
                self.s_arrays[k] = np.vstack((self.s_arrays[k], new_s_array))
            self.xmx_norm_array = np.vstack((self.xmx_norm_array, new_xmx_norm_array))

        # Expand observation fit status - new observations unfitted
        self.obs_fit_status = np.concatenate((self.obs_fit_status, np.zeros(num_new_obs, dtype=bool)))

        # Expand scale vectors
        new_scale_vector = np.ones(num_new_obs)
        for scale_key in ['coefficient', 'input_signal', 'response_signal', 'impedance']:
            scale_vector_name = f'{scale_key}_scale_vector'
            setattr(self, scale_vector_name, np.concatenate((getattr(self, scale_vector_name), new_scale_vector)))

        # Expand weight lists
        new_weight_list = [None] * num_new_obs
        self.chrono_est_weight_list += new_weight_list
        self.chrono_init_weight_list += new_weight_list
        self.chrono_init_outlier_t_list += new_weight_list
        self.eis_est_weight_list += new_weight_list
        self.eis_init_weight_list += new_weight_list
        self.eis_init_outlier_t_list += new_weight_list

        # Append data and determine overall data type
        if chrono_data_list is None and eis_data_list is None:
            raise ValueError('At least one of chrono or EIS data must be provided')
        elif eis_data_list is None:
            eis_data_list = [None] * num_new_obs
        elif chrono_data_list is None:
            chrono_data_list = [None] * num_new_obs

        self.chrono_data_list += chrono_data_list
        self.eis_data_list += eis_data_list

        # Determine type of new observations
        if chrono_data_list.count(None) == num_new_obs:
            new_data_type = 'eis'
        elif eis_data_list.count(None) == num_new_obs:
            new_data_type = 'chrono'
        else:
            new_data_type = 'hybrid'

        # Update overall data type
        if self.data_type is None:
            self.data_type = new_data_type
        elif self.data_type == 'eis' and new_data_type in ('chrono', 'hybrid'):
            self.data_type = 'hybrid'
        elif self.data_type == 'chrono' and new_data_type in ('eis', 'hybrid'):
            self.data_type = 'hybrid'

        # Define special parameters included in quadratic programming parameter vector
        # TODO: will need to zero out unused parameters (e.g. inductance for pure chrono, v_baseline for pure eis, etc.)
        self.set_special_parameter('R_inf')
        if self.fit_inductance:
            self.set_special_parameter('inductance')
        if self.data_type in ('hybrid', 'chrono'):
            self.set_special_parameter('v_baseline')
        if self.data_type == 'hybrid' and self.qphb_fit_kw['vz_offset']:
            self.set_special_parameter('vz_offset')

        # Preprocess chrono data and append to data lists/arrays
        chrono_sample_data_list, chrono_sample_index_list, chrono_step_info_list, chrono_baseline_vector = \
            self.process_chrono_signals(chrono_data_list, chrono_step_time_list)
        self.chrono_sample_index_list += chrono_sample_index_list
        self.chrono_step_info_list += chrono_step_info_list
        self.chrono_baseline_vector = np.concatenate((self.chrono_baseline_vector, chrono_baseline_vector))
        self.scaled_chrono_baseline_vector = np.concatenate((self.scaled_chrono_baseline_vector,
                                                             chrono_baseline_vector))
        self.scaled_chrono_offset_vector = np.concatenate((self.scaled_chrono_offset_vector, np.zeros(num_new_obs)))

        # Get data factors
        new_data_factors = np.array(
            [qphb.get_data_factor_from_data(utils.md.get_data_tuple_item(chrono_sample_data_list[i], 0),
                                            utils.md.get_data_tuple_item(chrono_step_info_list[i], 0),
                                            utils.md.get_data_tuple_item(eis_data_list[i], 0))
             for i in range(num_new_obs)]
        )
        self.obs_data_factors = np.concatenate((self.obs_data_factors, new_data_factors))

        # Delete sample data list - only retrieved for convenience in getting data factors and est weights
        # At time of fitting, chrono_sample_index should be used to obtain downsampled chrono data
        del chrono_sample_data_list

    def initialize_gp(self, task_cov_rank=None, task_noise_rank=0, **init_kw):
        # Use fitted observations as training data
        train_obs_indices = np.where(self.obs_fit_status)
        if len(train_obs_indices) == 0:
            raise Exception('Cannot initialize GP without fitted parameters')
        else:
            train_x = self.obs_psi_array[train_obs_indices]
            train_y = self.parameter_array[train_obs_indices]
            num_tasks = train_y.shape[1]

            if 'inductance' in self.special_qp_params.keys():
                train_y[:, self.special_qp_params['inductance']['index']] /= self.qphb_fit_kw['inductance_scale']

        print(train_x.shape, train_y.shape)

        # Define likelihood and covariance
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, has_global_noise=True,
                                                                      rank=task_noise_rank)

        # noise = torch.ones((train_y.shape[1], train_y.shape[0]))
        # noise_model = gpytorch.likelihoods.noise_models.FixedGaussianNoise(noise)
        # likelihood = gpytorch.likelihoods.multitask_gaussian_likelihood._MultitaskGaussianLikelihoodBase(
        #     num_tasks, noise_model, rank=task_noise_rank)

        if task_cov_rank is None:
            task_cov_rank = int(num_tasks / 2)

        covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=task_cov_rank,
            task_covar_prior=None
        )

        self.gpr = gp.MultitaskGP(train_x, train_y, likelihood, covar_module, **init_kw)

    def optimize_gp_hypers(self, update_train_data=True, iterations=100, lr=0.1, verbose=True):
        if self.gpr is None:
            raise RuntimeError('GP is not initialized')

        if update_train_data:
            self.update_gp_train_data()

        self.gpr.optimize_hypers(iterations, lr, verbose)

    def update_gp_train_data(self, obs_indices=None):
        if self.gpr is None:
            raise RuntimeError('GP is not initialized')

        # If obs_indices not specified, use all fitted observations for training
        if obs_indices is None:
            obs_indices = np.where(self.obs_fit_status)

        train_x = torch.Tensor(self.obs_psi_array[obs_indices])
        train_y = self.parameter_array[obs_indices]
        # Apply inductance scaling
        if 'inductance' in self.special_qp_params.keys():
            train_y[:, self.special_qp_params['inductance']['index']] /= self.qphb_fit_kw['inductance_scale']
        train_y = torch.Tensor(train_y)
        self.gpr.update_train_data(train_x, train_y, strict=True, update_normalization=True)

    # def get_gp_precision(self, obs_indices, remove_scaling=True):
    #     # Get psi_array for batch
    #     psi_array = self.obs_psi_array[obs_indices]
    #
    #     # Evaluate posterior distribution (disable fast pred to ensure covar matrix is invertible)
    #     mvn = self.gpr.evaluate_posterior_mvn(psi_array, fast_pred=False)
    #
    #     # Get posterior precision
    #     gp_precision = mvn.precision_matrix
    #
    #     # Remove GP target scaling
    #     if remove_scaling:
    #         gp_precision /= self.gpr.y_norm.raw_scale ** 2  # Reverse GP target scaling
    #
    #     gp_precision = gp_precision.detach().numpy()
    #
    #     return gp_precision

    def qphb_batch_fit(self, obs_indices, xtol=1e-2, max_iter=50,
                       start_from_prev=False, gp_assist=False,
                       gp_rho_alpha=1, gp_rho_0=1, gp_s_alpha=1, gp_s_0=1, gp_frac=0.1, include_gp_mean=True):

        batch_size = len(obs_indices)

        # Process data and calculate matrices for fit
        data, matrices = self._prep_for_batch_fit(obs_indices)
        chrono_data_list, eis_data_list = data
        rm, zm, chrono_vmm, eis_vmm, base_penalty_matrices, penalty_matrices = matrices

        if gp_assist:
            if self.gpr is None:
                raise ValueError('GP must be initialized to use gp_assist')
            # Get psi_array for batch
            psi_array = self.obs_psi_array[obs_indices]

            # Evaluate posterior distribution (disable fast pred to ensure covar matrix is invertible)
            mvn = self.gpr.evaluate_posterior_mvn(psi_array, fast_pred=False)

            # Get unscaled posterior mean
            if include_gp_mean:
                gp_mean = self.gpr.y_norm.inverse_transform(mvn.mean).detach().numpy().flatten()
            else:
                gp_mean = np.zeros(batch_size * self.params_per_obs)

            # Get unscaled posterior precision
            # gp_precision = self.get_gp_precision(obs_indices, remove_scaling=True)
            gp_precision = self.gpr.compute_precision(psi_array, remove_scaling=True)
            gp_precision = gp_precision.detach().numpy()

            # if not include_gp_mean:
            #     gp_precision = gp_precision - np.diag(np.diagonal(gp_precision)) \
            #                    + np.eye(batch_size * self.params_per_obs)

            # Apply QPHB data scaling
            x_scale_vector = self.coefficient_scale_vector[obs_indices]
            x_scale_diagonal = self.obs_vector_to_diagonal(x_scale_vector)
            x_inv_scale_diagonal = self.obs_vector_to_diagonal(1 / x_scale_vector)
            gp_mu = x_inv_scale_diagonal @ gp_mean
            gp_omega = x_scale_diagonal @ gp_precision @ x_scale_diagonal
            # if not include_gp_mean:
            #     gp_omega = gp_omega - np.diag(np.diagonal(gp_omega)) \
            #                    + np.eye(batch_size * self.params_per_obs)
            print('gp prec sum:', np.sum(gp_omega))
            # print('gp mu:', gp_mu)

        # Get default values of data-dependent hyperparameters
        default_hypers = qphb.get_default_hypers(self.obs_data_factors[obs_indices], True)

        # Use either user-supplied or default data-dependent hyperparameters
        if self.qphb_fit_kw['l2_lambda_0'] is None:
            l2_lambda_0_vector = default_hypers['l2_lambda_0']
        else:
            l2_lambda_0_vector = np.ones(batch_size) * self.qphb_fit_kw['l2_lambda_0']

        if self.qphb_fit_kw['iw_beta'] is None:
            iw_beta_vector = default_hypers['iw_beta']
        else:
            iw_beta_vector = np.ones(batch_size) * self.qphb_fit_kw['iw_beta']

        print('l2_lambda_0_vector:', np.round(l2_lambda_0_vector, 2))
        print('iw_beta_vector:', np.round(iw_beta_vector, 4))

        # Reshape lambda vector to diagonal matrix
        l2_lambda_0_diagonal = self.obs_vector_to_diagonal(l2_lambda_0_vector ** 0.5)

        # Get flattened data vectors
        z_vector = utils.md.data_list_to_vector(eis_data_list, 'eis', self.op_mode)
        response_vector = utils.md.data_list_to_vector(chrono_data_list, 'chrono', self.op_mode)

        # Stack chrono and eis matrices and data vectors
        # print(rm.shape, response_vector.shape)
        design_matrix = np.vstack((rm, zm))
        vmm = np.zeros((design_matrix.shape[0], design_matrix.shape[0]))
        # print(vmm.shape, chrono_vmm.shape, eis_vmm.shape)
        vmm[:chrono_vmm.shape[0], :chrono_vmm.shape[0]] = chrono_vmm
        vmm[chrono_vmm.shape[0]:, chrono_vmm.shape[0]:] = eis_vmm
        data_vector = np.concatenate((response_vector, z_vector))

        # Construct l1 lambda vector
        l1_lambda_vector = np.ones(design_matrix.shape[1]) * self.qphb_fit_kw['l1_lambda_0']

        # Initialize s and rho arrays at prior mode
        if start_from_prev:
            rho_array = self.rho_array[obs_indices]
            s_arrays = [s_array[obs_indices] for s_array in self.s_arrays]
            xmx_norm_array = self.xmx_norm_array[obs_indices]
            x = self.scaled_parameter_array[obs_indices]
        else:
            rho_array = np.ones((batch_size, self.num_derivatives)) * self.qphb_fit_kw['rho_0']
            s_arrays = [np.ones((batch_size, self.params_per_obs)) * self.qphb_fit_kw['s_0']] * self.num_derivatives
            xmx_norm_array = np.ones((self.num_obs, self.num_derivatives))
            x = np.zeros(design_matrix.shape[1]) + 1e-6

        # Flatten s_vectors for QPHB iteration
        s_vectors = [s_array.flatten() for s_array in s_arrays]

        # Initialize data weights
        chrono_weights, eis_weights = self.initialize_batch_weights(obs_indices, chrono_data_list, eis_data_list,
                                                                    rho_array, s_arrays, rm, zm, chrono_vmm, eis_vmm,
                                                                    base_penalty_matrices, iw_beta_vector
                                                                    )

        chrono_est_weights, chrono_init_weights, chrono_weight_factors, chrono_outlier_t = chrono_weights
        eis_est_weights, eis_init_weights, eis_weight_factors, eis_outlier_t = eis_weights

        # Concatenate weights
        est_weights = np.concatenate((chrono_est_weights, eis_est_weights))
        weights = np.concatenate((chrono_init_weights, eis_init_weights))
        weight_factors = np.concatenate((chrono_weight_factors, eis_weight_factors))
        outlier_t = np.concatenate((chrono_outlier_t, eis_outlier_t))

        # Identify any inactive parameters for batch. Delete corresponding columns from design matrix
        inactive_param_indices = self.get_inactive_param_indices(chrono_data_list, eis_data_list)
        if len(inactive_param_indices) > 0:
            inactive_param_values = np.zeros(len(inactive_param_indices))
        else:
            inactive_param_indices = None
            inactive_param_values = None

        # design_matrix = np.delete(design_matrix, inactive_param_indices, axis=1)

        # Initialize gp hypers
        if gp_assist:
            gp_rho = np.ones(batch_size) * gp_rho_0
            gp_s_array = np.ones((batch_size, self.params_per_obs)) * gp_s_0

            # Reshape for QPHB
            gp_s_vector = gp_s_array.flatten()

            # Pack into tuple
            gp_hypers = (gp_rho, gp_s_vector)

        self.qphb_history = []
        it = 0
        while it < max_iter:
            x_in = x.copy()

            # Apply weight factors
            weights *= weight_factors

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            # Wait until 2nd iteration is complete - first iteration with init_weights will be underfitted
            if it > 1 and all([self.qphb_fit_kw['scale_data'], self.qphb_fit_kw['update_scale']]):
                # Get scale factor
                rp_vector = self.get_batch_rp_vector(x, batch_size)
                scale_factor_vector = (self.qphb_fit_kw['rp_scale'] / rp_vector) ** 0.5  # Damp scale factor
                print('scale_factor_vector:', scale_factor_vector)
                print('rp_vector:', rp_vector)

                # Update data and qphb parameters to reflect new scale
                # Get scale vector of same size as data_vector
                data_scale_vector = np.concatenate((
                    utils.md.obs_vector_to_data_vector(scale_factor_vector, chrono_data_list),
                    utils.md.obs_vector_to_data_vector(scale_factor_vector, eis_data_list, expand_factor=2)
                ))

                # Update data, weights, and xmx_norm_array
                x_scale_vector = self.obs_vector_to_param_vector(scale_factor_vector)
                x_in *= x_scale_vector
                data_vector *= data_scale_vector
                xmx_norm_array *= scale_factor_vector[:, None]
                est_weights /= data_scale_vector
                weights /= data_scale_vector
                if gp_assist:
                    # x_scale_vector = self.obs_vector_to_param_vector(scale_factor_vector)
                    x_inv_scale_diag = self.obs_vector_to_diagonal(1 / scale_factor_vector)
                    gp_mu *= x_scale_vector
                    gp_omega = x_inv_scale_diag @ gp_omega @ x_inv_scale_diag
                    print('gp omega sum:', np.sum(gp_omega))

                # Update data scale attributes
                self.update_batch_data_scale(obs_indices, scale_factor_vector)

            # Format diagonal rho matrices
            rho_diagonals = [self.obs_vector_to_diagonal(rho_array[:, k] ** 0.5) for k in
                             range(self.num_derivatives)]

            if gp_assist:  # and it > 0:
                # if it == 1:
                #     gp_hypers = gp_init_hypers
                gp_rho, gp_s_vector = gp_hypers
                print('gp_rho:', gp_rho)
                gp_rho_diag = self.obs_vector_to_diagonal(gp_rho ** 0.5)
                gp_args = (
                gp_omega, gp_mu, gp_rho_diag, gp_s_vector, gp_rho_alpha, gp_rho_0, gp_s_alpha, gp_s_0, gp_frac)
            else:
                gp_args = None

            x, s_vectors, rho_array, weights, outlier_t, gp_hypers, cvx_result, converged = \
                qphb.iterate_md_qphb(x_in, s_vectors, rho_array, rho_diagonals, data_vector, weights, est_weights,
                                     outlier_t, design_matrix, vmm, penalty_matrices, penalty_type, l1_lambda_vector,
                                     l2_lambda_0_diagonal, self.qphb_fit_kw, gp_args, xmx_norm_array,
                                     inactive_param_indices, inactive_param_values, batch_size, self.params_per_obs,
                                     self.special_qp_params, xtol, 1, self.qphb_history)

            # Normalize to ordinary ridge solution
            if it == 0 and not start_from_prev:
                xmx_norm_array = qphb.calculate_md_xmx_norm_array(x, penalty_matrices,
                                                                  self.qphb_fit_kw['derivative_weights'],
                                                                  self.get_qp_mat_offset(), batch_size,
                                                                  self.params_per_obs)
                print('xmx', xmx_norm_array)

            if converged:
                break
            elif it == max_iter - 1:
                warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')

            it += 1

        # Insert default values for inactive parameters
        # x = np.insert(x, inactive_param_indices, 0)
        # for k in range(self.num_derivatives):
        #     s_vectors[k] = np.insert(s_vectors[k], inactive_param_indices, self.qphb_fit_kw['s_0'])

        # Reshape parameter array and insert into stored parameters
        x = self.reshape_batch_params(x, batch_size)
        self.scaled_parameter_array[obs_indices] = x.copy()
        # x_unscaled = x * self.coefficient_scale_vector[obs_indices, None]
        # if 'inductance' in self.special_qp_params.keys():
        #     x_unscaled[:, self.special_qp_indices['inductance']] *= self.qphb_fit_kw['inductance_scale']
        x_unscaled = self.unscale_batch_params(x, obs_indices)
        self.parameter_array[obs_indices] = x_unscaled.copy()

        # Get parameter variance and store
        rho_diagonals = [self.obs_vector_to_diagonal(rho_array[:, k] ** 0.5) for k in
                         range(self.num_derivatives)]
        p_matrix, q_vector = qphb.calculate_md_pq(design_matrix, data_vector, penalty_matrices, penalty_type,
                                                  self.qphb_fit_kw['derivative_weights'], l2_lambda_0_diagonal,
                                                  l1_lambda_vector, rho_diagonals, s_vectors, weights)
        cov_matrix = mat1d.invert_psd(p_matrix, use_cholesky=False)
        if cov_matrix is not None:
            x_var = self.reshape_batch_params(np.diag(cov_matrix), batch_size)
            x_var_unscaled = self.unscale_batch_params(x_var, obs_indices, apply_offsets=False, scale_exponent=2)
            # x_var *= self.coefficient_scale_vector[obs_indices, None] ** 2
            # if 'inductance' in self.special_qp_params.keys():
            #     x_var[:, self.special_qp_indices['inductance']] *= self.qphb_fit_kw['inductance_scale'] ** 2
            self.parameter_variance[obs_indices] = x_var_unscaled.copy()
        else:
            # Matrix could not be inverted. Should there be a placeholder value inserted?
            pass

        # Insert hyperparameters into stored attributes
        self.rho_array[obs_indices] = rho_array.copy()
        for k in range(self.num_derivatives):
            s_array = self.reshape_batch_params(s_vectors[k], batch_size)
            self.s_arrays[k][obs_indices] = s_array.copy()
        self.xmx_norm_array[obs_indices] = xmx_norm_array.copy()

        # Update observation fit status
        self.obs_fit_status[obs_indices] = True

    def initialize_batch_weights(self, obs_indices, chrono_data_list, eis_data_list, rho_array, s_arrays,
                                 rm, zm, chrono_vmm, eis_vmm, base_penalty_matrices, iw_beta_vector):
        # Initialize chrono and eis weights separately if not already done
        # ----------------------------------------------------------------
        batch_size = len(obs_indices)

        chrono_est_weights = np.empty(utils.md.get_data_list_size(chrono_data_list))
        chrono_init_weights = np.empty(utils.md.get_data_list_size(chrono_data_list))
        chrono_init_outlier_t = np.empty(utils.md.get_data_list_size(chrono_data_list))
        chrono_weight_factors = np.empty(utils.md.get_data_list_size(chrono_data_list))
        eis_est_weights = np.empty(utils.md.get_data_list_size(eis_data_list) * 2)
        eis_init_weights = np.empty(utils.md.get_data_list_size(eis_data_list) * 2)
        eis_init_outlier_t = np.empty(utils.md.get_data_list_size(eis_data_list) * 2)
        eis_weight_factors = np.empty(utils.md.get_data_list_size(eis_data_list) * 2)

        for i in range(batch_size):
            chrono_data = chrono_data_list[i]
            eis_data = eis_data_list[i]
            if chrono_data is not None:
                if self.chrono_est_weight_list[obs_indices[i]] is None:
                    # Weights not yet initialized
                    rho_vector = rho_array[i]
                    s_vec_i = [s_array[i] for s_array in s_arrays]
                    input_signal, response_signal = utils.chrono.get_input_and_response(chrono_data[1], chrono_data[2],
                                                                                        self.op_mode)
                    rm_i = rm[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i],
                           i * self.params_per_obs: (i + 1) * self.params_per_obs]
                    vmm_i = chrono_vmm[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i],
                            self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]]
                    w_est_i, w_init_i, x_over_i, outlier_t_i = \
                        qphb.initialize_weights(base_penalty_matrices, penalty_type,
                                                self.qphb_fit_kw['derivative_weights'], rho_vector, s_vec_i,
                                                response_signal, rm_i, vmm_i, self.qphb_fit_kw['nonneg'],
                                                self.special_qp_params, self.qphb_fit_kw['iw_alpha'], iw_beta_vector[i],
                                                self.qphb_fit_kw['outlier_p'])

                    # Insert est and init weights into batch vectors
                    chrono_est_weights[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]] = w_est_i
                    chrono_init_weights[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]] = w_init_i
                    chrono_init_outlier_t[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]] \
                        = outlier_t_i

                    # Insert est and init weights into attributes for reference and reuse
                    self.chrono_est_weight_list[obs_indices[i]] = w_est_i.copy()
                    self.chrono_init_weight_list[obs_indices[i]] = w_init_i.copy()
                    self.chrono_init_outlier_t_list[obs_indices[i]] = outlier_t_i.copy()

                    # Get overall chrono weight scale for observation
                    chrono_weight_scale = np.mean(w_est_i ** -2) ** -0.5
                else:
                    # Use existing weights
                    w_est_i = self.chrono_est_weight_list[obs_indices[i]].copy()
                    chrono_est_weights[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]] = w_est_i
                    chrono_init_weights[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]] = \
                        self.chrono_init_weight_list[obs_indices[i]].copy()
                    chrono_init_outlier_t[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]] = \
                        self.chrono_init_outlier_t_list[obs_indices[i]].copy()
                    chrono_weight_scale = np.mean(w_est_i ** -2) ** -0.5
                print('chrono est weight scale:', chrono_weight_scale)
            else:
                # No data
                chrono_weight_scale = None

            if eis_data is not None:
                if self.eis_est_weight_list[obs_indices[i]] is None:
                    # Weights not yet initialized
                    rho_vector = rho_array[i]
                    s_vec_i = [s_array[i] for s_array in s_arrays]
                    zm_i = zm[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i],
                           i * self.params_per_obs: (i + 1) * self.params_per_obs]
                    vmm_i = eis_vmm[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i],
                            self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]
                            ]
                    z = eis_data[1]
                    z = np.concatenate([z.real, z.imag])
                    w_est_i, w_init_i, x_over_i, outlier_t_i = \
                        qphb.initialize_weights(base_penalty_matrices, penalty_type,
                                                self.qphb_fit_kw['derivative_weights'], rho_vector, s_vec_i, z, zm_i,
                                                vmm_i, self.qphb_fit_kw['nonneg'], self.special_qp_params,
                                                self.qphb_fit_kw['iw_alpha'], iw_beta_vector[i],
                                                self.qphb_fit_kw['outlier_p'])

                    # Insert est and init weights into batch vectors
                    eis_est_weights[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]] = w_est_i
                    eis_init_weights[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]] = w_init_i
                    eis_init_outlier_t[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]] = outlier_t_i

                    # Insert est and init weights into attributes for reference and reuse
                    self.eis_est_weight_list[obs_indices[i]] = w_est_i.copy()
                    self.eis_init_weight_list[obs_indices[i]] = w_init_i.copy()
                    self.eis_init_outlier_t_list[obs_indices[i]] = outlier_t_i.copy()

                    # Get overall eis weight scale for observation
                    eis_weight_scale = np.mean(w_est_i ** -2) ** -0.5
                else:
                    # Use existing weights
                    w_est_i = self.eis_est_weight_list[obs_indices[i]].copy()
                    eis_est_weights[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]] = w_est_i
                    eis_init_weights[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]] = \
                        self.eis_init_weight_list[obs_indices[i]].copy()
                    eis_init_outlier_t[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]] = \
                        self.eis_init_outlier_t_list[obs_indices[i]].copy()
                    eis_weight_scale = np.mean(w_est_i ** -2) ** -0.5
                print('eis est weight scale:', eis_weight_scale)
            else:
                # No data
                eis_weight_scale = None

            # Get default weight factors
            if eis_weight_scale is not None and chrono_weight_scale is not None:
                # Hybrid data - scale weights accordingly
                eis_weight_factor = (chrono_weight_scale / eis_weight_scale) ** 0.25
                chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.25
                print('chrono weight factor: {:.3f}'.format(chrono_weight_factor))
                print('eis weight factor: {:.3f}'.format(eis_weight_factor))
            else:
                # Pure EIS or chrono data - use weights as obtained
                eis_weight_factor = 1
                chrono_weight_factor = 1

            # Overwrite if user specified weight factors
            if self.qphb_fit_kw['eis_weight_factor'] is not None:
                eis_weight_factor = self.qphb_fit_kw['eis_weight_factor']

            if self.qphb_fit_kw['chrono_weight_factor'] is not None:
                chrono_weight_factor = self.qphb_fit_kw['chrono_weight_factor']

            chrono_weight_factors[self.chrono_batch_start_indices[i]:self.chrono_batch_end_indices[i]] = \
                chrono_weight_factor
            eis_weight_factors[self.eis_batch_start_indices[i]:self.eis_batch_end_indices[i]] = eis_weight_factor

        return (chrono_est_weights, chrono_init_weights, chrono_weight_factors, chrono_init_outlier_t), \
               (eis_est_weights, eis_init_weights, eis_weight_factors, eis_init_outlier_t)

    def _prep_for_batch_fit(self, obs_indices):

        start_time = time.time()

        # Checks
        utils.validation.check_penalty_type(self.qphb_fit_kw['penalty_type'])

        # Get batch data
        psi_array = self.obs_psi_array[obs_indices]
        chrono_data_list = [self.chrono_data_list[i] for i in obs_indices]
        chrono_step_info_list = [self.chrono_step_info_list[i] for i in obs_indices]
        chrono_sample_index_list = [self.chrono_sample_index_list[i] for i in obs_indices]
        eis_data_list = [self.eis_data_list[i] for i in obs_indices]
        batch_size = psi_array.shape[0]

        # Check batch data
        utils.validation.check_md_data(psi_array, chrono_data_list, eis_data_list)

        # Get downsampled chrono data
        if self.downsample_chrono:
            chrono_sample_data_list = utils.md.get_sampled_chrono_data_list(chrono_data_list, chrono_sample_index_list)
        else:
            chrono_sample_data_list = chrono_data_list

        # Clear map samples
        self.map_samples = None
        self.map_sample_kw = None

        # Get observation indices
        self.chrono_batch_start_indices, self.chrono_batch_end_indices = \
            utils.md.get_data_obs_indices(chrono_sample_data_list)
        self.eis_batch_start_indices, self.eis_batch_end_indices = utils.md.get_data_obs_indices(eis_data_list,
                                                                                                 expand_factor=2)

        # Get parameter indices

        # Get matrix for chrono fit
        rm = matmd.construct_md_response_matrix(chrono_sample_data_list, chrono_step_info_list,
                                                self.step_model, self.basis_tau, self.tau_basis_type, self.tau_epsilon,
                                                self.special_qp_params, self.op_mode, self.integrate_method, 1000,
                                                self.zga_params, self.interpolate_lookups['response'],
                                                self.smooth_chrono_inf_response)

        # Get matrices for impedance fit
        frequency_list = [utils.md.get_data_tuple_item(eis_data, 0) for eis_data in eis_data_list]
        zm = matmd.construct_md_impedance_matrix(frequency_list, self.basis_tau, self.tau_basis_type, self.tau_epsilon,
                                                 self.special_qp_params, self.frequency_precision,
                                                 self.integrate_method, 1000, self.zga_params, self.interpolate_lookups)

        # Apply inductance scale
        if 'inductance' in self.special_qp_params.keys():
            rm[:, self.special_qp_indices['inductance']] *= self.qphb_fit_kw['inductance_scale']
            zm[:, self.special_qp_indices['inductance']] *= self.qphb_fit_kw['inductance_scale']

        # Calculate penalty matrices
        base_penalty_matrices, penalty_matrices = self._prep_penalty_matrices(batch_size)

        # # Store parameter indices
        # num_params = len(self.basis_tau) + self.get_qp_mat_offset()
        # self.parameter_obs_indices = np.arange(0, num_params * psi_array.shape[0], num_params, dtype=int)

        # Scale data
        scaled_chrono_data_list, scaled_eis_data_list = self.scale_batch_data(
            obs_indices, chrono_sample_data_list, eis_data_list, chrono_step_info_list
        )
        print('Finished signal scaling')

        # Scale each chrono response sub-matrix to corresponding observation input_signal_scale
        if rm.shape[0] > 0:
            for i, obs_index in enumerate(obs_indices):
                start_index = self.chrono_batch_start_indices[i]
                end_index = self.chrono_batch_end_indices[i]
                rm[start_index:end_index] = rm[start_index:end_index] / self.input_signal_scale_vector[obs_index]

        # Construct matrices for variance estimation
        eis_vmm = matmd.construct_md_eis_var_matrix(eis_data_list, self.qphb_fit_kw['eis_vmm_epsilon'],
                                                    self.qphb_fit_kw['eis_reim_cor'],
                                                    self.qphb_fit_kw['eis_error_structure'])
        chrono_vmm = matmd.construct_md_chrono_var_matrix(chrono_sample_data_list, chrono_step_info_list,
                                                          self.qphb_fit_kw['chrono_vmm_epsilon'],
                                                          self.qphb_fit_kw['chrono_error_structure'], )

        print('Finished prep_for_batch_fit in {:.2f} s'.format(time.time() - start_time))

        return (scaled_chrono_data_list, scaled_eis_data_list), \
               (rm, zm, chrono_vmm, eis_vmm, base_penalty_matrices, penalty_matrices)

    def _prep_penalty_matrices(self, num_obs):
        # Always calculate penalty matrices - fast, avoids any gaps in recalc logic
        base_penalty_matrices = {}
        penalty_matrices = {}
        for k in range(self.num_derivatives):
            # if self.qphb_fit_kw['penalty_type'] == 'discrete':
            #     dk = basis.construct_func_eval_matrix(np.log(self.basis_tau), None, self.tau_basis_type,
            #                                           self.tau_epsilon, k,
            #                                           zga_params=self.zga_params)
            #
            #     penalty_matrices[f'l{k}'] = dk.copy()
            if self.qphb_fit_kw['penalty_type'] == 'integral':
                dk_base, dk = matmd.construct_md_integrated_derivative_matrix(
                    num_obs, np.log(self.basis_tau), self.tau_basis_type, k, self.tau_epsilon, self.special_qp_params,
                    self.qphb_fit_kw['special_penalties'], zga_params=self.zga_params)

                base_penalty_matrices[f'm{k}'] = dk_base.copy()
                penalty_matrices[f'm{k}'] = dk.copy()

        self.fit_matrices.update(penalty_matrices)

        print('Constructed derivative matrices')
        return base_penalty_matrices, penalty_matrices

    def process_chrono_signals(self, chrono_data_list, step_time_list):
        sample_data_list = []
        sample_index_list = []
        step_info_list = []
        response_baseline_vector = np.zeros(len(chrono_data_list))

        for i, data in enumerate(chrono_data_list):
            if data is not None:
                # If chrono data provided, get input signal step information
                times, i_signal, v_signal = data
                if step_time_list is not None:
                    step_times = step_time_list[i]
                else:
                    step_times = None

                # Determine input signal from op_mode
                if self.op_mode == 'galvanostatic':
                    input_signal = i_signal
                else:
                    input_signal = v_signal

                # Determine step times and sizes in input signal
                if step_times is None:
                    # Step times not provided - determine from input signal
                    step_times, step_sizes, tau_rise = pp.process_input_signal(times, input_signal, self.step_model,
                                                                               self.offset_chrono_steps,
                                                                               fixed_tau_rise=self.tau_rise)
                else:
                    # Step times provided - only need to get step sizes
                    step_sizes = pp.get_step_sizes(times, input_signal, step_times)
                    tau_rise = None

                print('Step data:', step_times, step_sizes)
                print('Got step data')

                # Downsample data
                if self.downsample_chrono:
                    sample_times, sample_i, sample_v, sample_index = pp.downsample_data(times, i_signal, v_signal,
                                                                                        step_times=step_times,
                                                                                        op_mode=self.op_mode,
                                                                                        **self.downsample_chrono_kw)
                    print('Downsampled size:', len(sample_times))
                else:
                    sample_index = np.arange(0, len(times), 1, dtype=int)
                    sample_times = times.copy()
                    sample_i = i_signal.copy()
                    sample_v = v_signal.copy()

                input_signal, response_signal = utils.chrono.get_input_and_response(sample_i, sample_v, self.op_mode)
                response_baseline_vector[i] = np.mean(response_signal[sample_times < step_times[0]])

                sample_data_list.append((sample_times, sample_i, sample_v))
                sample_index_list.append(sample_index)
                step_info_list.append((step_times, step_sizes, tau_rise))
            else:
                # input_signal = None
                # response_signal = None
                sample_data_list.append(None)
                sample_index_list.append(None)
                step_info_list.append(None)

            # Set t_fit - must be done before setting raw_input_signal
            # self.set_chrono_fit_data(sample_time_list, input_signal_list, response_signal_list)

            # # Record sample index
            # self.chrono_sample_index_list = sample_index_list

            # # Store step data
            # self.step_time_list = deepcopy(step_time_list_out)
            # self.step_size_list = deepcopy(step_size_list)
            # self.tau_rise_list = deepcopy(tau_rise_list)

        return sample_data_list, sample_index_list, step_info_list, response_baseline_vector

    # ----------------
    # Scaling
    # ----------------
    def scale_batch_data(self, obs_indices, chrono_sample_data_list, eis_data_list, chrono_step_info_list):

        batch_size = len(obs_indices)

        scaled_chrono_data_list = []
        scaled_eis_data_list = []

        for i in range(batch_size):
            if chrono_sample_data_list[i] is None:
                times, i_signal, v_signal = None, None, None
                step_times, step_sizes, tau_rise = None, None, None
            else:
                times, i_signal, v_signal = chrono_sample_data_list[i]
                step_times, step_sizes, tau_rise = chrono_step_info_list[i]

            if eis_data_list[i] is None:
                frequencies, z = None, None
            else:
                frequencies, z = eis_data_list[i]

            i_scaled, v_scaled, z_scaled = self.scale_data(times, i_signal, v_signal, step_times, step_sizes, z,
                                                           self.qphb_fit_kw['scale_data'], self.qphb_fit_kw['rp_scale']
                                                           )

            self.coefficient_scale_vector[obs_indices[i]] = deepcopy(self.coefficient_scale)
            self.input_signal_scale_vector[obs_indices[i]] = deepcopy(self.input_signal_scale)
            self.response_signal_scale_vector[obs_indices[i]] = deepcopy(self.response_signal_scale)
            self.impedance_scale_vector[obs_indices[i]] = deepcopy(self.impedance_scale)

            # Scale chrono baseline
            scaled_chrono_baseline = self.chrono_baseline_vector[obs_indices[i]] / self.response_signal_scale
            self.scaled_chrono_baseline_vector[obs_indices[i]] = scaled_chrono_baseline

            # Subtract baseline
            if self.offset_chrono_baseline and times is not None:
                self.scaled_chrono_offset_vector[obs_indices[i]] = -scaled_chrono_baseline
                if self.op_mode == 'galvanostatic':
                    v_scaled -= scaled_chrono_baseline
                else:
                    i_scaled -= scaled_chrono_baseline
            else:
                self.scaled_chrono_offset_vector[obs_indices[i]] = 0

            if times is None:
                scaled_chrono_data_list.append(None)
            else:
                scaled_chrono_data_list.append((times, i_scaled, v_scaled))

            if frequencies is None:
                scaled_eis_data_list.append(None)
            else:
                scaled_eis_data_list.append((frequencies, z_scaled))

        # # Store scaled signal lists
        # if self.op_mode == 'galvanostatic':
        #     self.scaled_input_signal_list = deepcopy(i_scaled_list)
        #     self.scaled_response_signal_list = deepcopy(v_scaled_list)
        # else:
        #     self.scaled_input_signal_list = deepcopy(v_scaled_list)
        #     self.scaled_response_signal_list = deepcopy(i_scaled_list)

        # self.z_scaled_list = deepcopy(z_scaled_list)

        return scaled_chrono_data_list, scaled_eis_data_list

    def update_batch_data_scale(self, obs_indices, scale_factor_vector):
        self.coefficient_scale_vector[obs_indices] /= scale_factor_vector
        self.impedance_scale_vector[obs_indices] /= scale_factor_vector
        # self.eis_est_weight_list[obs_indices] /= scale_factor_vector
        # self.eis_init_weight_list[obs_indices] /= scale_factor_vector

        if self.op_mode == 'galvanostatic':
            self.response_signal_scale_vector[obs_indices] /= scale_factor_vector
            self.scaled_chrono_baseline_vector[obs_indices] *= scale_factor_vector
            self.scaled_chrono_offset_vector[obs_indices] *= scale_factor_vector
            # self.chrono_est_weight_list[obs_indices] /= scale_factor_vector
            # self.chrono_init_weight_list[obs_indices] /= scale_factor_vector
        else:
            # Not verified for potentiostatic mode
            self.response_signal_scale_vector[obs_indices] *= scale_factor_vector
            self.scaled_chrono_baseline_vector[obs_indices] /= scale_factor_vector
            self.scaled_chrono_offset_vector[obs_indices] /= scale_factor_vector
            # self.chrono_est_weight_list[obs_indices] /= scale_factor_vector
            # self.chrono_init_weight_list[obs_indices] /= scale_factor_vector

    # -------------------------
    # Prediction
    # -------------------------
    # def _prep_chrono_prediction_matrix(self, times, input_signal, op_mode, offset_steps, smooth_inf_response):
    #     # TODO: update for MD
    #     # If input signal is not provided, use self.raw_input_signal
    #     if input_signal is None:
    #         if self._t_fit_subset_index is not None:
    #             input_signal = self.raw_input_signal[self._t_fit_subset_index]
    #         else:
    #             input_signal = self.raw_input_signal
    #         use_fit_signal = True
    #     else:
    #         use_fit_signal = False
    #
    #     self.t_predict = times
    #     self.raw_prediction_input_signal = input_signal.copy()
    #     self.op_mode_predict = op_mode
    #
    #     print('recalc_response_prediction_matrix:', self._recalc_chrono_prediction_matrix)
    #
    #     if use_fit_signal:
    #         # Allow times to have a different length than input_signal if using fitted signal (input_signal = None)
    #         # This will break construct_inf_response_vector if smooth_inf_response==False
    #         step_times = self.step_times
    #         step_sizes = self.step_sizes
    #         tau_rise = self.tau_rise
    #     else:
    #         # Identify steps in applied signal
    #         step_times, step_sizes, tau_rise = pp.process_input_signal(times, input_signal, self.step_model,
    #                                                                    offset_steps)
    #
    #     if self._recalc_chrono_prediction_matrix:
    #         # Matrix recalculation is required
    #         rm, rm_layered = mat1d.construct_response_matrix(self.basis_tau, times, self.step_model, step_times,
    #                                                          step_sizes, basis_type=self.tau_basis_type,
    #                                                          epsilon=self.tau_epsilon, tau_rise=tau_rise,
    #                                                          op_mode=op_mode, zga_params=self.zga_params)
    #         if self.step_model == 'expdecay':
    #             # num_steps = len(step_times)
    #             # signal_fit = fit_signal_steps(times, input_signal)
    #             # # Get step time offsets
    #             # t_step_offset = signal_fit['x'][:num_steps] * 1e-6
    #             # step_times += t_step_offset
    #             # # Get rise times
    #             # tau_rise = np.exp(signal_fit['x'][num_steps:])
    #             induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times,
    #                                                                   step_sizes, tau_rise,
    #                                                                   op_mode)
    #         else:
    #             induc_rv = np.zeros(len(times))
    #         self.prediction_matrices = {'response': rm.copy(), 'inductance_response': induc_rv.copy()}
    #
    #         # With prediction matrices calculated, set recalc flag to False
    #         self._recalc_chrono_prediction_matrix = False
    #         print('Calculated response prediction matrices')
    #     elif self._t_predict_eq_t_fit:
    #         # times is the same as self.t_fit. Do not overwrite
    #         rm = self.fit_matrices['response'].copy()
    #         induc_rv = self.fit_matrices['inductance_response'].copy()
    #     elif self._t_predict_subset_index[0] == 'predict':
    #         # times is a subset of self.t_predict. Use sub-matrices of existing matrix; do not overwrite
    #         rm = self.prediction_matrices['response'][self._t_predict_subset_index[1], :].copy()
    #         induc_rv = self.prediction_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
    #     elif self._t_predict_subset_index[0] == 'fit':
    #         # times is a subset of self.t_fit. Use sub-matrices of existing matrix; do not overwrite
    #         rm = self.fit_matrices['response'][self._t_predict_subset_index[1], :].copy()
    #         induc_rv = self.fit_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
    #     else:
    #         # All matrix parameters are the same. Use existing matrix
    #         rm = self.prediction_matrices['response'].copy()
    #         induc_rv = self.prediction_matrices['inductance_response'].copy()
    #
    #     # Construct R_inf response vector
    #     inf_rv = mat1d.construct_inf_response_vector(times, self.step_model, step_times, step_sizes,
    #                                                  tau_rise, input_signal, smooth_inf_response,
    #                                                  op_mode)
    #
    #     self.prediction_matrices['inf_response'] = inf_rv.copy()
    #     print(len(induc_rv), len(inf_rv))
    #
    #     return rm, induc_rv, inf_rv

    def predict_response(self, chrono_data_list, chrono_step_info_list, psi_array=None, obs_indices=None, x=None,
                         model=None, split_matmul=True):
        # Get batch size
        batch_size = len(chrono_data_list)

        # Get full stacked matrix
        rm = matmd.construct_md_response_matrix(chrono_data_list, chrono_step_info_list,
                                                self.step_model, self.basis_tau, self.tau_basis_type, self.tau_epsilon,
                                                self.special_qp_params, self.op_mode, self.integrate_method, 1000,
                                                self.zga_params, self.interpolate_lookups['response'],
                                                self.smooth_chrono_inf_response)

        # Get parameters
        x = self.get_model_params(psi_array, obs_indices, x, model, None)

        # Ensure shape of x matches frequency_list
        if np.shape(x)[0] != batch_size:
            raise ValueError(f'Number of observations in chrono_data_list ({batch_size}) does not match number '
                             f'of apparent observations in x ({np.shape(x)[0]})')

        if split_matmul:
            # zm is a block diagonal matrix. Break into blocks for faster multiplication. Almost always faster
            obs_start, obs_end = utils.md.get_data_obs_indices(chrono_data_list)
            response_list = []
            for i in range(batch_size):
                rm_i = rm[obs_start[i]:obs_end[i], i * self.params_per_obs: (i + 1) * self.params_per_obs]
                r_i = rm_i @ x[i]
                response_list.append(r_i)
        else:
            # Multiply in single step. Less efficient for medium to large batches
            # Flatten parameters
            x_flat = np.array(x).flatten()

            # Get flattened output
            r_flat = rm @ x_flat

            # Split into list of vectors corresponding to frequency_list
            response_list = utils.md.reshape_vector_to_data(r_flat, chrono_data_list)

        return response_list

    def predict_impedance(self, frequency_list, psi_array=None, obs_indices=None, x=None, model=None,
                          split_matmul=True):
        # Get batch size
        batch_size = len(frequency_list)

        # Get full stacked matrix
        zm = matmd.construct_md_impedance_matrix(frequency_list, self.basis_tau, self.tau_basis_type, self.tau_epsilon,
                                                 self.special_qp_params, self.frequency_precision,
                                                 self.integrate_method, 1000, self.zga_params, self.interpolate_lookups)

        # Get parameters
        x = self.get_model_params(psi_array, obs_indices, x, model, None)

        # Ensure shape of x matches frequency_list
        if np.shape(x)[0] != batch_size:
            raise ValueError(f'Number of observations in frequency_list ({len(frequency_list)}) does not match number '
                             f'of apparent observations in x ({np.shape(x)[0]})')

        if split_matmul:
            # zm is a block diagonal matrix. Break into blocks for faster multiplication. Almost always faster
            obs_start, obs_end = utils.md.get_data_obs_indices(frequency_list, expand_factor=2)
            z_list = []
            for i in range(len(frequency_list)):
                zm_i = zm[obs_start[i]:obs_end[i], i * self.params_per_obs: (i + 1) * self.params_per_obs]
                z_i = zm_i @ x[i]
                # Convert concatenated vector to complex
                z_complex = utils.eis.concat_vector_to_complex(z_i)
                z_list.append(z_complex)
        else:
            # Multiply in single step. Less efficient for medium to large batches
            # Flatten parameters
            x_flat = np.array(x).flatten()

            # Get flattened output
            z_flat = zm @ x_flat

            # Split into list of vectors corresponding to frequency_list
            z_list = utils.md.reshape_vector_to_data(z_flat, frequency_list, expand_factor=2)

            # Convert concatenated vectors to complex vectors
            z_list = [utils.eis.concat_vector_to_complex(z) for z in z_list]

        return z_list

    def predict_distribution(self, tau, psi_array=None, obs_indices=None, x=None, model=None, quantiles=None):

        x = self.get_model_params(psi_array, obs_indices, x, model, quantiles)

        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            log_tau_min = np.min(np.log10(self.basis_tau)) - 1
            log_tau_max = np.max(np.log10(self.basis_tau)) + 1
            tau = np.logspace(log_tau_min, log_tau_max, int((log_tau_max - log_tau_min) * 20))


        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), np.log(tau), self.tau_basis_type,
                                                        self.tau_epsilon, 0, self.zga_params)

        # Get model parameters
        if quantiles is None:
            # Remove special parameters
            x = x[:, self.get_qp_mat_offset():]

            # Get 2d gamma array of size (batch_size, len(tau)))
            gamma = x @ basis_matrix.T
        else:
            # Remove special parameters
            x = x[:, :, self.get_qp_mat_offset():]

            gamma = np.empty((x.shape[0], x.shape[1], len(tau)))

            for i in range(len(quantiles)):
                gamma[i] = x[i] @ basis_matrix.T

        return gamma

    # def get_param_quantiles(self, quantiles, psi_array=None, obs_indices=None, model=None):
    #     model = self.check_param_spec(psi_array, obs_indices, None, model, quantiles)
    #
    #     if model == 'gp':
    #         x_quant = self.gpr.predict(psi_array, quantiles=quantiles)
    #     elif model == 'qphb':
    #         x_mean = self.parameter_array[obs_indices]
    #         x_sigma = np.sqrt(self.parameter_variance[obs_indices])
    #         s_quant = utils.stats.std_normal_quantile(quantiles)
    #
    #         x_quant = np.tile(x_mean, (len(quantiles), 1, 1)) + (
    #                 np.tile(x_sigma, (len(quantiles), 1, 1))
    #                 * np.tile(s_quant, (x_mean.shape[1], x_mean.shape[0], 1)).T
    #         )
    #
    #     return x_quant

    def get_model_params(self, psi_array, obs_indices, x, model, quantiles):
        # Ensure that one and only one x specification arg was provided
        utils.validation.check_md_x_spec(psi_array, obs_indices, x)

        # Check parameter specification
        model = self.check_param_spec(psi_array, obs_indices, x, model, quantiles)

        # Convert obs_indices to psi_array for GP prediction
        if obs_indices is not None and model == 'gp':
            psi_array = self.obs_psi_array[obs_indices]
            obs_indices = None

        # Get parameters
        if quantiles is None:
            if model == 'gp':
                x = self.gpr.predict(psi_array)
                # Apply inductance scale
                if 'inductance' in self.special_qp_params.keys():
                    x[:, self.special_qp_indices['inductance']] *= self.qphb_fit_kw['inductance_scale']
            elif model == 'qphb':
                x = self.parameter_array[obs_indices]
            else:
                x = self.reshape_batch_params(x, None)

            return x
        else:
            if model == 'gp':
                x_quant = self.gpr.predict(psi_array, quantiles=quantiles)
                # Apply inductance scale
                if 'inductance' in self.special_qp_params.keys():
                    x_quant[:, :, self.special_qp_indices['inductance']] *= self.qphb_fit_kw['inductance_scale']

            elif model == 'qphb':
                x_mean = self.parameter_array[obs_indices]
                x_sigma = np.sqrt(self.parameter_variance[obs_indices])
                s_quant = utils.stats.std_normal_quantile(quantiles)

                x_quant = np.tile(x_mean, (len(quantiles), 1, 1)) + (
                        np.tile(x_sigma, (len(quantiles), 1, 1))
                        * np.tile(s_quant, (x_mean.shape[1], x_mean.shape[0], 1)).T
                )
            return x_quant

    def check_param_spec(self, psi_array, obs_indices, x, model, quantiles):
        # Select model to use for prediction if not specified
        if model is None:
            # If psi_array specified, use GP. If obs_indices specified, use QPHB
            if psi_array is not None and self.gpr is not None:
                model = 'gp'
            elif obs_indices is not None:
                model = 'qphb'

        # Check compatibility of model type and parameter specification
        if model == 'gp' and self.gpr is None:
            raise ValueError('GP model is not initialized')
        elif model == 'qphb' and obs_indices is None:
            raise ValueError('Prediction with QPHB model can only be performed when obs_indices are provided')
        elif model is not None and x is not None:
            raise ValueError('model and x cannot be specified together')
        elif x is not None and quantiles is not None:
            raise ValueError('Cannot specify both x and quantiles')

        return model

    # -------------
    # Plotting
    # -------------
    def plot_distribution_1d(self, tau=None, psi_array=None, obs_indices=None, x=None, model=None, ax=None,
                             scale_prefix=None, plot_bounds=False, quantile=None, plot_ci=False,
                             ci_quantiles=[0.025, 0.975], colors=None, area=None,
                             **plot_kw):

        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            log_tau_min = np.min(np.log10(self.basis_tau)) - 1
            log_tau_max = np.max(np.log10(self.basis_tau)) + 1
            tau = np.logspace(log_tau_min, log_tau_max, int((log_tau_max - log_tau_min) * 20))

        # Get predicted distribution
        if quantile is not None:
            quantile = [quantile]
        gamma = self.predict_distribution(tau, psi_array, obs_indices, x, model, quantile)

        # Make/get figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        # Scale to appropriate magnitude
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(gamma.flatten())
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        if area is not None:
            scale_factor /= area

        batch_size = gamma.shape[0]

        # Plot distribution
        if colors is None:
            colors = [None] * batch_size
        lines = []
        for i in range(batch_size):
            line = ax.plot(tau, gamma[i] / scale_factor, color=colors[i], **plot_kw)
            lines.append(line)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (s)')

        if area is not None:
            ax.set_ylabel(fr'$\gamma$ ({scale_prefix}$\Omega \cdot \mathrm{{cm}}^2$)')
        else:
            ax.set_ylabel(fr'$\gamma$ ({scale_prefix}$\Omega$)')

        if plot_bounds:
            # Indicate measurement range ONLY if specific observation were specified
            if obs_indices is not None:
                for i, index in enumerate(obs_indices):
                    # Get min and max tau across all measurement types
                    eis_data = self.eis_data_list[index]
                    chrono_data = self.chrono_data_list[index]
                    chrono_step_info = self.chrono_step_info_list[index]

                    tau_min, tau_max = pp.get_tau_lim(utils.md.get_data_tuple_item(eis_data, 0),
                                                      utils.md.get_data_tuple_item(chrono_data, 0),
                                                      utils.md.get_data_tuple_item(chrono_step_info, 0)
                                                      )

                    # Indicate bounds with vertical lines
                    color = lines[i][0].get_color()
                    ax.axvline(tau_min, ls=':', color=color, alpha=0.6, lw=1.5, zorder=-1)
                    ax.axvline(tau_max, ls=':', color=color, alpha=0.6, lw=1.5, zorder=-1)

        if plot_ci:
            if len(ci_quantiles) != 2:
                raise ValueError('ci_quantiles must contain 2 quantile values')

            gamma_quant = self.predict_distribution(tau, psi_array, obs_indices, x, model, ci_quantiles)
            gamma_lo = gamma_quant[0]
            gamma_hi = gamma_quant[1]

            if self.qphb_fit_kw['nonneg']:
                gamma_lo = np.maximum(gamma_lo, 0)

            for i in range(batch_size):
                ax.fill_between(tau, gamma_lo[i] / scale_factor, gamma_hi[i] / scale_factor,
                                color=lines[i][0].get_color(), lw=0.5,
                                alpha=0.2, zorder=-10)

        fig.tight_layout()

        return ax

    def plot_distribution_2d(self, psi_array, psi_plot_index, tau=None, ax=None,
                             scale_prefix=None, quantile=None, colorbar=True, cb_kw=None,
                             log_scale=False, area=None,
                             **plot_kw):
        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            log_tau_min = np.min(np.log10(self.basis_tau)) - 1
            log_tau_max = np.max(np.log10(self.basis_tau)) + 1
            tau = np.logspace(log_tau_min, log_tau_max, int((log_tau_max - log_tau_min) * 20))

        # Make axis
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        # Get predicted distribution
        if quantile is not None:
            quantile = [quantile]
        gamma = self.predict_distribution(tau, psi_array, None, None, 'gp', quantile)

        # Get scale prefix
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(gamma)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        if area is not None:
            scale_factor /= area

        # Make tau-psi mesh
        tau_mesh, psi_mesh = np.meshgrid(tau, psi_array[:, psi_plot_index])

        if log_scale:
            # Make log norm
            norm = mpl.colors.LogNorm(vmin=plot_kw.get('vmin', None), vmax=plot_kw.get('vmax', None))
            for key in ['vmin', 'vmax']:
                if key in plot_kw.keys():
                    del plot_kw[key]
            cm = ax.pcolormesh(tau_mesh, psi_mesh, gamma / scale_factor, norm=norm, **plot_kw)
        else:
            cm = ax.pcolormesh(tau_mesh, psi_mesh, gamma / scale_factor, **plot_kw)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (s)')
        ax.set_ylabel(rf'$\psi_{psi_plot_index}$')

        if colorbar:
            if area is not None:
                units = r'$\Omega \cdot \mathrm{cm}^2$'
            else:
                units = r'$\Omega$'
            if cb_kw is None:
                cb_kw = {'aspect': 25}
            fig.colorbar(cm, ax=ax, label=fr'$\gamma$ ({scale_prefix}{units})', **cb_kw)

        fig.tight_layout()

        return ax

    def plot_distribution_surf(self, psi_array, psi_plot_index, tau=None, ax=None,
                               scale_prefix=None, quantile=None, colorbar=False,
                               log_scale=False,
                               **plot_kw):
        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if tau is None:
            log_tau_min = np.min(np.log10(self.basis_tau)) - 1
            log_tau_max = np.max(np.log10(self.basis_tau)) + 1
            tau = np.logspace(log_tau_min, log_tau_max, int((log_tau_max - log_tau_min) * 20))

        # Make axis
        if ax is None:
            fig = plt.figure(figsize=(4, 3.5))
            ax = fig.add_subplot(projection='3d')
        else:
            fig = ax.get_figure()

        # Get predicted distribution
        if quantile is not None:
            quantile = [quantile]
        gamma = self.predict_distribution(tau, psi_array, None, None, 'gp', quantile)

        # Get scale prefix
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(gamma)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Make tau-psi mesh
        tau_mesh, psi_mesh = np.meshgrid(tau, psi_array[:, psi_plot_index])

        if log_scale:
            # Make log norm
            norm = mpl.colors.LogNorm(vmin=plot_kw.get('vmin', None), vmax=plot_kw.get('vmax', None))
            for key in ['vmin', 'vmax']:
                if key in plot_kw.keys():
                    del plot_kw[key]
            cm = ax.plot_surface(np.log10(tau_mesh), psi_mesh, gamma / scale_factor, norm=norm, **plot_kw)
        else:
            cm = ax.plot_surface(np.log10(tau_mesh), psi_mesh, gamma / scale_factor, **plot_kw)

        # Manual tau ticks - set_xscale('log') doesn't work in 3D
        log_tau_min = np.log10(np.min(tau))
        log_tau_max = np.log10(np.max(tau))
        pwr_min = np.ceil(log_tau_min)
        pwr_max = np.floor(log_tau_max)
        num_ticks = 4
        tick_interval = int((pwr_max - pwr_min) / num_ticks)

        log_tau_ticks = np.arange(pwr_min, pwr_max, tick_interval, dtype=int)
        tau_tick_labels = [f'$10^{{{pwr}}}$' for pwr in log_tau_ticks]
        ax.set_xticks(log_tau_ticks)
        ax.set_xticklabels(tau_tick_labels)

        ax.set_xlabel(r'$\tau$ (s)')
        ax.set_ylabel(rf'$\psi_{psi_plot_index}$')
        ax.set_zlabel(fr'$\gamma$ ({scale_prefix}$\Omega$)')

        if colorbar:
            fig.colorbar(cm, ax=ax, label=fr'$\gamma$ ({scale_prefix}$\Omega$)')

        fig.tight_layout()

        return ax

    def plot_chrono_fit(self, chrono_data_list=None, chrono_step_info_list=None, psi_array=None, obs_indices=None,
                        x=None, model=None, transform_time=True, linear_time_axis=True, overlay=True, axes=None,
                        data_label='', data_kw=None,
                        scale_prefix=None, plot_data=True, **plot_kw):

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # If frequency_list not provided and obs_indices specified, use observation frequencies
        if chrono_data_list is None:
            if obs_indices is not None:
                chrono_data_list = [self.chrono_data_list[i] for i in obs_indices]
                chrono_step_info_list = [self.chrono_step_info_list[i] for i in obs_indices]
            else:
                raise ValueError('frequency_list must be provided if obs_indices is not provided')

        # Get model response
        r_hat_list = self.predict_response(chrono_data_list, chrono_step_info_list, psi_array, obs_indices, x, model)

        # Define axes
        if not overlay:
            ax_per_obs = 1

            if axes is None:
                ncol = 3
                col_width = 3.
                row_height = 2.75
                nax = ax_per_obs * len(chrono_data_list)
                nrow = int(np.ceil(nax / ncol))
                ncol = min(nax, ncol)
                fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * col_width, nrow * row_height))

        # Get data to plot
        if plot_data:
            r_list = [utils.chrono.get_input_and_response(utils.md.get_data_tuple_item(data, 1),
                                                          utils.md.get_data_tuple_item(data, 2), self.op_mode)[1]
                      for data in chrono_data_list]

        # Get scale prefix
        if scale_prefix is None:
            if plot_data:
                r_data_concat = np.concatenate(r_list)
                r_hat_concat = np.concatenate(r_hat_list)
                scale_prefix = utils.scale.get_common_scale_prefix([r_data_concat, r_hat_concat])
            else:
                r_hat_concat = np.concatenate(r_hat_list)
                scale_prefix = utils.scale.get_scale_prefix(r_hat_concat)

        # Plot model impedance and data (if requested)
        for i, r_hat in enumerate(r_hat_list):
            if not overlay:
                axes_i = axes.ravel()[i * ax_per_obs: (i + 1) * ax_per_obs]
                if ax_per_obs == 1:
                    axes_i = axes_i[0]
            else:
                axes_i = axes

            # Plot model response
            if len(r_hat) > 0:
                times = chrono_data_list[i][0]
                step_times = chrono_step_info_list[i][0]

                # Plot response only
                if self.op_mode == 'galvanostatic':
                    i_plot = None
                    v_plot = r_hat
                else:
                    i_plot = r_hat
                    v_plot = None

                axes_i = plot_chrono((times, i_plot, v_plot), self.op_mode, step_times, axes_i, 'plot', transform_time,
                                     linear_time_axis, scale_prefix, **plot_kw)

                # Plot data if requested
                if plot_data:
                    if self.op_mode == 'galvanostatic':
                        i_plot = None
                        v_plot = r_list[i]
                    else:
                        i_plot = r_list[i]
                        v_plot = None
                    axes_i = plot_chrono((times, i_plot, v_plot), self.op_mode, step_times, axes_i, 'scatter',
                                         transform_time, linear_time_axis, scale_prefix, label=data_label, **data_kw)
                if overlay:
                    axes = axes_i

        fig = np.atleast_1d(axes).ravel()[0].get_figure()
        fig.tight_layout()

        return axes

    def plot_eis_fit(self, frequency_list=None, z_list=None, psi_array=None, obs_indices=None, x=None, model=None,
                     plot_type='nyquist', bode_cols=['Zmod', 'Zphz'],
                     overlay=True, axes=None, data_label='', data_kw=None,
                     scale_prefix=None, plot_data=True, **plot_kw):

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # If frequency_list not provided and obs_indices specified, use observation frequencies
        if frequency_list is None:
            if obs_indices is not None:
                frequency_list = [utils.md.get_data_tuple_item(self.eis_data_list[i], 0) for i in obs_indices]
            else:
                raise ValueError('frequency_list must be provided if obs_indices is not provided')

        # Get model impedance
        z_hat_list = self.predict_impedance(frequency_list, psi_array, obs_indices, x, model)

        # Define axes
        if not overlay:
            if plot_type == 'nyquist':
                ax_per_obs = 1
            elif plot_type == 'bode':
                ax_per_obs = len(bode_cols)
            else:
                ax_per_obs = 1 + len(bode_cols)

            if axes is None:
                ncol = 3
                col_width = 3.
                row_height = 2.75
                nax = ax_per_obs * len(frequency_list)
                nrow = int(np.ceil(nax / ncol))
                ncol = min(nax, ncol)
                fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * col_width, nrow * row_height))

        # Get data to plot
        if plot_data:
            # If z_list not provided and obs_indices specified, use observation data
            if z_list is None and obs_indices is not None:
                z_list = [utils.md.get_data_tuple_item(self.eis_data_list[i], 1) for i in obs_indices]

        # Get scale prefix
        if scale_prefix is None:
            if plot_data:
                z_data_concat = np.concatenate(z_list)
                z_hat_concat = np.concatenate(z_hat_list)
                scale_prefix = utils.scale.get_common_scale_prefix([z_data_concat, z_hat_concat])
            else:
                z_hat_concat = np.concatenate(z_hat_list)
                scale_prefix = utils.scale.get_scale_prefix(z_hat_concat)

        # Plot model impedance and data (if requested)
        for i, z_hat in enumerate(z_hat_list):
            pred_df = utils.eis.construct_eis_df(frequency_list[i], z_hat)
            if not overlay:
                axes_i = axes.ravel()[i * ax_per_obs: (i + 1) * ax_per_obs]
                if ax_per_obs == 1:
                    axes_i = axes_i[0]
            else:
                axes_i = axes
            # Plot model impedance
            axes_i = plot_eis(pred_df, plot_type, axes=axes_i, plot_func='plot', scale_prefix=scale_prefix,
                              bode_cols=bode_cols, **plot_kw)

            # Plot data if requested
            if plot_data:
                data_df = utils.eis.construct_eis_df(frequency_list[i], z_list[i])
                axes_i = plot_eis(data_df, plot_type, axes=axes_i, scale_prefix=scale_prefix, label=data_label,
                                  bode_cols=bode_cols, **data_kw)
            if overlay:
                axes = axes_i

        fig = np.atleast_1d(axes).ravel()[0].get_figure()
        fig.tight_layout()

        return axes

    # ---------------------------------------
    # Parameter management
    # ---------------------------------------
    def set_special_parameter(self, parameter):
        """
        Method to declare a special parameter. If parameter does not already exist, insert default value into arrays as
        necessary. If parameter already exists, do nothing. Called by add_observations
        :param str parameter: parameter name
        """
        nonneg_dict = {
            'R_inf': True,
            'v_baseline': False,
            'inductance': True,
            'vz_offset': False
        }

        if parameter not in self.special_qp_params.keys():
            # Add new parameter
            new_index = self.get_qp_mat_offset()
            self.special_qp_params[parameter] = {'index': new_index, 'nonneg': nonneg_dict[parameter]}
            self.parameter_array = np.insert(self.parameter_array, new_index, 0, axis=1)
            self.scaled_parameter_array = np.insert(self.scaled_parameter_array, new_index, 0, axis=1)
            self.parameter_variance = np.insert(self.parameter_variance, new_index, 0, axis=1)

            for k in range(self.num_derivatives):
                self.s_arrays[k] = np.insert(self.s_arrays[k], new_index, self.qphb_fit_kw['s_0'], axis=1)

    def get_inactive_param_indices(self, chrono_data_list, eis_data_list):
        """
        Get indices of inactive special parameters for a batch of data
        :param list chrono_data_list: list of chrono datasets in batch
        :param eis_data_list: list of EIS datasets in batch
        """
        batch_size = len(chrono_data_list)
        obs_data_types = [utils.md.get_data_type(chrono_data_list[i], eis_data_list[i])
                          for i in range(batch_size)]
        inactive_indices = []
        for i in range(batch_size):
            inactive_params = self.inactive_parameters_for_data_type(obs_data_types[i])
            add_indices = [i * self.params_per_obs + self.special_qp_params[param]['index']
                           for param in inactive_params]
            inactive_indices += add_indices

        inactive_indices = np.array(inactive_indices)

        return inactive_indices

    def inactive_parameters_for_data_type(self, data_type):
        """
        Return names of parameters which are inactive for the given data type.
        :param str data_type: data type for which to return inactive parameters
        :return:
        """
        inactive_params = []
        if data_type == 'eis':
            for param in ['v_baseline', 'vz_offset']:
                if param in self.special_qp_params.keys():
                    inactive_params.append(param)
        elif data_type == 'chrono':
            if 'vz_offset' in self.special_qp_params.keys():
                inactive_params.append('vz_offset')
            if 'inductance' in self.special_qp_params.keys() and self.step_model == 'ideal':
                inactive_params.append('inductance')

        return inactive_params

    def extract_qphb_parameters(self, x):
        # Reshape parameter vector
        num_params = len(self.basis_tau) + self.get_qp_mat_offset()
        x_scaled_2d = np.reshape(x, (self.num_obs, num_params))

        # Apply parameter scales to get true (data-scale) parameter values
        x_2d = x_scaled_2d.copy()

        # DRT coefficients
        x_2d[:, self.get_qp_mat_offset():] *= self.coefficient_scale_vector[:, None]

        # Special parameters
        special_scales = {
            'R_inf': self.coefficient_scale_vector,
            'inductance': self.coefficient_scale_vector * self.inductance_scale,
            'v_baseline': self.response_signal_scale_vector,
            'vz_offset': np.ones(self.num_obs)
        }
        for key in self.special_qp_indices.keys():
            # Add offset to v_baseline before scaling
            if key == 'v_baseline':
                x_2d[:, self.special_qp_indices[key]] -= self.scaled_response_offset_vector
            # Scale by corresponding scale vector
            x_2d[:, self.special_qp_indices[key]] *= special_scales[key]

        # Split and place in dict
        fit_parameters = {
            'x_full_scaled': x_scaled_2d,  # all scaled parameters
            'x_full': x_2d,  # all parameters
            'x': x_2d[:, self.get_qp_mat_offset():]  # DRT coefficients only
        }

        for key in special_scales.keys():
            if key in self.special_qp_indices.keys():
                val = x_2d[:, self.special_qp_indices[key]]
            else:
                val = np.zeros(self.num_obs)
            fit_parameters[key] = val

        return fit_parameters

    # ---------------------------------------
    # Convenience properties
    # ---------------------------------------
    @property
    def special_qp_indices(self):
        return {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}

    @property
    def num_obs(self):
        if self.obs_psi_array is None:
            return 0
        else:
            return self.obs_psi_array.shape[0]

    # @property
    # def batch_num_eis(self):
    #     if self.obs_psi_array is None:
    #         return 0
    #     else:
    #         return self.eis_batch_end_indices[-1] - self.eis_batch_start_indices[0]
    #
    # @property
    # def batch_num_chrono(self):
    #     if self.obs_psi_array is None:
    #         return 0
    #     else:
    #         return self.chrono_batch_end_indices[-1] - self.chrono_batch_start_indices[0]

    @property
    def params_per_obs(self):
        return self.get_qp_mat_offset() + len(self.basis_tau)

    @property
    def num_derivatives(self):
        return len(self.qphb_fit_kw['derivative_weights'])

    def obs_vector_to_param_vector(self, vector):
        """Construct diagonal matrix"""
        return np.repeat(vector, self.params_per_obs)

    def obs_vector_to_diagonal(self, vector):
        """Construct diagonal matrix"""
        rep_vector = np.repeat(vector, self.params_per_obs)
        return np.diag(rep_vector)

    def reshape_batch_params(self, x, batch_size):
        x = np.array(x)
        if batch_size is None:
            if len(x.shape) == 1:
                # Check shape of provided x. Expect shape (batch_size, params_per_obs)
                # If 1d array received, attempt to reshape
                batch_size = int(len(x) / self.params_per_obs)
            else:
                batch_size = x.shape[0]

        try:
            return x.reshape((batch_size, self.params_per_obs))
        except ValueError:
            raise ValueError('Shape of x does not match model parameters and/or batch size. Expected either a 1d array '
                             f'of length {batch_size} * {self.params_per_obs} or a 2d array with shape '
                             f'({batch_size}, {self.params_per_obs}). Received array of shape {x.shape}')

    def unscale_batch_params(self, x, obs_indices, apply_offsets=True, scale_exponent=1):
        """
        Transform scaled parameters (model scale) to unscaled parameters (true scale)
        :param x: scaled parameteres
        :param obs_indices: indices of observations in batch
        :param apply_offsets: if True, apply parameter offsets (e.g. v_baseline). Should only be set to False for
        unscaling variance
        :param scale_exponent: power to which to raise applied scale factor. Set to 2 for variance
        :return:
        """
        x_unscaled = x.copy()

        if apply_offsets:
            if 'v_baseline' in self.special_qp_params.keys():
                x_unscaled[:, self.special_qp_indices['v_baseline']] -= self.scaled_chrono_offset_vector[obs_indices]

        if 'inductance' in self.special_qp_params.keys():
            x_unscaled[:, self.special_qp_indices['inductance']] *= \
                self.qphb_fit_kw['inductance_scale'] ** scale_exponent

        if 'v_baseline' in self.special_qp_params.keys():
            # v_baseline must be scaled by response_signal_scale
            x_unscaled[:, self.special_qp_indices['v_baseline']] *= \
                self.response_signal_scale_vector[obs_indices] ** scale_exponent
            # All other parameters are scaled by coefficient_scale
            x_unscaled[:, :self.special_qp_indices['v_baseline']] *= \
                self.coefficient_scale_vector[obs_indices, None] ** scale_exponent
            x_unscaled[:, self.special_qp_indices['v_baseline'] + 1:] *= \
                self.coefficient_scale_vector[obs_indices, None] ** scale_exponent
        else:
            x_unscaled *= self.coefficient_scale_vector[obs_indices, None] ** scale_exponent

        return x_unscaled

    def get_batch_rp_vector(self, x, batch_size):
        x_2d = self.reshape_batch_params(x, batch_size)
        x_drt = x_2d[:, self.get_qp_mat_offset():]
        return np.sum(x_drt, axis=1) * np.sqrt(np.pi) / self.tau_epsilon

    # def get_fit_frequencies(self):
    #     return np.concatenate(self.fit_frequency_list)
    #
    # def get_fit_times(self):
    #     return np.concatenate(self.fit_time_list)

    # ------------------------
    # Getters and setters
    # ------------------------
    # def set_chrono_fit_data(self, time_list, input_signal_list, response_signal_list):
    #     if hasattr(self, 'fit_time_list'):
    #         # Check if chrono data has changed
    #         if not utils.array.check_equality(
    #                 [utils.array.rel_round(t, self.time_precision) for t in self.fit_time_list],
    #                 [utils.array.rel_round(t, self.time_precision) for t in time_list]
    #         ) or not utils.array.check_equality(
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in self.fit_input_signal_list],
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in input_signal_list]
    #         ):
    #             # If time or input signal list has changed, recalculate matrix.
    #             # Logic to reuse matrices is complex, may implement in future
    #             self.fit_time_list = time_list
    #             self.fit_input_signal_list = input_signal_list
    #             self._recalc_chrono_fit_matrix = True
    #     else:
    #         self.fit_time_list = time_list
    #         self.fit_input_signal_list = input_signal_list
    #         self._recalc_chrono_fit_matrix = True
    #
    #     # Store response signals - doesn't affect matrix recalculation
    #     self.fit_response_signal_list = response_signal_list
    #
    # def set_chrono_predict_data(self, time_list, input_signal_list):
    #     self._t_predict_eq_t_fit = False
    #     # Check if prediction chrono data matches fit chrono data
    #     if utils.array.check_equality(
    #             [utils.array.rel_round(t, self.time_precision) for t in self.fit_time_list],
    #             [utils.array.rel_round(t, self.time_precision) for t in time_list]
    #     ) and utils.array.check_equality(
    #         [utils.array.rel_round(s, self.input_signal_precision) for s in self.fit_input_signal_list],
    #         [utils.array.rel_round(s, self.input_signal_precision) for s in input_signal_list]
    #     ):
    #         self._t_predict_eq_t_fit = True
    #     elif hasattr(self, 'predict_time_list'):
    #         # Check if prediction chrono data has changed
    #         if not utils.array.check_equality(
    #                 [utils.array.rel_round(t, self.time_precision) for t in self.predict_time_list],
    #                 [utils.array.rel_round(t, self.time_precision) for t in time_list]
    #         ) or not utils.array.check_equality(
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in self.predict_input_signal_list],
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in input_signal_list]
    #         ):
    #             # If time or input signal list has changed, recalculate matrix.
    #             # Logic to reuse matrices is complex, may implement in future
    #             self.predict_time_list = time_list
    #             self.predict_input_signal_list = input_signal_list
    #             self._recalc_chrono_prediction_matrix = True
    #     else:
    #         self.predict_time_list = time_list
    #         self.predict_input_signal_list = input_signal_list
    #         self._recalc_chrono_prediction_matrix = True

    # def get_fit_time_list(self):
    #     return self._fit_time_list
    # 
    # def set_fit_time_list(self, time_list):
    #     if hasattr(self, 'fit_time_list'):
    #         # Check if time_list has changed
    #         if not utils.array.check_equality([utils.array.rel_round(t, self.time_precision) for t in self.fit_time_list],
    #                                     [utils.array.rel_round(t, self.time_precision) for t in time_list]):
    #             # If time list has changed, recalculate matrix.
    #             # Logic to reuse matrices is complex, may implement in future
    #             self.fit_time_list = time_list
    #             self._recalc_chrono_fit_matrix = True
    #     else:
    #         self.fit_time_list = time_list
    #         self._recalc_chrono_fit_matrix = True
    # 
    # fit_time_list = property(get_fit_time_list, set_fit_time_list)
    # 
    # def get_predict_time_list(self):
    #     return self._predict_time_list
    # 
    # def set_predict_time_list(self, time_list):
    #     self._t_predict_eq_t_fit = False
    #     # Check if time_list matches fit_time_list
    #     if utils.array.check_equality(
    #             [utils.array.rel_round(t, self.time_precision) for t in self.fit_time_list],
    #             [utils.array.rel_round(t, self.time_precision) for t in time_list]
    #     ):
    #         self._t_predict_eq_t_fit = True
    #     elif hasattr(self, 'predict_time_list'):
    #         # Check if time_list matches predict_time_list
    #         if not utils.array.check_equality(
    #                 [utils.array.rel_round(t, self.time_precision) for t in self.predict_time_list],
    #                 [utils.array.rel_round(t, self.time_precision) for t in time_list]):
    #             # If time list has changed, recalculate matrix.
    #             # Logic to reuse matrices is complex, may implement later
    #             self.predict_time_list = time_list
    #             self._recalc_chrono_prediction_matrix = True
    #     else:
    #         self.predict_time_list = time_list
    #         self._recalc_chrono_prediction_matrix = True
    # 
    # predict_time_list = property(get_predict_time_list, set_predict_time_list)
    # 
    # def get_fit_raw_input_signal_list(self):
    #     return self._fit_raw_input_signal_list
    # 
    # def set_fit_raw_input_signal_list(self, raw_input_signal_list):
    #     if hasattr(self, 'fit_raw_input_signal_list'):
    #         # Check if raw_input_signal_list has changed
    #         if not utils.array.check_equality(
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in self.fit_raw_input_signal_list],
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in raw_input_signal_list]
    #         ):
    #             # If raw_input_signal list has changed, recalculate matrix.
    #             # Logic to reuse matrices is complex, may implement in future
    #             self.fit_raw_input_signal_list = raw_input_signal_list
    #             self._recalc_chrono_fit_matrix = True
    #     else:
    #         self.fit_raw_input_signal_list = raw_input_signal_list
    #         self._recalc_chrono_fit_matrix = True
    # 
    # fit_raw_input_signal_list = property(get_fit_raw_input_signal_list, set_fit_raw_input_signal_list)
    # 
    # def get_predict_raw_input_signal_list(self):
    #     return self._predict_raw_input_signal_list
    # 
    # def set_predict_raw_input_signal_list(self, raw_input_signal_list):
    #     
    #     if self._t_predict_eq_t_fit:
    #         # Check if raw_input_signal_list matches fit_raw_input_signal_list
    #         if not utils.array.check_equality(
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in self.fit_raw_input_signal_list],
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in raw_input_signal_list]
    #         ):
    #             self._t_predict_eq_t_fit = False
    #     elif hasattr(self, 'predict_raw_input_signal_list'):
    #         # Check if raw_input_signal_list matches predict_raw_input_signal_list
    #         if not utils.array.check_equality(
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in self.predict_raw_input_signal_list],
    #                 [utils.array.rel_round(s, self.input_signal_precision) for s in raw_input_signal_list]):
    #             # If raw_input_signal list has changed, recalculate matrix.
    #             # Logic to reuse matrices is complex, may implement later
    #             self.predict_raw_input_signal_list = raw_input_signal_list
    #             self._recalc_chrono_prediction_matrix = True
    #     else:
    #         self.predict_raw_input_signal_list = raw_input_signal_list
    #         self._recalc_chrono_prediction_matrix = True
    # 
    # predict_raw_input_signal_list = property(get_predict_raw_input_signal_list, set_predict_raw_input_signal_list)

    # def get_fit_frequency_list(self):
    #     return self._fit_frequency_list
    #
    # def set_fit_frequency_list(self, frequency_list):
    #     if hasattr(self, 'fit_frequency_list'):
    #         # Check if frequency_list has changed
    #         if not utils.array.check_equality([utils.array.rel_round(f, self.frequency_precision) for f in self.fit_frequency_list],
    #                                     [utils.array.rel_round(f, self.frequency_precision) for f in frequency_list]):
    #             # If frequency list has changed, recalculate matrix.
    #             # Recalculating EIS matrix is fast and logic to reuse matrices is complex
    #             self._fit_frequency_list = frequency_list
    #             self._recalc_eis_fit_matrix = True
    #     else:
    #         self._fit_frequency_list = frequency_list
    #         self._recalc_eis_fit_matrix = True
    #
    # fit_frequency_list = property(get_fit_frequency_list, set_fit_frequency_list)
    #
    # def get_predict_frequency_list(self):
    #     return self._predict_frequency_list
    #
    # def set_predict_frequency_list(self, frequency_list):
    #     self._f_predict_eq_f_fit = False
    #     # Check if frequency_list matches fit_frequency_list
    #     if utils.array.check_equality(
    #             [utils.array.rel_round(f, self.frequency_precision) for f in self.fit_frequency_list],
    #             [utils.array.rel_round(f, self.frequency_precision) for f in frequency_list]):
    #         self._f_predict_eq_f_fit = True
    #     elif hasattr(self, 'predict_frequency_list'):
    #         # Check if frequency_list matches predict_frequency_list
    #         if not utils.array.check_equality(
    #                 [utils.array.rel_round(f, self.frequency_precision) for f in self.predict_frequency_list],
    #                 [utils.array.rel_round(f, self.frequency_precision) for f in frequency_list]):
    #             # If frequency list has changed, recalculate matrix.
    #             # Recalculating EIS matrix is fast and logic to reuse matrices is complex
    #             self._predict_frequency_list = frequency_list
    #             self._recalc_eis_prediction_matrix = True
    #     else:
    #         self._predict_frequency_list = frequency_list
    #         self._recalc_eis_prediction_matrix = True
    #
    # predict_frequency_list = property(get_predict_frequency_list, set_predict_frequency_list)

    def get_qphb_fit_kw(self):
        return self._qphb_fit_kw

    def update_qphb_fit_kw(self, **qphb_fit_kw):
        # Reset init weights, est weights, and outlier_t to ensure that batch weights are re-initialized
        # with new qphb kw
        reset_list = [None] * self.num_obs
        self.chrono_est_weight_list = reset_list
        self.chrono_init_weight_list = reset_list
        self.chrono_init_outlier_t_list = reset_list
        # self.chrono_outlier_t_list = []
        self.eis_est_weight_list = reset_list
        self.eis_init_weight_list = reset_list
        self.eis_init_outlier_t_list = reset_list

        self._qphb_fit_kw.update(qphb_fit_kw)

    qphb_fit_kw = property(get_qphb_fit_kw, update_qphb_fit_kw)
