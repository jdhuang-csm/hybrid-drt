import numpy as np
from numpy import ndarray
import warnings
from typing import Union, Optional, Callable

from .drt1d import DRT
from . import background, qphb
from ..matrices import mat1d, basis
from .. import utils, evaluation, preprocessing as pp


class NonlinearDRT(DRT):
    def extract_qphb_parameters(self, x):
        M = int(len(x) / 2)
        x_n, x_p = np.split(x, 2)
        param_n = super().extract_qphb_parameters(x_n)
        param_p = super().extract_qphb_parameters(x_p)
        param_n = {f'{k}_neg': v for k, v in param_n.items()}
        param_p = {f'{k}_pos': v for k, v in param_p.items()}
        return param_n | param_p
    
    def get_linear_params(self, kind: str):
        options = ['neg', 'pos']
        if kind not in options:
            raise ValueError('Received invalid kind argument {kind}. Valid options: {options}')
        
        # Get parameters for specified bias
        params = {
            '_'.join(k.split('_')[:-1]): v 
            for k , v in self.fit_parameters.items() if k.split('_')[-1] == kind
        }
        
        return params
    
    def to_linear(self, kind: str):
        options = ['neg', 'pos', 'mean']
        if kind not in options:
            raise ValueError('Received invalid kind argument {kind}. Valid options: {options}')
        
        # Get parameters for specified bias
        if kind == 'mean':
            neg_params = self.get_linear_params('neg')
            pos_params = self.get_linear_params('pos')
            params = {k: 0.5 * (neg_params[k] + pos_params[k]) for k in neg_params.keys()}
        else:
            params = self.get_linear_params(kind)
            
        
        # Create linear DRT instance
        lin_drt = DRT(interpolate_integrals=False)
        lin_drt.set_attributes(self.get_attributes('all'))
        
        # Update with single-bias parameters
        lin_drt.fit_parameters = params
        
        return lin_drt
    
    # def _prep_chrono_prediction_matrix(self, times, input_signal, step_times, step_sizes,
    #                                    op_mode, offset_steps, smooth_inf_response):
        
    #     rm_drt, induc_rv, inf_rv, cap_rv, rm_dop = self._prep_chrono_prediction_matrix(
    #         times, input_signal, step_times, step_sizes, op_mode,
    #         offset_steps, smooth_inf_response
    #     )
        
    #     rm_drt = np.concatenate()
        
    def predict_response(self, kind: str = None, 
                         times=None, input_signal=None, step_times=None, step_sizes=None, op_mode=None,
                         offset_steps=None,
                         smooth_inf_response=None, x=None, include_vz_offset=True, subtract_background=True,
                         y_bkg=None, v_baseline=None):
        options = ['net', 'mean', 'neg', 'pos']
        if kind is not None:
            if kind not in options:
                raise ValueError('Received invalid kind argument {kind}. Valid options: {options}')
        else:
            if input_signal is None and step_sizes is None:
                # Predict fitted signal, use net
                kind = 'net'
            else:
                # Predict new signal, use mean (can't use net)
                kind = 'mean'
                
        pred_kw = dict(
            times=times, input_signal=input_signal,
            step_times=step_times, step_sizes=step_sizes,
            op_mode=op_mode, offset_steps=offset_steps,
            smooth_inf_response=smooth_inf_response, x=x, 
            include_vz_offset=include_vz_offset, 
            subtract_background=subtract_background,
            y_bkg=y_bkg, v_baseline=v_baseline
        )
        
        if kind == 'net':
            # Predict response for fitted signal
            ndrt = self.to_linear('neg')
            pdrt = self.to_linear('pos')
            pw = self.nonlin_chrono_weights
            nw = 1 - pw
            
            rv_neg = ndrt.predict_response(**pred_kw)
            rv_pos = pdrt.predict_response(**pred_kw)
            
            return rv_neg * nw + rv_pos * pw
        else:
            lin_drt = self.to_linear(kind)
            return lin_drt.predict_response(**pred_kw)
            
        
    # def predict_response(self, times=None, input_signal=None, step_times=None, step_sizes=None, op_mode=None,
    #                      offset_steps=None,
    #                      smooth_inf_response=None, x=None, include_vz_offset=True, subtract_background=True,
    #                      y_bkg=None, v_baseline=None):
    #     # If chrono_mode is not provided, use fitted chrono_mode
    #     if op_mode is None:
    #         op_mode = self.chrono_mode
    #     utils.validation.check_ctrl_mode(op_mode)

    #     # If times is not provided, use self.t_fit
    #     if times is None:
    #         times = self.get_fit_times()

    #     # If kwargs not provided, use same values used in fitting
    #     if offset_steps is None:
    #         offset_steps = self.fit_kwargs['offset_steps']
    #     if smooth_inf_response is None:
    #         smooth_inf_response = self.fit_kwargs['smooth_inf_response']

    #     # Get prediction matrix and vectors
    #     rm_drt, induc_rv, inf_rv, cap_rv, rm_dop = self._prep_chrono_prediction_matrix(times, input_signal,
    #                                                                            step_times, step_sizes, op_mode,
    #                                                                            offset_steps, smooth_inf_response)
    #     # Response matrices from _prep_response_prediction_matrix will be scaled. Rescale to data scale
    #     # rm *= self.input_signal_scale
    #     # induc_rv *= self.input_signal_scale
    #     # inf_rv *= self.input_signal_scale

    #     # Get parameters
    #     if x is not None:
    #         fit_parameters = self.extract_qphb_parameters(x)
    #     else:
    #         fit_parameters = self.fit_parameters

    #     x_drt = fit_parameters['x']
    #     x_dop = fit_parameters.get('x_dop', None)
    #     r_inf = fit_parameters.get('R_inf', 0)
    #     induc = fit_parameters.get('inductance', 0)
    #     c_inv = fit_parameters.get('C_inv', 0)

    #     if v_baseline is None:
    #         v_baseline = fit_parameters.get('v_baseline', 0)

    #     response = rm_drt @ x_drt + inf_rv * r_inf + induc * induc_rv + c_inv * cap_rv

    #     if x_dop is not None:
    #         response += rm_dop @ x_dop

    #     # if not subtract_background:
    #     #     if not np.array_equal(times, self.get_fit_times()):
    #     #         raise ValueError('Background can only be included if prediction times are same as fit times')
    #     #     response += self.raw_response_background

    #     if include_vz_offset:
    #         # # Need to back out the offset to get the fit parameter, apply offset to fit parameter by offset
    #         # v_baseline_param = (v_baseline + self.scaled_response_offset * self.response_signal_scale)
    #         # v_baseline_offset = v_baseline_param * (1 + self.fit_parameters.get('vz_offset', 0)) \
    #         #                     - self.scaled_response_offset * self.response_signal_scale
    #         # Apply vz_offset before adding baseline
    #         vz_strength_vec, _ = self._get_vz_strength_vec(
    #             times, vz_offset_eps=self.fit_parameters.get('vz_offset_eps', None)
    #         )
    #         response *= (1 + fit_parameters.get('vz_offset', 0) * vz_strength_vec)

    #     response += v_baseline

    #     if not subtract_background:
    #         if y_bkg is None:
    #             y_bkg = self.predict_chrono_background(times)
    #         if len(times) != len(y_bkg):
    #             raise ValueError('Length of background does not match length of times')
    #         response += y_bkg

    #     return response
        
    def _qphb_fit_core(self, times, i_signal, v_signal, frequencies, z, step_times=None,
                       nonneg=True, series_neg=False, scale_data=True, update_scale=False, solve_rp=False,
                       # Nonlinear args
                       nonlin_function: Union[str, Callable] = 'v_exp', nl_lambda_0=100,
                       # chrono args
                       offset_steps=True, offset_baseline=True, downsample=False, downsample_kw=None,
                       subtract_background=False, background_type='static', background_corr_power=None,
                       estimate_background_kw=None, smooth_inf_response=True,
                       # penalty settings
                       v_baseline_penalty=1e-6, ohmic_penalty=1e-6,
                       inductance_penalty=1e-6, capacitance_penalty=1e-6,
                       inductance_scale=1e-5, capacitance_scale=1e-3,
                       background_penalty=1,
                       penalty_type='integral',
                       remove_extremes=False, extreme_kw=None,
                       # error structure
                       chrono_error_structure='uniform', eis_error_structure=None,
                       remove_outliers=False, return_outlier_index=False, outlier_thresh=0.75,
                       chrono_vmm_epsilon=4, eis_vmm_epsilon=0.25, eis_reim_cor=0.25,
                       iw_l1_lambda_0=1e-4, iw_l2_lambda_0=1e-4,
                       # Hybrid settings
                       vz_offset=True, vz_offset_scale=1, vz_offset_eps=1,
                       eis_weight_factor=None, chrono_weight_factor=None,
                       hybrid_weight_factor_method=None,
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
        
        if times is None:
            raise ValueError('Nonlinear DRT fit is only applicable to hybrid or chrono data')

        # Check inputs
        utils.validation.check_chrono_data(times, i_signal, v_signal)
        utils.validation.check_eis_data(frequencies, z)
        for err_struct in [chrono_error_structure, eis_error_structure]:
            utils.validation.check_error_structure(err_struct)
        utils.validation.check_penalty_type(penalty_type)

        if solve_rp and not scale_data and self.warn:
            warnings.warn('solve_rp is ignored if scale_data=False')
        if (self.fit_dop or not nonneg) and not solve_rp and self.warn:
            warnings.warn('For best results, set solve_rp=True when performing DRT-DOP fits '
                          'or DRT fits without a non-negativity constraint')

        if series_neg and not nonneg:
            raise ValueError('Only one of series_neg and nonneg may be True')

        background_types = ['static', 'dynamic', 'scaled']
        if background_type not in background_types:
            raise ValueError(f"Invalid background_type {background_type}. Options: {background_types}")

        if remove_outliers and 'outlier_p' not in kw.keys():
            raise ValueError('If remove_outliers is True, the prior probability of outlier presence, outlier_p, '
                             'must be specified. A good starting value might be 0.01-0.05')

        # Copy data
        if times is not None:
            times = np.array(times)
            i_signal = np.array(i_signal)
            v_signal = np.array(v_signal)
        if frequencies is not None:
            frequencies = np.array(frequencies)
            z = np.array(z)

        # Find and remove extreme values (rough)
        if remove_extremes:
            if extreme_kw is None:
                extreme_kw = {'qr_size': 0.8, 'qr_thresh': 1.5}
            if times is not None:
                i_flag = pp.identify_extreme_values(i_signal, **extreme_kw)
                v_flag = pp.identify_extreme_values(v_signal, **extreme_kw)
                chrono_flag = i_flag | v_flag
                if np.sum(chrono_flag) > 0:
                    if self.warn:
                        warnings.warn(f'Identified extreme values in chrono data at the following '
                                      f'indices: {np.where(chrono_flag)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )
                    times = times[~chrono_flag]
                    i_signal = i_signal[~chrono_flag]
                    v_signal = v_signal[~chrono_flag]
            if frequencies is not None:
                re_flag = pp.identify_extreme_values(z.real, **extreme_kw)
                im_flag = pp.identify_extreme_values(z.imag, **extreme_kw)
                eis_flag = re_flag | im_flag
                if np.sum(eis_flag) > 0:
                    if self.warn:
                        warnings.warn(f'Identified extreme values in EIS data at the following '
                                      f'indices: {np.where(eis_flag)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )
                    frequencies = frequencies[~eis_flag]
                    z = z[~eis_flag]

        # Find and remove outliers (more precise)
        if remove_outliers:
            # Only need to supply kwargs that matter for initial weights estimation
            chrono_outlier_index, eis_outlier_index = self._qphb_fit_core(times, i_signal, v_signal, frequencies, z,
                                                                          step_times=step_times, nonneg=nonneg,
                                                                          series_neg=series_neg, scale_data=scale_data,
                                                                          solve_rp=solve_rp, offset_steps=offset_steps,
                                                                          offset_baseline=offset_baseline,
                                                                          downsample=downsample,
                                                                          downsample_kw=downsample_kw,
                                                                          subtract_background=False,
                                                                          smooth_inf_response=smooth_inf_response,
                                                                          v_baseline_penalty=v_baseline_penalty,
                                                                          ohmic_penalty=ohmic_penalty,
                                                                          inductance_penalty=inductance_penalty,
                                                                          capacitance_penalty=capacitance_penalty,
                                                                          background_penalty=background_penalty,
                                                                          inductance_scale=inductance_scale,
                                                                          capacitance_scale=capacitance_scale,
                                                                          penalty_type=penalty_type,
                                                                          chrono_error_structure=chrono_error_structure,
                                                                          eis_error_structure=eis_error_structure,
                                                                          remove_outliers=False,
                                                                          return_outlier_index=True,
                                                                          outlier_thresh=outlier_thresh,
                                                                          chrono_vmm_epsilon=chrono_vmm_epsilon,
                                                                          eis_vmm_epsilon=eis_vmm_epsilon,
                                                                          eis_reim_cor=eis_reim_cor, eff_hp=eff_hp,
                                                                          **kw)

            self.eis_outlier_index = eis_outlier_index
            self.chrono_outlier_index = chrono_outlier_index

            # Preserve step times determined before outlier removal
            step_times = self.step_times

            if times is not None:
                if np.sum(chrono_outlier_index) > 0:
                    if self.warn:
                        warnings.warn('Found outliers in chrono data at the following '
                                      f'indices: {np.where(chrono_outlier_index)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )

                    # Store outliers for reference
                    self.chrono_outliers = (
                        times[chrono_outlier_index],
                        i_signal[chrono_outlier_index],
                        v_signal[chrono_outlier_index]
                    )

                    # Remove outliers from data to fit
                    times = times[~chrono_outlier_index]
                    i_signal = i_signal[~chrono_outlier_index]
                    v_signal = v_signal[~chrono_outlier_index]
                else:
                    self.chrono_outliers = None
            if frequencies is not None:
                if np.sum(eis_outlier_index) > 0:
                    if self.warn:
                        warnings.warn('Found outliers in EIS data at the following '
                                      f'indices: {np.where(eis_outlier_index)[0].tolist()}. '
                                      f'These data points will be removed before fitting'
                                      )

                    # Store outliers for reference
                    self.eis_outliers = (
                        frequencies[eis_outlier_index],
                        z[eis_outlier_index]
                    )

                    # Remove outliers from data to fit
                    frequencies = frequencies[~eis_outlier_index]
                    z = z[~eis_outlier_index]
                else:
                    self.eis_outliers = None

            # After identifying outliers, disable outlier error structure
            kw['outlier_p'] = None
        else:
            self.eis_outlier_index = None
            self.eis_outliers = None
            self.chrono_outlier_index = None
            self.chrono_outliers = None

        # Subtract chrono background
        if subtract_background and times is not None:
            if estimate_background_kw is None:
                estimate_background_kw = {}
            estimate_background_defaults = {
                'step_times': step_times,
                'nonneg': nonneg, 'series_neg': series_neg,
                'downsample': downsample, 'downsample_kw': downsample_kw
            }
            estimate_background_kw = dict(estimate_background_defaults, **estimate_background_kw)

            if estimate_background_kw.get('bkg_iter', 1) > 1:
                raise ValueError('When fitting with background subtraction, bkg_iter must be set to 1. If there'
                                 'are multiple background length scales, set kernel_size > 1')

            drt_bkg, bkg_gps, y_bkg = self.estimate_chrono_background(times, i_signal, v_signal, copy_self=True,
                                                                      **estimate_background_kw)
            if self.print_diagnostics:
                print('Finished initial background estimation')
            y_pred_bkg = drt_bkg.predict_response()

            if background_corr_power is None and background_type != 'static':
                bkg_std = np.std(y_bkg)
                v_std = np.std(y_pred_bkg)
                std_ratio = bkg_std / v_std
                print('std ratio:', std_ratio)
                # background_corr_power = 1.1 * np.exp(-5 * std_ratio)
                # background_corr_power = 1.25 * np.exp(-10 * std_ratio) + 0.25
                background_corr_power = np.log(0.02 / std_ratio + 1) + 0.25
                print('background_corr_power:', background_corr_power)
                # background_corr_power = 1  # Default: penalty for dynamic background

            self.background_gp = bkg_gps[0]
            if background_type == 'static':
                if background_corr_power is not None:
                    rm_bkg = background.get_background_matrix(bkg_gps, drt_bkg.get_fit_times()[:, None],
                                                              y_drt=y_pred_bkg,
                                                              corr_power=background_corr_power)
                    y_resid = drt_bkg.raw_response_signal - y_pred_bkg
                    self.raw_response_background = rm_bkg @ y_resid
                else:
                    self.raw_response_background = y_bkg.copy()

                # Subtract the background from the signal to fit
                if self.chrono_mode == 'galv':
                    v_signal[drt_bkg.sample_index] -= self.raw_response_background
                else:
                    i_signal[drt_bkg.sample_index] = i_signal[drt_bkg.sample_index] - self.raw_response_background
        else:
            bkg_gps = None
            y_pred_bkg = None
            self.background_gp = None

        # Store series_neg
        self.series_neg = series_neg

        # Determine data type
        # Get num_eis. num_chrono must be determined after downsampling
        if times is None:
            data_type = 'eis'
            num_eis = len(frequencies)
        elif frequencies is None:
            data_type = 'chrono'
            num_eis = 0
        else:
            data_type = 'hybrid'
            num_eis = len(frequencies)

        # Define special parameters included in quadratic programming parameter vector
        self.special_qp_params = {}

        if times is not None:
            self._add_special_qp_param('v_baseline', False)

        if vz_offset and data_type == 'hybrid':
            self._add_special_qp_param('vz_offset', False)

        if subtract_background and background_type == 'scaled':
            self._add_special_qp_param('background_scale', True)

        # DOP replaces R_inf and L
        if self.fit_ohmic:
            self._add_special_qp_param('R_inf', True)

        if self.fit_inductance:
            self._add_special_qp_param('inductance', True)

        if self.fit_capacitance:
            self._add_special_qp_param('C_inv', True)

        if self.fit_dop:
            if self.fixed_basis_nu is None:
                # self.basis_nu = np.linspace(-1, 1, 41)
                self.basis_nu = np.concatenate([np.linspace(-1, -0.4, 25), np.linspace(0.4, 1, 25)])
            else:
                self.basis_nu = self.fixed_basis_nu

            # Set nu_epsilon
            if self.nu_epsilon is None and self.nu_basis_type != 'delta':
                dnu = np.median(np.diff(np.sort(self.basis_nu)))
                self.nu_epsilon = 1 / dnu

            self._add_special_qp_param('x_dop', True, size=len(self.basis_nu))
        else:
            self.basis_nu = None

        # Get preprocessing hyperparameters. Won't know data factor until chrono data has been downsampled
        pp_hypers = qphb.get_default_hypers(eff_hp, self.fit_dop, self.nu_basis_type)

        # Validate provided kwargs
        for key in kw.keys():
            if key not in pp_hypers.keys():
                raise ValueError(f'Invalid keyword argument {key}')
        pp_hypers.update(kw)

        # Process data and calculate matrices for fit
        sample_data, matrices = self._prep_for_fit(times, i_signal, v_signal, frequencies, z,
                                                   step_times=step_times, downsample=downsample,
                                                   downsample_kw=downsample_kw, offset_steps=offset_steps,
                                                   smooth_inf_response=smooth_inf_response,
                                                   scale_data=scale_data, rp_scale=pp_hypers['rp_scale'],
                                                   penalty_type=penalty_type,
                                                   derivative_weights=pp_hypers['derivative_weights'])
        # Unpack processed data
        sample_times, sample_i, sample_v, response_baseline, z_scaled = sample_data
        # Unpack partial matrices
        rm_drt, induc_rv, inf_rv, cap_rv, rm_dop, zm_drt, induc_zv, cap_zv, zm_dop, base_penalty_matrices = matrices

        # Get num_chrono after downsampling
        if sample_times is not None:
            num_chrono = len(sample_times)
        else:
            num_chrono = 0

        # Downsample static background
        if subtract_background and downsample and background_type == 'static' \
                and not estimate_background_kw.get('downsample', False):
            self.raw_response_background = self.raw_response_background[self.sample_index]

        # Get data factor after downsampling
        data_factor = qphb.get_data_factor_from_data(sample_times, self.step_times, frequencies)
        if self.print_diagnostics:
            print('data factor:', data_factor)

        # Get default hyperparameters and update with user-specified values
        qphb_hypers = qphb.get_default_hypers(eff_hp, self.fit_dop, self.nu_basis_type)
        qphb_hypers.update(kw)

        # Store fit kwargs for reference (after prep_for_fit creates self.fit_kwargs)
        self.fit_kwargs.update(qphb_hypers)
        self.fit_kwargs['nonneg'] = nonneg
        self.fit_kwargs['eff_hp'] = eff_hp
        self.fit_kwargs['penalty_type'] = penalty_type
        self.fit_kwargs['subtract_background'] = subtract_background
        self.fit_kwargs['background_type'] = background_type
        self.fit_kwargs['background_corr_power'] = background_corr_power

        if self.print_diagnostics:
            print('lambda_0, iw_beta:', qphb_hypers['l2_lambda_0'], qphb_hypers['iw_beta'])

        # Format matrices for QP fit
        rm, zm, penalty_matrices = self._format_qp_matrices(rm_drt, inf_rv, induc_rv, cap_rv, rm_dop, zm_drt, induc_zv,
                                                            cap_zv, zm_dop, base_penalty_matrices, v_baseline_penalty,
                                                            ohmic_penalty, inductance_penalty, capacitance_penalty,
                                                            vz_offset_scale, background_penalty, inductance_scale,
                                                            capacitance_scale, penalty_type,
                                                            qphb_hypers['derivative_weights'])

        if subtract_background and times is not None and background_type != 'static':
            rm_bkg = background.get_background_matrix(bkg_gps, sample_times[:, None], y_drt=y_pred_bkg,
                                                      corr_power=background_corr_power)
            if background_type == 'dynamic':
                rm_orig = rm.copy()
                rm = rm - rm_bkg @ rm
            else:
                rm_orig = None
        else:
            rm_bkg = None
            rm_orig = None

        # Construct hybrid response-impedance matrix
        if rm is None:
            rzm = zm.copy()
        elif zm is None:
            rzm = rm.copy()
        else:
            rzm = np.vstack((rm, zm))

        # Make a copy for vz_offset calculation
        if data_type == 'hybrid' and vz_offset:
            if subtract_background and background_type == 'dynamic':
                # The vz_offset is between z and the raw voltage, not the dynamic baseline transformed voltage
                rzm_vz = np.vstack((rm_orig, zm))
            else:
                rzm_vz = rzm.copy()
                # Remove v_baseline from rzm_vz - don't want to scale the baseline, only the delta
                rzm_vz[:, self.special_qp_params['v_baseline']['index']] = 0

            # VZ offset decays as we move away from overlap
            chrono_vz_strength, eis_vz_strength = self._get_vz_strength_vec(
                sample_times, frequencies, fit_times=sample_times, fit_frequencies=frequencies,
                vz_offset_eps=vz_offset_eps
            )
            eis_vz_strength = np.tile(eis_vz_strength, 2)
            vz_strength_vec = np.concatenate([chrono_vz_strength, eis_vz_strength])
        else:
            rzm_vz = None
            vz_strength_vec = 1

        # Offset voltage baseline
        if times is not None:
            if offset_baseline:
                self.scaled_response_offset = -response_baseline
            else:
                self.scaled_response_offset = 0
            # print('scaled_response_offset:', self.scaled_response_offset * self.response_signal_scale)
            rv = self.scaled_response_signal + self.scaled_response_offset

            if subtract_background and background_type == 'dynamic':
                rv_orig = rv.copy()
                rv = rv - rm_bkg @ rv
            else:
                rv_orig = None
        else:
            rv = None
            rv_orig = None

        # Flatten impedance vector
        if frequencies is not None:
            zv = np.concatenate([z_scaled.real, z_scaled.imag])
        else:
            zv = None

        # Construct hybrid response-impedance vector
        if times is None:
            rzv = zv.copy()
        elif frequencies is None:
            rzv = rv.copy()
        else:
            rzv = np.concatenate([rv, zv])

        # Construct lambda vectors
        l1_lambda_vector = np.zeros(rzm.shape[1])  # No penalty applied to v_baseline, R_inf, and inductance
        l1_lambda_vector[self.get_qp_mat_offset():] = qphb_hypers['l1_lambda_0']
        if self.fit_dop:
            dop_start, dop_end = self.dop_indices
            l1_lambda_vector[dop_start:dop_end] = qphb_hypers['dop_l1_lambda_0']

        # Initialize rho and s vectors at prior mode
        k_range = len(qphb_hypers['derivative_weights'])
        rho_vector = qphb_hypers['rho_0'].copy()
        s_vectors = [np.ones(rzm.shape[1]) * qphb_hypers['s_0'][k] for k in range(k_range)]

        if self.fit_dop:
            dop_rho_vector = qphb_hypers['dop_rho_0'].copy()
        else:
            dop_rho_vector = None

        # Update the Rp estimate by solving for the DRT coefficients. Important if fitting the DOP
        if scale_data and solve_rp:
            rp_est, dop_rescale_factor = self._solve_data_scale(qphb_hypers, penalty_matrices, penalty_type, rho_vector,
                                                                dop_rho_vector, s_vectors, rzv, rzm, nonneg)
            scale_factor = (qphb_hypers['rp_scale'] / rp_est)
            if self.print_diagnostics:
                print('Solution-based Rp estimate: {:.3f}'.format(rp_est * self.coefficient_scale))
                print('Data rescale factor:', scale_factor)

            # Update data and qphb parameters to reflect new scale
            for data_vec in [rv, zv, rzv, rv_orig]:
                if data_vec is not None:
                    data_vec *= scale_factor
            self.update_data_scale(scale_factor)

            # Update DOP scale to match DRT scale
            if self.fit_dop and self.normalize_dop:
                if self.print_diagnostics:
                    print('DOP rescale factor:', dop_rescale_factor)
                self.dop_scale_vector /= dop_rescale_factor
                dop_start, dop_end = self.dop_indices

                for mat in [rm, zm, rzm, rzm_vz, rm_orig]:
                    if mat is not None:
                        mat[:, dop_start:dop_end] /= dop_rescale_factor

                for mat_name in penalty_matrices.keys():
                    if mat_name[-3:] == 'dop':
                        penalty_matrices[mat_name][dop_start:dop_end, dop_start:dop_end] /= dop_rescale_factor

        # elif scale_data and not solve_rp:
        #     pp.estimate_rp(times, step_times, step_sizes, response_signal, self.step_model, z)

        # NONLINEAR MATRIX MODIFICATION
        # ----------------------
        rzm_lin = rzm # copy original
        M = rzm_lin.shape[1]
        # Get nonlin weighting function
        if type(nonlin_function) == str:
            nonlin_function = get_func_from_string(nonlin_function)
            
        # Store callable
        self.nonlin_function = nonlin_function
            
        # Get weights for chrono data
        nl_weights_p = nonlin_function(times, i_signal, v_signal)
        nl_weights_n = 1 - nl_weights_p
        print('max nl weights:', np.max(nl_weights_n), np.max(nl_weights_p))
        
        # Store positive weights
        self.nonlin_chrono_weights = nl_weights_p
        
        # Stack chrono matrices: left matrix is for negative bias, right matrix is for positive bias
        rm = np.concatenate((rm * nl_weights_n[:, None], rm * nl_weights_p[:, None]), axis=1)
        
        # Modify rm_orig if using dynamic bkg
        if rm_orig is not None:
            rm_orig = np.concatenate((rm_orig * nl_weights_n[:, None], 
                                      rm_orig * nl_weights_p[:, None]), 
                                     axis=1)
        
        # Extend weights for EIS data
        if frequencies is not None:
            z_weights = np.ones(2 * len(frequencies)) * 0.5
            nl_weights_n = np.concatenate((nl_weights_n, z_weights))
            nl_weights_p = np.concatenate((nl_weights_p, z_weights))
            
        # Stack the matrices: left matrix is for negative bias, right matrix is for positive bias
        rzm = np.concatenate((rzm * nl_weights_n[:, None], rzm * nl_weights_p[:, None]), axis=1)
        print('rzm shape:', rzm.shape)
        if zm is not None:
            zm = np.concatenate((zm * 0.5, zm * 0.5), axis=1)
        
        
        # Modify rzm_vz if using vz_offset
        if rzm_vz is not None:
            rzm_vz = np.concatenate((rzm_vz * nl_weights_n[:, None], 
                                      rzm_vz * nl_weights_p[:, None]), axis=1)
            
        # Expand penalty matrices
        for k in list(penalty_matrices.keys()):
            mat = penalty_matrices[k]
            new_mat = np.zeros((2 * M, 2 * M))
            new_mat[:M, :M] = mat.copy()
            new_mat[M:, M:] = mat.copy()
                
            penalty_matrices[k] = new_mat
            
        # Add nonlin cross-correlation matrix
        m_cross = np.eye(2 * M)
        m_cross[:M, M:] = -1 * np.eye(M)
        m_cross[M:, :M] = -1 * np.eye(M)
        
        m_cross = m_cross * nl_lambda_0
        
        # Apply higher penalties to parameters that shouldn't change with bias
        # NOTE: it would be better to simply have just one of each parameter, 
        # but this would require more invasive modification of the special parameter framework
        fixed_lambda = max(nl_lambda_0, 1) * 1e6
        for name in ['v_baseline', 'vz_offset']:
            if name in self.special_qp_params.keys():
                index = self.special_qp_params[name]['index']
                print(name, index)
                m_cross[index, index] = fixed_lambda
                m_cross[index, M + index] = -fixed_lambda
                m_cross[M + index, index] = -fixed_lambda
                m_cross[M + index, M + index] = fixed_lambda
                
        penalty_matrices['m1_nl'] = m_cross
            
        # Extend L1 and s vectors        
        l1_lambda_vector = np.tile(l1_lambda_vector, 2)
        s_vectors = [np.tile(sv, 2) for sv in s_vectors]
        
        # Store updated matrices
        self._qp_matrices = {
                    'rm': rm,
                    'zm': zm,
                    'penalty_matrices': penalty_matrices
                }
            
        # END NONLINEAR MATRIX MODIFICATION
        # --------------------------
        
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
        iw_hypers = qphb_hypers.copy()
        iw_hypers['l1_lambda_0'] = iw_l1_lambda_0
        iw_hypers['l2_lambda_0'] = iw_l2_lambda_0
        if 'dop_l2_lambda_0' in qphb_hypers.keys():
            dop_drt_ratio = qphb_hypers['dop_l2_lambda_0'] / qphb_hypers['l2_lambda_0']
            iw_hypers['dop_l2_lambda_0'] = dop_drt_ratio * iw_hypers['l2_lambda_0']
        if times is not None:
            chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = \
                qphb.initialize_weights(iw_hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                                        s_vectors,
                                        rv, rm, chrono_vmm, nonneg, self.special_qp_params)

            chrono_weight_scale = np.mean(chrono_est_weights ** -2) ** -0.5
        else:
            chrono_est_weights, chrono_init_weights, x_overfit_chrono, chrono_outlier_t = None, None, None, None
            chrono_weight_scale = None

        if frequencies is not None:
            eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = \
                qphb.initialize_weights(iw_hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                                        s_vectors,
                                        zv, zm, eis_vmm, nonneg, self.special_qp_params)

            eis_weight_scale = np.mean(eis_est_weights ** -2) ** -0.5
        else:
            eis_est_weights, eis_init_weights, x_overfit_eis, eis_outlier_t = None, None, None, None
            eis_weight_scale = None

        # TEMP TEST
        # ---------
        # est_weights, init_weights, x_overfit, outlier_t = \
        #     qphb.initialize_weights(penalty_matrices, penalty_type, qphb_hypers['derivative_weights'], rho_vector,
        #                             s_vectors, rzv, rzm, vmm, nonneg, self.special_qp_params,
        #                             qphb_hypers['iw_alpha'], qphb_hypers['iw_beta'], qphb_hypers['outlier_p'])
        # 
        # basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)
        # fig, ax = plt.subplots()
        # ax.plot(x_overfit)
        # rp = np.sum(np.abs(x_overfit[self.get_qp_mat_offset():])) * basis_area
        # rp = self.predict_r_p(absolute=True, x=x_overfit, raw=True)
        # # rp = 0.045 / self.impedance_scale
        # print('estimated rp from x_overfit:', rp * self.impedance_scale)
        # scale_factor = (qphb_hypers['rp_scale'] / rp)
        # scale_factor = 1
        # print('rescale factor:', scale_factor)
        # 
        # # Update data and qphb parameters to reflect new scale
        # for x_t in [x_overfit_eis, x_overfit_chrono]:
        #     if x_t is not None:
        #         x_t *= scale_factor
        # rzv *= scale_factor
        # est_weights /= scale_factor
        # init_weights /= scale_factor  # update for reference only
        # for weights in [eis_est_weights, eis_init_weights, eis_weight_scale,
        #                 chrono_est_weights, chrono_init_weights, chrono_weight_scale]:
        #     if weights is not None:
        #         weights /= scale_factor
        # self.update_data_scale(scale_factor)
        # ---------
        # END TEST

        # chrono_est_weights = est_weights[:len(rv)]
        # eis_est_weights = est_weights[len(rv):]
        # chrono_init_weights = init_weights[:len(rv)]
        # eis_init_weights = init_weights[len(rv):]

        # eis_est_weights = eis_est_weights_raw * eis_weight_factor
        # eis_init_weights = eis_init_weights_raw * eis_weight_factor

        # Get weight factors
        if data_type == 'hybrid':
            if eis_weight_factor is None or chrono_weight_factor is None:
                if hybrid_weight_factor_method == 'weight':
                    # Set based on ratio of EIS to chrono weights
                    ratio = (eis_weight_scale / chrono_weight_scale) ** 0.25
                    if eis_weight_factor is None:
                        eis_weight_factor = 1 / ratio
                        # eis_weight_factor = (chrono_weight_scale / eis_weight_scale) ** 0.5 * (num_chrono / num_eis) ** 0.25

                    if chrono_weight_factor is None:
                        # chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.25
                        # chrono_weight_factor = (eis_weight_scale / chrono_weight_scale) ** 0.5 * (num_eis / num_chrono) ** 0.25
                        chrono_weight_factor = ratio

                elif hybrid_weight_factor_method == 'rp':
                    rp_eis = pp.estimate_rp(None, None, None, None, None, self.z_fit)
                    rp_chrono = pp.estimate_rp(sample_times, self.step_times, self.step_sizes, self.raw_response_signal,
                                               self.step_model, None)
                    rp_tot = self.coefficient_scale * qphb_hypers['rp_scale']

                    if self.print_diagnostics:
                        print('rp_eis: {:.2e}'.format(rp_eis))
                        print('rp_chrono: {:.2e}'.format(rp_chrono))
                        print('rp_tot: {:.2e}'.format(rp_tot))
                        print('rp_chrono / rp_eis:', rp_chrono / rp_eis)
                    # ratio = (rp_chrono / rp_eis) ** 0.5

                    # rp_eis += rp_tot * 0.25
                    # rp_chrono += rp_tot * 0.25

                    # print('rp_tot:', rp_tot)
                    # print('rp_tot factor:', (rp_chrono * rp_eis) ** 0.5 / rp_tot)
                    # rp_tot_factor = (rp_chrono * rp_eis) ** 0.5 / rp_tot

                    if eis_weight_factor is None:
                        # eis_weight_factor = ratio ** -1 * rp_tot_factor ** 0.5
                        # eis_weight_factor = (rp_eis / rp_tot) ** 0.25
                        eis_weight_factor = rp_eis ** 0.75 / (rp_chrono ** 0.25 * rp_tot ** 0.5)
                        # eis_weight_factor = (rp_eis / rp_chrono) ** 0.5

                    if chrono_weight_factor is None:
                        # chrono_weight_factor = ratio * rp_tot_factor ** 0.5
                        # chrono_weight_factor = (rp_chrono / rp_tot) ** 0.25
                        chrono_weight_factor = rp_chrono ** 0.75 / (rp_eis ** 0.25 * rp_tot ** 0.5)
                        # chrono_weight_factor = (rp_chrono / rp_eis) ** 0.5
                elif hybrid_weight_factor_method is None:
                    eis_weight_factor = 1
                    chrono_weight_factor = 1
                else:
                    raise ValueError(f"Invalid hybrid_weight_factor_method argument {hybrid_weight_factor_method}. "
                                     f"Options: 'weight', 'rp', None")

            if self.print_diagnostics:
                print('num_chrono / num_eis:', num_chrono / num_eis)
                print('w_eis / w_chrono:', eis_weight_scale / chrono_weight_scale)
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

        # Return outliers if requested
        # ----------------------------
        if return_outlier_index:
            outlier_index = (1 - outlier_t) > outlier_thresh
            if times is None:
                eis_outlier_index = outlier_index
                chrono_outlier_index = None
            elif frequencies is None:
                eis_outlier_index = None
                chrono_outlier_index = outlier_index
            else:
                chrono_outlier_index = outlier_index[:num_chrono]
                eis_outlier_index = outlier_index[num_chrono:]

            # Consider any z data point with bad real OR imag value to be an outlier
            if eis_outlier_index is not None:
                eis_outlier_index = eis_outlier_index[:len(frequencies)] | eis_outlier_index[len(frequencies):]

            return chrono_outlier_index, eis_outlier_index

        weights = init_weights.copy()

        # Initialize outlier tvt
        if kw.get('outlier_p', None) is not None:
            outlier_tvt = qphb.outlier_tvt(vmm, outlier_t)
        else:
            outlier_tvt = None

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
        dop_xmx_norms = np.ones(k_range)
        # xmx_norms = [10, 0.2, 0.1]

        # TEST: curvature constraints
        if peak_locations is not None:
            drt_curv_matrix = basis.construct_func_eval_matrix(np.log(self.basis_tau), None, self.tau_basis_type,
                                                               self.tau_epsilon, 2, self.zga_params)
            curv_matrix = np.zeros((len(self.basis_tau), len(x)))
            curv_matrix[:, self.get_qp_mat_offset():] = drt_curv_matrix
            peak_indices = np.array([utils.array.nearest_index(np.log(self.basis_tau), np.log(pl))
                                     for pl in peak_locations])
            curv_spread_func = evaluation.get_similarity_function('gaussian')
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

            # # Set the outlier weights to nearly zero
            # if remove_outliers:
            #     weights[outlier_index] = 0

            # Apply overall weight scaling factor
            if it > 0:
                weights = weights * weight_factor
            # print(np.mean(weights))
            # weights = np.ones(len(rzv))

            # TEST: enforce curvature constraint
            if peak_locations is not None:  # and it > 5:
                curv = curv_matrix @ x_in
                peak_curv = curv[peak_indices]
                curv_limit = [2.5 * pc * curv_spread_func(np.log(self.basis_tau / pl), 1.5, 2)
                              for pc, pl in zip(peak_curv, peak_locations)]
                curv_limit = np.sum(curv_limit, axis=0)
                # curv_limit = 0.5 * (curv_limit + curv)

                curv_limit = np.zeros(len(self.basis_tau))
                for index in peak_indices:
                    curv_limit[max(0, index - 5):index + 5] = -100
                print(curv_limit)
                curvature_constraint = (-curv_matrix, -curv_limit)
            else:
                curvature_constraint = None

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and scale_data and update_scale:
                # Get scale factor
                # basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)
                # rp = np.sum(np.abs(x[self.get_qp_mat_offset():])) * basis_area
                rp = self.predict_r_p(absolute=True, x=x, raw=True)
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                if self.print_diagnostics:
                    print('Iter {} scale factor: {:.3f}'.format(it, scale_factor))
                # Update data and qphb parameters to reflect new scale
                for x_t in [x_in, x_overfit_eis, x_overfit_chrono]:
                    if x_t is not None:
                        x_t *= scale_factor
                rzv *= scale_factor
                if rv_orig is not None:
                    rv_orig *= scale_factor
                xmx_norms *= scale_factor ** 0.5  # shouldn't this be scale_factor ** 2?
                if self.fit_dop:
                    dop_xmx_norms *= scale_factor ** 0.5
                est_weights /= scale_factor
                init_weights /= scale_factor  # update for reference only
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            # Perform actual QPHB operation
            x, s_vectors, rho_vector, dop_rho_vector, weights, outlier_t, outlier_tvt, cvx_result, converged = \
                qphb.iterate_qphb(x_in, s_vectors, rho_vector, dop_rho_vector, rzv, weights, est_weights, outlier_tvt,
                                  rzm, vmm, penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers,
                                  eff_hp, xmx_norms, dop_xmx_norms, None, None, curvature_constraint, nonneg,
                                  self.special_qp_params, xtol, 1, self.qphb_history, nonlin=True)

            # Normalize to ordinary ridge solution
            if it == 0:
                # Only include DRT penalty in XMX norms - zero out special params
                x_drt = x.copy()
                x_drt[:self.get_qp_mat_offset()] = 0
                x_drt[M: M + self.get_qp_mat_offset()] = 0
                
                pm_drt_list = [penalty_matrices[f'm{k}'] for k in range(k_range)]
                xmx_norms = np.array([x_drt.T @ pm_drt_list[k] @ x_drt for k in range(k_range)])
                if self.print_diagnostics:
                    print('xmx', xmx_norms)

                if self.fit_dop:
                    dop_start, dop_end = self.dop_indices
                    x_dop = subset_vector(x, dop_start, dop_end, M)
                    pm_dop_list = [subset_penalty_matrix(penalty_matrices[f'm{k}'], 
                                                         dop_start, dop_end, M)
                                   for k in range(k_range)]
                    dop_xmx_norms = np.array([x_dop.T @ pm_dop_list[k] @ x_dop for k in range(k_range)])
                    if self.print_diagnostics:
                        print('dop_xmx', dop_xmx_norms)
                
                # TODO: recalculate nl_weights using v_pred in place of v_signal

            # Update background estimate
            if times is not None and subtract_background and background_type == 'scaled':
                # TODO: this would need to be updated but has not been, 
                # since scaled background is obsolete.
                # Use the residuals from the current iteration to estimate the background
                y_hat = rzm @ x
                y_err_chrono = (rzv - y_hat)[:num_chrono]
                y_bkg = rm_bkg @ y_err_chrono
                rzm[:num_chrono, self.special_qp_params['background_scale']['index']] = y_bkg

            # Update vz_offset column
            if data_type == 'hybrid' and vz_offset:
                # Update the response matrix with the current predicted y vector
                # vz_offset offsets chrono and eis predictions
                y_hat = rzm_vz @ x
                vz_sep = y_hat.copy()
                vz_sep[len(rv):] *= -1  # vz_offset > 0 means EIS Rp smaller chrono Rp
                # TODO: currently we have 2 vz_offset columns and 2 vz_offset parameters. 
                # ideally would only have one of each
                rzm[:, self.special_qp_params['vz_offset']['index']] = 0.5 * vz_sep * vz_strength_vec
                rzm[:, M + self.special_qp_params['vz_offset']['index']] = 0.5 * vz_sep * vz_strength_vec
                # if rm_orig is not None:
                #     rm_orig[:, self.special_qp_params['vz_offset']['index']] = vz_sep[:len(rv)]

            if converged:
                break
            elif it == max_iter - 1 and self.warn:
                warnings.warn(f'Solution did not converge within {max_iter} iterations')

            it += 1

        # # Set the outlier weights to nearly zero (final update)
        # if remove_outliers:
        #     weights[outlier_index] = 1e-10

        # Store the scaled weights for reference
        scaled_weights = weights.copy()
        if data_type == 'hybrid':
            # weights from iterate_qphb does not have weight factors applied
            scaled_weights[:len(rv)] *= chrono_weight_factor
            scaled_weights[len(rv):] *= eis_weight_factor

        # Store QPHB diagnostic parameters
        # TODO: should scaled_weights or weights be used here? weights seems to make sense...
        p_matrix, q_vector = qphb.calculate_pq(rzm, rzv, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
                                               rho_vector, dop_rho_vector, s_vectors, scaled_weights,
                                               self.special_qp_params)

        # post_lp = qphb.evaluate_posterior_lp(x, penalty_matrices, penalty_type, qphb_hypers, l1_lambda_vector,
        #                                      rho_vector, s_vectors, weights, rzm, rzv, xmx_norms)

        # Calculate the estimated background
        if subtract_background and times is not None:
            if background_type == 'dynamic':
                resid = rv_orig - rm_orig @ x
                self.raw_response_background = (rm_bkg @ resid) * self.response_signal_scale
            elif background_type == 'scaled':
                rzm_resid = rzm.copy()
                x_bkg_index = self.special_qp_params['background_scale']['index']
                rzm_resid[:, x_bkg_index] = 0
                resid = (rzv - rzm_resid @ x)[:num_chrono]
                self.raw_response_background = (rm_bkg @ resid) * self.response_signal_scale * x[x_bkg_index]
            else:
                # Add the background back to the raw data
                self.raw_response_signal = self.raw_response_signal + self.raw_response_background
        elif times is not None:
            self.raw_response_background = np.zeros(len(sample_times))
        else:
            self.raw_response_background = None

        if self.print_diagnostics:
            w_mat = np.diag(scaled_weights)
            wtw = w_mat.T @ w_mat
            rss = rzv.T @ wtw @ rzv - 2 * rzv.T @ wtw @ rzm @ x + x.T @ rzm.T @ wtw @ rzm @ x
            wtd_resid = w_mat @ (rzv - rzm @ x)
            print('rss:', rss)
            print('Mean wtd. resid:', np.mean(np.abs(wtd_resid)))

        self.qphb_params = {'est_weights': est_weights.copy(),
                            'init_weights': init_weights.copy(),
                            'weights': scaled_weights.copy(),  # scaled weights
                            'true_weights': weights.copy(),  # unscaled weights
                            'data_factor': data_factor,
                            'chrono_weight_factor': chrono_weight_factor,
                            'eis_weight_factor': eis_weight_factor,
                            'xmx_norms': xmx_norms.copy(),
                            'dop_xmx_norms': dop_xmx_norms,
                            'x_overfit_chrono': x_overfit_chrono,
                            'x_overfit_eis': x_overfit_eis,
                            'p_matrix': p_matrix,
                            'q_vector': q_vector,
                            'rho_vector': rho_vector,
                            'dop_rho_vector': dop_rho_vector,
                            's_vectors': s_vectors,
                            'outlier_t': outlier_t,
                            'vmm': vmm,
                            'l1_lambda_vector': l1_lambda_vector,
                            # 'posterior_lp': post_lp,
                            'rm': rzm,
                            'rv': rzv,
                            'penalty_matrices': penalty_matrices,
                            'hypers': qphb_hypers,
                            'num_eis': num_eis,
                            'num_chrono': num_chrono,
                            'rm_bkg': rm_bkg,
                            'rm_orig': rm_orig,
                            'rv_orig': rv_orig,
                            'vz_strength_vec': vz_strength_vec,
                            # 'outlier_index': outlier_index,
                            # 'wtd_resid': wtd_resid
                            }

        if self.print_diagnostics:
            print('rho:', rho_vector)
            if self.fit_dop:
                print('dop_rho:', dop_rho_vector)

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
        self.fit_parameters['vz_offset_eps'] = vz_offset_eps
        self.fit_parameters['p_matrix'] = p_matrix
        self.fit_parameters['q_vector'] = q_vector

        self.fit_type = f'qphb_{data_type}'
        
        
def get_func_from_string(func_str: str):
    try:
        sig, func = func_str.split('_')
        
        if sig not in ('v', 'i'):
            raise ValueError(f'Received invalid signal name {sig}')
        
        if func == 'exp':
            base_func = exponential_func
        elif func == 'lin':
            base_func = linear_func
        else:
            raise ValueError(f'Received invalid function name {func}')
        
        def nl_func(times, i_signal, v_signal):
            x = v_signal if sig == 'v' else i_signal
            return base_func(x)
        
        return nl_func
    except Exception as err:
        raise ValueError(f'Could not parse nonlin_func string {func_str}') from err
            
    
def subset_vector(x, start, end, M):
    return np.concatenate((x[start:end], x[M + start:M + end]))

def subset_penalty_matrix(mat, start, end, M):
    m11 = mat[start:end, start:end]
    m12 = mat[start:end, M + start:M + end]
    m21 = mat[M + start:M + end, start:end]
    m22 = mat[M + start:M + end, M + start:M + end]
    size = m11.shape[0]
    out = np.zeros((2 * size, 2 * size))
    out[:size, :size] = m11
    out[:size, size:] = m12
    out[size:, :size] = m21
    out[size:, size:] = m22
    
    return out
    
def minmax_normalize(x, percentiles=(1, 99), range=(0.0, 1.0)):
    x_min = np.percentile(x, percentiles[0])
    x_max = np.percentile(x, percentiles[1])
    
    y = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    
    y = y * (range[1] - range[0]) + range[0]
    print(np.min(y), np.max(y))
    return y
        
def exponential_func(x, margin=0.0):
    # Normalize to [0, 1] range
    x = minmax_normalize(x, range=(margin, 1 - margin))
    
    # Return exponential scaled and offset to [0, 1] range
    return (np.exp(x) - 1) / (np.exp(1) - 1)

def linear_func(x, margin=0.0):
    return minmax_normalize(x, range=(margin, 1 - margin))
    
    