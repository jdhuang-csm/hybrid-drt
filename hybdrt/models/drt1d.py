import inspect
import itertools
import time
import warnings
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal, ndimage
from copy import deepcopy
import pickle
from typing import Optional

from numpy import ndarray

from .. import utils, preprocessing as pp
from ..utils import stats
from ..matrices import mat1d, basis, phasance
from . import qphb, peaks, elements, pfrt, background
from .. import evaluation
from .drtbase import DRTBase
from ..plotting import get_transformed_plot_time, add_linear_time_axis, plot_eis, plot_distribution, plot_chrono


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

        self.background_gp = None

        self.pfrt_result = None
        self.pfrt_history = None
        self.pfrt_candidate_df = None
        self.pfrt_candidate_dict = None

        self._qp_matrices = None

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

    # def _qphb_loop(self, data_type, x, weights, rzv, rzm, num_eis, chrono_weight_factor, eis_weight_factor,
    #                weight_factor, max_iter, xmx_norms, dop_xmx_norms,
    #                peak_locations, peak_indices, curv_matrix, curv_spread_func,
    #                ):
    #     it = 0
    #     # fixed_prior = False
    #
    #     while it < max_iter:
    #
    #         x_in = x.copy()
    #
    #         # Apply chrono/eis weight adjustment factors
    #         if data_type == 'hybrid':
    #             weights[:num_eis] *= chrono_weight_factor
    #             weights[num_eis:] *= eis_weight_factor
    #
    #         # Apply overall weight scaling factor
    #         if it > 0:
    #             weights = weights * weight_factor
    #
    #         # TEST: enforce curvature constraint
    #         if peak_locations is not None and it > 5:
    #             curv = curv_matrix @ x_in
    #             peak_curv = curv[peak_indices]
    #             curv_limit = [2.5 * pc * curv_spread_func(np.log(self.basis_tau / pl), 1.5, 2)
    #                           for pc, pl in zip(peak_curv, peak_locations)]
    #             curv_limit = np.sum(curv_limit, axis=0)
    #             # curv_limit = 0.5 * (curv_limit + curv)
    #             curvature_constraint = (-curv_matrix, -curv_limit)
    #         else:
    #             curvature_constraint = None
    #
    #         # Update data scale as Rp estimate improves to maintain specified rp_scale
    #         if it > 1 and scale_data and update_scale:
    #             # Get scale factor
    #             # basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)
    #             # rp = np.sum(np.abs(x[self.get_qp_mat_offset():])) * basis_area
    #             rp = self.predict_r_p(absolute=True, x=x, raw=True)
    #             scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
    #             if self.print_diagnostics:
    #                 print('Iter {} scale factor: {:.3f}'.format(it, scale_factor))
    #             # Update data and qphb parameters to reflect new scale
    #             for x_t in [x_in, x_overfit_eis, x_overfit_chrono]:
    #                 if x_t is not None:
    #                     x_t *= scale_factor
    #             rzv *= scale_factor
    #             xmx_norms *= scale_factor ** 0.5  # shouldn't this be scale_factor ** 2?
    #             if self.fit_dop:
    #                 dop_xmx_norms *= scale_factor ** 0.5
    #             est_weights /= scale_factor
    #             init_weights /= scale_factor  # update for reference only
    #             weights /= scale_factor
    #             # Update data scale attributes
    #             self.update_data_scale(scale_factor)
    #
    #         # Perform actual QPHB operation
    #         x, s_vectors, rho_vector, dop_rho_vector, weights, outlier_t, cvx_result, converged = \
    #             qphb.iterate_qphb(x_in, s_vectors, rho_vector, dop_rho_vector, rzv, weights, est_weights, outlier_t,
    #                               rzm, vmm, penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers, eff_hp,
    #                               xmx_norms, dop_xmx_norms,
    #                               None, None, curvature_constraint, nonneg, self.special_qp_params, xtol, 1,
    #                               self.qphb_history)
    #
    #         # Normalize to ordinary ridge solution
    #         if it == 0:
    #             # Only include DRT penalty in XMX norms
    #             x_drt = x[self.get_qp_mat_offset():]
    #             pm_drt_list = [penalty_matrices[f'm{k}'][self.get_qp_mat_offset():, self.get_qp_mat_offset():]
    #                            for k in range(k_range)]
    #             xmx_norms = np.array([x_drt.T @ pm_drt_list[k] @ x_drt for k in range(k_range)])
    #             if self.print_diagnostics:
    #                 print('xmx', xmx_norms)
    #
    #             if self.fit_dop:
    #                 dop_start, dop_end = self.dop_indices
    #                 x_dop = x[dop_start:dop_end]
    #                 pm_dop_list = [penalty_matrices[f'm{k}'][dop_start:dop_end, dop_start:dop_end]
    #                                for k in range(k_range)]
    #                 dop_xmx_norms = np.array([x_dop.T @ pm_dop_list[k] @ x_dop for k in range(k_range)])
    #                 if self.print_diagnostics:
    #                     print('dop_xmx', dop_xmx_norms)
    #
    #         # Update vz_offset column
    #         if data_type == 'hybrid' and vz_offset:
    #             # Update the response matrix with the current predicted y vector
    #             # vz_offset offsets chrono and eis predictions
    #             y_hat = rzm_vz @ x
    #             vz_sep = y_hat.copy()
    #             vz_sep[len(rv):] *= -1  # vz_offset > 0 means EIS Rp smaller chrono Rp
    #             rzm[:, self.special_qp_params['vz_offset']['index']] = vz_sep
    #
    #         if converged:
    #             break
    #         elif it == max_iter - 1:
    #             warnings.warn(f'Hyperparametric solution did not converge within {max_iter} iterations')
    #
    #         it += 1
    #
    #     return x, s_vectors, rho_vector, dop_rho_vector, weights, outlier_t, cvx_result

    def _qphb_fit_core(self, times, i_signal, v_signal, frequencies, z, step_times=None,
                       nonneg=True, series_neg=False, scale_data=True, update_scale=False, solve_rp=False,
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

        # Check inputs
        utils.validation.check_chrono_data(times, i_signal, v_signal)
        utils.validation.check_eis_data(frequencies, z)
        for err_struct in [chrono_error_structure, eis_error_structure]:
            utils.validation.check_error_structure(err_struct)
        utils.validation.check_penalty_type(penalty_type)

        if solve_rp and not scale_data and self.warn:
            warnings.warn('solve_rp is ignored if scale_data=False')
        # if (self.fit_dop or not nonneg) and not solve_rp and self.warn:
        #     warnings.warn('For best results, set solve_rp=True when performing DRT-DOP fits '
        #                   'or DRT fits without a non-negativity constraint')

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

                self._qp_matrices = {
                    'rm': rm,
                    'zm': zm,
                    'penalty_matrices': penalty_matrices
                }
        # elif scale_data and not solve_rp:
        #     pp.estimate_rp(times, step_times, step_sizes, response_signal, self.step_model, z)

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
                                  self.special_qp_params, xtol, 1, self.qphb_history)

            # Normalize to ordinary ridge solution
            if it == 0:
                # Only include DRT penalty in XMX norms
                x_drt = x[self.get_qp_mat_offset():]
                pm_drt_list = [penalty_matrices[f'm{k}'][self.get_qp_mat_offset():, self.get_qp_mat_offset():]
                               for k in range(k_range)]
                xmx_norms = np.array([x_drt.T @ pm_drt_list[k] @ x_drt for k in range(k_range)])
                if self.print_diagnostics:
                    print('xmx', xmx_norms)

                if self.fit_dop:
                    dop_start, dop_end = self.dop_indices
                    x_dop = x[dop_start:dop_end]
                    pm_dop_list = [penalty_matrices[f'm{k}'][dop_start:dop_end, dop_start:dop_end]
                                   for k in range(k_range)]
                    dop_xmx_norms = np.array([x_dop.T @ pm_dop_list[k] @ x_dop for k in range(k_range)])
                    if self.print_diagnostics:
                        print('dop_xmx', dop_xmx_norms)

            # Update background estimate
            if times is not None and subtract_background and background_type == 'scaled':
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
                rzm[:, self.special_qp_params['vz_offset']['index']] = vz_sep * vz_strength_vec
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

    def get_offset_pq(self, weights=None, filter_weights=False, unscale=False):
        # Get necessary matrices/vectors
        rzm = self.qphb_params['rm'].copy()
        rzv = self.qphb_params['rv'].copy()
        penalty_matrices = deepcopy(self.qphb_params['penalty_matrices'])
        penalty_type = self.fit_kwargs['penalty_type']
        qphb_hypers = self.qphb_params['hypers']
        l1_lambda_vector = self.qphb_params['l1_lambda_vector'].copy()
        rho_vector = self.qphb_params['rho_vector']
        dop_rho_vector = self.qphb_params['dop_rho_vector']
        s_vectors = self.qphb_params['s_vectors'].copy()

        if weights is None:
            scaled_weights = self.qphb_params['weights']
            # print(np.mean(scaled_weights))

            # TODO: compare and determine how to simplify weights. Seems like mean of all weights is safest...

            if filter_weights:
                num_chrono = self.num_chrono
                chrono_weights = scaled_weights[:num_chrono]
                eis_weights = scaled_weights[num_chrono:]
                # chrono_weights = ndimage.median_filter(chrono_weights, size=5, mode='reflect')
                # eis_weights = ndimage.median_filter(eis_weights, size=5, mode='reflect')
                chrono_weights = np.ones_like(chrono_weights) * np.mean(chrono_weights)
                eis_weights = np.ones_like(eis_weights) * np.mean(eis_weights)
                scaled_weights = np.concatenate([chrono_weights, eis_weights])
            else:
                scaled_weights = np.ones_like(scaled_weights) * np.mean(scaled_weights)
        else:
            scaled_weights = weights

        if unscale:
            # rzv = rzv * self.coefficient_scale
            scaled_weights = scaled_weights / self.coefficient_scale
            for key in list(penalty_matrices.keys()):
                mat = penalty_matrices[key]
                penalty_matrices[key] = mat / self.coefficient_scale ** 2

        # Get raw parameter values
        x_raw = np.array(list(self.cvx_result['x']))

        # For each offset parameter, add its influence to the data vector
        del_index = []
        for name in ['v_baseline', 'vz_offset']:
            if name in self.special_qp_params.keys():
                index = self.special_qp_params[name]['index']
                val = x_raw[index]
                rzv -= rzm[:, index] * val

                del_index.append(index)

        if len(del_index) > 0:
            # After accounting for offsets, delete the corresponding matrix/vector entries
            # Response matrix - delete columns
            rzm = np.delete(rzm, del_index, axis=1)

            # Penalty matrices - delete rows and columns
            for key in list(penalty_matrices.keys()):
                mat = penalty_matrices[key]
                penalty_matrices[key] = np.delete(np.delete(mat, del_index, axis=0), del_index, axis=1)

            # Vectors - delete entries
            l1_lambda_vector = np.delete(l1_lambda_vector, del_index)
            for i, sv in enumerate(s_vectors):
                s_vectors[i] = np.delete(sv, del_index)

            # Remove entries from special_qp dict
            special_qp = deepcopy(self.special_qp_params)
            for name in ['v_baseline', 'vz_offset']:
                if name in special_qp.keys():
                    del special_qp[name]

            # Shift indices of remaining special params
            for key in list(special_qp.keys()):
                index = special_qp[key]['index']
                shift = np.sum([1 if di < index else 0 for di in del_index])
                special_qp[key]['index'] = index - shift

            p_matrix, q_vector = qphb.calculate_pq(rzm, rzv, penalty_matrices, penalty_type, qphb_hypers,
                                                   l1_lambda_vector,
                                                   rho_vector, dop_rho_vector, s_vectors, scaled_weights,
                                                   special_qp)
        else:
            # No offsets to remove
            p_matrix = self.fit_parameters['p_matrix'].copy()
            q_vector = self.fit_parameters['q_vector'].copy()

        return p_matrix, q_vector

    def fit_chrono(self, times, i_signal, v_signal, step_times=None,
                   nonneg=True, scale_data=True, update_scale=False,
                   offset_baseline=True, offset_steps=True, subtract_background=False, estimate_background_kw=None,
                   downsample=False, downsample_kw=None, smooth_inf_response=True,
                   error_structure='uniform', vmm_epsilon=4,
                   **kwargs):

        self._qphb_fit_core(times, i_signal, v_signal, None, None, step_times=step_times, nonneg=nonneg,
                            scale_data=scale_data, update_scale=update_scale, offset_steps=offset_steps,
                            offset_baseline=offset_baseline, downsample=downsample, downsample_kw=downsample_kw,
                            subtract_background=subtract_background, estimate_background_kw=estimate_background_kw,
                            smooth_inf_response=smooth_inf_response, chrono_error_structure=error_structure,
                            chrono_vmm_epsilon=vmm_epsilon, **kwargs)

    def fit_eis(self, frequencies, z, nonneg=True, scale_data=True, update_scale=False,
                error_structure=None, vmm_epsilon=0.25, vmm_reim_cor=0.25, **kwargs):
        """
        Perform a conventional DRT fit of EIS data.
        :param ndarray frequencies: array of frequencies
        :param ndarray z: complex array of impedance values
        :param bool nonneg: if True, constrain the DRT to be non-negative. If False, allow negative DRT values
        :param bool scale_data: if True, scale the impedance prior to fitting. If set to False, the model tuning will
        not work as intended and may yield unexpected results.
        :param bool update_scale: if True, update the data scale during solution iterations. Setting this to False
        provides more stable performance, but setting this to True can be helpful if your data is truncated or contains
        outliers
        :param str error_structure: error structure to use for the data. If None, a flexible error structure based on
        residual filtering is used. If 'uniform', a uniform error structure is assumed
        :param float vmm_epsilon: inverse scale parameter for flexible error structure filter. Larger values will allow
        greater local variations in error scale
        :param float vmm_reim_cor: correlation factor between real and imaginary error structures. A value of 1 forces
        the reak and imaginary error structures to be the same, while a value of 0 corresponds to completely independent
        real and imaginary error structures
        :param kwargs: additional keyword args passed to _qphb_fit_core
        :return:
        """

        self._qphb_fit_core(None, None, None, frequencies, z, nonneg=nonneg, scale_data=scale_data,
                            update_scale=update_scale, eis_error_structure=error_structure, eis_vmm_epsilon=vmm_epsilon,
                            eis_reim_cor=vmm_reim_cor, **kwargs)

    def fit_hybrid(self, times, i_signal, v_signal, frequencies, z, step_times=None,
                   nonneg=True, scale_data=True, update_scale=False,
                   # chrono parameters
                   offset_steps=True, offset_baseline=True, subtract_background=False, estimate_background_kw=None,
                   downsample=False, downsample_kw=None, smooth_inf_response=True,
                   # vz offset
                   vz_offset=True, vz_offset_scale=1, vz_offset_eps=1,
                   # Error structure
                   chrono_error_structure='uniform', eis_error_structure=None,
                   chrono_vmm_epsilon=4, eis_vmm_epsilon=0.25, eis_reim_cor=0.25,
                   eis_weight_factor=None, chrono_weight_factor=None, **kwargs):

        self._qphb_fit_core(times, i_signal, v_signal, frequencies, z, step_times=step_times, nonneg=nonneg,
                            scale_data=scale_data, update_scale=update_scale, offset_steps=offset_steps,
                            offset_baseline=offset_baseline, downsample=downsample, downsample_kw=downsample_kw,
                            subtract_background=subtract_background, estimate_background_kw=estimate_background_kw,
                            smooth_inf_response=smooth_inf_response, chrono_error_structure=chrono_error_structure,
                            eis_error_structure=eis_error_structure, chrono_vmm_epsilon=chrono_vmm_epsilon,
                            eis_vmm_epsilon=eis_vmm_epsilon, eis_reim_cor=eis_reim_cor, vz_offset=vz_offset,
                            vz_offset_scale=vz_offset_scale, vz_offset_eps=vz_offset_eps,
                            eis_weight_factor=eis_weight_factor, chrono_weight_factor=chrono_weight_factor, **kwargs)

    def _continue_from_init(self, qphb_hypers, x_init, rv, rm, vmm, rho_vector, dop_rho_vector, s_vectors, outlier_t,
                            penalty_matrices, xmx_norms, dop_xmx_norms,
                            est_weights, weights, l1_lambda_vector,
                            nonneg=True, update_scale=False, weight_factor=1,
                            eis_weight_factor=None, chrono_weight_factor=None,
                            penalty_type='integral', eff_hp=True, xtol=1e-2, max_iter=10, min_iter=2, **kw):

        # Update hyperparameters with user-specified values
        qphb_hypers = qphb_hypers.copy()
        qphb_hypers.update(kw)
        # print(qphb_hypers)
        # print(rv[-10:])
        # print(self.coefficient_scale)

        if eis_weight_factor is None:
            eis_weight_factor = self.qphb_params['chrono_weight_factor']
        if chrono_weight_factor is None:
            chrono_weight_factor = self.qphb_params['chrono_weight_factor']

        x = x_init.copy()
        continue_history = []
        it = 0

        if 'vz_offset' in self.special_qp_params.keys():
            # Make a copy for vz_offset calculation
            rzm_vz = rm.copy()
            # Remove v_baseline from rzm_vz - don't want to scale the baseline, only the delta
            rzm_vz[:, self.special_qp_params['v_baseline']['index']] = 0
            vz_strength_vec = self.qphb_params['vz_strength_vec']
        else:
            rzm_vz = None
            vz_strength_vec = 1

        # Initialize outlier_tvt
        if qphb_hypers.get('outlier_p', None) is not None:
            outlier_tvt = qphb.outlier_tvt(vmm, outlier_t)
        else:
            outlier_tvt = None

        while it < max_iter:

            x_in = x.copy()

            # Apply chrono/eis weight adjustment factors
            if self.fit_type.find('hybrid') > -1:
                weights[:self.qphb_params['num_chrono']] *= chrono_weight_factor
                weights[self.qphb_params['num_chrono']:] *= eis_weight_factor

            weights = weights * weight_factor

            # Update data scale as Rp estimate improves to maintain specified rp_scale
            if it > 1 and update_scale:
                # Get scale factor
                rp = self.predict_r_p(absolute=True, x=x, raw=True)
                scale_factor = (qphb_hypers['rp_scale'] / rp) ** 0.5  # Damp the scale factor to avoid oscillations
                if self.print_diagnostics:
                    print('Iter {} scale factor: {:.3f}'.format(it, scale_factor))

                # Update data and qphb parameters to reflect new scale
                x_in *= scale_factor
                rv *= scale_factor
                xmx_norms *= scale_factor ** 0.5
                if self.fit_dop:
                    dop_xmx_norms *= scale_factor ** 0.5
                est_weights /= scale_factor
                weights /= scale_factor
                # Update data scale attributes
                self.update_data_scale(scale_factor)

            # Perform actual QPHB operation
            x, s_vectors, rho_vector, dop_rho_vector, weights, outlier_t, outlier_tvt, cvx_result, converged = \
                qphb.iterate_qphb(x_in, s_vectors, rho_vector, dop_rho_vector, rv, weights, est_weights, outlier_tvt,
                                  rm,
                                  vmm, penalty_matrices, penalty_type, l1_lambda_vector, qphb_hypers, eff_hp, xmx_norms,
                                  dop_xmx_norms, None, None, None, nonneg, self.special_qp_params, xtol, 1,
                                  continue_history)

            # Update vz_offset column
            if self.fit_type.find('hybrid') > -1 and 'vz_offset' in self.special_qp_params.keys():
                # Update the response matrix with the current predicted y vector
                # vz_offset offsets chrono and eis predictions
                y_hat = rzm_vz @ x
                vz_sep = y_hat.copy()
                vz_sep[self.qphb_params['num_chrono']:] *= -1  # vz_offset > 0 means EIS Rp smaller than chrono Rp
                rm[:, self.special_qp_params['vz_offset']['index']] = vz_sep * vz_strength_vec

            if converged and it >= min_iter - 1:
                break
            elif it == max_iter - 1 and self.warn:
                warnings.warn(f'Solution did not converge within {max_iter} iterations')

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

        if self.fit_dop:
            dop_rho_in = self.qphb_params['dop_rho_vector'].copy()
        else:
            dop_rho_in = None

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

            hist = self._continue_from_init(self.qphb_params['hypers'], x_in, self.qphb_params['rv'],
                                            self.qphb_params['rm'], self.qphb_params['vmm'], rho_in, dop_rho_in, s_in,
                                            self.qphb_params['outlier_t'], self.qphb_params['penalty_matrices'],
                                            self.qphb_params['xmx_norms'], self.qphb_params['dop_xmx_norms'],
                                            self.qphb_params['est_weights'], weights_in,
                                            self.qphb_params['l1_lambda_vector'], nonneg=self.fit_kwargs['nonneg'],
                                            update_scale=False, penalty_type=self.fit_kwargs['penalty_type'],
                                            eff_hp=self.fit_kwargs['eff_hp'], xtol=xtol, max_iter=max_iter,
                                            **new_hypers, **kw)
            x_in = hist[-1]['x'].copy()
            rho_in = hist[-1]['rho_vector'].copy()
            dop_rho_in = deepcopy(hist[-1]['dop_rho_vector'])
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
        # s_in = self.qphb_params['s_vectors'].copy()
        weights_in = self.qphb_params['weights'].copy()
        if self.fit_dop:
            dop_rho_in = self.qphb_params['dop_rho_vector'].copy()
        else:
            dop_rho_in = None

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

            hist = self._continue_from_init(self.qphb_params['hypers'], x_in, self.qphb_params['rv'],
                                            self.qphb_params['rm'], self.qphb_params['vmm'], rho_in, dop_rho_in, s_in,
                                            self.qphb_params['outlier_t'], self.qphb_params['penalty_matrices'],
                                            self.qphb_params['xmx_norms'], self.qphb_params['dop_xmx_norms'],
                                            self.qphb_params['est_weights'], weights_in,
                                            self.qphb_params['l1_lambda_vector'], nonneg=self.fit_kwargs['nonneg'],
                                            update_scale=False, penalty_type=self.fit_kwargs['penalty_type'],
                                            eff_hp=self.fit_kwargs['eff_hp'], xtol=xtol, max_iter=max_iter,
                                            **new_hypers, **kw)
            x_in = hist[-1]['x'].copy()
            rho_in = hist[-1]['rho_vector'].copy()
            # s_in = hist[-1]['s_vectors'].copy()
            weights_in = hist[-1]['weights'].copy()
            dop_rho_in = deepcopy(hist[-1]['dop_rho_vector'])

            history += hist
            hypers += [new_hypers] * len(hist)

        candidate_x = [h['x'] for h in history]

        return candidate_x, history, hypers

    def generate_candidates(self, s0_multiplier=4, s0_steps=2, weight_multiplier=0.5, weight_steps=3,
                            include_qphb_history=True, fill=True, min_fill_num=None,
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
        # down_x, down_history, down_hypers = self._generate_candidates_s0(s0_multiplier ** -1, s0_steps,
        #                                                                       xtol, max_iter, **kw)
        up_x, up_history, up_hypers = self._generate_candidates_s0(s0_multiplier, s0_steps, xtol, max_iter, **kw)

        # Get relevant hyperparameters for default solution
        hypers_keys = list(down_hypers[0].keys()) + list(up_hypers[0].keys())
        default_hypers = [{k: self.fit_kwargs.get(k, None) for k in hypers_keys}] * len(qphb_x)

        # Gather all candidates
        candidate_history = qphb_history + up_history + down_history
        candidate_hypers = default_hypers + up_hypers + down_hypers
        candidate_x = np.array(qphb_x + up_x + down_x)

        # Evaluate llh
        # TODO: incorporate outlier_t into llh calculation
        if llh_kw is None:
            llh_kw = {}
        cand_weights = [
            qphb.estimate_weights(x, self.qphb_params['rv'], self.qphb_params['vmm'], self.qphb_params['rm'])[0]
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
        candidate_peak_results = [
            self.find_peaks(x=self.extract_qphb_parameters(x)['x'], return_info=True, **find_peaks_kw)
            for x in candidate_x
        ]
        candidate_peak_tau = [cpr[0] for cpr in candidate_peak_results]
        candidate_peak_info = [cpr[3] for cpr in candidate_peak_results]
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
            'peak_info': candidate_peak_info,
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

        # Identify discrete models
        unique_num_peaks = np.unique(candidate_num_peaks)

        # Get best candidate for each num_peaks
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
                'peak_info': candidate_peak_info[best_index[0][0]],
                'history': candidate_history[best_index[0][0]],
                'hypers': candidate_hypers[best_index[0][0]]
            }

        # Fill in missing num_peaks
        if fill:
            new_candidates = {}

            # Fill to the requested minimum number of peaks
            if min_fill_num is None:
                # Don't fill below simplest candidate
                min_fill_num = unique_num_peaks[0]
            elif min_fill_num < 0:
                # Fill by the specified number below the simplest candidate
                min_fill_num = max(1, unique_num_peaks[0] + min_fill_num)
            if min_fill_num < unique_num_peaks[0]:
                # Insert a dummy entry to force the below loop to fill in from min_fill_num
                unique_num_peaks = np.insert(unique_num_peaks, 0, min_fill_num - 1)

            fill_index = np.where(np.diff(unique_num_peaks) > 1)[0]
            print('fill_index:', fill_index)
            for fi in fill_index:
                lo_num = unique_num_peaks[fi]
                hi_num = unique_num_peaks[fi + 1]
                # lo_peaks = self.best_candidate_dict[lo_num]['peak_tau']
                hi_peaks = self.best_candidate_dict[hi_num]['peak_tau']
                hi_peak_info = self.best_candidate_dict[hi_num]['peak_info']

                # # Find peaks that are in hi_peaks but not in lo_peaks
                # new_peak_index = peaks.find_new_peaks(np.log(hi_peaks), np.log(lo_peaks))
                # new_peak_info = {k: v[new_peak_index] for k, v in self.best_candidate_dict[hi_num]['peak_info'].items()}

                # Sort new peaks by min(prominence, height)
                min_prom = np.minimum(hi_peak_info['prominences'], hi_peak_info['peak_heights'])
                sort_index = np.argsort(min_prom)[::-1]

                # # Start with the [lo_num] most prominent peaks in the hi_num candidate
                # base_peaks = hi_peaks[sort_index[:lo_num]]

                # Add peaks one at a time, starting from most prominent
                for j in range(lo_num + 1, hi_num):
                    # new_peak_tau = np.append(base_peaks, hi_peaks[sort_index[j]])
                    new_peak_tau = hi_peaks[sort_index[:j]]
                    new_peak_info = {
                        k: v[sort_index[:j]] for k, v in self.best_candidate_dict[hi_num]['peak_info'].items()
                    }

                    # New candidate needs subsetted peak_tau and peak_info,
                    # but will take all other info from hi_num candidate
                    new_candidates[j] = {
                        'x': self.best_candidate_dict[hi_num]['x'],
                        'llh': self.best_candidate_dict[hi_num]['llh'],
                        'bic': self.best_candidate_dict[hi_num]['bic'],
                        'peak_tau': new_peak_tau,
                        'peak_info': new_peak_info,
                        'history': self.best_candidate_dict[hi_num]['history'],
                        'hypers': self.best_candidate_dict[hi_num]['hypers'],
                    }

            # print('new candidates:', new_candidates)
            self.best_candidate_dict.update(new_candidates)

            # Sort by num_peaks
            sorted_keys = sorted(list(self.best_candidate_dict.keys()))
            self.best_candidate_dict = {k: self.best_candidate_dict[k] for k in sorted_keys}

        self.best_candidate_df = pd.DataFrame(
            np.vstack((candidate_num_peaks[best_indices], candidate_num_peaks[best_indices],
                       candidate_llh[best_indices], candidate_bic[best_indices],
                       candidate_llh[best_indices] - best_llh, candidate_bic[best_indices] - best_bic)).T,
            columns=['model_id', 'num_peaks', 'llh', 'bic', 'rel_llh', 'rel_bic']
        )

        return self.candidate_dict.copy()

    def convert_candidate_to_discrete(self, candidate_num_peaks, model_init_kw=None, prior=True,
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
            dem.fit_eis(self.get_fit_frequencies(), self.z_fit, from_drt=True, prior=prior, **fit_kw)
        else:
            # TODO: implement fit_chrono and fit_hybrid methods for DiscreteElementModel
            raise ValueError('dual_fit is currently only implemented for EIS data')

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
                'lml-bic': 0.5 * (lml - 0.5 * bic),
                'peak_tau': dem.get_peak_tau(),
                'time_constants': dem.get_time_constants()
            }

        # Get best metrics across models
        discrete_lb = 0.5 * (discrete_lml - 0.5 * discrete_bic)
        best_llh = np.max(discrete_llh)
        best_lml = np.max(discrete_lml)
        best_bic = np.min(discrete_bic)
        best_lb = np.max(discrete_lb)

        # Fill in metrics relative to best
        for i, candidate in enumerate(candidates):
            self.discrete_candidate_dict[candidate]['rel_llh'] = discrete_llh[i] - best_llh
            self.discrete_candidate_dict[candidate]['rel_bic'] = discrete_bic[i] - best_bic
            self.discrete_candidate_dict[candidate]['rel_lml'] = discrete_lml[i] - best_lml
            self.discrete_candidate_dict[candidate]['rel_lml-bic'] = discrete_lb[i] - best_lb

        self.discrete_candidate_df = pd.DataFrame(
            np.vstack([
                candidates, np.array(candidates).astype(int),
                discrete_llh, discrete_bic, discrete_lml, discrete_lb,
                discrete_llh - best_llh, discrete_bic - best_bic, discrete_lml - best_lml, discrete_lb - best_lb
            ]).T,
            columns=['model_id', 'num_peaks', 'llh', 'bic', 'lml', 'lml-bic',
                     'rel_llh', 'rel_bic', 'rel_lml', 'rel_lml-bic']
        )

        if self.print_diagnostics:
            print('Discrete models created in {:.3f} s'.format(time.time() - start))

        return self.discrete_candidate_dict.copy()

    def _dual_fit_core(self, times, i_signal, v_signal, frequencies, z, generate_kw=None, discrete_kw=None, **qphb_kw):
        start_time = time.time()

        if qphb_kw is None:
            qphb_kw = {}

        if times is None:
            self.fit_eis(frequencies, z, **qphb_kw)
        elif frequencies is None:
            self.fit_chrono(times, i_signal, v_signal, **qphb_kw)
        else:
            self.fit_hybrid(times, i_signal, v_signal, frequencies, z, **qphb_kw)

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

    def dual_fit_eis(self, frequencies, z, generate_kw=None, discrete_kw=None, **qphb_kw):
        self._dual_fit_core(None, None, None, frequencies, z, generate_kw=generate_kw,
                            discrete_kw=discrete_kw, **qphb_kw)

    def dual_fit_chrono(self, times, i_signal, v_signal, generate_kw=None, discrete_kw=None, **qphb_kw):
        self._dual_fit_core(times, i_signal, v_signal, None, None, generate_kw=generate_kw,
                            discrete_kw=discrete_kw, **qphb_kw)

    def dual_fit_hybrid(self, times, i_signal, v_signal, frequencies, z, generate_kw=None, discrete_kw=None, **qphb_kw):
        self._dual_fit_core(times, i_signal, v_signal, frequencies, z, generate_kw=generate_kw,
                            discrete_kw=discrete_kw, **qphb_kw)

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
                                    tau=None, ppd=20, **kw):
        candidate_info = self.get_candidate(candidate_id, candidate_type)

        if candidate_type == 'continuous':
            candidate_x = self.extract_qphb_parameters(candidate_info['x'])['x']

            mark_peaks_default = {'peak_tau': candidate_info['peak_tau']}
            if mark_peaks_kw is None:
                mark_peaks_kw = {}
            mark_peaks_kw = dict(mark_peaks_default, **mark_peaks_kw)

            return self.plot_distribution(tau=tau, x=candidate_x, mark_peaks=mark_peaks, mark_peaks_kw=mark_peaks_kw,
                                          **kw)
        else:
            dem = candidate_info['model']

            if tau is None:
                tau = self.get_tau_eval(ppd)

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

    def evaluate_norm_bayes_factors(self, candidate_type, criterion=None, candidate_id=None,
                                    na_val=None):
        cand_df = self.get_candidate_df(candidate_type)

        if criterion is None:
            criterion = 'bic'

        if candidate_id is None:
            return stats.norm_bayes_factors(cand_df[criterion].values, criterion)
        else:
            cand_index = np.where(cand_df['model_id'] == candidate_id)
            bf = stats.norm_bayes_factors(cand_df[criterion].values, criterion)
            if na_val is not None and len(cand_index) == 0:
                return na_val
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
            raise ValueError(f"Invalid candidate_type {candidate_type}. Options: 'continuous', 'discrete', 'pfrt'")

    def get_best_candidate_id(self, candidate_type, criterion=None):
        candidate_types = ['discrete', 'continuous']
        if candidate_type not in candidate_types:
            raise ValueError(f'Invalid candidate_type {candidate_type}. Options: {candidate_types}')

        criterion_directions = {
            'bic': -1,
            'lml': 1,
            'lml-bic': 1
        }

        if criterion is not None:
            if criterion_directions.get(criterion, None) is None:
                raise ValueError(f'Invalid criterion {criterion}. Options: {criterion_directions.keys()}')

        if candidate_type == 'discrete':
            if criterion is None:
                criterion = 'lml-bic'
            model_df = self.discrete_candidate_df
        else:
            if criterion is None:
                criterion = 'bic'
            model_df = self.best_candidate_df

        crit_direction = criterion_directions[criterion]

        # if criterion == 'bic-lml':
        #     crit_values = (-0.5 * model_df['bic'] + model_df['lml']).values
        # else:
        crit_values = model_df[criterion].values
        best_index = np.argmax(crit_direction * crit_values)
        best_id = model_df.loc[model_df.index[best_index], 'model_id']

        return best_id

    def predict_pdrt(self, tau=None, ppd=20, criterion='bic', criterion_factor=1):
        if tau is None:
            tau = self.get_tau_eval(ppd)

        spread_func = evaluation.get_similarity_function('gaussian')

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
    def _pfrt_fit_core(self, times, i_signal, v_signal, frequencies, z, factors=None, max_iter_per_step=10,
                       max_init_iter=20, xtol=1e-2, nonneg=True, series_neg=False, **kw):

        # Get default hyperparameters
        qphb_hypers = qphb.get_default_hypers(True, self.fit_dop, self.nu_basis_type)
        init_kw = dict(qphb_hypers, **kw)

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
            # new_hypers = {}
            return new_hypers

        # Initialize fit at first factor
        factor = factors[0]
        init_hypers = prep_step_hypers(factor)
        init_kw.update(init_hypers)

        # Perform qphb fit corresponding to data type
        if times is None:
            self.fit_eis(frequencies, z, nonneg=nonneg, series_neg=series_neg, max_iter=max_init_iter, xtol=xtol,
                         **init_kw)
        elif frequencies is None:
            self.fit_chrono(times, i_signal, v_signal, nonneg=nonneg, series_neg=series_neg, max_iter=max_init_iter,
                            xtol=xtol, **init_kw)
        else:
            self.fit_hybrid(times, i_signal, v_signal, frequencies, z, nonneg=nonneg, series_neg=series_neg,
                            max_iter=max_init_iter, xtol=xtol, **init_kw)

        # Initialize history and insert initial fit
        pfrt_history = []
        step_x = []
        step_llh = []
        step_hypers = []
        step_backgrounds = []
        # hyper_history = []
        step_p_mat = []

        def step_update(old_history, new_history, new_hypers):
            current_history = old_history + new_history
            # hyper_history += [new_hypers] * len(new_history)
            step_hypers.append(new_hypers)
            step_x.append(new_history[-1]['x'])

            # Get weights estimate based only on current x for llh calculation
            # TODO: incorporate outlier_t into pfrt_fit LLH
            weights, _, _ = qphb.estimate_weights(new_history[-1]['x'], self.qphb_params['rv'], self.qphb_params['vmm'],
                                                  self.qphb_params['rm'])

            step_llh.append(self.evaluate_llh(weights, x=step_x[-1], marginalize_weights=True))

            # Get P matrix
            p_matrix, _ = qphb.calculate_pq(self.qphb_params['rm'], self.qphb_params['rv'],
                                            self.qphb_params['penalty_matrices'], self.fit_kwargs['penalty_type'],
                                            self.qphb_params['hypers'], self.qphb_params['l1_lambda_vector'],
                                            new_history[-1]['rho_vector'], new_history[-1]['dop_rho_vector'],
                                            new_history[-1]['s_vectors'],
                                            weights, self.special_qp_params)

            step_p_mat.append(p_matrix)

            if kw.get('subtract_background', False) and kw.get('background_type', 'static') == 'dynamic':
                resid = self.qphb_params['rv_orig'] - self.qphb_params['rm_orig'] @ new_history[-1]['x']
                y_bkg = (self.qphb_params['rm_bkg'] @ resid) * self.response_signal_scale
            elif kw.get('subtract_background', False):
                y_bkg = self.raw_response_background.copy()
            else:
                y_bkg = None
            step_backgrounds.append(y_bkg)

            return current_history

        pfrt_history = step_update(pfrt_history, self.qphb_history, init_hypers)

        # print(factor, len(pfrt_history))

        # Proceed through remaining factors
        for factor in factors[1:]:
            update_hypers = prep_step_hypers(factor)
            # print(update_hypers)

            x_in = pfrt_history[-1]['x'].copy()
            rho_in = pfrt_history[-1]['rho_vector'].copy()
            s_in = pfrt_history[-1]['s_vectors'].copy()
            weights_in = pfrt_history[-1]['weights'].copy()
            outlier_t_in = pfrt_history[-1]['outlier_t'].copy()
            dop_rho_in = deepcopy(pfrt_history[-1]['dop_rho_vector'])

            if self.fit_type.find('hybrid') > -1:
                # At lower factors, increase the chrono-eis weight refactor to give chrono more weight to push
                #   against the stronger regularization
                # At higher factors, decrease the chrono weight so that it doesn't go crazy, increase the EIS weight
                #   so that it has a chance to shift the DRT
                # TODO: decide how to set weight factors as the pfrt factor changes
                eis_weight_factor = self.qphb_params['eis_weight_factor']  # ** (factor ** -0.5)
                chrono_weight_factor = self.qphb_params['chrono_weight_factor']  # ** (factor ** -0.5)
                # print('factors:', eis_weight_factor, chrono_weight_factor)
            else:
                eis_weight_factor = None
                chrono_weight_factor = None
            hist = self._continue_from_init(self.qphb_params['hypers'], x_in, self.qphb_params['rv'],
                                            self.qphb_params['rm'], self.qphb_params['vmm'], rho_in, dop_rho_in, s_in,
                                            outlier_t_in, self.qphb_params['penalty_matrices'],
                                            self.qphb_params['xmx_norms'], self.qphb_params['dop_xmx_norms'],
                                            self.qphb_params['est_weights'], weights_in,
                                            self.qphb_params['l1_lambda_vector'], nonneg=self.fit_kwargs['nonneg'],
                                            update_scale=False,
                                            penalty_type=self.fit_kwargs['penalty_type'],
                                            eff_hp=self.fit_kwargs['eff_hp'], xtol=xtol, max_iter=max_iter_per_step,
                                            eis_weight_factor=eis_weight_factor,
                                            chrono_weight_factor=chrono_weight_factor,
                                            **update_hypers)

            pfrt_history = step_update(pfrt_history, hist, update_hypers)
            # print(factor, len(hist))

        self.pfrt_history = pfrt_history

        self.pfrt_result = {
            'factors': factors,
            'step_x': step_x,
            'step_llh': step_llh,
            'step_p_mat': step_p_mat,
            'step_hypers': step_hypers,
            'step_backgrounds': step_backgrounds
        }

    def pfrt_fit_eis(self, frequencies, z, factors=None, max_iter_per_step=10,
                     max_init_iter=20, xtol=1e-2, nonneg=True, **kw):
        self._pfrt_fit_core(None, None, None, frequencies, z, factors=factors, max_iter_per_step=max_iter_per_step,
                            max_init_iter=max_init_iter, xtol=xtol, nonneg=nonneg, **kw)

    def pfrt_fit_chrono(self, times, i_signal, v_signal, factors=None, max_iter_per_step=10,
                        max_init_iter=20, xtol=1e-2, nonneg=True, **kw):
        self._pfrt_fit_core(times, i_signal, v_signal, None, None, factors=factors, max_iter_per_step=max_iter_per_step,
                            max_init_iter=max_init_iter, xtol=xtol, nonneg=nonneg, **kw)

    def pfrt_fit_hybrid(self, times, i_signal, v_signal, frequencies, z, factors=None, max_iter_per_step=10,
                        max_init_iter=20, xtol=1e-2, nonneg=True, **kw):
        self._pfrt_fit_core(times, i_signal, v_signal, frequencies, z, factors=factors,
                            max_iter_per_step=max_iter_per_step, max_init_iter=max_init_iter,
                            xtol=xtol, nonneg=nonneg, **kw)

    def predict_pfrt(self, tau=None, tau_pfrt=None, sign=None, prior_mu=-4, prior_sigma=0.5, find_peaks_kw=None,
                     n_eff_factor=0.5,
                     fxx_var_floor=1e-5, extend_var=True,
                     smooth=True, smooth_kw=None, integrate=False, integrate_threshold=1e-6,
                     normalize=True):

        if sign is None:
            sign = self.default_dist_sign

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

        # TODO: temp test - remove
        # log_post_eff = log_prior

        # Normalize to posterior area
        if len(factors) > 1:
            post_area = np.trapz(np.exp(log_post_eff), x=np.log(factors))
        else:
            post_area = np.exp(log_post_eff[0])
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
            # x_drt = self.get_drt_params(x_raw, sign)
            fxx = self.predict_distribution(tau_pfrt, x=x_drt, sign=sign, order=2, normalize=True)

            # Get curvature std
            fxx_cov = self.estimate_distribution_cov(tau_pfrt, p_matrix=step_p_mat[i], order=2, sign=sign,
                                                     normalize=True,
                                                     var_floor=fxx_var_floor,
                                                     extend_var=extend_var
                                                     )
            fxx_sigma = np.diag(fxx_cov) ** 0.5
            fxx_sigmas.append(fxx_sigma)

            # Get dist and std
            f = self.predict_distribution(tau_pfrt, x=x_drt, sign=sign, order=0, normalize=True)
            f_cov = self.estimate_distribution_cov(tau_pfrt, p_matrix=step_p_mat[i], order=0, sign=sign,
                                                   normalize=True,
                                                   var_floor=fxx_var_floor,
                                                   extend_var=extend_var
                                                   )
            f_sigma = np.diag(f_cov) ** 0.5

            # Find peaks
            if self.fit_kwargs['nonneg'] and sign != 0:
                # This covers both the standard nonneg case and the series_neg case
                peak_index, peak_info = signal.find_peaks(-sign * fxx, **find_peaks_kw)
                # peak_index, peak_info = signal.find_peaks(-fxx / fxx_sigma, **find_peaks_kw)
            else:
                # Signed distribution - Find negative and positive peaks separately
                peak_index_list = []
                peak_info_list = []
                for peak_sign in [-1, 1]:
                    peak_index, peak_info = signal.find_peaks(-peak_sign * fxx, **find_peaks_kw)
                    # Limit to peaks that are positive in the direction of the current sign
                    pos_index = peak_sign * f[peak_index] > 0
                    # print(pos_index)
                    peak_index = peak_index[pos_index]
                    peak_info = {k: v[pos_index] for k, v in peak_info.items()}

                    peak_index_list.append(peak_index)
                    peak_info_list.append(peak_info)
                peak_index = np.concatenate(peak_index_list)
                peak_info = {k: np.concatenate([pi[k] for pi in peak_info_list]) for k in peak_info.keys()}

            # Use prominence or height, whichever is smaller
            min_prom = np.minimum(peak_info['prominences'], peak_info['peak_heights'])

            # Evaluate peak credibility
            # Get probability that a peak in the curvature exists at the identified location
            # peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, fxx_sigma[peak_index])
            fxx_prob = 1 - 2 * stats.cdf_normal(0, min_prom, fxx_sigma[peak_index])
            # peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, 1)
            # peak_prob=1

            # Get probability that the height of the function (not curvature) is greater than zero
            peak_heights = f[peak_index]
            f_prob = 1 - 2 * stats.cdf_normal(0, peak_heights * np.sign(peak_heights), f_sigma[peak_index])

            # Take the lower of the two probabilities (don't multiply - not independent)
            # peak_prob = fxx_prob * f_prob
            peak_prob = np.minimum(f_prob, fxx_prob)

            step_pfrt[i, peak_index] = peak_prob

            tot_pfrt[peak_index] += post_prob_eff[i] * peak_prob

        # Divide by summed posterior probs
        tot_pfrt /= np.sum(post_prob_eff)

        self.pfrt_result['tau_pfrt'] = tau_pfrt
        self.pfrt_result['raw_pfrt'] = tot_pfrt.copy()
        self.pfrt_result['step_pfrt'] = step_pfrt

        if smooth:
            # Smooth to aggregate neighboring peak probs, which may arise due to
            # slight peak shifts with hyperparameter changes
            spread_func = evaluation.get_similarity_function('gaussian')
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

        return tot_pfrt  # , step_pfrt, post_prob_eff, step_x, step_p_mat, factors, fxx_sigmas

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

    # Prediction
    # --------------------------------------------
    @property
    def num_drt_params(self):
        return len(self.basis_tau) * (1 + int(self.series_neg))

    def get_drt_params(self, x, sign=1):
        if x is not None:
            if len(x) > self.num_drt_params:
                x = self.extract_qphb_parameters(x)['x']
        else:
            x = self.fit_parameters['x']

        if self.series_neg:
            # Get the coefficients corresponding to the requested sign
            if sign == 1:
                return x[:len(self.basis_tau)]
            elif sign == -1:
                return -x[len(self.basis_tau):]
            elif sign == 0:
                # Net distribution (pos + neg)
                return x[:len(self.basis_tau)] - x[len(self.basis_tau):]
            else:
                raise ValueError(f'Invalid sign {sign}. Options: -1, 0, 1')

        return x

    @property
    def default_dist_sign(self):
        if self.series_neg:
            return 0
        else:
            return 1

    @property
    def dop_indices(self):
        if self.fit_dop:
            start_index = self.special_qp_params['x_dop']['index']
            end_index = start_index + len(self.basis_nu)
        else:
            start_index, end_index = None, None

        return start_index, end_index

    def get_special_indices(self, key):
        start_index = self.special_qp_params[key]['index']
        end_index = start_index + self.special_qp_params[key].get('size', 1)
        return start_index, end_index

    def get_dop_params(self, x=None):
        if x is not None:
            if len(x) > len(self.basis_nu):
                x = self.extract_qphb_parameters(x)['x_dop']
        else:
            x = self.fit_parameters['x_dop']

        return x
    
    def get_drt_norm(self, normalize: bool, normalize_by: Optional[float] = None, 
                     x: Optional[ndarray] = None):
        if normalize_by is not None:
            normalize = True
            
        if normalize:
            if normalize_by is None:
                normalize_by = self.predict_r_p(x=x)
        else:
            normalize_by = 1
            
        return normalize_by

    def predict_distribution(self, tau=None, ppd=20, x=None, order=0, sign=1, 
                             normalize=False, normalize_by: Optional[float] = None):
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

        x = self.get_drt_params(x, sign)

        normalize_by = self.get_drt_norm(normalize, normalize_by, x=x)

        return basis_matrix @ x / normalize_by

    def estimate_distribution_cov(self, tau=None, ppd=20, p_matrix=None, sign=1, order=0, 
                                  normalize=False, normalize_by: Optional[float] = None,
                                  var_floor=0.0, tau_data_limits=None, extend_var=False):
        """
        DRT parameter covariance matrix
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

        # Get normalization - square for covariance
        normalize_by = self.get_drt_norm(normalize, normalize_by)
        normalize_by **= 2

        # Get parameter covariance
        x_cov = self.estimate_param_cov(p_matrix)

        if x_cov is not None:
            # Limit to DRT parameters
            x_cov = x_cov[self.get_qp_mat_offset():, self.get_qp_mat_offset():]

            # Limit to requested sign
            if self.series_neg:
                tau_len = len(self.basis_tau)
                if sign == 1:
                    x_cov = x_cov[:tau_len, :tau_len]
                elif sign == -1:
                    x_cov = x_cov[tau_len:, tau_len:]
                elif sign == 0:
                    x_cov_pos = x_cov[:tau_len, :tau_len]
                    x_cov_neg = x_cov[tau_len:, tau_len:]
                    x_cov_pn = x_cov[:tau_len, tau_len:]
                    x_cov_np = x_cov[tau_len:, :tau_len]
                    x_cov = x_cov_pos + x_cov_neg - (x_cov_pn + x_cov_np)

            # Distribution covariance given by matrix product
            dist_cov = basis_matrix @ x_cov @ basis_matrix.T

            # Normalize
            dist_cov = dist_cov / normalize_by

            # Add variance beyond measurement bounds
            # if extrapolation_var_scale > 0:
            #     add_var = np.zeros(len(tau))
            #     if tau_data_limits is None:
            #         tau_data_limits = pp.get_tau_lim(self.get_fit_frequencies(True), self.get_fit_times(True),
            #                                          self.step_times)
            #     t_start, t_end = tau_data_limits
            #     add_var[tau > t_end] = np.log(tau[tau > t_end] / t_end) ** 2
            #     add_var[tau < t_start] = (-np.log(tau[tau < t_start] / t_start)) ** 2
            #     add_var *= extrapolation_var_scale
            #     dist_cov += np.diag(add_var)

            # Extend variance beyond measurement bounds
            if extend_var:
                if tau_data_limits is None:
                    tau_data_limits = pp.get_tau_lim(self.get_fit_frequencies(True), self.get_fit_times(True),
                                                     self.step_times)
                t_left, t_right = tau_data_limits
                left_index = utils.array.nearest_index(tau, t_left) + 1
                right_index = utils.array.nearest_index(tau, t_right)
                var = np.diag(dist_cov).copy()
                # Set the variance outside the measurement bounds to the variance at the corresponding bound
                var[:left_index] = np.maximum(var[:left_index], var[left_index])
                var[right_index:] = np.maximum(var[right_index:], var[right_index])
                dist_cov[np.diag_indices(dist_cov.shape[0])] = var

            # Set variance floor
            if var_floor > 0:
                dist_var = np.diag(dist_cov).copy()
                dist_var[dist_var < var_floor] = var_floor
                np.fill_diagonal(dist_cov, dist_var)

            return dist_cov
        else:
            # Error in precision matrix inversion
            return None

    def estimate_dop_cov(self, nu=None, p_matrix=None, normalize=False, 
                         normalize_tau=None, normalize_quantiles=(0.25, 0.75), 
                         var_floor=0.0, order=0,
                         delta_density=False):
        # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
        if nu is None:
            nu = self.basis_nu

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(self.basis_nu, nu,
                                                        self.nu_basis_type, epsilon=self.nu_epsilon,
                                                        order=order, zga_params=None)

        # Normalize
        normalize_by = self.get_dop_norm(nu, normalize, normalize_tau, normalize_quantiles)
        normalize_by **= 2

        # Get parameter covariance
        x_cov = self.estimate_param_cov(p_matrix)

        if x_cov is not None:
            # Limit to DOP parameters
            dop_start, dop_end = self.dop_indices
            x_cov = x_cov[dop_start:dop_end, dop_start:dop_end]

            # Normalize delta function magnitude to grid spacing
            if delta_density and self.nu_basis_type == 'delta':
                dnu = self.get_nu_basis_spacing()
                x_cov = x_cov / dnu

            # Distribution covariance given by matrix product
            dist_cov = basis_matrix @ x_cov @ basis_matrix.T

            # Normalize
            dist_cov = dist_cov / normalize_by

            # Set variance floor
            if var_floor > 0:
                dist_var = np.diag(dist_cov).copy()
                dist_var[dist_var < var_floor] = var_floor
                np.fill_diagonal(dist_cov, dist_var)

            return dist_cov
        else:
            # Error in precision matrix inversion
            return None

    def predict_distribution_ci(self, tau=None, ppd=20, x=None, order=0, sign=1, 
                                normalize=False, normalize_by: Optional[float] = None,
                                quantiles=[0.025, 0.975]):
        # Get distribution std
        dist_cov = self.estimate_distribution_cov(tau=tau, ppd=ppd, order=order, sign=sign, 
                                                  normalize=normalize, normalize_by=normalize_by)
        if dist_cov is not None:
            dist_sigma = np.diag(dist_cov) ** 0.5

            # Distribution mean
            dist_mu = self.predict_distribution(tau=tau, ppd=ppd, x=x, order=order, sign=sign, 
                                                normalize=normalize, normalize_by=normalize_by)

            # Determine number of std devs to obtain quantiles
            s_lo, s_hi = stats.std_normal_quantile(quantiles)

            dist_lo = dist_mu + s_lo * dist_sigma
            dist_hi = dist_mu + s_hi * dist_sigma

            return dist_lo, dist_hi
        else:
            # Error in covariance calculation
            return None, None

    def predict_dop_ci(self, nu=None, x=None, normalize=False, normalize_tau=None, quantiles=[0.025, 0.975],
                       order=0, normalize_quantiles=(0.25, 0.75), delta_density=False, include_ideal=True):
        # Get distribution std
        dist_cov = self.estimate_dop_cov(nu, order=order, normalize=normalize, 
                                         normalize_tau=normalize_tau,
                                         normalize_quantiles=normalize_quantiles,
                                         delta_density=delta_density)
        if dist_cov is not None:
            dist_sigma = np.diag(dist_cov) ** 0.5

            # Distribution mean
            dist_mu = self.predict_dop(nu=nu, x=x, normalize=normalize,
                                       normalize_tau=normalize_tau, order=order,
                                       normalize_quantiles=normalize_quantiles,
                                       delta_density=delta_density, include_ideal=include_ideal)

            # Determine number of std devs to obtain quantiles
            s_lo, s_hi = stats.std_normal_quantile(quantiles)

            dist_lo = dist_mu + s_lo * dist_sigma
            dist_hi = dist_mu + s_hi * dist_sigma

            return dist_lo, dist_hi
        else:
            # Error in covariance calculation
            return None, None

    def get_nu_basis_spacing(self):
        # For each grid point, get the minimum distance to next basis location
        if self.fixed_basis_nu is not None:
            basis_nu = self.fixed_basis_nu
        else:
            basis_nu = self.basis_nu

        dnu = np.diff(np.sort(basis_nu))
        dnu = np.minimum(dnu[1:], dnu[:-1])
        median_dnu = np.median(dnu)
        dnu = np.append(np.insert(dnu, 0, median_dnu), median_dnu)
        return dnu

    def predict_dop(self, nu=None, x=None, normalize=False, normalize_tau=None, order=0, return_nu=False,
                    normalize_quantiles=(0, 1), delta_density=False, include_ideal=True):
        if nu is None:
            nu = np.linspace(-1, 1, 401)
            # Ensure basis_nu is included
            nu = np.unique(np.concatenate([self.basis_nu, nu]))
            # Ensure pure inductance, resistance, and capacitance are included
            nu = np.unique(np.concatenate([nu, np.array([-1, 0, 1])]))
        else:
            nu = np.sort(nu)

        # Construct basis matrix
        basis_matrix = basis.construct_func_eval_matrix(self.basis_nu, nu,
                                                        self.nu_basis_type, epsilon=self.nu_epsilon,
                                                        order=order, zga_params=None)

        x = self.get_dop_params(x=x)

        if delta_density:
            dnu = self.get_nu_basis_spacing()
            if self.nu_basis_type == 'delta':
                x = x / dnu

        dop = basis_matrix @ x


        # TODO: revisit normalization
        normalize_by = self.get_dop_norm(nu, normalize, normalize_tau, normalize_quantiles)

        # print(normalize_by)

        dop /= normalize_by
        
        # Add pure inductance, resistance, and capacitance
        if include_ideal:
            ohmic_index = np.where(nu == 0)
            if len(ohmic_index) == 1:
                r_inf = self.fit_parameters['R_inf']
                if delta_density:
                    r_inf = r_inf / dnu[utils.array.nearest_index(self.basis_nu, 0)]
                if normalize:
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                    r_inf = r_inf / (normalize_by[ohmic_index] * self.nu_basis_area)
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                dop[ohmic_index] += r_inf
    
    
            induc_index = np.where(nu == 1)
            if len(induc_index) == 1:
                induc = self.fit_parameters['inductance']
                if delta_density:
                    induc = induc / dnu[utils.array.nearest_index(self.basis_nu, 1)]
                if normalize:
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                    induc = induc / (normalize_by[induc_index] * self.nu_basis_area)
                dop[induc_index] += induc
    
            cap_index = np.where(nu == -1)
            if len(cap_index) == 1:
                c_inv = self.fit_parameters['C_inv']
                if delta_density:
                    c_inv = c_inv / dnu[utils.array.nearest_index(self.basis_nu, -1)]
                if normalize:
                    # Since ideal elements are delta functions, they should not be 
                    # scaled by the non-ideal basis function area
                    c_inv = c_inv / (normalize_by[cap_index] * self.nu_basis_area)
                dop[cap_index] += c_inv

        if return_nu:
            return nu, dop
        else:
            return dop
        
    def get_dop_norm(self, nu, normalize: bool = False, normalize_tau: Optional[tuple] = None, 
                     normalize_quantiles: tuple = (0, 1)):
        if normalize:
            if normalize_tau is None:
                data_tau_lim = pp.get_tau_lim(self.get_fit_frequencies(), self.get_fit_times(), self.step_times)
                normalize_tau = np.array(data_tau_lim)
            
            normalize_by = phasance.phasor_scale_vector(nu, normalize_tau, normalize_quantiles)
            normalize_by /= self.nu_basis_area
        else:
            normalize_by = 1
            
        return normalize_by

    def predict_response(self, times=None, input_signal=None, step_times=None, step_sizes=None, op_mode=None,
                         offset_steps=None,
                         smooth_inf_response=None, x=None, include_vz_offset=True, subtract_background=True,
                         y_bkg=None, v_baseline=None):
        # If chrono_mode is not provided, use fitted chrono_mode
        if op_mode is None:
            op_mode = self.chrono_mode
        utils.validation.check_ctrl_mode(op_mode)

        # If times is not provided, use self.t_fit
        if times is None:
            times = self.get_fit_times()

        # If kwargs not provided, use same values used in fitting
        if offset_steps is None:
            offset_steps = self.fit_kwargs['offset_steps']
        if smooth_inf_response is None:
            smooth_inf_response = self.fit_kwargs['smooth_inf_response']

        # Get prediction matrix and vectors
        rm_drt, induc_rv, inf_rv, cap_rv, rm_dop = self._prep_chrono_prediction_matrix(times, input_signal,
                                                                               step_times, step_sizes, op_mode,
                                                                               offset_steps, smooth_inf_response)
        # Response matrices from _prep_response_prediction_matrix will be scaled. Rescale to data scale
        # rm *= self.input_signal_scale
        # induc_rv *= self.input_signal_scale
        # inf_rv *= self.input_signal_scale

        # Get parameters
        if x is not None:
            fit_parameters = self.extract_qphb_parameters(x)
        else:
            fit_parameters = self.fit_parameters

        x_drt = fit_parameters['x']
        x_dop = fit_parameters.get('x_dop', None)
        r_inf = fit_parameters.get('R_inf', 0)
        induc = fit_parameters.get('inductance', 0)
        c_inv = fit_parameters.get('C_inv', 0)

        if v_baseline is None:
            v_baseline = fit_parameters.get('v_baseline', 0)

        response = rm_drt @ x_drt + inf_rv * r_inf + induc * induc_rv + c_inv * cap_rv

        if x_dop is not None:
            response += rm_dop @ x_dop

        # if not subtract_background:
        #     if not np.array_equal(times, self.get_fit_times()):
        #         raise ValueError('Background can only be included if prediction times are same as fit times')
        #     response += self.raw_response_background

        if include_vz_offset:
            # # Need to back out the offset to get the fit parameter, apply offset to fit parameter by offset
            # v_baseline_param = (v_baseline + self.scaled_response_offset * self.response_signal_scale)
            # v_baseline_offset = v_baseline_param * (1 + self.fit_parameters.get('vz_offset', 0)) \
            #                     - self.scaled_response_offset * self.response_signal_scale
            # Apply vz_offset before adding baseline
            vz_strength_vec, _ = self._get_vz_strength_vec(
                times, vz_offset_eps=self.fit_parameters.get('vz_offset_eps', None)
            )
            response *= (1 + fit_parameters.get('vz_offset', 0) * vz_strength_vec)

        response += v_baseline

        if not subtract_background:
            if y_bkg is None:
                y_bkg = self.predict_chrono_background(times)
            if len(times) != len(y_bkg):
                raise ValueError('Length of background does not match length of times')
            response += y_bkg

        return response

    def predict_chrono_background(self, times):
        if self.background_gp is None:
            return np.zeros(len(times))
            # raise ValueError('Chrono background was not fitted. To fit the chrono background, '
            #                  'call fit_chrono or fit_hybrid with subtract_background=True')

        if np.array_equal(times, self.get_fit_times()):
            # If times are fit times, use precalculated background
            return self.raw_response_background
        else:
            # Predict using fitted GP
            if self.fit_kwargs['background_type'] == 'static':
                return self.background_gp.predict(times[:, None])
            else:
                # Need to account for correlation between DRT and background
                y_pred = self.predict_response(times)
                rm_bkg = background.get_background_matrix([self.background_gp], times[:, None],
                                                          y_drt=y_pred,
                                                          corr_power=self.fit_kwargs['background_corr_power'])
                # Get residuals for training data
                y_resid = self.raw_response_signal - self.predict_response()

                return rm_bkg @ y_resid

    def predict_z(self, frequencies, include_vz_offset=True, x=None,
                  include_dop=True, include_drt=True, include_inductance=True,
                  include_ohmic=True, include_cap=True
                  ):
        # Get matrix
        zm, zm_dop = self._prep_impedance_prediction_matrix(frequencies)

        # Get parameters
        if x is not None:
            fit_parameters = self.extract_qphb_parameters(x)
        else:
            fit_parameters = self.fit_parameters

        x_drt = fit_parameters['x']
        x_dop = fit_parameters.get('x_dop', None)
        r_inf = fit_parameters.get('R_inf', 0)
        induc = fit_parameters.get('inductance', 0)
        c_inv = fit_parameters.get('C_inv', 0)

        z = np.zeros(len(frequencies), dtype=complex)

        if include_drt:
            z += zm @ x_drt

        if include_ohmic:
            z += r_inf
        if include_inductance:
            z += induc * 2j * np.pi * frequencies
        if include_cap:
            z+= c_inv * (2j * np.pi * frequencies) ** -1

        if x_dop is not None and include_dop:
            z += zm_dop @ x_dop

        if include_vz_offset:
            _, vz_strength_vec = self._get_vz_strength_vec(None, frequencies,
                                                           vz_offset_eps=self.fit_parameters.get('vz_offset_eps', None))
            z *= (1 - fit_parameters.get('vz_offset', 0) * vz_strength_vec)

        return z

    def predict_sigma(self, measurement):
        if measurement == 'chrono':
            key = 'v_sigma_tot'
        elif measurement == 'eis':
            key = 'z_sigma_tot'

        return self.fit_parameters.get(key, None)

    def predict_r_p(self, sign=None, absolute=False, x=None, raw=False):
        basis_area = basis.get_basis_func_area(self.tau_basis_type, self.tau_epsilon, self.zga_params)

        if sign is None:
            if self.series_neg:
                sign = 0
            else:
                sign = 1

        if raw:
            if len(x) > self.num_drt_params:
                x = x[self.get_qp_mat_offset():]
        else:
            x = self.get_drt_params(x, sign)

        if absolute:
            sum_x = np.sum(np.abs(x))
        else:
            sum_x = np.sum(x)
        return sum_x * basis_area

    def predict_r_inf(self):
        r_inf = self.fit_parameters.get('R_inf', 0)

        if self.fit_dop and self.nu_basis_type == 'delta':
            zero_index = np.where(self.basis_nu == 0)
            if len(zero_index) == 1:
                r_inf += np.sum(self.fit_parameters['x_dop'][zero_index])

        return r_inf

    def predict_r_tot(self):
        return self.predict_r_inf() + self.predict_r_p()

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
    # Fit quantification
    # ----------------------------------------------------
    def evaluate_chi_sq(self, frequencies=None, z=None, x=None, weights=None, **predict_kw):
        # Get fit frequencies
        if frequencies is None:
            frequencies = self.get_fit_frequencies()

        if z is None:
            z = self.z_fit

        # Get weights
        if weights is not None:
            if weights == 'modulus':
                weights = 1 / np.abs(z)
            elif np.shape(weights) != np.shape(z):
                raise ValueError('Weights must have same shape as frequencies')

        # Get model impedance
        z_hat = self.predict_z(frequencies, x=x, **predict_kw)

        # Calculate residuals
        return evaluation.chi_sq(z, z_hat, weights=weights)

    # ----------------------------------------------------
    # Peak finding
    # ----------------------------------------------------
    def find_peaks(self, tau=None, x=None, normalize=True, ppd=10, prominence=None, height=None, sign=1,
                   return_info=False,
                   method='thresh', prob_thresh=None, p_matrix=None, fxx_var_floor=1e-5, extend_var=True,
                   **kw):
        method_options = ['thresh', 'prob']
        if method not in method_options:
            raise ValueError(f'Invalid method {method}. Options: {method_options}')

        # If tau is not provided, go one decade beyond basis_tau with standard spacing
        # Finer spacing can cause many minor peaks to appear in 2nd derivative
        if tau is None:
            tau = self.get_tau_eval(ppd)

        # fx = self.predict_distribution(tau=tau, x=x, order=1)
        fxx = self.predict_distribution(tau=tau, x=x, order=2, sign=sign, normalize=normalize)

        # if normalize:
        # fx = fx / self.predict_r_p()
        # fxx = fxx / self.predict_r_p()

        # peak_indices = peaks.find_peaks(fx, fxx, fx_kw, fxx_kw)
        # if fxx_kw is None:
        #     fxx_kw = {}

        # f_mix = 0.5 * (f - fxx)

        # if prominence is None:
        #     prominence = -np.min(fxx) * 2.5e-2
        #     print('prom:', prominence)

        if prominence is None:
            # prominence = np.median(np.abs(fxx)) + 0.05 * np.percentile(-fxx[fxx > -np.inf], 95)
            if method == 'thresh':
                prominence = 0.05 * np.std(fxx[~np.isinf(fxx)]) + 5e-3
            else:
                prominence = 5e-3
            # print(prominence)

        if height is None:
            if method == 'thresh':
                height = 0
            else:
                height = 1e-3

        # if self.series_neg:
        #     # Find positive and negative peaks separately
        #     peak_index_list = []
        #     peak_info_list = []
        #     for sign in [-1, 1]:
        #         f = self.predict_distribution(tau=tau, x=x, order=0, sign=sign, normalize=normalize)
        #         fxx = self.predict_distribution(tau=tau, x=x, order=2, sign=sign, normalize=normalize)
        #         peak_index, peak_info = signal.find_peaks(-sign * fxx, height=height, prominence=prominence, **kw)
        #         # Limit to peaks that are positive in the direction of the current sign
        #         pos_index = sign * f[peak_index] > 0
        #         # print(pos_index)
        #         peak_index = peak_index[pos_index]
        #         peak_info = {k: v[pos_index] for k, v in peak_info.items()}
        #
        #         peak_index_list.append(peak_index)
        #         peak_info_list.append(peak_info)
        #     # Concatenate pos and neg peaks
        #     peak_indices = np.concatenate(peak_index_list)
        #     peak_info = {k: np.concatenate([pi[k] for pi in peak_info_list]) for k in peak_info.keys()}
        #
        #     # Sort ascending
        #     sort_index = np.argsort(peak_indices)
        #     peak_indices = peak_indices[sort_index]
        #     peak_info = {k: v[sort_index] for k, v in peak_info.items()}
        # else:

        if self.fit_kwargs['nonneg'] and sign != 0:
            # This captures both the standard nonneg case and the series_neg case
            # peak_indices = peaks.find_peaks_simple(fxx, order=2, height=0, prominence=prominence, **kw)
            peak_indices, peak_info = signal.find_peaks(-sign * fxx, height=height, prominence=prominence, **kw)
            # peak_indices = peaks.find_peaks_compound(fx, fxx, **kw)
        else:
            f = self.predict_distribution(tau=tau, x=x, order=0, sign=sign, normalize=normalize)
            # Find positive and negative peaks separately
            peak_index_list = []
            peak_info_list = []
            for peak_sign in [-1, 1]:
                peak_index, peak_info = signal.find_peaks(-peak_sign * fxx, height=height, prominence=prominence, **kw)
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

        if method == 'prob':
            if prob_thresh is None:
                prob_thresh = 0.25
            # Use prominence or height, whichever is smaller
            min_prom = np.minimum(peak_info['prominences'], peak_info['peak_heights'])

            # Get curvature std
            fxx_cov = self.estimate_distribution_cov(tau, p_matrix=p_matrix, order=2, sign=sign, normalize=normalize,
                                                     var_floor=fxx_var_floor,
                                                     extend_var=extend_var
                                                     )
            fxx_sigma = np.diag(fxx_cov) ** 0.5

            # Evaluate peak confidence
            peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, fxx_sigma[peak_indices])
            print(peak_prob)

            # TODO: incorporate probability that peak height is greater than zero
            # Keep peaks that exceed probability threshold
            peak_indices = peak_indices[peak_prob > prob_thresh]

            # peak_prob = 1 - 2 * stats.cdf_normal(0, min_prom, 1)

        if return_info:
            return tau[peak_indices], tau, peak_indices, peak_info
        else:
            return tau[peak_indices]

    def estimate_peak_coef(self, tau=None, peak_indices=None, x=None, sign=1, epsilon_factor=1.25, max_epsilon=1.25,
                           epsilon_uniform=None,
                           **find_peaks_kw):
        if peak_indices is not None and tau is None:
            raise ValueError('If peak_indices are provided, the corresponding tau grid must also be provided')

        x = self.get_drt_params(x, sign)

        if peak_indices is None:
            _, tau, peak_indices, _ = self.find_peaks(x=x, sign=sign, return_info=True, **find_peaks_kw)

        f = self.predict_distribution(tau, x=x, sign=sign)
        fxx = self.predict_distribution(tau, x=x, sign=sign, order=2)
        peak_weights = peaks.estimate_peak_weight_distributions(tau, f, fxx, peak_indices, self.basis_tau,
                                                                epsilon_factor, max_epsilon, epsilon_uniform)

        x_peaks = x * peak_weights

        return x_peaks

    def estimate_peak_distributions(self, tau=None, ppd=10, tau_find_peaks=None, peak_indices=None, x=None, sign=None,
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

        if sign is None:
            if self.series_neg:
                sign = 0
            else:
                sign = 1

        x_peaks = self.estimate_peak_coef(tau_find_peaks, peak_indices, x, sign, epsilon_factor,
                                          max_epsilon, epsilon_uniform,
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

    def mark_peaks(self, ax, x=None, sign=1, peak_tau=None, find_peaks_kw=None, scale_prefix=None, area=None,
                   normalize=False, normalize_by: Optional[float] = None,
                   **plot_kw):
        if find_peaks_kw is None:
            find_peaks_kw = {}

        # Find peaks
        if peak_tau is None:
            peak_tau = self.find_peaks(x=x, sign=sign, **find_peaks_kw)

        gamma_peaks = self.predict_distribution(peak_tau, 
                                                normalize=normalize, normalize_by=normalize_by, 
                                                x=x, sign=sign)

        if area is not None:
            gamma_peaks *= area

        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(gamma_peaks)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        ax.scatter(peak_tau, gamma_peaks / scale_factor, **plot_kw)

    def plot_peak_distributions(self, ax=None, tau=None, ppd=10, peak_gammas=None, estimate_peak_distributions_kw=None,
                                scale_prefix=None, x=None, sign=None, **plot_kw):

        if estimate_peak_distributions_kw is None:
            estimate_peak_distributions_kw = {}

        if tau is None:
            tau = self.get_tau_eval(ppd)

        if sign is None:
            sign = self.default_dist_sign

        if peak_gammas is None:
            peak_gammas = self.estimate_peak_distributions(tau=tau, x=x, sign=sign, **estimate_peak_distributions_kw)

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
            p_matrix = self.fit_parameters.get('p_matrix', None)

        if p_matrix is not None:
            try:
                p_inv = np.linalg.inv(p_matrix)
                # Invert DOP scaling
                if self.fit_dop:
                    dop_start, dop_end = self.dop_indices
                    dop_scale_mat = np.tile(self.dop_scale_vector, (len(p_inv), 1))
                    p_inv[:, dop_start:dop_end] *= dop_scale_mat
                    p_inv[dop_start:dop_end, :] *= dop_scale_mat.T
                return p_inv * self.coefficient_scale ** 2
            except np.linalg.LinAlgError:
                warnings.warn('Singular P matrix - could not obtain covariance estimate')
                return None
        else:
            raise Exception('Parameter covariance estimation is only available for qphb fits')
        
    def fisher_matrix(self, weighted: bool = True):
        rm = self.qphb_params['rm']
        if weighted:
            w = self.qphb_params['weights']
            rm = np.diag(w) @ rm
        return rm.T @ rm

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
                    np.ones(self.get_qp_mat_offset()) * self.basis_tau[0] * 1e-2,
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

            for primary_index in range(self.get_qp_mat_offset(), len(x_hat)):
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

            # for fixed_x_index in range(self.get_qp_mat_offset(), len(x_hat), x_interval):
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

    # def estimate_ci(self, method, quantiles=[0.025, 0.975], **sample_kw):
    #     if self.fit_type.find('qphb') > -1:
    #         if method == 'cov':
    #             # Estimate covariance matrix for coefficients
    #             cov = self.estimate_param_cov()
    #             if cov is not None:
    #                 sigma_x = np.diag(cov) ** 0.5
    #
    #                 x_hat = np.empty(len(sigma_x))
    #                 for key, info in self.special_qp_params.items():
    #                     if key == 'dop':
    #                         print('get dop')
    #                         dop_start, dop_end = self.dop_indices
    #                         x_hat[dop_start:dop_end] = self.fit_parameters['x_dop']
    #                     else:
    #                         start = info['index']
    #                         end = start + info.get('size', 1)
    #                         x_hat[start:end] = self.fit_parameters[key]
    #                 x_hat[self.get_qp_mat_offset():] = self.fit_parameters['x']
    #
    #                 # Determine number of std devs to obtain quantiles
    #                 s_quant = stats.std_normal_quantile(quantiles)
    #                 s_lo = s_quant[0]
    #                 s_hi = s_quant[1]
    #
    #                 x_lo = x_hat + s_lo * sigma_x
    #                 x_hi = x_hat + s_hi * sigma_x
    #             else:
    #                 x_lo, x_hi = None, None
    #         elif method == 'map_sample':
    #             # Use QPHB algorithm to sample MAP solutions with different parameters fixed
    #             self.generate_map_samples(**sample_kw)
    #
    #             x_array = self.map_samples['x']
    #             lp_array = self.map_samples['lp']
    #             lp_hat = self.qphb_params['posterior_lp']
    #             x_lo, x_hi = utils.weighted_quantile_2d(x_array, quantiles, np.exp(lp_array - lp_hat), axis=0)
    #
    #     return x_lo, x_hi

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
        dop_rho_vector = history_entry['dop_rho_vector']

        if weights is None:
            weights = self.qphb_params['est_weights']

        lml = qphb.evaluate_lml(x_hat, penalty_matrices, penalty_type, qphb_hypers, 
                                l1_lambda_vector, rho_vector, dop_rho_vector,
                                s_vectors, weights, rm, rv, self.special_qp_params)

        return lml

    # Plotting
    # --------------------------------------------
    def get_chrono_transforms(self, include_outliers=True):
        times = self.get_fit_times()

        # Add outlier times
        if include_outliers and self.chrono_outliers is not None:
            t_out = self.chrono_outliers[0]
            times = np.unique(np.concatenate([times, t_out]))

        trans_functions = utils.chrono.get_time_transforms(times, self.nonconsec_step_times)

        return trans_functions

    def plot_chrono_fit(self, ax=None, transform_time=False, linear_time_axis=None,
                        display_linear_ticks=None, linear_tick_kw=None,
                        plot_data=True, data_kw=None, data_label='',
                        scale_prefix=None, x=None, subtract_background=True, y_bkg=None,
                        predict_kw=None, c='k', tight_layout=True, show_outliers=False, outlier_kw=None, **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if display_linear_ticks is None:
            if transform_time and not linear_time_axis:
                display_linear_ticks = True
            else:
                display_linear_ticks = False

        if linear_time_axis is None:
            if transform_time and not display_linear_ticks:
                linear_time_axis = True
            else:
                linear_time_axis = False

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # Get fit times
        times = self.get_fit_times()

        # # Transform time to visualize each step on a log scale
        # if transform_time:
        #     # Transforms are not defined for times outside input times.
        #     # Must pad times to prevent issues with transforms if making secondary x-axis
        #     x_plot, trans_functions = get_transformed_plot_time(times, self.step_times, linear_time_axis)
        # else:
        #     x_plot = times
        # Get time transform functions
        if transform_time:
            trans_functions = self.get_chrono_transforms(show_outliers)
        else:
            trans_functions = None

        # Get fitted response
        if predict_kw is None:
            predict_kw = {}
        y_hat = self.predict_response(x=x, subtract_background=subtract_background, y_bkg=None, **predict_kw)

        # # Add the fitted background to the predicted response
        # if not subtract_background:
        #     y_hat += self.raw_response_background

        # Get appropriate scale
        if scale_prefix is None:
            if plot_data:
                scale_prefix = utils.scale.get_common_scale_prefix([y_hat, self.raw_response_signal])
            else:
                scale_prefix = utils.scale.get_scale_prefix(y_hat)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot data
        if plot_data:
            y_meas = self.raw_response_signal
            if subtract_background:
                # Add the removed background back to the data
                if y_bkg is not None:
                    y_meas = y_meas - y_bkg
                else:
                    y_meas = y_meas - self.raw_response_background

            plot_tuple = utils.chrono.signals_to_tuple(times, None, y_meas, self.chrono_mode)
            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                        linear_time_axis=False, scale_prefix=scale_prefix,
                        label=data_label, tight_layout=tight_layout, **data_kw)

        # Plot outliers
        if show_outliers and self.chrono_outliers is not None:
            if outlier_kw is None:
                outlier_kw = {'c': 'r', 'label': 'Outliers'}
            _, y_out = utils.chrono.get_input_and_response(self.chrono_outliers[1], self.chrono_outliers[2],
                                                           self.chrono_mode)
            plot_tuple = utils.chrono.signals_to_tuple(self.chrono_outliers[0], None, y_out, self.chrono_mode)
            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                        linear_time_axis=False, scale_prefix=scale_prefix,
                        tight_layout=tight_layout, **outlier_kw)

        # Plot fitted response
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_hat, self.chrono_mode)
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, plot_func='plot', transform_time=transform_time, trans_functions=trans_functions,
                    linear_time_axis=linear_time_axis,
                    display_linear_ticks=display_linear_ticks, linear_tick_kw=linear_tick_kw,
                    scale_prefix=scale_prefix,
                    c=c, tight_layout=tight_layout, **kw)
        # ax.plot(x_plot, y_hat / scale_factor, c=c, **kw)

        # # Add linear time axis
        # if linear_time_axis and transform_time:
        #     axt = add_linear_time_axis(ax, times, self.step_times, trans_functions)
        # else:
        #     axt = None

        # Labels
        # if transform_time:
        #     ax.set_xlabel('$f(t)$')
        # else:
        #     ax.set_xlabel('$t$ (s)')
        #
        # if self.chrono_mode == 'galv':
        #     ax.set_ylabel(f'$v$ ({scale_prefix}V)')
        # elif self.chrono_mode == 'pot':
        #     ax.set_ylabel(f'$i$ ({scale_prefix}A)')

        # if tight_layout:
        #     fig.tight_layout()

        # if return_second_ax:
        #     return ax, axt
        # else:
        return ax

    def plot_chrono_residuals(self, plot_sigma=True, ax=None, x=None, transform_time=True,
                              linear_time_axis=False, display_linear_ticks=True, linear_tick_kw=None,
                              subtract_background=True, y_bkg=None,
                              predict_kw=None,
                              s=10, alpha=0.5, scale_prefix=None, show_outliers=False, outlier_kw=None,
                              tight_layout=True, **kw):

        # Check x_axis string
        # x_axis_options = ['t', 'f(t)', 'index']
        # if x_axis not in x_axis_options:
        #     raise ValueError(f'Invalid x_axis option {x_axis}. Options: {x_axis_options}')

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if display_linear_ticks is None:
            if transform_time and not linear_time_axis:
                display_linear_ticks = True
            else:
                display_linear_ticks = False

        if linear_time_axis is None:
            if transform_time and not display_linear_ticks:
                linear_time_axis = True
            else:
                linear_time_axis = False

        # Get fit times
        times = self.get_fit_times()

        # Get time transforms
        if transform_time:
            trans_functions = self.get_chrono_transforms(show_outliers)
        else:
            trans_functions = None

        # Get outlier times
        # if show_outliers and self.chrono_outliers is not None:
        #     t_out = self.chrono_outliers[0]
        #     # Get sorted indices for plotting vs. index
        #     all_times, t_sort_index = np.unique(np.concatenate([times, t_out]), return_inverse=True)
        #     t_index = t_sort_index[:len(times)]
        #     t_out_index = t_sort_index[len(times):]
        # else:
        #     t_index = np.arange(len(times))
        #     t_out_index = None
        #     t_out = None

        # Get x values for plotting
        # trans_functions = None
        # if x_axis == 'f(t)':
        #     # Transform time to visualize each step on a log scale
        #     x_plot, trans_functions = get_transformed_plot_time(times, self.nonconsec_step_times)
        #     ax.set_xlabel('$f(t)$')
        # elif x_axis == 'index':
        #     # Uniform spacing
        #     x_plot = t_index
        #     ax.set_xlabel('Sample index')
        # else:
        #     x_plot = times
        #     ax.set_xlabel('$t$ (s)')

        # Get model response
        if predict_kw is None:
            predict_kw = {}
        y_hat = self.predict_response(x=x, subtract_background=subtract_background, y_bkg=y_bkg, **predict_kw)

        # Get measured response
        y_meas = self.raw_response_signal

        # Remove estimated background from data if specified
        if subtract_background:
            if y_bkg is not None:
                y_meas = y_meas - y_bkg
            else:
                y_meas = y_meas - self.raw_response_background

        # Calculate residuals
        y_err = y_meas - y_hat

        # Get appropriate scale
        if scale_prefix is None:
            scale_prefix, scale_factor = utils.scale.get_scale_prefix_and_factor(y_err)
        else:
            scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot outliers
        if show_outliers and self.chrono_outliers is not None:
            t_out = self.chrono_outliers[0]
            _, y_out = utils.chrono.get_input_and_response(self.chrono_outliers[1], self.chrono_outliers[2],
                                                           self.chrono_mode)
            y_err_out = self.predict_response(times=t_out, x=x, subtract_background=True, **predict_kw) - y_out

            # Get x values for plotting
            # if x_axis == 'f(t)':
            #     x_out = trans_functions[1](t_out)
            # elif x_axis == 'index':
            #     x_out = t_out_index
            # else:
            #     x_out = t_out

            # Plot residuals
            if outlier_kw is None:
                outlier_kw = {'s': s, 'alpha': alpha, 'c': 'r', 'label': 'Outliers'}
            # ax.scatter(x_out, y_err_out / scale_factor, **outlier_kw)

            plot_tuple = utils.chrono.signals_to_tuple(t_out, None, y_err_out, self.chrono_mode)
            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                        linear_time_axis=False,
                        display_linear_ticks=False,
                        scale_prefix=scale_prefix,
                        tight_layout=tight_layout, **outlier_kw)

        # Plot residuals
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_err, self.chrono_mode)
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, transform_time=transform_time, trans_functions=trans_functions,
                    linear_time_axis=linear_time_axis,
                    display_linear_ticks=display_linear_ticks, linear_tick_kw=linear_tick_kw,
                    scale_prefix=scale_prefix,
                    tight_layout=tight_layout, alpha=alpha, **kw)

        # ax.scatter(x_plot, y_err / scale_factor, s=s, alpha=alpha, **kw)

        # Indicate zero
        ax.axhline(0, c='k', lw=1, zorder=-10)

        # Show error structure
        if plot_sigma:
            sigma = self.predict_sigma('chrono') / scale_factor
            if sigma is not None:
                if transform_time:
                    x_plot = trans_functions[1](times)
                else:
                    x_plot = times
                ax.fill_between(x_plot, -3 * sigma, 3 * sigma, color='k', lw=0, alpha=0.15)

        # # Add linear time axis
        # if linear_time_axis and x_axis == 'f(t)':
        #     axt = add_linear_time_axis(ax, times, self.step_times, trans_functions)

        # Labels
        if self.chrono_mode == 'galv':
            ax.set_ylabel(f'$v - \hat{{v}}$ ({scale_prefix}V)')
        elif self.chrono_mode == 'pot':
            ax.set_ylabel(f'$i - \hat{{i}}$ ({scale_prefix}A)')

        fig.tight_layout()

        return ax

    def plot_chrono_correction(self, ax=None, transform_time=False, linear_time_axis=True,
                               display_linear_ticks=False, linear_tick_kw=None,
                               scale_prefix=None, show_background=True, bkg_twin_ax=False,
                               tight_layout=True, raw_kw=None, corrected_kw=None, background_kw=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2.75))
        else:
            fig = ax.get_figure()

        # Get fit times
        times = self.get_fit_times()

        y_meas = self.raw_response_signal
        y_bkg = self.raw_response_background

        # Get appropriate scale
        if scale_prefix is None:
            scale_prefix = utils.scale.get_common_scale_prefix([y_meas, y_meas - y_bkg])
        # scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)

        # Plot raw data
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_meas, self.chrono_mode)
        if raw_kw is None:
            raw_kw = {'label': 'Raw', 'plot_func': 'plot'}
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, transform_time=transform_time,
                    linear_time_axis=False, scale_prefix=scale_prefix,
                    tight_layout=False, **raw_kw)

        # Plot the background
        if show_background:
            if bkg_twin_ax:
                ax_bkg = ax.twinx()
                bkg_scale_prefix = None
                y_bkg_plot = y_bkg
            else:
                ax_bkg = ax
                bkg_scale_prefix = scale_prefix
                y_bkg_plot = y_bkg + self.fit_parameters['v_baseline']

            plot_tuple = utils.chrono.signals_to_tuple(times, None, y_bkg_plot,
                                                       self.chrono_mode)
            if background_kw is None:
                background_kw = {'label': 'Background', 'plot_func': 'plot', 'c': 'r', 'alpha': 0.75}

            plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                        axes=ax_bkg, transform_time=transform_time,
                        linear_time_axis=False, scale_prefix=bkg_scale_prefix,
                        tight_layout=False, **background_kw)

            if bkg_twin_ax:
                units = ax_bkg.get_ylabel().split(' ')[1]
                ax_bkg.set_ylabel(f'Background {units}')
        else:
            ax_bkg = None

        # Plot the corrected signal
        plot_tuple = utils.chrono.signals_to_tuple(times, None, y_meas - y_bkg, self.chrono_mode)
        if corrected_kw is None:
            corrected_kw = {'label': 'Corrected', 'plot_func': 'plot', 'c': 'k', 'alpha': 0.75}
        plot_chrono(plot_tuple, chrono_mode=self.chrono_mode, step_times=self.nonconsec_step_times,
                    axes=ax, transform_time=transform_time,
                    linear_time_axis=linear_time_axis,
                    display_linear_ticks=display_linear_ticks, linear_tick_kw=linear_tick_kw,
                    scale_prefix=scale_prefix,
                    tight_layout=tight_layout, **corrected_kw)

        if show_background and bkg_twin_ax:
            handles, labels = ax.get_legend_handles_labels()
            h_bkg, l_bkg = ax_bkg.get_legend_handles_labels()
            leg = ax.legend(handles=handles + h_bkg, labels=labels + l_bkg)
            leg.set_zorder(10)
        else:
            ax.legend()

        return ax

    def plot_eis_fit(self, frequencies=None, axes=None, plot_type='nyquist', plot_data=True, data_kw=None,
                     bode_cols=['Zreal', 'Zimag'], data_label='', scale_prefix=None, area=None, normalize=False,
                     x=None, predict_kw={}, c='k', tight_layout=True, show_outliers=False, outlier_kw=None, **kw):

        # Set default data plotting kwargs if not provided
        if data_kw is None:
            data_kw = dict(s=10, alpha=0.5)

        # Get model impedance
        if frequencies is None:
            frequencies = self.get_fit_frequencies()
        z_hat = self.predict_z(frequencies, x=x, **predict_kw)

        # Get rp for normalization
        if normalize:
            normalize_rp = self.predict_r_p()
            scale_prefix = ''
        else:
            normalize_rp = None

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

        # Plot data if requested
        if plot_data:
            axes = plot_eis((self.get_fit_frequencies(), self.z_fit), plot_type, axes=axes, scale_prefix=scale_prefix,
                            label=data_label,
                            bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                            tight_layout=False, **data_kw)

        # Plot outliers if requested
        if show_outliers and self.eis_outliers is not None:
            if outlier_kw is None:
                outlier_kw = {'c': 'r', 'label': 'Outliers'}
            axes = plot_eis(self.eis_outliers, plot_type, axes=axes, scale_prefix=scale_prefix,
                            bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                            tight_layout=False, **outlier_kw)

        # Plot fit
        axes = plot_eis((frequencies, z_hat), plot_type, axes=axes, plot_func='plot', c=c, scale_prefix=scale_prefix,
                        bode_cols=bode_cols, area=area, normalize=normalize, normalize_rp=normalize_rp,
                        tight_layout=tight_layout, **kw)

        # fig = np.atleast_1d(axes)[0].get_figure()
        # fig.tight_layout()

        return axes

    def plot_eis_residuals(self, plot_sigma=True, axes=None, scale_prefix=None, x=None,
                           predict_kw={}, part='both', show_outliers=False, outlier_kw=None,
                           s=10, alpha=0.5, **kw):

        if part == 'both':
            bode_cols = ['Zreal', 'Zimag']
        elif part == 'real':
            bode_cols = ['Zreal']
        elif part == 'imag':
            bode_cols = ['Zimag']
        else:
            raise ValueError(f"Invalid part argument {part}. Options: 'both', 'real', 'imag'")

        if axes is None:
            fig, axes = plt.subplots(1, len(bode_cols), figsize=(3 * len(bode_cols), 2.75))
        else:
            fig = np.atleast_1d(axes)[0].get_figure()
        axes = np.atleast_1d(axes)

        # Get fit frequencies
        f_fit = self.get_fit_frequencies()

        # Get model impedance
        y_hat = self.predict_z(f_fit, x=x, **predict_kw)

        # Calculate residuals
        y_err = self.z_fit - y_hat

        # Get scale prefix
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(np.concatenate([y_err.real, y_err.imag]))

        # Plot residuals
        plot_eis((f_fit, y_err), axes=axes, plot_type='bode', bode_cols=bode_cols,
                 s=s, alpha=alpha, label='Residuals', scale_prefix=scale_prefix, **kw)

        # Plot outliers
        if show_outliers and self.eis_outliers is not None:
            f_out, z_out = self.eis_outliers

            # Calculate residuals
            y_err_out = self.predict_z(f_out, x=x, **predict_kw) - z_out

            # Plot residuals
            if outlier_kw is None:
                outlier_kw = {'c': 'r', 'label': 'Outliers'}
            plot_eis((f_out, y_err_out), axes=axes, plot_type='bode', bode_cols=bode_cols,
                     s=s, alpha=alpha, scale_prefix=scale_prefix, **outlier_kw)

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
            axes[bode_cols.index('Zreal')].set_ylabel(fr'$Z^{{\prime}} - \hat{{Z}}^{{\prime}}$ ({scale_prefix}$\Omega$)')
        if 'Zimag' in bode_cols:
            axes[bode_cols.index('Zimag')].set_ylabel(
                fr'$-(Z^{{\prime\prime}} - \hat{{Z}}^{{\prime\prime}})$ ({scale_prefix}$\Omega$)')

        fig.tight_layout()

        return axes

    def plot_distribution(self, tau=None, ppd=20, x=None, ax=None, scale_prefix=None, plot_bounds=False,
                          normalize=False, normalize_by=None, sign=None,
                          plot_ci=False, ci_kw=None, ci_quantiles=[0.025, 0.975],  # sample_kw={},
                          area=None, order=0, mark_peaks=False, mark_peaks_kw=None, tight_layout=True,
                          return_line=False, freq_axis=False,
                          **kw):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        if tau is None:
            # If tau is not provided, go one decade beyond self.basis_tau with finer spacing
            tau = self.get_tau_eval(ppd)

        if sign is None:
            if self.series_neg:
                sign = [-1, 1]
            else:
                sign = [1]
        signs = np.atleast_1d(sign)

        # if sample_kw == {} and self.map_sample_kw is not None:
        #     sample_kw = self.map_sample_kw

        # Normalize
        if normalize_by is not None:
            normalize = True
        
        if normalize:
            if normalize_by is None:
                normalize_by = self.predict_r_p(x=x, absolute=True)
            scale_prefix = ''
            # print('r_p:', r_p)

        # Get common scale prefix if plotting multiple signs
        if self.series_neg and len(signs) > 1 and scale_prefix is None:
            gamma_pos = self.predict_distribution(tau, x=x, order=order, sign=1)
            gamma_neg = self.predict_distribution(tau, x=x, order=order, sign=-1)
            scale_prefix = utils.scale.get_common_scale_prefix([gamma_pos, gamma_neg])

        lines = []
        for sign in signs:
            # Calculate MAP distribution at evaluation points
            gamma = self.predict_distribution(tau, x=x, order=order, sign=sign)
            # if line == 'mode':
            #     # Calculate MAP distribution at evaluation points
            #     gamma = self.predict_distribution(tau, x=x, order=order, sign=sign)
            # elif line == 'mean':
            #     # Estimate posterior mean from MAP samples
            #     x_mean = self.estimate_posterior_mean(**sample_kw)
            #     gamma = self.predict_distribution(tau, x=x_mean[self.get_qp_mat_offset():], order=order, sign=sign)
            # else:
            #     raise ValueError(f"Invalid line argument {line}. Options: 'mode', 'mean'")

            ax, info = plot_distribution(tau, gamma, ax, area, scale_prefix, 
                                         normalize_by=normalize_by, freq_axis=freq_axis,
                                         return_info=True, **kw)
            line, scale_prefix, scale_factor = info
            lines.append(line)

            if plot_ci:
                if self.fit_type.find('qphb') > -1:
                    gamma_lo, gamma_hi = self.predict_distribution_ci(
                        tau, ppd, x=x, order=order, sign=sign, 
                        normalize=normalize, normalize_by=normalize_by,
                        quantiles=ci_quantiles
                    )
                    
                    if area is not None:
                        for g in (gamma_lo, gamma_hi):
                            g *= area
                            
                    # if normalize:
                    #     for g in (gamma_lo, gamma_hi):
                    #         g /= r_p

                    # Enforce sign constraint in CI
                    if gamma_lo is not None:
                        if self.series_neg:
                            if sign == 1:
                                gamma_lo = np.maximum(gamma_lo, 0)
                            elif sign == -1:
                                gamma_hi = np.minimum(gamma_hi, 0)
                        elif self.fit_kwargs['nonneg'] and order == 0:
                            gamma_lo = np.maximum(gamma_lo, 0)

                        if ci_kw is None:
                            ci_kw = {}
                        ci_defaults = dict(color=line[0].get_color(), lw=0.5, alpha=0.25, zorder=-10)
                        ci_defaults.update(ci_kw)
                        ci_kw_sign = ci_defaults

                        ax.fill_between(tau, gamma_lo / scale_factor, gamma_hi / scale_factor, **ci_kw_sign)

            if mark_peaks:
                if mark_peaks_kw is None:
                    mark_peaks_kw = {}
                self.mark_peaks(ax, x=x, sign=sign, scale_prefix=scale_prefix, area=area, 
                                normalize=normalize, normalize_by=normalize_by,
                                **mark_peaks_kw)

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

        if tight_layout:
            fig.tight_layout()

        if return_line:
            return ax, lines
        else:
            return ax

    def plot_dop(self, nu=None, x=None, ax=None, scale_prefix=None, normalize=False, normalize_tau=None,
                 invert_nu=True, phase=True, area=None,
                 plot_ci=False, ci_kw=None, ci_quantiles=[0.025, 0.975], order=0,
                 delta_density=False, include_ideal=True,
                 tight_layout=True, return_line=False, normalize_quantiles=(0, 1), **kw):

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        nu, dop = self.predict_dop(nu=nu, x=x, normalize=normalize, normalize_tau=normalize_tau,
                                   order=order, return_nu=True, normalize_quantiles=normalize_quantiles,
                                   delta_density=delta_density, include_ideal=include_ideal)

        # Invert nu for more intuitive visualization
        if invert_nu:
            nu_plot = -nu
            x_label_sign = '-'
        else:
            nu_plot = nu
            x_label_sign = ''

        # Convert nu to phase for easier interpretation
        if phase:
            nu_plot = nu_plot * 90 #np.pi / 2
            x_label = fr'${x_label_sign}\theta$ ($^\circ$)'
        else:
            x_label = fr'${x_label_sign}\nu$'

        # Get scale factor
        if scale_prefix is None:
            scale_prefix = utils.scale.get_scale_prefix(dop)
        scale_factor = utils.scale.get_factor_from_prefix(scale_prefix)
        
        if area is not None:
            # Multiply by area
            scale_factor = scale_factor / area

        line = ax.plot(nu_plot, dop / scale_factor, **kw)

        # Plot CI
        if plot_ci:
            if self.fit_type.find('qphb') > -1:
                dop_lo, dop_hi = self.predict_dop_ci(nu=nu, x=x, normalize=normalize, normalize_tau=normalize_tau,
                                                     quantiles=ci_quantiles, order=order,
                                                     normalize_quantiles=normalize_quantiles,
                                                     delta_density=delta_density, include_ideal=include_ideal)

                if dop_lo is not None:
                    # Enforce sign constraint in CI
                    if order == 0:
                        dop_lo = np.maximum(dop_lo, 0)

                    if ci_kw is None:
                        ci_kw = {}
                    ci_defaults = dict(color=line[0].get_color(), lw=0.5, alpha=0.2, zorder=-10)
                    ci_defaults.update(ci_kw)
                    ci_kw = ci_defaults

                    ax.fill_between(nu_plot, dop_lo / scale_factor, dop_hi / scale_factor, **ci_kw)

        if invert_nu:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel(x_label)

        if area is not None:
            area_units = '$\cdot \mathrm{cm}^2$'
        else:
            area_units = ''
            
        if normalize:
            ax.set_ylabel(fr'$\tilde{{\rho}}$ ({scale_prefix}$\Omega${area_units})')
        else:
            ax.set_ylabel(fr'$\rho$ ({scale_prefix}$\Omega \cdot \mathrm{{s}}^\nu${area_units})')

        if tight_layout:
            fig.tight_layout()

        if return_line:
            return ax, line
        else:
            return ax

    def plot_results(self, axes=None, x=None, show_outliers=False, outlier_kw=None,
                     distribution_kw=None, eis_fit_kw=None, eis_resid_kw=None,
                     chrono_fit_kw=None, chrono_resid_kw=None):

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
        else:
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
        if distribution_kw is None and self.series_neg:
            for sign, c in zip([1, -1], ['b', 'r']):
                self.plot_distribution(x=x, sign=sign, ax=drt_ax, plot_ci=True, c=c)
        else:
            if distribution_kw is None:
                distribution_kw = {}
            self.plot_distribution(x=x, ax=drt_ax, **dict(plot_ci=True, c='k') | distribution_kw)
            
        drt_ax.set_title('DRT')

        # Plot EIS fit and residuals
        if eis_fit_ax is not None:
            if eis_fit_kw is None:
                eis_fit_kw = {}
            if eis_resid_kw is None:
                eis_resid_kw = {}

            self.plot_eis_fit(axes=eis_fit_ax, plot_type='nyquist', x=x,
                              show_outliers=show_outliers, outlier_kw=outlier_kw,
                              **eis_fit_kw)
            eis_fit_ax.set_title('EIS Fit')

            self.plot_eis_residuals(axes=eis_resid_axes, x=x,
                                    show_outliers=show_outliers, outlier_kw=outlier_kw,
                                    **eis_resid_kw)
            eis_resid_axes[0].set_title('$Z^\prime$ Residuals')
            eis_resid_axes[1].set_title('$Z^{\prime\prime}$ Residuals')
            eis_resid_axes[0].legend()
            eis_resid_axes[1].get_legend().remove()

        # Plot chrono fit and residuals
        if chrono_fit_ax is not None:
            if chrono_fit_kw is None:
                chrono_fit_kw = {'linear_time_axis': False}
            if chrono_resid_kw is None:
                chrono_resid_kw = {'linear_time_axis': False}

            self.plot_chrono_fit(ax=chrono_fit_ax, x=x,
                                 show_outliers=show_outliers, outlier_kw=outlier_kw,
                                 **chrono_fit_kw)
            chrono_fit_ax.set_title('Chrono Fit')

            self.plot_chrono_residuals(ax=chrono_resid_ax, x=x,
                                       show_outliers=show_outliers, outlier_kw=outlier_kw,
                                       **chrono_resid_kw)
            chrono_resid_ax.set_title('Chrono Residuals')

        fig.tight_layout()

        return axes

    # Preprocessing (matrix calculations, scaling)
    # --------------------------------------------
    def _solve_data_scale(self, hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                          s_vectors, rzv, rzm, nonneg):
        # Perform a quick Elastic Net solution to estimate Rp
        x_rp = qphb.estimate_x_rp(hypers, penalty_matrices, penalty_type, rho_vector, dop_rho_vector,
                                  s_vectors, rzv, rzm, nonneg, self.special_qp_params, l2_lambda_0=1e-4,
                                  l1_lambda_0=1e-3)
        rp_est = self.predict_r_p(absolute=True, x=x_rp, raw=True)

        if self.fit_dop:
            dop_start, dop_end = self.dop_indices
            x_drt_max = np.max(np.abs(x_rp[self.get_qp_mat_offset():]))
            x_dop_max = np.max(np.abs(x_rp[dop_start:dop_end]))
            dop_rescale_factor = x_drt_max / x_dop_max
        else:
            dop_rescale_factor = None

        return rp_est, dop_rescale_factor

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
        if self.fixed_basis_tau is not None:
            self.basis_tau = self.fixed_basis_tau
        else:
            # Default: 10 ppd basis grid. Extend basis tau one decade beyond data on each end
            self.basis_tau = pp.get_basis_tau(frequencies, times, step_times, tau_grid=self.tau_supergrid)

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

        # Set t_fit again to ensure matrix recalc status is updated
        self.t_fit = sample_times

        if sample_times is not None:
            # Get matrix and vectors for chrono fit
            rm_drt, inf_rv, induc_rv, cap_rv, rm_dop = \
                self._prep_chrono_fit_matrix(sample_times, step_times, step_sizes,
                                             tau_rise, smooth_inf_response)

            if self.series_neg:
                rm_drt = np.hstack((rm_drt, -rm_drt))
        else:
            # No chrono data
            self.t_fit = []
            rm_drt, inf_rv, induc_rv, cap_rv, rm_dop = None, None, None, None, None

        if frequencies is not None:
            # Get matrix and vector for impedance fit
            zm_drt, induc_zv, cap_zv, zm_dop = self._prep_impedance_fit_matrix(frequencies)

            if self.series_neg:
                zm_drt = np.hstack((zm_drt, -zm_drt))
        else:
            # No EIS data
            self.f_fit = []
            zm_drt, induc_zv, cap_zv, zm_dop = None, None, None, None

        # Calculate penalty matrices
        penalty_matrices = self._prep_penalty_matrices(penalty_type, derivative_weights)
        if self.series_neg:
            for key in list(penalty_matrices.keys()):
                if key.find('dop') == -1:
                    mat = penalty_matrices[key]
                    new_mat = np.kron(np.eye(2), mat)
                    penalty_matrices[key] = new_mat

        # Perform scaling
        i_signal_scaled, v_signal_scaled, z_scaled = self.scale_data(sample_times, sample_i, sample_v, step_times,
                                                                     step_sizes, z, scale_data, rp_scale)
        if self.print_diagnostics:
            print('Finished signal scaling')

        # Estimate chrono baseline after scaling
        if sample_times is not None:
            if self.chrono_mode == 'galv':
                response_baseline = np.median(v_signal_scaled[sample_times < step_times[0]])
            else:
                response_baseline = np.median(i_signal_scaled[sample_times < step_times[0]])
        else:
            response_baseline = None

        # scale chrono response matrix/vectors to input_signal_scale
        if rm_drt is not None:
            rm_drt = rm_drt / self.input_signal_scale
            induc_rv = induc_rv / self.input_signal_scale
            inf_rv = inf_rv / self.input_signal_scale
            cap_rv = cap_rv / self.input_signal_scale
            if rm_dop is not None:
                rm_dop = rm_dop / self.input_signal_scale

        if self.print_diagnostics:
            print('Finished prep_for_fit in {:.2f} s'.format(time.time() - start_time))

        return (sample_times, i_signal_scaled, v_signal_scaled, response_baseline, z_scaled), \
               (rm_drt, induc_rv, inf_rv, cap_rv, rm_dop, zm_drt, induc_zv, cap_zv, zm_dop, penalty_matrices)

    def _prep_chrono_fit_matrix(self, times, step_times, step_sizes, tau_rise, smooth_inf_response):
        # Recalculate matrices if necessary
        # TODO: incorporate step_times and step_sizes logic
        #  This is a placeholder to avoid issues with corner cases
        #  (e.g. if input_signal remains the same but step_times is manually specified)
        self._recalc_chrono_fit_matrix = True

        if self._recalc_chrono_fit_matrix:
            if self.print_diagnostics:
                print('Calculating chrono response matrix')
            rm, rm_layered = mat1d.construct_response_matrix(self.basis_tau, times, self.step_model, step_times,
                                                             step_sizes, basis_type=self.tau_basis_type,
                                                             epsilon=self.tau_epsilon, tau_rise=tau_rise,
                                                             op_mode=self.chrono_mode,
                                                             integrate_method=self.integrate_method,
                                                             zga_params=self.zga_params,
                                                             interpolate_grids=self.interpolate_lookups['response'])
            if self.print_diagnostics:
                print('Constructed chrono response matrix')
            self.fit_matrices['response'] = rm.copy()
            self.fit_matrices['rm_layered'] = rm_layered.copy()

            induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                  tau_rise, self.chrono_mode)
            cap_rv = mat1d.construct_capacitance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                 tau_rise, self.chrono_mode)

            self.fit_matrices['inductance_response'] = induc_rv
            self.fit_matrices['capacitance_response'] = cap_rv

            # With all matrices calculated, set recalc flags to False
            self._recalc_chrono_fit_matrix = False
            # self._recalc_chrono_prediction_matrix = False
            # We shouldn't reset _recalc_eis_prediction_matrix here.
            #  Example case: basis_tau changes during fit (triggers recalc, which is then overwritten here),
            #  but f_predict is unchanged
            #  When we next try to call predict_z, recalc status will be False even though basis_tau changed

        # Otherwise, reuse existing matrices as appropriate
        elif self._t_fit_subset_index is not None:
            # times is a subset of self.t_fit. Use sub-matrices of existing A array; do not overwrite
            rm = self.fit_matrices['response'][self._t_fit_subset_index, :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._t_fit_subset_index].copy()
            cap_rv = self.fit_matrices['capacitance_response'][self._t_fit_subset_index].copy()
        else:
            # All matrix parameters are the same. Use existing matrices
            rm = self.fit_matrices['response'].copy()
            induc_rv = self.fit_matrices['inductance_response'].copy()
            cap_rv = self.fit_matrices['capacitance_response'].copy()

        # Construct R_inf response vector. Fast - always calculate
        inf_rv = mat1d.construct_ohmic_response_vector(times, self.step_model, step_times, step_sizes, tau_rise,
                                                       self.raw_input_signal, smooth_inf_response, self.chrono_mode)
        self.fit_matrices['inf_response'] = inf_rv.copy()

        if self.fit_dop:
            rm_dop, rm_dop_layered = phasance.construct_phasor_v_matrix(times, self.basis_nu,
                                                                        self.nu_basis_type,
                                                                        self.nu_epsilon,
                                                                        self.step_model, step_times,
                                                                        step_sizes, self.chrono_mode)
            self.fit_matrices['rm_dop'] = rm_dop.copy()
            self.fit_matrices['rm_dop_layered'] = rm_dop_layered.copy()
        else:
            rm_dop = None

        return rm, inf_rv, induc_rv, cap_rv, rm_dop

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
            # self._recalc_eis_prediction_matrix = False
            # We shouldn't reset _recalc_eis_prediction_matrix here.
            #  Example case: basis_tau changes during fit (triggers recalc, which is then overwritten here),
            #  but f_predict is unchanged
            #  When we next try to call predict_z, recalc status will be False even though basis_tau changed
        elif self._f_fit_subset_index is not None:
            # frequencies is a subset of self.f_fit. Use sub-matrices of existing matrix; do not overwrite
            zm = self.fit_matrices['impedance'][self._f_fit_subset_index, :].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            zm = self.fit_matrices['impedance'].copy()

        # Construct inductance impedance vector
        induc_zv = mat1d.construct_inductance_impedance_vector(frequencies)
        cap_zv = mat1d.construct_capacitance_impedance_vector(frequencies)

        # Distribution of phasances
        if self.fit_dop:
            zm_dop = phasance.construct_phasor_z_matrix(frequencies, self.basis_nu,
                                                        self.nu_basis_type, self.nu_epsilon)
        else:
            zm_dop = None
        self.fit_matrices['zm_dop'] = zm_dop

        return zm, induc_zv, cap_zv, zm_dop

    def _prep_penalty_matrices(self, penalty_type, derivative_weights, truncate=False):
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

                if self.fit_dop:
                    # Use Gaussian derivatives for delta basis
                    dnu = np.median(np.diff(np.sort(self.basis_nu)))
                    dk_dop = basis.construct_func_eval_matrix(self.basis_nu, None,
                                                              basis_type='gaussian',
                                                              order=k, epsilon=1 / dnu)
                    penalty_matrices[f'l{k}_dop'] = dk_dop.copy()
                    penalty_matrices[f'm{k}_dop'] = dk_dop.T @ dk_dop
            elif penalty_type == 'integral':
                if truncate:
                    integration_limits = (np.log(self.basis_tau[0]), np.log(self.basis_tau[-1]))
                else:
                    integration_limits = None
                dk = mat1d.construct_integrated_derivative_matrix(np.log(self.basis_tau),
                                                                  basis_type=self.tau_basis_type,
                                                                  order=k, epsilon=self.tau_epsilon,
                                                                  zga_params=self.zga_params,
                                                                  integration_limits=integration_limits)
                penalty_matrices[f'm{k}'] = dk.copy()

                if self.fit_dop:
                    # TODO: truncate dop penalty?
                    if self.nu_basis_type == 'delta':
                        # Use Gaussian derivatives for delta basis
                        dnu = np.median(np.diff(np.sort(self.basis_nu)))
                        dk_dop = mat1d.construct_integrated_derivative_matrix(self.basis_nu,
                                                                              basis_type='gaussian',
                                                                              order=k, epsilon=1 / dnu)
                    else:
                        dk_dop = mat1d.construct_integrated_derivative_matrix(self.basis_nu,
                                                                              basis_type=self.nu_basis_type,
                                                                              order=k, epsilon=self.nu_epsilon)
                    penalty_matrices[f'm{k}_dop'] = dk_dop.copy()

                    # Make a smoothing matrix for the 0th-derivative s vector
                    if k == 0:
                        dnu = np.mean(np.abs(np.diff(self.basis_nu)))
                        gmat = mat1d.construct_integrated_derivative_matrix(self.basis_nu,
                                                                            basis_type='gaussian',
                                                                            order=1, epsilon=1 / dnu)
                        penalty_matrices[f'gmat{k}_dop'] = gmat.copy()
                    # except ValueError:
                    #     penalty_matrices[f'm{k}_dop'] = 0

        self.fit_matrices.update(penalty_matrices)

        if self.print_diagnostics:
            print('Constructed penalty matrices')
        return penalty_matrices

    def _format_qp_matrices(self, rm_drt, inf_rv, induc_rv, cap_rv, rm_dop,
                            zm_drt, induc_zv, cap_zv, zm_dop,
                            base_penalty_matrices, v_baseline_penalty, ohmic_penalty,
                            inductance_penalty, capacitance_penalty,
                            vz_offset_scale, background_penalty,
                            inductance_scale, capacitance_scale, penalty_type, derivative_weights):
        """
        Format matrices for quadratic programming solution
        :param capacitance_scale:
        :param capacitance_penalty:
        :param cap_rv:
        :param cap_zv:
        :param background_penalty:
        :param derivative_weights:
        :param rm_drt:
        :param base_penalty_matrices:
        :param inductance_scale:
        :param penalty_type:
        :return:
        """
        # Count number of special params for padding
        num_special = self.get_qp_mat_offset()

        # Extract indices for convenience
        special_indices = {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}

        # Store inductance scale for reference
        self.inductance_scale = inductance_scale
        self.capacitance_scale = capacitance_scale

        # Get DOP scale vector
        if self.fit_dop:
            if self.normalize_dop:
                # If tau_supergrid is set, we should use this for DOP normalization
                # to ensure consistency across spectra (e.g. for DRTMD).
                # Otherwise, we can just use basis_tau
                if self.tau_supergrid is not None:
                    dop_eval_tau = self.tau_supergrid
                else:
                    dop_eval_tau = self.basis_tau
                
                self.dop_scale_vector = phasance.phasor_scale_vector(self.basis_nu, dop_eval_tau)
                
                # Normalize for basis function area
                # print('nu basis area:', basis.get_basis_func_area(self.nu_basis_type, self.nu_epsilon))
                self.dop_scale_vector /= self.nu_basis_area
                
                # self.dop_scale_vector *= 100
                # print(self.dop_scale_vector)
            else:
                self.dop_scale_vector = np.ones(len(self.basis_nu))
        else:
            self.dop_scale_vector = None
        dop_start_index, dop_end_index = self.dop_indices

        # Add columns to rm for v_baseline, R_inf, and inductance
        if rm_drt is not None:
            rm = np.empty((rm_drt.shape[0], rm_drt.shape[1] + num_special))

            # Add entries for special parameters
            if 'v_baseline' in special_indices.keys():
                rm[:, special_indices['v_baseline']] = 1  # v_baseline
            if 'inductance' in special_indices.keys():
                rm[:, special_indices['inductance']] = induc_rv * inductance_scale  # inductance response
            if 'R_inf' in special_indices.keys():
                rm[:, special_indices['R_inf']] = inf_rv  # R_inf response. only works for galvanostatic mode
            if 'C_inv' in special_indices.keys():
                rm[:, special_indices['C_inv']] = cap_rv * capacitance_scale

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                rm[:, special_indices['vz_offset']] = 0

            if 'background_scale' in special_indices.keys():
                rm[:, special_indices['background_scale']] = 0

            # Add DOP
            if self.fit_dop:
                rm[:, dop_start_index:dop_end_index] = rm_dop * self.dop_scale_vector

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

            if 'R_inf' in special_indices.keys():
                zm[:, special_indices['R_inf']] = 1  # R_inf contribution to impedance

            if 'C_inv' in special_indices.keys():
                zm[:, special_indices['C_inv']] = cap_zv * capacitance_scale

            # Add entry for vz_offset if applicable
            if 'vz_offset' in special_indices.keys():
                zm[:, special_indices['vz_offset']] = 0

            if 'background_scale' in special_indices.keys():
                zm[:, special_indices['background_scale']] = 0

            # Add DOP
            if self.fit_dop:
                zm[:, dop_start_index:dop_end_index] = zm_dop * self.dop_scale_vector

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
                m_drt = base_penalty_matrices[f'm{k}']

                # Add rows/columns for v_baseline, inductance, and R_inf
                m_k = np.zeros((m_drt.shape[0] + num_special, m_drt.shape[1] + num_special))

                # Insert penalties for special parameters
                if 'v_baseline' in special_indices.keys():
                    m_k[special_indices['v_baseline'], special_indices['v_baseline']] = v_baseline_penalty
                if 'inductance' in special_indices.keys():
                    m_k[special_indices['inductance'], special_indices['inductance']] = inductance_penalty
                if 'R_inf' in special_indices.keys():
                    m_k[special_indices['R_inf'], special_indices['R_inf']] = ohmic_penalty
                if 'C_inv' in special_indices.keys():
                    m_k[special_indices['C_inv'], special_indices['C_inv']] = capacitance_penalty

                # Add entry for vz_offset if applicable
                if 'vz_offset' in special_indices.keys():
                    m_k[special_indices['vz_offset'], special_indices['vz_offset']] = 1 / vz_offset_scale

                # Add entry for background_scale if applicable
                if 'background_scale' in special_indices.keys():
                    m_k[special_indices['background_scale'], special_indices['background_scale']] = background_penalty

                # Add DOP matrix
                if self.fit_dop:
                    m_dop = base_penalty_matrices.get(f'm{k}_dop', 0)
                    m_k[dop_start_index:dop_end_index, dop_start_index:dop_end_index] = m_dop
                    # penalty_matrices[f'm{k}_dop'] = m_dop.copy()

                # Insert main DRT matrix
                m_k[num_special:, num_special:] = m_drt

                penalty_matrices[f'm{k}'] = m_k.copy()
                # penalty_matrices[f'm{k}_drt'] = m_drt.copy()
        elif penalty_type == 'discrete':
            if self.fit_dop:
                raise ValueError('DOP fit not implemented with discrete penalty')

            for k in range(len(derivative_weights)):
                # Get penalty matrix for DRT coefficients
                l_drt = base_penalty_matrices[f'l{k}']

                # Add rows/columns for v_baseline, inductance, and R_inf
                l_k = np.zeros((l_drt.shape[0] + num_special, l_drt.shape[1] + num_special))

                # Insert penalties for special parameters
                if 'v_baseline' in special_indices.keys():
                    l_k[special_indices['v_baseline'], special_indices['v_baseline']] = v_baseline_penalty ** 0.5
                if 'inductance' in special_indices.keys():
                    l_k[special_indices['inductance'], special_indices['inductance']] = inductance_penalty ** 0.5
                if 'R_inf' in special_indices.keys():
                    l_k[special_indices['R_inf'], special_indices['R_inf']] = ohmic_penalty ** 0.5
                if 'C_inv' in special_indices.keys():
                    l_k[special_indices['C_inv'], special_indices['C_inv']] = capacitance_penalty ** 0.5

                # Add entry for vz_offset if applicable
                if 'vz_offset' in special_indices.keys():
                    l_k[special_indices['vz_offset'], special_indices['vz_offset']] = 1 / vz_offset_scale ** 0.5

                # Add entry for background_scale if applicable
                if 'background_scale' in special_indices.keys():
                    l_k[special_indices['background_scale'], special_indices[
                        'background_scale']] = background_penalty ** 0.5

                # Add DOP matrix
                if self.fit_dop:
                    l_dop = base_penalty_matrices.get(f'l{k}_dop', 0)
                    l_k[dop_start_index:dop_end_index] = l_dop

                # Insert main DRT matrix
                l_k[num_special:, num_special:] = l_drt

                penalty_matrices[f'l{k}'] = l_k.copy()

                # Calculate norm matrix
                penalty_matrices[f'm{k}'] = l_k.T @ l_k

        self._qp_matrices = {
            'rm': rm,
            'zm': zm,
            'penalty_matrices': penalty_matrices
        }

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

    def _prep_chrono_prediction_matrix(self, times, input_signal, step_times, step_sizes,
                                       op_mode, offset_steps, smooth_inf_response):
        # TODO: incorporate step_times and step_sizes logic
        #  This is a placeholder to avoid issues with corner cases
        #  (e.g. if input_signal remains the same but step_times is manually specified)
        self._recalc_chrono_prediction_matrix = True

        # Validate input signal
        if input_signal is not None and step_times is not None:
            raise ValueError('Either input_signal OR (step_times and step_sizes) should be provided; '
                             'received input_signal and step_times')
        elif step_times is not None and step_sizes is None:
            raise ValueError('If input signal steps are provided, both step_times and step_sizes must be provided; '
                             'received step_times only')

        # If input signal is not provided, use self.raw_input_signal
        if input_signal is None and step_times is None:
            if self._t_fit_subset_index is not None:
                input_signal = self.raw_input_signal[self._t_fit_subset_index]
            else:
                input_signal = self.raw_input_signal
            use_fit_signal = True
        else:
            use_fit_signal = False

        # If signal steps are provided instead of input_signal, create dummy signal for validation
        if step_times is not None:
            input_signal = pp.generate_model_signal(times, step_times, step_sizes, None, 'ideal')

        self.t_predict = times
        self.raw_prediction_input_signal = input_signal
        self.chrono_mode_predict = op_mode
        # Final update of t_predict in case recalc status changed
        self.t_predict = times

        if self.print_diagnostics:
            print('recalc_response_prediction_matrix:', self._recalc_chrono_prediction_matrix)

        # Process signal to identify steps
        if use_fit_signal:
            # Allow times to have a different length than input_signal if using fitted signal (input_signal = None)
            # This will break construct_inf_response_vector if smooth_inf_response==False
            step_times = self.step_times
            step_sizes = self.step_sizes
            tau_rise = self.tau_rise
        else:
            if step_times is None:
                # Identify steps in applied signal
                step_times, step_sizes, tau_rise = pp.process_input_signal(times, input_signal, self.step_model,
                                                                           offset_steps)
            else:
                tau_rise = None

        if self._recalc_chrono_prediction_matrix:
            # Matrix recalculation is required
            rm, rm_layered = mat1d.construct_response_matrix(self.basis_tau, times, self.step_model, step_times,
                                                             step_sizes, basis_type=self.tau_basis_type,
                                                             epsilon=self.tau_epsilon, tau_rise=tau_rise,
                                                             op_mode=op_mode,
                                                             integrate_method=self.integrate_method,
                                                             zga_params=self.zga_params,
                                                             interpolate_grids=self.interpolate_lookups['response'])

            induc_rv = mat1d.construct_inductance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                  tau_rise, op_mode)
            cap_rv = mat1d.construct_capacitance_response_vector(times, self.step_model, step_times, step_sizes,
                                                                 tau_rise, op_mode)

            self.prediction_matrices.update({
                'response': rm.copy(),
                'inductance_response': induc_rv.copy(),
                'capacitance_response': cap_rv.copy(),
            })

            # With prediction matrices calculated, set recalc flag to False
            self._recalc_chrono_prediction_matrix = False
            if self.print_diagnostics:
                print('Calculated response prediction matrices')
        elif self._t_predict_eq_t_fit:
            print('eq')
            # times is the same as self.t_fit. Do not overwrite
            rm = self.fit_matrices['response'].copy()
            induc_rv = self.fit_matrices['inductance_response'].copy()
            cap_rv = self.fit_matrices['capacitance_response'].copy()
        elif self._t_predict_subset_index[0] == 'predict':
            # times is a subset of self.t_predict. Use sub-matrices of existing matrix; do not overwrite
            rm = self.prediction_matrices['response'][self._t_predict_subset_index[1], :].copy()
            induc_rv = self.prediction_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
            cap_rv = self.prediction_matrices['capacitance_response'][self._t_predict_subset_index[1]].copy()
        elif self._t_predict_subset_index[0] == 'fit':
            print('fit')
            # times is a subset of self.t_fit. Use sub-matrices of existing matrix; do not overwrite
            rm = self.fit_matrices['response'][self._t_predict_subset_index[1], :].copy()
            induc_rv = self.fit_matrices['inductance_response'][self._t_predict_subset_index[1]].copy()
            cap_rv = self.fit_matrices['capacitance_response'][self._t_predict_subset_index[1]].copy()
        else:
            # All matrix parameters are the same. Use existing matrix
            print('predict')
            rm = self.prediction_matrices['response'].copy()
            induc_rv = self.prediction_matrices['inductance_response'].copy()
            cap_rv = self.prediction_matrices['capacitance_response'].copy()

        # Construct R_inf response vector
        inf_rv = mat1d.construct_ohmic_response_vector(times, self.step_model, step_times, step_sizes,
                                                       tau_rise, input_signal, smooth_inf_response,
                                                       op_mode)

        self.prediction_matrices['inf_response'] = inf_rv.copy()

        # Construct DOP matrix
        if self.fit_dop:
            rm_dop, _ = phasance.construct_phasor_v_matrix(times, self.basis_nu, self.nu_basis_type,
                                                           self.nu_epsilon, self.step_model, step_times,
                                                           step_sizes, op_mode)
            self.prediction_matrices['rm_dop'] = rm_dop.copy()
        else:
            rm_dop = None

        if self.series_neg:
            rm = np.hstack((rm, -rm))

        return rm, induc_rv, inf_rv, cap_rv, rm_dop

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
        else:
            try:
                if self._f_predict_eq_f_fit:
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
            except KeyError:
                # The matrix we're trying to reuse doesn't exist. Recalculate explicitly
                # This can happen if the matrix was deleted or if the DRT instance was loaded from file
                self._recalc_eis_prediction_matrix = True
                zm, _ = self._prep_impedance_prediction_matrix(frequencies)
                self._f_predict_eq_f_fit = False
                self._f_predict_subset_index = ('', [])

        if self.fit_dop:
            zm_dop = phasance.construct_phasor_z_matrix(frequencies, self.basis_nu,
                                                        self.nu_basis_type, self.nu_epsilon)
            self.prediction_matrices['zm_dop'] = zm_dop.copy()
        else:
            zm_dop = None

        if self.series_neg:
            zm = np.hstack((zm, -zm))

        return zm, zm_dop

    def _get_vz_strength_vec(self, times=None, frequencies=None, fit_times=None, step_times=None,
                             fit_frequencies=None, vz_offset_eps=1):
        if fit_times is None:
            fit_times = self.get_fit_times(True)

        if step_times is None:
            step_times = self.nonconsec_step_times

        if fit_frequencies is None:
            fit_frequencies = self.get_fit_frequencies(True)

        if fit_times is None or fit_frequencies is None or vz_offset_eps is None:
            if times is not None:
                chrono_vz_strength = np.ones(len(times))
            else:
                chrono_vz_strength = None

            if frequencies is not None:
                eis_vz_strength = np.ones(len(frequencies))
            else:
                eis_vz_strength = None

            return chrono_vz_strength, eis_vz_strength
        else:
            # VZ offset decays as we move away from overlap
            rbf = basis.get_basis_func('gaussian')

            # Get effective timescale of each data point to determine data limits
            fit_time_deltas = pp.get_time_since_step(fit_times, step_times, prestep_value=-1)
            chrono_tau_min = np.min(fit_time_deltas[fit_time_deltas > 0])
            eis_tau_max = np.max(1 / (2 * np.pi * fit_frequencies))

            # Make vz_offset strength vectors
            # Strength is 1 in overlapped region and then decays in log space
            if times is not None:
                time_deltas = pp.get_time_since_step(times, step_times, prestep_value=-1)
                chrono_vz_strength = np.ones(len(time_deltas))
                chrono_vz_strength[(time_deltas >= eis_tau_max)] = rbf(
                    np.log(time_deltas[time_deltas >= eis_tau_max] / eis_tau_max), vz_offset_eps
                )
                chrono_vz_strength[time_deltas == -1] = 0
            else:
                chrono_vz_strength = None

            if frequencies is not None:
                f_inv = 1 / (2 * np.pi * frequencies)
                eis_vz_strength = np.ones(len(frequencies))
                eis_vz_strength[f_inv <= chrono_tau_min] = rbf(
                    np.log(f_inv[f_inv <= chrono_tau_min] / chrono_tau_min), vz_offset_eps
                )
            else:
                eis_vz_strength = None

            return chrono_vz_strength, eis_vz_strength

    def extract_qphb_parameters(self, x):
        special_indices = {k: self.special_qp_params[k]['index'] for k in self.special_qp_params.keys()}
        fit_parameters = {'x': x[self.get_qp_mat_offset():] * self.coefficient_scale}

        if 'R_inf' in special_indices.keys():
            fit_parameters['R_inf'] = x[special_indices['R_inf']] * self.coefficient_scale
        else:
            fit_parameters['R_inf'] = 0

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

        if 'C_inv' in special_indices.keys():
            fit_parameters['C_inv'] = x[special_indices['C_inv']] \
                                      * self.coefficient_scale * self.capacitance_scale
        else:
            fit_parameters['C_inv'] = 0

        if 'background_scale' in special_indices.keys():
            fit_parameters['background_scale'] = x[special_indices['background_scale']]

        if self.fit_dop:
            dop_start, dop_end = self.dop_indices
            fit_parameters['x_dop'] = x[dop_start:dop_end] * self.dop_scale_vector * self.coefficient_scale

        return fit_parameters

    def estimate_chrono_background(self, times, i_signal, v_signal, bkg_iter=1,
                                   gp=None, kernel_type='gaussian', length_scale_bounds=(0.01, 10),
                                   periodicity_bounds=(1e-3, 1e3), noise_level_bounds=(0.1, 10),
                                   kernel_size=1, n_restarts=1, kernel_scale_factor=1, y_err_thresh=1e-3,
                                   linear_downsample=True, linear_sample_interval=None, copy_self=False,
                                   **fit_kw):
        fit_defaults = {'max_iter': 10, 'error_structure': None}
        fit_kw = dict(fit_defaults, **fit_kw)

        # Make a copy to avoid overwriting any attributes during fit
        if copy_self:
            drt_bkg = deepcopy(self)
        else:
            drt_bkg = self
        gps, y_bkg = background.estimate_chrono_background(drt_bkg, times, i_signal, v_signal, max_iter=bkg_iter,
                                                           gp=gp, kernel_type=kernel_type,
                                                           length_scale_bounds=length_scale_bounds,
                                                           periodicity_bounds=periodicity_bounds,
                                                           noise_level_bounds=noise_level_bounds,
                                                           kernel_size=kernel_size, n_restarts=n_restarts,
                                                           kernel_scale_factor=kernel_scale_factor,
                                                           y_err_thresh=y_err_thresh,
                                                           linear_downsample=linear_downsample,
                                                           linear_sample_interval=linear_sample_interval, fit_kw=fit_kw)

        if copy_self:
            return drt_bkg, gps, y_bkg
        else:
            return gps, y_bkg

    # -------------------------------------------------------
    # Attribute management for storing/loading fit parameters
    # -------------------------------------------------------
    @property
    def attribute_categories(self):
        att = {
            'config': [
                'fixed_basis_tau', 'basis_tau', 'tau_basis_type', 'tau_epsilon', 'tau_supergrid',
                'fixed_basis_nu', 'basis_nu', 'nu_basis_type', 'nu_epsilon',
                'series_neg', 'fit_dop', 'normalize_dop', 'fit_inductance',
                'step_model', 'chrono_mode',
                'frequency_precision', 'time_precision', 'input_signal_precision',
                'integrate_method'
            ],
            'fit_core': [
                'fit_parameters', 'fit_type', 'fit_kwargs',
                'special_qp_params',
                'pfrt_result',
                # These are necessary for extract_qphb_parameters
                'coefficient_scale', 'inductance_scale', 'input_signal_scale', 'response_signal_scale',
                'scaled_response_offset', 'impedance_scale', 'dop_scale_vector',
                # dual fit attributes
                'discrete_candidate_df', 'discrete_candidate_dict',
                'best_candidate_df', 'best_candidate_dict', 'discrete_reordered_candidates',
                'pfrt_candidate_df', 'pfrt_candidate_dict'
            ],
            'fit_detail': [
                'qphb_params', 'cvx_result', 'qphb_history', 'pfrt_history', 'interpolate_lookups', 'fit_matrices'
            ],
            'data': [
                't_fit', 'raw_input_signal', 'raw_response_signal', 'scaled_input_signal', 'scaled_response_signal',
                'raw_response_background', 'step_times', 'nonconsec_step_times', 'step_sizes', 'tau_rise',
                'f_fit', 'z_fit', 'z_fit_scaled',
                'chrono_outlier_index', 'chrono_outliers', 'eis_outlier_index', 'eis_outliers',
                '_t_fit_subset_index', '_f_fit_subset_index'
            ],
        }

        return att

    def get_attributes(self, which):
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
        for k, v in att_dict.items():
            setattr(self, k, deepcopy(v))

    def save_attributes(self, which, dest):
        att_dict = self.get_attributes(which)
        with open(dest, 'wb') as f:
            pickle.dump(att_dict, f, pickle.DEFAULT_PROTOCOL)

    def load_attributes(self, source):
        with open(source, 'rb') as f:
            att_dict = pickle.load(f)
        self.set_attributes(att_dict)

    def copy(self):
        return deepcopy(self)

