import numpy as np
from scipy.integrate import cumtrapz
from scipy import signal
from scipy.stats import hmean, iqr, median_abs_deviation
import warnings

from .models import peaks
from .plotting import plot_distribution
from .utils import scale, stats
from .utils.array import check_equality


# =======================
# Scorer class
# =======================
class DrtScorer:
    def __init__(self, normalize=True, find_peaks_kw=None, sim_function_kw=None, reg_kw=None):

        self.normalize = normalize

        # if find_peaks_kw is None:
        #     find_peaks_kw = {}
        self.find_peaks_kw = find_peaks_kw

        # if sim_function_kw is None:
        #     sim_function_kw = {}
            #     'sim_function_type': 'gaussian',
            #     'order': 2,
            #     'epsilon': 0.75
            # }
        self.sim_function_kw = sim_function_kw

        if reg_kw is None:
            reg_kw = {}
            #     'pad': 1e-5,
            #     'sigma_uniform': 0.1
            # }
        self.reg_kw = reg_kw

        self.tau_reg = None
        self.tau_cls = None
        self.y_exact = None
        self.y_est = None
        self.y_is_discretized = False
        self.exact_peak_tau = None
        self.est_peak_tau = None
        self.exact_model = None
        self.est_model = None

        self.cls_detail = None
        self.cls_summary = None
        self.reg_detail = None
        self.reg_summary = None
        self.metric_summary = None

        self.rc_curve_args = None
        self.rc_curve_metrics = None


    # def calculate_classification_metrics(self, est_peak_tau, exact_peak_tau):
    #     self.exact_peak_tau = exact_peak_tau
    #     self.est_peak_tau = est_peak_tau
    #     # Calculate peak similarity function
    #     est_to_exact_sim, exact_to_est_sim = calc_peak_similarity(est_peak_tau, exact_peak_tau,
    #                                                               self.sim_function_kw['function_type'],
    #                                                               self.sim_function_kw['order'],
    #                                                               self.sim_function_kw['epsilon'])
    #
    #     # Get TP, FP, and FN character for all peaks
    #     est_tp_character = est_to_exact_sim.copy()
    #     est_fp_character = 1 - est_tp_character
    #     exact_tp_character = exact_to_est_sim.copy()
    #     exact_fn_character = 1 - exact_tp_character
    #
    #     self.classification_detail = {
    #         'exact_peak_tau': exact_peak_tau.copy(),
    #         'est_peak_tau': est_peak_tau.copy(),
    #         'est_to_exact_sim': est_to_exact_sim.copy(),
    #         'exact_to_est_sim': exact_to_est_sim.copy(),
    #         'est_tp_character': est_tp_character.copy(),
    #         'est_fp_character': est_fp_character.copy(),
    #         'exact_tp_character': exact_tp_character.copy(),
    #         'exact_fn_character': exact_fn_character.copy()
    #     }
    #
    #     # Sum character values to get total TP, FP, and FN
    #     tp_count = np.sum(est_tp_character)
    #     fp_count = np.sum(est_fp_character)
    #     fn_count = np.sum(exact_fn_character)
    #
    #     # Calculate metrics
    #     # True positive rate, AKA sensitivity or recall
    #     if tp_count + fn_count == 0:
    #         # If no predicted positives, TPR is 1
    #         tpr = 1
    #     else:
    #         tpr = tp_count / (tp_count + fn_count)
    #     ppv = tp_count / (tp_count + fp_count)  # Positive predictive value, precision
    #     fmi = np.sqrt(tpr * ppv)  # Fowlkes-Mallows Index
    #     f_score = 2 * ppv * tpr / (tpr + ppv)  # F-score
    #
    #     self.classification_summary = {
    #         'tp': tp_count,
    #         'fp': fp_count,
    #         'fn': fn_count,
    #         'tpr': tpr,
    #         'ppv': ppv,
    #         'fmi': fmi,
    #         'f1': f_score
    #     }
    #
    #     return self.classification_summary.copy()

    # def calculate_regression_metrics(self, tau, y_exact, y_est, y_is_discretized=False,
    #                                  normalize=None, kld_kw=None, sqed_kw=None):
    #     self.tau_reg = tau
    #     self.y_exact = y_exact
    #     self.y_est = y_est
    #     self.y_is_discretized = y_is_discretized
    #
    #     if kld_kw is None:
    #         kld_kw = self.divergence_kw.get('kld_kw', {})
    #
    #     if sqed_kw is None:
    #         sqed_kw = self.divergence_kw.get('sqed_kw', {})
    #
    #     if normalize is None:
    #         normalize = self.divergence_kw.get('normalize', True)
    #
    #     r2_resid, r2_dev = r2_dev_arrays(y_exact, y_est)
    #     kld_array = kl_div_array(np.log(tau), y_exact, y_est, normalize=normalize, **kld_kw)
    #     jsd_array = js_div_array(np.log(tau), y_exact, y_est, normalize=normalize, **kld_kw)
    #     sqed_array = sqe_distance_array(np.log(tau), y_exact, y_est, normalize=normalize, **sqed_kw)
    #
    #     self.regression_detail = {
    #         'r2_resid': r2_resid.copy(),
    #         'r2_dev': r2_dev.copy(),
    #         'kld_array': kld_array.copy(),
    #         'jsd_array': jsd_array.copy(),
    #         'sqed_array': sqed_array.copy()
    #     }
    #
    #     self.regression_summary = {
    #         'r2': r2_score(y_exact, y_est),
    #         'rss': np.sum(sqed_array),
    #         'kld': np.trapz(kld_array, x=np.log(tau)),
    #         'jsd': np.trapz(jsd_array, x=np.log(tau)),
    #         'sqed': np.trapz(sqed_array, x=np.log(tau))
    #     }
    #
    #     return self.regression_summary.copy()

    def evaluate_data(self, regression_data, classification_data, is_discretized=False):
        # Unpack data
        tau_reg, y_exact_reg, y_est_reg = regression_data
        tau_cls, exact_peak_tau, est_peak_tau = classification_data

        # Store info
        self.tau_reg = tau_reg
        self.tau_cls = tau_cls
        self.y_exact = y_exact_reg
        self.y_est = y_est_reg
        self.y_is_discretized = is_discretized
        self.exact_peak_tau = exact_peak_tau
        self.est_peak_tau = est_peak_tau

        # Update normalize kw for regression metrics
        # If provided, override instance attribute; otherwise, use self.normalize.
        # This matters because when evaluate_models is used to calculate metrics, normalization is performed using
        # the model methods for improved accuracy. In this case, the data should not be re-normalized.
        # if normalize is None:
        #     normalize = self.normalize

        # Calculate regression metrics
        self.reg_detail, self.reg_summary = calculate_reg_metrics(tau_reg, y_exact_reg, y_est_reg, is_discretized,
                                                                  self.normalize, **self.reg_kw)

        # Process classification data
        self.cls_detail, self.cls_summary = calculate_cls_metrics(est_peak_tau, exact_peak_tau, self.sim_function_kw)

        # Calculate combination metrics
        combo_metrics = calculate_combo_metrics(self.reg_summary, self.cls_summary)

        # Merge summary metrics
        self.metric_summary = self.reg_summary | self.cls_summary | combo_metrics

        return self.metric_summary.copy()

    def evaluate_models(self, exact_model, est_model, tau_reg, tau_cls):
        self.exact_model = exact_model
        self.est_model = est_model

        # Check for singularities
        if getattr(exact_model, 'is_singular', False) or getattr(est_model, 'is_singular', False):
            discretize = True
        else:
            discretize = False

        # print(discretize)

        # Process models
        y_exact_reg, exact_peak_tau = process_model_instance(exact_model, tau_reg, tau_cls, discretize,
                                                             self.normalize, self.find_peaks_kw)
        y_est_reg, est_peak_tau = process_model_instance(est_model, tau_reg, tau_cls, discretize,
                                                         self.normalize, self.find_peaks_kw)

        # Package output and pass to evaluate_data
        reg_data = (tau_reg, y_exact_reg, y_est_reg)
        cls_data = (tau_cls, exact_peak_tau, est_peak_tau)

        # Don't re-normalize regression vectors in evaluate_data
        return self.evaluate_data(reg_data, cls_data, discretize)

    def get_regression_vector(self, model_name, tau_reg=None, discretize=False):
        if model_name not in ('exact', 'est'):
            raise ValueError(f"Invalid model_name {model_name}: must be 'exact' or 'est'")

        if tau_reg is None:
            tau_reg = self.tau_reg

        model = getattr(self, f'{model_name}_model', None)
        if model is not None:
            if discretize:
                # Use relaxation mass functions in place of distribution
                if hasattr(model, 'predict_mass'):
                    y_reg = model.predict_mass(tau_reg)
                else:
                    if getattr(model, 'is_singular', False) \
                            and getattr(model, 'singularity_info', None) is None:
                        warnings.warn('Model distribution is singular, but neither a predict_mass method nor '
                                      'singularity_info attribute is defined in the model instance.')
                    y_reg = discretize_distribution(tau_reg, model.predict_distribution(tau_reg),
                                                    getattr(model, 'singularity_info', None))
            else:
                # Evaluate distribution as normal
                y_reg = model.predict_distribution(tau_reg)
        else:
            # No model instance stored. Use stored y vector
            if not check_equality(tau_reg, self.tau_reg):
                raise ValueError('No model instance available for prediction over custom tau array')

            y_stored = getattr(self, f'y_{model_name}').copy()
            if discretize:
                if self.y_is_discretized:
                    y_reg = y_stored
                else:
                    y_reg = discretize_distribution(tau_reg, y_stored)
            else:
                y_reg = y_stored

        return y_reg

    def compute_rc_curve(self, exact_model, data_list, fit_function, arg_array, tau_reg, tau_cls):
        # discretize=False,
        #                  normalize=True, find_peaks_kw=None, sim_function_kw=None, reg_kw=None):
        # fit_function: signature fit_function(data, *args), must return est_model instance

        # y_reg_exact, exact_peak_tau = process_model_instance(exact_model, tau_reg, tau_cls, discretize,
        #                                                      normalize, find_peaks_kw=find_peaks_kw)
        # if reg_kw is None:
        #     reg_kw = {}

        vec_dict = None

        for i, arg_vec in enumerate(arg_array):
            print(arg_vec)
            arg_vec = np.atleast_1d(arg_vec)

            metric_dicts = []
            for j, data in enumerate(data_list):
                # Fit each dataset with current args
                est_model = fit_function(data, *arg_vec)
                self.evaluate_models(exact_model, est_model, tau_reg, tau_cls)
                # y_reg_est, est_peak_tau = process_model_instance(est_model, tau_reg, tau_cls, discretize,
                #                                                  normalize, find_peaks_kw=find_peaks_kw)
                #
                # reg_detail, reg_summary = calculate_reg_metrics(tau_reg, y_reg_exact, y_reg_est, discretize, normalize,
                #                                                 **reg_kw)
                # cls_detail, cls_summary = calculate_cls_metrics(est_peak_tau, exact_peak_tau, sim_function_kw)
                # combo_metrics = calculate_combo_metrics(reg_summary, reg_detail)
                # # Merge all metrics
                # all_metrics = reg_summary | cls_summary | combo_metrics

                metric_dicts.append(self.metric_summary.copy())

            # Aggregate all metrics
            agg_metrics = aggregate_metrics(metric_dicts)

            # Initialize vector dict
            if vec_dict is None:
                vec_dict = {k: np.empty(len(arg_array)) for k in agg_metrics.keys()}

            for k, v in agg_metrics.items():
                vec_dict[k][i] = v

            self.rc_curve_args = arg_array.copy()
            self.rc_curve_metrics = vec_dict.copy()

        return vec_dict

    # --------------------
    # Plotting
    # --------------------
    def plot_drt_comparison(self, tau=None, discretize=False, ax=None, scale_prefix=None, show_singularities=None,
                            singularity_height=None,
                            mark_est_peaks=True, mark_est_peaks_kw=None,
                            mark_exact_peaks=True, mark_exact_peaks_kw=None,
                            color_est_peaks=True, color_exact_peaks=False,
                            return_cmap_sm=False,
                            exact_kw=None, est_kw=None):

        if tau is None:
            tau = self.tau_reg

        if show_singularities is None:
            # By default, show singularities if distributions are not discretized
            show_singularities = not discretize

        # Get distribution vectors. Use model.predict_distribution if available; otherwise use regression vectors
        y_exact = self.get_regression_vector('exact', tau, discretize)
        y_est = self.get_regression_vector('est', tau, discretize)
        # if self.exact_model is not None:
        #     y_exact = self.exact_model.predict_distribution(tau)
        # else:
        #     y_exact = self.y_exact
        #
        # if self.est_model is not None:
        #     y_est = self.est_model.predict_distribution(tau)
        # else:
        #     y_est = self.y_est

        # Get appropriate scale
        if scale_prefix is None:
            scale_prefix = scale.get_common_scale_prefix([y_exact, y_est])
        scale_factor = scale.get_factor_from_prefix(scale_prefix)

        # Set infinite values to slightly above max visible value to avoid gaps in the plot due to singularities
        if singularity_height is None:
            y_concat = np.concatenate((y_exact, y_est))
            singularity_height = 1. * np.max(np.abs(y_concat[~np.isinf(y_concat)]))
        y_exact[np.isinf(y_exact)] = np.sign(y_exact[np.isinf(y_exact)]) * singularity_height
        y_est[np.isinf(y_est)] = np.sign(y_est[np.isinf(y_est)]) * singularity_height

        # Plot exact distribution
        if exact_kw is None:
            exact_kw = {'ls': '--', 'zorder': -10}
        if 'label' not in exact_kw.keys():
            exact_kw['label'] = 'Exact'
        ax, info = plot_distribution(tau, y_exact, ax, None, scale_prefix, None, return_info=True, **exact_kw)
        exact_line, _, _ = info

        # Plot estimated distribution
        if est_kw is None:
            est_kw = {'c': 'k'}
        if 'label' not in est_kw.keys():
            est_kw['label'] = 'Estimate'
        ax, info = plot_distribution(tau, y_est, ax, None, scale_prefix, None, return_info=True, **est_kw)
        est_line, _, _ = info

        # Indicate singularities with vertical lines
        if show_singularities:
            for model_name in ('exact', 'est'):
                if getattr(self, f'{model_name}_model') is not None:
                    # Use model to get singularity locations
                    model = getattr(self, f'{model_name}_model')
                    sing_info = getattr(model, 'singularity_info', None)
                    if sing_info is not None:
                        for si in sing_info:
                            r_s, tau_s = si
                            # Start at distribution value just past singularity
                            y_start = model.predict_distribution(tau_s * (1 + 1e-6)) / scale_factor

                            # End at max value
                            if r_s != 0:
                                y_end = np.sign(r_s) * singularity_height / scale_factor
                            else:
                                y_end = y_start

                            sing_kw = locals()[f'{model_name}_kw'].copy()
                            sing_kw['label'] = ''  # Don't double-label

                            if sing_kw.get('c', sing_kw.get('color', None)) is None:
                                sing_kw['c'] = locals()[f'{model_name}_line'][0].get_color()

                            ax.plot([tau_s, tau_s], [y_start, y_end], **sing_kw)

        # Mark peaks
        for model_name in ('exact', 'est'):
            mark_peaks = locals()[f'mark_{model_name}_peaks']

            if mark_peaks:
                mark_peaks_kw = locals()[f'mark_{model_name}_peaks_kw']
                if mark_peaks_kw is None:
                    mark_peaks_kw = {'alpha': 0.8, 'facecolor': 'none'}
                    if model_name == 'exact':
                        mark_peaks_kw['marker'] = '^'
                        mark_peaks_kw['s'] = 50
                    else:
                        mark_peaks_kw['marker'] = 'o'

                # y_concat = np.concatenate((y_exact, y_est))
                # gamma_max = np.max(y_concat[y_concat < np.inf]) / scale_factor
                # gamma_min = np.min(y_concat[y_concat > -np.inf]) / scale_factor
                # print(gamma_max, gamma_min)

                peak_tau = getattr(self, f'{model_name}_peak_tau')
                if not discretize and getattr(self, f'{model_name}_model') is not None:
                    # Use model to get distribution values at peaks
                    y_peaks = self.get_regression_vector(model_name, peak_tau, discretize)

                    # Handle singularities
                    y_peaks[np.isinf(y_peaks)] = np.sign(y_peaks[np.isinf(y_peaks)]) * singularity_height
                    # y_peaks[y_peaks == -np.inf] = gamma_min
                else:
                    # Find nearest distribution values in tau array
                    # When discretizing, must do it this way even if a model instance is available since the
                    # discretization output is dependent on the tau grid
                    peak_index = peaks.index_closest_peaks(peak_tau, tau)

                    # Get the nearest index that maximizes the y value
                    y_model = locals()[f'y_{model_name}']
                    peak_index = np.array([idx - 1 + np.argmax(y_model[idx - 1: idx + 2]) for idx in peak_index])

                    peak_tau = tau[peak_index]
                    y_peaks = y_model[peak_index]
                # Scale peak y values
                y_peaks /= scale_factor

                # Match color of corresponding model line
                line = locals()[f'{model_name}_line']

                if locals()[f'color_{model_name}_peaks']:
                    mark_peaks_kw['c'] = self.cls_detail[f'{model_name}_tp_character']
                    mark_peaks_kw['vmin'] = 0
                    mark_peaks_kw['vmax'] = 1
                    if mark_peaks_kw.get('cmap', mark_peaks_kw.get('colormap', None)) is None:
                        mark_peaks_kw['cmap'] = 'bwr_r'

                cmap_sm = ax.scatter(peak_tau, y_peaks, edgecolors=line[0].get_color(), **mark_peaks_kw)
            else:
                cmap_sm = None

        # Format
        if discretize:
            ax.set_ylabel(f'$p$ ({scale_prefix}$\Omega$)')
        else:
            ax.set_ylabel(f'$\gamma$ ({scale_prefix}$\Omega$)')

        fig = ax.get_figure()
        fig.tight_layout()

        if return_cmap_sm:
            return ax, cmap_sm
        else:
            return ax


def process_model_instance(model, tau_reg, tau_cls, discretize, normalize_find_peaks, find_peaks_kw=None):
    """
    Process a model instance for evaluation.
    :param model: Model instance. The instance must define a predict_distribution method with the signature
    model.predict_distribution(tau). The instance may optionally define the attribute model.is_singular. If the instance
    is singular, it must also define either a method model.predict_mass(tau) or an attribute model.singularity_info
    :param tau_reg: tau array for regression
    :param tau_cls: tau array for classification (used for peak finding)
    :param discretize:
    :param normalize_find_peaks:
    :param find_peaks_kw:
    :return:
    """
    # Don't normalize y_reg here - some metrics require common norm (r2, rss),
    # others require independent norms (kl_div, js_div, sqed)
    # # Get R_p for normalization
    # if normalize:
    #     normalize_by = get_model_r_p(model, tau_reg)
    # else:
    #     normalize_by = 1

    # Get regression vectors
    if discretize:
        # Use relaxation mass functions in place of distribution due to singular distribution
        if hasattr(model, 'predict_mass'):
            y_reg = model.predict_mass(tau_reg)
        else:
            if getattr(model, 'is_singular', False) \
                    and getattr(model, 'singularity_info', None) is None:
                warnings.warn('Model distribution is singular, but neither a predict_mass method nor '
                              'singularity_info attribute is defined in the model instance.')
            y_reg = discretize_distribution(tau_reg, model.predict_distribution(tau_reg),
                                            getattr(model, 'singularity_info', None))
    else:
        # No singularities. Evaluate distribution as normal
        y_reg = model.predict_distribution(tau_reg)

    # # Normalize by R_p
    # y_reg /= normalize_by

    # Find model peaks for classification
    peak_tau = find_model_peaks(model, tau_cls, normalize_find_peaks, find_peaks_kw)

    return y_reg, peak_tau


def get_model_r_p(model, tau=None):
    if hasattr(model, 'predict_r_p'):
        r_p = model.predict_r_p()
    else:
        if tau is None:
            raise ValueError('tau must be provided if model does not have a predict_r_p method')
        y_norm = model.predict_distribution(tau)
        r_p = np.trapz(y_norm, x=np.log(tau))
        # Add singularity mass to R_p
        if getattr(model, 'singularity_info', None) is not None:
            r_p += np.sum([si[0] for si in getattr(model, 'singularity_info')])

    return r_p


def find_model_peaks(model, tau_cls, normalize, find_peaks_kw=None, return_prominence=False):
    if normalize:
        normalize_by = get_model_r_p(model, tau_cls)
    else:
        normalize_by = 1

    # Get curvature for peak identification
    try:
        fxx = model.predict_distribution(tau_cls, order=2) / normalize_by
        index_offset = 0
    except TypeError:
        gamma_cls = model.predict_distribution(tau_cls) / normalize_by

        # Find peaks
        fx = np.diff(gamma_cls) / np.diff(np.log(tau_cls))
        fxx = np.diff(fx) / np.diff(np.log(tau_cls[1:]))

        index_offset = 1

    if find_peaks_kw is None:
        # print('min_curv:', np.min(fxx[fxx > -np.inf]))
        # print('med_curv:', np.median(np.abs(fxx)))
        # print('std_curv:', np.std(fxx[~np.isinf(fxx)]))
        # print('iqr_curv:', iqr(fxx[~np.isinf(fxx)]))
        # print('mad_curv:', median_abs_deviation(fxx[~np.isinf(fxx)]))
        prom_thresh = 0.05 * np.std(fxx[~np.isinf(fxx)]) + 5e-3
        # prom_thresh = 0.5 * iqr(fxx[~np.isinf(fxx)]) + 5e-3
        # print('thresh:', prom_thresh)
        find_peaks_kw = {'height': 0, 'prominence': prom_thresh}
    # peak_index = peaks.find_peaks_compound(fx[1:], fxx, **find_peaks_kw) + 1
    peak_index = peaks.find_peaks_simple(fxx, 2, **find_peaks_kw) + index_offset
    if len(peak_index) == 0:
        peak_tau = np.array([])
    else:
        peak_tau = tau_cls[peak_index]

    # Merge singular peaks
    peak_tau, sing_index = merge_singular_peaks(tau_cls, peak_tau, getattr(model, 'singularity_info', None))

    if return_prominence:
        # Get peak prominences
        peak_prom = signal.peak_prominences(-fxx, peak_index - 1)[0]
        peak_prom = np.insert(peak_prom, sing_index, np.inf)
        return peak_tau, peak_prom
    else:
        return peak_tau


def calculate_reg_metrics(tau, y_exact, y_est, discrete, normalize=True, pad=1e-5, sigma_uniform=None):

    r2_resid, r2_dev = r2_dev_arrays(y_exact, y_est)
    kld_array = kl_div_array(np.log(tau), y_exact, y_est, pad=pad, normalize=normalize, discrete=discrete)
    jsd_array = js_div_array(np.log(tau), y_exact, y_est, normalize=normalize, pad=pad, discrete=discrete)
    sqed_array = sqe_distance_array(np.log(tau), y_exact, y_est, normalize=normalize, discrete=discrete)
    wrss_array = rss_array(y_exact, y_est, weights=None, normalize=normalize, sigma_uniform=sigma_uniform)
    urss_array = rss_array(y_exact, y_est, weights=1, normalize=normalize, sigma_uniform=sigma_uniform)

    def aggregate_div(div_array):
        if discrete:
            return np.sum(div_array)
        else:
            return np.trapz(div_array, x=np.log(tau))

    reg_detail = {
        'r2_resid': r2_resid.copy(),
        'r2_dev': r2_dev.copy(),
        'kld_array': kld_array.copy(),
        'jsd_array': jsd_array.copy(),
        'sqed_array': sqed_array.copy(),
        'wrss_array': wrss_array.copy(),
        'urss_array': urss_array.copy()
    }

    reg_summary = {
        'r2': r2_score(y_exact, y_est),
        'urss': np.sum(urss_array),
        'wrss': np.sum(wrss_array),
        'kld': aggregate_div(kld_array),
        'jsd': aggregate_div(jsd_array),
        'sqed': aggregate_div(sqed_array)
    }

    reg_summary['f_kl'] = np.exp(-2 * reg_summary['kld'])

    return reg_detail, reg_summary


def calculate_cls_metrics(est_peak_tau, exact_peak_tau, sim_function_kw=None):
    if sim_function_kw is None:
        sim_function_kw = {
            'sim_function_type': 'gaussian',
            'order': 2,
            'epsilon': 0.75
        }
    # Calculate peak similarity function
    est_to_exact_sim, exact_to_est_sim = peak_similarity(est_peak_tau, exact_peak_tau, **sim_function_kw)

    # Get TP, FP, and FN character for all peaks
    est_tp_character = est_to_exact_sim.copy()
    est_fp_character = 1 - est_tp_character
    exact_tp_character = exact_to_est_sim.copy()
    exact_fn_character = 1 - exact_tp_character

    cls_detail = {
        'exact_peak_tau': exact_peak_tau.copy(),
        'est_peak_tau': est_peak_tau.copy(),
        'est_to_exact_sim': est_to_exact_sim.copy(),
        'exact_to_est_sim': exact_to_est_sim.copy(),
        'est_tp_character': est_tp_character.copy(),
        'est_fp_character': est_fp_character.copy(),
        'exact_tp_character': exact_tp_character.copy(),
        'exact_fn_character': exact_fn_character.copy()
    }

    # Sum character values to get total TP, FP, and FN
    tp_count = np.sum(est_tp_character)
    fp_count = np.sum(est_fp_character)
    fn_count = np.sum(exact_fn_character)

    # Calculate metrics
    tpr, ppv, fmi, f1_score = cls_metrics_from_counts(tp_count, fp_count, fn_count)

    cls_summary = {
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'tpr': tpr,
        'ppv': ppv,
        'fmi': fmi,
        'f1': f1_score
    }

    return cls_detail, cls_summary


def cls_metrics_from_counts(tp_count, fp_count, fn_count):
    # True positive rate, AKA sensitivity or recall
    if tp_count + fn_count == 0:
        # If no real positives, TPR is 1
        tpr = 1
    else:
        tpr = tp_count / (tp_count + fn_count)

    # Positive predictive value, AKA precision
    if tp_count + fp_count == 0:
        # If no predicted positives, PPV is 1
        ppv = 1
    else:
        ppv = tp_count / (tp_count + fp_count)

    fmi = np.sqrt(tpr * ppv)  # Fowlkes-Mallows Index
    f1_score = try_hmean([tpr, ppv])  # F1-score

    return tpr, ppv, fmi, f1_score


def try_hmean(x):
    try:
        return hmean(x)
    except ValueError:
        return np.nan


def calculate_combo_metrics(reg_summary, cls_summary):
    combo_dict = {
        'h_r2f1': try_hmean([reg_summary['r2'], cls_summary['f1']]),
        'g_r2fmi': np.sqrt(reg_summary['r2'] * cls_summary['fmi']),
        'h_klf1': try_hmean([reg_summary['f_kl'], cls_summary['f1']]),
        'g_klfmi': np.sqrt(reg_summary['f_kl'] * cls_summary['fmi']),
    }
    return combo_dict


def aggregate_metrics(metric_dicts, weights=None):
    if weights is None:
        weights = np.ones(len(metric_dicts))

    # Regression metrics: average
    agg_metrics = {}
    reg_metrics = ['r2', 'urss', 'wrss', 'kld', 'jsd', 'sqed']
    for k in reg_metrics:
        values = np.array([md[k] for md in metric_dicts])
        agg_metrics[k] = np.average(values, weights=weights)

    agg_metrics['f_kl'] = np.exp(-2 * agg_metrics['kld'])

    # Classification metrics: sum pseudo-counts
    cls_counts = ['tp', 'fp', 'fn']
    for k in cls_counts:
        values = np.array([md[k] for md in metric_dicts])
        agg_metrics[k] = np.sum(weights * values)

    tpr, ppv, fmi, f1_score = cls_metrics_from_counts(agg_metrics['tp'], agg_metrics['fp'], agg_metrics['fn'])
    agg_metrics['tpr'] = tpr
    agg_metrics['ppv'] = ppv
    agg_metrics['fmi'] = fmi
    agg_metrics['f1'] = f1_score

    # Combined metrics
    combo_metrics = calculate_combo_metrics(agg_metrics, agg_metrics)
    agg_metrics.update(combo_metrics)

    return agg_metrics


def peakthresh_pr_curve(exact_model, est_model, tau_cls, normalize=True, find_exact_peaks_kw=None):
    """
    Compute a precision-recall curve for a static DRT estimate by varying the peak identification threshold
    :param exact_model:
    :param est_model:
    :param tau_cls:
    :param normalize:
    :param find_exact_peaks_kw:
    :return:
    """
    # Find real peaks
    if find_exact_peaks_kw is None:
        find_exact_peaks_kw = {}
    exact_peak_tau = find_model_peaks(exact_model, tau_cls, normalize, find_exact_peaks_kw, False)

    # Find all possible predicted peaks
    find_est_peaks_kw = {'order1_kw': {}, 'order2_kw': {'height': 0}}
    est_peak_tau, est_peak_prom = find_model_peaks(est_model, tau_cls, normalize, find_est_peaks_kw, True)

    prom_thresholds = np.unique(est_peak_prom)
    tpr_array = np.empty(len(prom_thresholds) + 1)
    ppv_array = np.empty(len(prom_thresholds) + 1)
    # No predicted peaks at max threshold: TPR = 0, PPV = 1
    tpr_array[-1] = 0
    ppv_array[-1] = 1
    for i, thresh in enumerate(prom_thresholds):
        thresh_peak_tau = est_peak_tau[est_peak_prom >= thresh]
        cls_detail, cls_summary = calculate_cls_metrics(thresh_peak_tau, exact_peak_tau)
        tpr_array[i] = cls_summary['tpr']
        ppv_array[i] = cls_summary['ppv']

    return ppv_array, tpr_array


# def compute_rc_curve(exact_model, tau_reg, tau_cls, data_list, fit_function, arg_array, scorer):
#     # discretize=False,
#     #                  normalize=True, find_peaks_kw=None, sim_function_kw=None, reg_kw=None):
#     # fit_function: signature fit_function(data, *args), must return est_model instance
#
#     # y_reg_exact, exact_peak_tau = process_model_instance(exact_model, tau_reg, tau_cls, discretize,
#     #                                                      normalize, find_peaks_kw=find_peaks_kw)
#     # if reg_kw is None:
#     #     reg_kw = {}
#
#     vec_dict = None
#
#     for i, arg_vec in enumerate(arg_array):
#         print(arg_vec)
#         arg_vec = np.atleast_1d(arg_vec)
#
#         metric_dicts = []
#         for j, data in enumerate(data_list):
#             # Fit each dataset with current args
#             est_model = fit_function(data, *arg_vec)
#             scorer.evaluate_models(exact_model, est_model, tau_reg, tau_cls)
#             # y_reg_est, est_peak_tau = process_model_instance(est_model, tau_reg, tau_cls, discretize,
#             #                                                  normalize, find_peaks_kw=find_peaks_kw)
#             #
#             # reg_detail, reg_summary = calculate_reg_metrics(tau_reg, y_reg_exact, y_reg_est, discretize, normalize,
#             #                                                 **reg_kw)
#             # cls_detail, cls_summary = calculate_cls_metrics(est_peak_tau, exact_peak_tau, sim_function_kw)
#             # combo_metrics = calculate_combo_metrics(reg_summary, reg_detail)
#             # # Merge all metrics
#             # all_metrics = reg_summary | cls_summary | combo_metrics
#
#             metric_dicts.append(scorer.metric_summary.copy())
#
#         # Aggregate all metrics
#         agg_metrics = aggregate_metrics(metric_dicts)
#
#         # Initialize vector dict
#         if vec_dict is None:
#             vec_dict = {k: np.empty(len(arg_array)) for k in agg_metrics.keys()}
#
#         for k, v in agg_metrics.items():
#             vec_dict[k][i] = v
#
#     return vec_dict  # ppv_array, tpr_array, r2_array, kld_array


# ========================
# Divergence functions
# ========================
def normalize_distributions(x, *distributions, common_norm=False, discrete=False):
    if len(distributions) > 1:
        if discrete:
            # Sum masses, don't integrate
            areas = [np.sum(p) for p in distributions]
        else:
            areas = [np.trapz(p, x=x) for p in distributions]

        if common_norm:
            # Normalize all distributions to mean area (maintains relative scales)
            norm_area = [np.mean(areas)] * len(distributions)
        else:
            # Normalize each distribution to unit area (does not maintain relative scales)
            norm_area = areas
        return [distributions[i] / norm_area[i] for i in range(len(distributions))]
    else:
        p = distributions[0]
        if discrete:
            # Sum masses, don't integrate
            area = np.sum(p)
        else:
            area = np.trapz(p, x=x)
        return p / area


def kl_div_array(x, p, q, pad=1e-5, normalize=False, discrete=False):
    """
    Calculate array of KL divergence contributions
    :param ndarray x: Independent (integration) variable
    :param ndarray p: First (reference) distribution
    :param ndarray q: Second distribution
    :param float pad: amount by which to pad both distributions to reduce influence of zero or near-zero density regions
    :param bool normalize: Ff True, normalize both distributions to unit area.
    If False, assume distributions are already normalized
    :return:
    """
    if normalize:
        # Normalize p and q to unit mean area
        p, q = normalize_distributions(x, p, q, discrete=discrete)

    if pad > 0:
        # Pad both distributions and re-normalize to unit area
        x_range = np.max(x) - np.min(x)
        p = (p + pad) / (1 + pad * x_range)
        q = (q + pad) / (1 + pad * x_range)

    div = p * np.log(p / q)
    div[p == 0] = 0  # Divergence goes to zero when p==0
    return div


def kl_divergence(x, p, q, pad=1e-5, normalize=False, discrete=False):
    """
    Calculate KL divergence
    :param ndarray x: Independent (integration) variable
    :param ndarray p: First (reference)  distribution
    :param ndarray q: Second distribution
    :param float pad: amount by which to pad both distributions to reduce influence of zero or near-zero density regions
    :param bool normalize: Ff True, normalize both distributions to unit area.
    If False, assume distributions are already normalized
    :return:
    """
    kl_div = kl_div_array(x, p, q, pad, normalize, discrete)
    if discrete:
        return np.sum(kl_div)
    else:
        return np.trapz(kl_div, x=x)


def js_div_array(x, p, q, **kl_kw):
    m = 0.5 * (p + q)
    return 0.5 * (kl_div_array(x, p, m, **kl_kw) + kl_div_array(x, q, m, **kl_kw))


def js_divergence(x, p, q, **kl_kw):
    return np.trapz(js_div_array(x, p, q, **kl_kw), x=x)


def sqe_distance_array(x, p, q, normalize=False, discrete=False):
    if normalize:
        # Normalize p and q to unit mean area
        p, q = normalize_distributions(x, p, q, common_norm=True, discrete=discrete)

    return (p - q) ** 2


def sqe_distance(x, p, q, normalize=False, discrete=False):
    sqe = sqe_distance_array(x, p, q, normalize, discrete)
    if discrete:
        return np.sum(sqe)
    else:
        return np.trapz(sqe, x=x)


def divergence_index(div):
    """Convert the divergence (0 <= div <= inf) to an index (0 <= index <= 1)"""
    return np.exp(-div)


# ========================
# Regression functions
# ========================
def discretize_distribution(tau, gamma, singularity_info=None):
    """
    Discretize a distribution into a discrete mass function
    :param ndarray tau: Tau array
    :param ndarray gamma: Array of distribution values
    :param list singularity_info: List of 2-tuples indicating mass (R) and location (tau) of singularities in the
    distribution which are not captured by gamma
    :return:
    """
    # Get cumulative integral
    cum_mass = cumtrapz(gamma, x=np.log(tau), initial=0)

    # Add singularity mass
    if singularity_info is not None:
        for si in singularity_info:
            r_s, tau_s = si
            cum_mass[tau >= tau_s] += r_s

    inc_mass = np.diff(cum_mass)

    # Pad with zero to maintain array size
    inc_mass = np.concatenate(([0], inc_mass))

    return inc_mass


def r2_dev_arrays(y_true, y_est, weights=None):
    if weights is None:
        weights = 1

    y_resid = weights * (y_true - y_est)
    y_dev = weights * (y_true - np.mean(y_true))

    return y_resid, y_dev


def r2_score(y_true, y_est, weights=None):
    y_resid, y_dev = r2_dev_arrays(y_true, y_est, weights)

    ss_resid = np.sum(y_resid ** 2)
    ss_tot = np.sum(y_dev ** 2)

    return 1 - ss_resid / ss_tot


def chi_sq(y_true, y_est, weights=None):
    if weights is None:
        weights = 1

    return np.sum((weights * np.abs(y_true - y_est)) ** 2)


def rss_array(p, q, weights=None, sigma_uniform=None, normalize=False):
    if weights is None:
        # Assume the variance has proportional and uniform components. Sum them to get the total variance
        if sigma_uniform is None:
            # Set sigma_uniform to 25% of R_p
            sigma_uniform = np.sum(p) * 0.25
        var = p ** 2 + sigma_uniform ** 2
        # var = normalize_distributions(x, var, discrete=discrete)
        weights = var ** -0.5
        # weights /= np.mean(weights)

    if normalize:
        normalize_by = np.sum((weights * p) ** 2)
    else:
        normalize_by = 1

    return (weights * (p - q)) ** 2 / normalize_by


def rss(p, q, weights=None, sigma_uniform=None, normalize=False):
    resid_sq = rss_array(p, q, weights, sigma_uniform, normalize)
    return np.sum(resid_sq)


# ========================
# Classification functions
# ========================
def merge_singular_peaks(tau_cls, peak_tau, singularity_info):
    """
    Merge any peaks due to singularities with peaks identified in the non-singular distribution
    :param ndarray tau_cls: tau grid used for peak identification
    :param peak_tau: Locations of identified peaks
    :param singularity_info: List of 2-tuples indicating mass (R) and location (tau) of singularities in the
    distribution which are not captured by gamma. None indicates no singularities are present
    :return:
    """
    if singularity_info is not None:
        # Determine grid spacing
        dx = np.mean(np.abs(np.diff(np.log(tau_cls))))

        # Get singularity locations
        sing_tau = np.array([si[1] for si in singularity_info])

        # Check if each singularity is already captured in peak_tau
        # If it is, it should be within dx of the true location
        add_peak_index = peaks.find_new_peaks(np.log(sing_tau), np.log(peak_tau), dx)

        # Add singular peaks
        peak_tau = np.concatenate((peak_tau, sing_tau[add_peak_index]))

        # Sort by tau and return indices of singular peaks
        sort_index = np.argsort(peak_tau)
        sing_index = sort_index[-len(add_peak_index):]

        return peak_tau[sort_index], sing_index
    else:
        # No singularities
        return peak_tau, []


def get_similarity_function(function_type):
    if function_type == 'gaussian':
        def sim_func(x, order, epsilon):
            return np.exp(-(epsilon * np.abs(x)) ** (2 * order))
    elif function_type == 'inv_quad':
        def sim_func(x, order, epsilon):
            return 1 / (1 + (epsilon * np.abs(x)) ** (2 * order))
    elif function_type == 'pulse':
        def sim_func(x, order, epsilon):
            # order is ignored - included for compatibility only
            out = np.zeros_like(x)
            out[np.abs(x) <= epsilon ** -1] = 1
            return out
    else:
        raise ValueError(f'Invalid similarity function_type {function_type}')

    return sim_func


def match_peaks(est_peak_tau, true_peak_tau):
    est_ln_tau = np.log(est_peak_tau)
    true_ln_tau = np.log(true_peak_tau)

    # Proceed until all possible pairs have been created
    est_to_true_match_index = np.zeros(len(est_peak_tau), dtype=int) - 1
    true_to_est_match_index = np.zeros(len(true_ln_tau), dtype=int) - 1
    # match_index = 0
    while min(len(est_ln_tau), len(true_ln_tau)) > 0:
        # print(est_ln_tau, true_ln_tau)
        index_est_to_true = peaks.index_closest_peaks(est_ln_tau, true_ln_tau)
        index_true_to_est = peaks.index_closest_peaks(true_ln_tau, est_ln_tau)
        paired_est_index = []
        paired_true_index = []
        for est_index, true_index in enumerate(index_est_to_true):
            # Match peaks IFF they are mutual nearest neighbors
            # (est_peak is closest estimated peak to true_peak AND true_peak is closest true peak to est_peak)
            if index_true_to_est[true_index] == est_index:
                # print(est_index, true_index)
                orig_est_index = np.nonzero(np.log(est_peak_tau) == est_ln_tau[est_index])
                orig_true_index = np.nonzero(np.log(true_peak_tau) == true_ln_tau[true_index])
                # print(orig_est_index, orig_true_index)
                est_to_true_match_index[orig_est_index] = orig_true_index
                true_to_est_match_index[orig_true_index] = orig_est_index

                paired_est_index.append(est_index)
                paired_true_index.append(true_index)

        # print(paired_est_index, paired_true_index)

        # Remove paired peaks from remaining peak arrays
        est_ln_tau = np.delete(est_ln_tau, paired_est_index)
        true_ln_tau = np.delete(true_ln_tau, paired_true_index)

    return est_to_true_match_index, true_to_est_match_index


def p2p_distance(tau_a, tau_b, a2b_index):
    """
    For each peak in tau_a, get the peak-to-peak distance to its matched peak in tau_b.
    Distance is returned in log-tau space
    :param ndarray tau_a: tau values for set a
    :param ndarray tau_b: tau values for set b
    :param ndarray a2b_index: Indices to match peaks in tau_a to tau_b. Must contain one index for each peak in tau_a
    corresponding to the matching peak in tau_b. Unmatched peaks in tau_a should be assigned to index -1
    :return:
    """
    if len(tau_a) == 0:
        # No peaks in a --> no distances
        dist = np.array([])
    elif len(tau_b) == 0:
        # No peaks in b --> infinite distance for all peaks in a
        dist = np.ones(len(tau_a)) * np.inf
    else:
        if len(tau_a) != len(a2b_index):
            raise ValueError('a2b_index must have same length as tau_a')
        # Get tau for peaks in set b matched to set a
        tau_match = tau_b[a2b_index]

        # Get distance in ln(tau) space
        dist = np.abs(np.log(tau_a / tau_match))

        # Set infinite distance for unmatched peaks
        dist[a2b_index == -1] = np.inf

    return dist


def peak_similarity(est_peak_tau, true_peak_tau, sim_function_type, order, epsilon):
    # Match estimated and true peaks
    est_to_true_index, true_to_est_index = match_peaks(est_peak_tau, true_peak_tau)

    # Get activation function for peak character calculation
    sim_func = get_similarity_function(sim_function_type)

    # Get peak-to-peak distances
    est_to_true_dist = p2p_distance(est_peak_tau, true_peak_tau, est_to_true_index)
    true_to_est_dist = p2p_distance(true_peak_tau, est_peak_tau, true_to_est_index)

    return sim_func(est_to_true_dist, order, epsilon), sim_func(true_to_est_dist, order, epsilon)

# def similarity_index(tau_1, tau_2, activation_function):
#     act_func = get_activation_function(activation_function)


# def true_positives(est_peak_tau, true_peak_tau, )
