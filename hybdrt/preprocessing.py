import numpy as np
import warnings
from scipy.optimize import least_squares
from scipy import ndimage

from .utils import stats
from .utils.array import unit_step, nearest_index
from .utils.validation import check_step_model, check_ctrl_mode
from .utils.chrono import get_input_and_response
from .filters import nonuniform_gaussian_filter1d, masked_filter


# Chrono data preprocessing
# -------------------------
def identify_steps(y, allow_consecutive=True, rthresh=50, athresh=1e-10):
    """
    Identify steps in signal
    :param ndarray y: signal
    :param bool allow_consecutive: if False, do not allow consecutive steps
    :param float rthresh: relative threshold for identifying steps
    :param float athresh: absolute threshold for step size
    :return: step indices
    """
    dy = np.diff(y)
    # Identify indices where diff exceeds threshold
    # athresh defaults to 1e-10 in case median diff is zero
    step_idx = np.where((np.abs(dy) >= np.median(np.abs(dy)) * rthresh) & (np.abs(dy) >= athresh))[0] + 1

    if not allow_consecutive:
        # eliminate consecutive steps - these arise due to finite rise time and do not represent distinct steps
        idx_diff = np.diff(step_idx)
        idx_diff = np.concatenate(([2], idx_diff))
        step_idx = step_idx[idx_diff > 1]

    return step_idx


def split_steps(x, step_index):
    """
    Split x by step indices
    :param ndarray x: array to split
    :param ndarray step_index: step indices
    :return:
    """
    step_index = np.array(step_index)
    # Add start and end indices
    if step_index[0] > 0:
        step_index = np.insert(step_index, 0, 0)
    if step_index[-1] < len(x):
        step_index = np.append(step_index, len(x))

    return [x[start:end] for start, end in zip(step_index[:-1], step_index[1:])]


def get_step_info(times, y, allow_consecutive=True, offset_step_times=False, rthresh=50, athresh=1e-10):
    """
    Get step times and sizes from signal
    :param ndarray times: measurement times
    :param ndarray y: signal
    :return: array of step times, array of step magnitudes
    """
    step_idx = identify_steps(y, allow_consecutive, rthresh, athresh)

    step_times = times[step_idx]

    if offset_step_times:
        # Get minimum sample period
        t_sample = np.min(np.diff(times))
        # print('t_sample:', t_sample)
        # Assume actual step time occurred 1 sample period before observed
        # Multiply by 0.999 to ensure that step time isn't exactly equal to previous sample time,
        # as this causes trouble with R_inf response (R_inf response at times >= step_time)
        step_times -= t_sample * (1 - 1e-8)
        # step_times = times[step_idx - 1] + 1e-8

    # # Get step size for each step
    # for n in range(n_steps):
    #     # If last step, stop at end of data
    #     if n == n_steps - 1:
    #         end = len(y)
    #     else:
    #         end = step_idx[n+1]
    #
    #     # If first step, start at
    #     if n == 0:
    #         prev_start = 0
    #     else:
    #         prev_start = step_idx[n-1]
    #     # print('step stuff:', n, step_idx[n], prev_start, end)
    #     # print('step values:', y[step_idx[n]:end])
    #     # print('prev values:', y[prev_start:step_idx[n]])
    #     step_sizes[n] = np.mean(y[step_idx[n]:end]) - np.mean(y[prev_start:step_idx[n]])

    step_sizes = get_step_sizes(times, y, step_times)

    return step_times, step_sizes


def get_step_sizes(times, y, step_times):
    # Get step indices from step times
    step_index = get_step_indices_from_step_times(times, step_times)
    n_steps = len(step_times)

    step_sizes = np.zeros(n_steps)
    # Get step size for each step
    for n in range(n_steps):
        # If last step, stop at end of data
        if n == n_steps - 1:
            end = len(y)
        else:
            end = step_index[n + 1]

        # If first step, start at
        if n == 0:
            prev_start = 0
        else:
            prev_start = step_index[n - 1]

        step_sizes[n] = np.mean(y[step_index[n]:end]) - np.mean(y[prev_start:step_index[n]])

    return step_sizes


def process_input_signal(times, input_signal, step_model, offset_steps, rthresh=50, fixed_tau_rise=None):
    check_step_model(step_model)
    if step_model == 'ideal':
        # If using ideal step model, model each real step as a series of ideal steps
        allow_consecutive_steps = True
    else:
        # If using non-ideal step model, apply step model to each real step
        allow_consecutive_steps = False
    step_times, step_sizes = get_step_info(times, input_signal, allow_consecutive_steps, offset_steps, rthresh)

    # If using non-ideal step model, fit input signal to step model
    if step_model != 'ideal':
        num_steps = len(step_times)
        # Fit step model
        signal_fit = fit_signal_steps(times, input_signal, fixed_tau_rise=fixed_tau_rise)
        # Get step time offsets from fit
        t_step_offset = signal_fit['x'][1: num_steps + 1] * 1e-6
        step_times += t_step_offset
        # Get rise times from fit
        tau_rise = np.exp(signal_fit['x'][num_steps + 1:])
        print('Fitted {} step model'.format(step_model))
        print('t_step_offset:', t_step_offset)
        print('tau_rise:', tau_rise)
    else:
        tau_rise = None

    return step_times, step_sizes, tau_rise


def get_step_indices_from_step_times(times, step_times):
    """
    Get array indices corresponding to step times
    :param times:
    :param step_times:
    :return:
    """
    # Determine step index by getting measurement time closest to step time
    # Each step_index must start at or after step time - cannot start before
    def pos_delta(x, x0):
        out = np.empty(len(x))
        out[x < x0] = np.inf
        out[x >= x0] = x[x >= x0] - x0
        return out

    step_index = np.array([np.argmin(pos_delta(times, st)) for st in step_times])

    return step_index


def generate_model_signal(times, step_times, step_sizes, tau_rise, step_model):
    """
    Generate model input signal
    :param times:
    :param step_times:
    :param step_sizes:
    :param tau_rise:
    :param step_model:
    :return:
    """
    signal = np.zeros(len(times))
    if step_model == 'ideal':
        # Ideal step model
        for st, sa in zip(step_times, step_sizes):
            signal += sa * unit_step(times, st)
    elif step_model == 'expdecay':
        pass
        # Exponential decay model
        # Construct coefficient vector
        num_steps = len(step_times)
        x = np.zeros(1 + 2 * num_steps)
        x[1:num_steps + 1] = 0  # step offsets are already incorporated into step_times - pass zeros
        x[num_steps + 1:] = np.log(tau_rise)  # remaining coefficients are log of rise times
        # Evaluate model
        signal = evaluate_step_fit(times, step_times, step_sizes, x)

    return signal


def downsample_data(times, i_signal, v_signal, target_times=None, target_size=None, stepwise_sample_times=True,
                    step_times=None, step_model=None, method='match',
                    decimation_interval=10, decimation_factor=2, decimation_max_period=None,
                    antialiased=True, filter_kw=None,
                    op_mode='galv', prestep_samples=20):
    """
    Downsample data to match desired sample times
    :param stepwise_sample_times:
    :param times: measurement times
    :param i_signal: current signal
    :param v_signal: voltage signal
    :param target_times: ideal (desired) sample times after each step
    :param step_times: list or array of step times. If not provided, determine from input signal
    :param step_model: Which step model to apply to the input signal. Ignored if step_times is provided
    :param op_mode: Operation mode. Options: galvanostatic, potentiostatic
    :param prestep_samples: number of samples to keep from pre-step period
    :return: sample_times, sample_i, sample_v
    """

    if stepwise_sample_times:
        # Sample ideal_times from start of each step
        check_ctrl_mode(op_mode)
        # Identify step times
        if step_times is None:
            check_step_model(step_model)
            if step_model == 'ideal':
                # If using ideal step model, model each real step as a series of ideal steps
                allow_consecutive_steps = True
            else:
                # If using non-ideal step model, apply step model to each real step
                allow_consecutive_steps = False

            if op_mode == 'galv':
                step_indices = identify_steps(i_signal, allow_consecutive_steps)
            else:
                step_indices = identify_steps(v_signal, allow_consecutive_steps)

            step_times = times[step_indices]
            # print('Step times:', step_times)
        else:
            step_indices = get_step_indices_from_step_times(times, step_times)
    else:
        # Treat as a single step
        step_times = [0]
        step_indices = [0]

    if method == 'match':
        # Determine matching post-step sample times
        if target_times is not None:
            # Apply target_times to each step
            target_times = np.unique(np.concatenate([target_times + ts for ts in step_times]))

            # Get sample times closest to ideal times
            sample_index = np.array([[nearest_index(times, target_times[i])] for i in range(len(target_times))])
            sample_index = np.unique(sample_index)
        else:
            # If target_times is None, keep all post-step samples
            sample_index = np.arange(step_indices[0], len(times), dtype=int)

        # Keep some samples immediately prior to first step
        if step_indices[0] > 0 and prestep_samples > 0:
            num_prestep = step_indices[0]
            prestep_samples = min(prestep_samples, num_prestep)
            # prestep_index = np.arange(0, num_prestep, num_prestep // prestep_samples, dtype=int)  # uniformly spaced
            prestep_index = np.arange(step_indices[0] - prestep_samples, step_indices[0], dtype=int)
            sample_index = np.unique(np.concatenate((prestep_index, sample_index)))
    elif method == 'decimate':
        t_sample = np.min(np.diff(times))
        if target_size is not None:
            decimation_interval = -1
            while decimation_interval == -1:
                decimation_interval = select_decimation_interval(times, step_times, t_sample, prestep_samples,
                                                                 decimation_factor, decimation_max_period,
                                                                 target_size)
        # print(decimation_interval)
        sample_index = get_decimation_index(times, step_times, t_sample, prestep_samples, decimation_interval,
                                            decimation_factor, decimation_max_period)
    else:
        raise ValueError(f"Invalid downsample method {method}. Options: 'match', 'decimate'")
    # else:
    #     # Get sample times closest to ideal times
    #     sample_index = np.array([[nearest_index(times, target_times[i])] for i in range(len(target_times))])
    #     sample_index = np.unique(sample_index)

    if antialiased and stepwise_sample_times:
        # Apply an antialiasing filter before downsampling
        if filter_kw is None:
            filter_kw = {}
        input_signal, _ = get_input_and_response(i_signal, v_signal, op_mode)
        step_index = identify_steps(input_signal, allow_consecutive=False)
        i_signal = filter_chrono_signal(times, i_signal, step_index=step_index,
                                        decimate_index=sample_index, **filter_kw)
        v_signal = filter_chrono_signal(times, v_signal, step_index=step_index,
                                        decimate_index=sample_index, **filter_kw)

    sample_times = times[sample_index].flatten()
    sample_i = i_signal[sample_index].flatten()
    sample_v = v_signal[sample_index].flatten()

    return sample_times, sample_i, sample_v, sample_index


def filter_chrono_signal(times, y, step_index=None, input_signal=None, decimate_index=None,
                         sigma_factor=0.01, max_sigma=None,
                         remove_outliers=False, outlier_kw=None, median_prefilter=False, **kw):
    if step_index is None and input_signal is None:
        raise ValueError('Either step_index or input_signal must be provided')

    if step_index is None:
        step_index = identify_steps(input_signal, allow_consecutive=False)

    if remove_outliers:
        y = y.copy()

        # First, remove obvious extreme values
        # ext_index = identify_extreme_values(y, qr_size=0.8)
        # print('extreme value indices:', np.where(ext_index))
        # y[ext_index] = ndimage.median_filter(y, size=31)[ext_index]

        # Find outliers with difference from filtered signal
        # Use median prefilter to avoid spread of outliers
        y_filt = filter_chrono_signal(times, y, step_index=step_index,
                                      sigma_factor=sigma_factor, max_sigma=max_sigma,
                                      remove_outliers=False,
                                      empty=False, median_prefilter=True, **kw)
        if outlier_kw is None:
            outlier_kw = {}

        outlier_flag = flag_chrono_outliers(y, y_filt, **outlier_kw)

        print('outlier indices:', np.where(outlier_flag))

        # Set outliers to filtered value
        y[outlier_flag] = y_filt[outlier_flag]

    y_steps = split_steps(y, step_index)
    t_steps = split_steps(times, step_index)
    t_sample = np.median(np.diff(times))

    if max_sigma is None:
        max_sigma = sigma_factor / t_sample

    # Get sigmas corresponding to decimation index
    if decimate_index is not None:
        decimate_sigma = sigma_from_decimate_index(y, decimate_index)
        step_dec_sigmas = split_steps(decimate_sigma, step_index)
    else:
        step_dec_sigmas = None

    y_filt = []
    for i, (t_step, y_step) in enumerate(zip(t_steps, y_steps)):
        # Ideal sigma from inverse sqrt of maximum curvature of RC relaxation
        sigma_ideal = np.exp(1) * (t_step - (t_step[0] - t_sample)) / 2
        sigmas = sigma_factor * (sigma_ideal / t_sample)
        sigmas[sigmas > max_sigma] = max_sigma

        # Use decimation index to cap sigma
        if step_dec_sigmas is not None:
            sigmas = np.minimum(step_dec_sigmas[i], sigmas)

        if median_prefilter:
            y_in = ndimage.median_filter(y_step, 3, mode='nearest')
        else:
            y_in = y_step

        yf = nonuniform_gaussian_filter1d(y_in, sigmas, **kw)

        y_filt.append(yf)

    return np.concatenate(y_filt)


def sigma_from_decimate_index(y, decimate_index, truncate=4.0):
    sigmas = np.zeros(len(y)) #+ 0.25

    # Determine distance to nearest sample
    diff = np.diff(decimate_index)
    ldiff = np.insert(diff, 0, diff[0])
    rdiff = np.append(diff, diff[-1])
    min_diff = np.minimum(ldiff, rdiff)

    # Set sigma such that truncate * sigma reaches halfway to nearest sample
    sigma_dec = min_diff / (2 * truncate)
    sigma_dec[min_diff < 2] = 0  # Don't filter undecimated regions
    sigmas[decimate_index] = sigma_dec

    return sigmas


def flag_chrono_outliers(y_raw, y_filt, thresh=0.75, p_prior=0.01):
    dev = y_filt - y_raw
    std = stats.robust_std(dev)
    sigma_out = np.maximum(np.abs(dev), 0.01 * std)
    p_out = outlier_prob(dev, 0, std, sigma_out, p_prior)

    return p_out > thresh


def select_decimation_interval(times, step_times, t_sample, prestep_points, decimation_factor, max_t_sample,
                               target_size):
    intervals = np.logspace(np.log10(2), np.log10(1000), 12).astype(int)
    sizes = [len(get_decimation_index(times, step_times, t_sample, prestep_points,
                                      interval, decimation_factor, max_t_sample)
                 )
             for interval in intervals]
    if target_size > sizes[-1]:
        warnings.warn(f'Cannot achieve target size of {target_size} with selected decimation factor of '
                      f'{decimation_factor}. Decrease the decimation factor and/or decrease the maximum period')
    if target_size < sizes[0]:
        warnings.warn(f'Cannot achieve target size of {target_size} with selected decimation factor of '
                      f'{decimation_factor}. Increase the decimation factor and/or increase the maximum period'
                      )
    return int(np.interp(target_size, sizes, intervals))


def get_decimation_index(times, step_times, t_sample, prestep_points, decimation_interval, decimation_factor,
                         max_t_sample):
    # Get evenly spaced samples from pre-step period
    prestep_times = times[times < np.min(step_times)]
    prestep_index = np.linspace(0, len(prestep_times) - 1, prestep_points).round(0).astype(int)

    # Determine index of first sample time after each step
    def pos_delta(x, x0):
        out = np.empty(len(x))
        out[x < x0] = np.inf
        out[x >= x0] = x[x >= x0] - x0
        return out

    step_index = [np.argmin(pos_delta(times, st)) for st in step_times]

    # Limit sample interval to max_t_sample
    if max_t_sample is None:
        max_sample_interval = np.inf
    else:
        max_sample_interval = int(max_t_sample / t_sample)

    # Build array of indices to keep
    keep_indices = [prestep_index]
    for i, start_index in enumerate(step_index):
        # Decimate samples after each step
        if start_index == step_index[-1]:
            next_step_index = len(times)
        else:
            next_step_index = step_index[i + 1]

        # Keep first decimation_interval points without decimation
        undec_index = np.arange(start_index, min(start_index + decimation_interval + 1, next_step_index), dtype=int)

        keep_indices.append(undec_index)
        # sample_interval = 1
        last_index = undec_index[-1]
        j = 1
        while last_index < next_step_index - 1:
            # Increment sample_interval
            # sample_interval = min(int(sample_interval * decimation_factor), max_sample_interval)
            sample_interval = min(int(decimation_factor ** j), max_sample_interval)
            # print('sample_interval:', sample_interval)

            if sample_interval == max_sample_interval:
                # Sample interval has reached maximum. Continue through end of step
                interval_end_index = next_step_index #- 1
            else:
                # Continue with current sampling rate until decimation_interval points acquired
                interval_end_index = min(last_index + decimation_interval * sample_interval + 1,
                                         next_step_index)

            keep_index = np.arange(last_index + sample_interval, interval_end_index, sample_interval, dtype=int)

            if len(keep_index) == 0:
                # sample_interval too large - runs past end of step. Keep last sample
                keep_index = [interval_end_index - 1]

            # If this is the final interval, ensure that last point before next step is included
            if interval_end_index == next_step_index and keep_index[-1] < next_step_index - 1:
                keep_index = np.append(keep_index, next_step_index - 1)

            keep_indices.append(keep_index)

            # Increment last_index
            last_index = keep_index[-1]
            j += 1

    decimate_index = np.unique(np.concatenate(keep_indices))

    return decimate_index


def get_signal_scales(times, step_times, input_step_sizes, response_signal, step_model):
    """Should be OBSOLETE"""
    # Scale input signal such that mean step size is 1
    if step_model == 'ideal':
        # Check for consecutive steps - only applies to ideal step model
        # Align indices with step_times
        new_step_index = np.concatenate(([0], np.where(np.diff(step_times) > 2e-5)[0] + 1))

        if len(new_step_index) < len(step_times):
            # If consecutive steps exist, condense into single step to get accurate step sizes
            step_times = np.array([step_times[i] for i in new_step_index])
            step_sizes_new = np.zeros_like(step_times)
            for i, start_index in enumerate(new_step_index):
                if i == len(new_step_index) - 1:
                    end_index = len(input_step_sizes)
                else:
                    end_index = new_step_index[i + 1]
                step_sizes_new[i] = np.sum(input_step_sizes[start_index:end_index])
            input_step_sizes = step_sizes_new
            # print('Scaling step times:', step_times)
        # print('step sizes:', input_step_sizes)
    input_signal_scale = np.mean(np.abs(input_step_sizes))

    # Scale response signal such that mean range within each step is 1
    # Each step_index must start at or after step time
    def pos_delta(x, x0):
        out = np.empty(len(x))
        out[x < x0] = np.inf
        out[x >= x0] = x[x >= x0] - x0
        return out

    step_index = [np.argmin(pos_delta(times, st)) for st in step_times]
    response_step_ranges = np.zeros(len(step_index))
    for i, start_index in enumerate(step_index):
        if i == len(step_index) - 1:
            end_index = len(times)
        else:
            end_index = step_index[i + 1]
        # print(start_index, times[start_index], times[start_index + 1])
        step_response = response_signal[start_index:end_index]
        response_step_ranges[i] = np.max(step_response) - np.min(step_response)
    print('response step ranges:', response_step_ranges)
    response_signal_scale = np.mean(np.abs(response_step_ranges))

    return input_signal_scale, response_signal_scale


def get_input_signal_scale(times, step_times, input_step_sizes, step_model):
    # Scale input signal such that mean step size is 1
    if step_model == 'ideal':
        # Check for consecutive steps - only applies to ideal step model
        # Align indices with step_times
        new_step_index = np.concatenate(([0], np.where(np.diff(step_times) > 2e-5)[0] + 1))

        if len(new_step_index) < len(step_times):
            # If consecutive steps exist, condense into single step to get accurate step sizes
            step_times = np.array([step_times[i] for i in new_step_index])
            step_sizes_new = np.zeros_like(step_times)
            for i, start_index in enumerate(new_step_index):
                if i == len(new_step_index) - 1:
                    end_index = len(input_step_sizes)
                else:
                    end_index = new_step_index[i + 1]
                step_sizes_new[i] = np.sum(input_step_sizes[start_index:end_index])
            input_step_sizes = step_sizes_new
            # print('Scaling step times:', step_times)
        # print('step sizes:', input_step_sizes)
    input_signal_scale = np.mean(np.abs(input_step_sizes))

    return input_signal_scale


def estimate_rp(times, step_times, input_step_sizes, response_signal, step_model, z):
    if times is not None:
        # i_cum = np.cumsum(input_step_sizes)
        # i_range = np.max(i_cum) - np.min(i_cum)
        # v_range = np.percentile(response_signal, 99) - np.percentile(response_signal, 1)
        # Estimate R_min and R_max from chrono data
        if step_model == 'ideal':
            # Check for consecutive steps - only applies to ideal step model
            # Align indices with step_times
            new_step_index = np.concatenate(([0], np.where(np.diff(step_times) > 2e-5)[0] + 1))

            if len(new_step_index) < len(step_times):
                # If consecutive steps exist, condense into single step to get accurate step sizes
                step_times = np.array([step_times[i] for i in new_step_index])
                step_sizes_new = np.zeros_like(step_times)
                for i, start_index in enumerate(new_step_index):
                    if i == len(new_step_index) - 1:
                        end_index = len(input_step_sizes)
                    else:
                        end_index = new_step_index[i + 1]
                    step_sizes_new[i] = np.sum(input_step_sizes[start_index:end_index])
                input_step_sizes = step_sizes_new

        # Identify response data corresponding to each step
        # Each step_index must start at or after step time
        def pos_delta(x, x0):
            out = np.empty(len(x))
            out[x < x0] = np.inf
            out[x >= x0] = x[x >= x0] - x0
            return out

        # Get apparent Rp for each step
        step_index = [np.argmin(pos_delta(times, st)) for st in step_times]
        step_r_min = np.zeros(len(step_index))
        step_r_max = np.zeros(len(step_index))
        for i, start_index in enumerate(step_index):
            if i == len(step_index) - 1:
                end_index = len(times)
            else:
                end_index = step_index[i + 1]
            # Get response value prior to step
            pre_step_val = response_signal[start_index - 1]

            # Identify response data corresponding to each step
            step_response = response_signal[start_index:end_index]

            # Get min and max resistance observed in step
            step_r_min[i] = np.min((step_response - pre_step_val) / input_step_sizes[i])
            step_r_max[i] = np.max((step_response - pre_step_val) / input_step_sizes[i])

        r_min_chrono = np.mean(step_r_min)
        r_max_chrono = np.percentile(step_r_max, 99)
    else:
        # Set limits such that they will not influence aggregate
        r_min_chrono = np.inf
        r_max_chrono = 0

    if z is not None:
        # Estimate R_min and R_max from EIS data
        r_min_eis = np.min(z.real)
        r_max_eis = np.max(z.real)
    else:
        # Set limits such that they will not influence aggregate
        r_min_eis = np.inf
        r_max_eis = 0

    # Get min and max resistance observed across datasets
    r_min = min(r_min_chrono, r_min_eis)
    r_max = max(r_max_chrono, r_max_eis)

    return r_max - r_min


def get_quantile_limits(y, qr_size=0.5, qr_thresh=1.5):
    q_lo = np.percentile(y, 50 - 100 * qr_size / 2)
    q_hi = np.percentile(y, 50 + 100 * qr_size / 2)
    qr = q_hi - q_lo
    y_min = q_lo - qr * qr_thresh
    y_max = q_hi + qr * qr_thresh

    return y_min, y_max


def identify_extreme_values(y, qr_size=0.5, qr_thresh=1.5):
    y_min, y_max = get_quantile_limits(y, qr_size, qr_thresh)

    return (y < y_min) | (y > y_max)


def outlier_prob(x, mu_in, sigma_in, sigma_out, p_prior):
    """
    Estimate outlier probability using a Bernoulli prior
    :param ndarray x: data
    :param ndarray mu_in: mean of inlier distribution
    :param ndarray sigma_in: standard deviation of inlier distribution
    :param ndarray sigma_out: standard deviation of outlier distribution
    :param float p_prior: prior probability of any point being an outlier
    :return:
    """
    pdf_in = stats.pdf_normal(x, mu_in, sigma_in)
    pdf_out = stats.pdf_normal(x, mu_in, sigma_out)
    p_out = p_prior * pdf_out / ((1 - p_prior) * pdf_in + p_prior * pdf_out)
    dev = np.abs(x - mu_in)
    # Don't consider data points with smaller deviations than sigma_in to be outliers
    p_out[dev <= sigma_in] = 0
    return p_out


# =======================
# Data limits and spacing
# =======================
def get_ppd(x):
    lf_max = np.log10(np.max(x))
    lf_min = np.log10(np.min(x))
    num_decades = lf_max - lf_min

    return (len(x) - 1) / num_decades


def get_time_ppd(times, step_times, aggregate=True):
    time_deltas = []
    # Get min sample period
    t_sample = np.min(np.diff(times))

    # For each step, get time delta from step time
    for i, start_time in enumerate(step_times):
        if i == len(step_times) - 1:
            end_time = np.inf
        else:
            end_time = step_times[i + 1]

        step_index = np.where((times >= start_time) & (times < end_time))
        if len(step_index[0]) > 0:
            # Don't allow time_delta smaller than t_sample - this will inflate number of decades
            time_deltas.append(np.maximum(times[step_index] - start_time, t_sample))

    if aggregate:
        # Get points per decade across all steps
        # Concatenate time deltas for all steps
        time_deltas = np.concatenate(time_deltas)
        return get_ppd(time_deltas)
    else:
        # Get ppd for each step
        ppds = [get_ppd(td) for td in time_deltas]
        return ppds


def get_time_since_step(times, step_times, prestep_value=None):
    """
    Convert elapsed times to time delta since last step
    :param times:
    :param step_times:
    :return:
    """
    time_deltas = []
    # Get min sample period
    if len(times) > 1:
        t_sample = np.min(np.diff(times))
    else:
        t_sample = times[0]

    # Populate specified value for all times prior to first step
    if prestep_value is not None:
        time_deltas.append(np.tile(prestep_value, len(times[times < step_times[0]])))

    # For each step, get time delta from step time
    for i, start_time in enumerate(step_times):
        if i == len(step_times) - 1:
            end_time = np.inf
        else:
            end_time = step_times[i + 1]

        step_index = np.where((times >= start_time) & (times < end_time))
        if len(step_index[0]) > 0:
            # Don't allow time_delta smaller than t_sample - this will inflate number of decades
            time_deltas.append(np.maximum(times[step_index] - start_time, t_sample))

    time_deltas = np.concatenate(time_deltas)

    return time_deltas


def get_tau_lim(frequencies, times, step_times):
    if frequencies is not None:
        eis_tau_min = 1 / (2 * np.pi * np.max(frequencies))
        eis_tau_max = 1 / (2 * np.pi * np.min(frequencies))
    else:
        eis_tau_min = np.inf
        eis_tau_max = -np.inf

    if times is not None:
        time_deltas = get_time_since_step(times, step_times)
        chrono_tau_min = np.min(time_deltas)
        chrono_tau_max = np.max(time_deltas)
    else:
        chrono_tau_min = np.inf
        chrono_tau_max = -np.inf

    tau_min = min(eis_tau_min, chrono_tau_min)
    tau_max = max(eis_tau_max, chrono_tau_max)

    return tau_min, tau_max


def get_num_decades(frequencies, times, step_times):
    tau_min, tau_max = get_tau_lim(frequencies, times, step_times)
    num_decades = np.log10(tau_max) - np.log10(tau_min)

    return num_decades


def get_basis_tau(frequencies, times, step_times, ppd=10, extend_decades=1, tau_grid=None):
    # Get tau limits of measurement
    tau_min, tau_max = get_tau_lim(frequencies, times, step_times)

    # Get min and max log tau for basis
    log_tau_min = np.log10(tau_min) - extend_decades
    log_tau_max = np.log10(tau_max) + extend_decades

    if tau_grid is not None:
        # If tau_grid provided, select range from grid
        if 10 ** log_tau_min < np.min(tau_grid):
            left_index = 0
        else:
            left_index = nearest_index(tau_grid, 10 ** log_tau_min, constraint=-1)

        if 10 ** log_tau_max > np.max(tau_grid):
            right_index = len(tau_grid)
        else:
            right_index = nearest_index(tau_grid, 10 ** log_tau_max, constraint=1) + 1

        return tau_grid[left_index:right_index]
    else:
        # Determine number of basis points based on spacing
        num_points_exact = (log_tau_max - log_tau_min) * ppd + 1
        num_points = int(np.ceil(num_points_exact))

        # Extend the basis range to ensure exact spacing
        add_decades = 0.5 * (num_points - num_points_exact) / ppd
        log_tau_min -= add_decades
        log_tau_max += add_decades

        return np.logspace(log_tau_min, log_tau_max, num_points)


def get_epsilon_from_ppd(ppd, factor=1):
    return factor / np.log(10 ** (1 / ppd))


# Functions for fitting non-ideal current/voltage steps
# -----------------------------------------------------
def evaluate_step_fit(times, step_times, step_sizes, x):
    """
    Evaluate decaying exponential model for non-ideal step
    :param ndarray times: times at which to evaulate model
    :param ndarray step_times: times at which steps occur
    :param ndarray step_sizes: sizes of steps
    :param x: fit parameters from fit_control_steps
    :return: ndarray of fitted signal values
    """
    num_steps = len(step_times)
    signal_offset = x[0]
    t_step_offset = x[1: num_steps + 1] * 1e-6
    tau_rise = np.exp(x[num_steps + 1:])
    t_step = step_times + t_step_offset

    y_hat = np.zeros(len(times)) + signal_offset
    for n in range(num_steps):
        y_hat[times >= t_step[n]] += step_sizes[n] * \
                                     (1 - np.exp(-(times[times >= t_step[n]] - t_step[n]) / tau_rise[n]))

    return y_hat


def fit_signal_steps(times, signal, tau_var_penalty=0.1, t_step_offset_penalty=1e-5, fixed_tau_rise=None):
    """
    Fit decaying exponential model to non-ideal control step(s).
    :param ndarray times: measurement times
    :param ndarray signal: measured signal
    :param float tau_var_penalty: penalty strength for variance in tau_rise between steps.
    Penalty is applied to variations in ln(tau_rise).
    :param float t_step_offset_penalty: penalty strength for step time offset from nominal step times.
    Penalty is applied to offsets in microseconds.
    :param float fixed_tau_rise: constant value to use for tau_rise. If provided, only the signal offset and step time
    offset will be calculated. If None, tau_rise will be optimized.
    :return: scipy.least_squares result
    """
    # Identify steps in signal
    step_indices = identify_steps(signal, allow_consecutive=False)
    step_times, step_sizes = get_step_info(times, signal, allow_consecutive=False)
    num_steps = len(step_indices)
    print('step indices:', step_indices)

    # Scale signal to unity step size (why am I doing this? seems unnecessary)
    # signal_scale = np.mean(np.abs(step_sizes))
    # scaled_step_sizes /= step_sizes / signal_scale
    # scaled_signal = signal / signal_scale

    # Define residual function for optimization
    def residuals(x, t, y):
        """
        Evaluate residuals
        :param ndarray x: parameter vector. The first num_steps values are step time offsets in microseconds,
        and the remaining num_steps values are rise time constants
        :param ndarray t: times to fit
        :param ndarray y: signal values to fit
        :return: ndarray of residuals
        """
        # Evaluate fit signal
        if fixed_tau_rise is not None:
            x = np.concatenate(x, np.ones(num_steps) * fixed_tau_rise)
        y_hat = evaluate_step_fit(t, step_times, step_sizes, x)
        # Extract parameters
        log_tau_rise = x[num_steps + 1:]
        tau_rise = np.exp(log_tau_rise)

        # Penalize signal misfit
        y_resid = y_hat - y
        # Penalize time step offsets
        t_offset_resid = x[1: num_steps + 1] * t_step_offset_penalty
        # Penalize variance in tau_rise
        tau_resid = (log_tau_rise - np.mean(log_tau_rise)) * tau_var_penalty

        return np.concatenate((y_resid, t_offset_resid, tau_resid))

    # Select times and values to fit
    fit_times = np.empty(num_steps * 25)
    fit_signal = np.empty(num_steps * 25)
    for i, step_index in enumerate(step_indices):
        start_index = step_index - 5
        end_index = step_index + 20
        print(start_index, end_index)
        fit_times[i * 25:(i + 1) * 25] = times[start_index:end_index]
        fit_signal[i * 25:(i + 1) * 25] = signal[start_index:end_index]

    print('fit times:', fit_times)
    print('fit signal:', fit_signal)

    # Optimize
    if fixed_tau_rise is not None:
        x0 = np.empty(1 + num_steps)
    else:
        x0 = np.empty(1 + num_steps * 2)
    x0[0] = np.mean(signal[times < step_times[0]])  # signal offset (initial value)
    x0[1:num_steps + 1] = 0  # t_step_offset initialized at zero
    if fixed_tau_rise is None:
        x0[num_steps + 1:] = np.log(1e-5)  # tau_rise initialized at 10 us
    print('x0:', x0)
    result = least_squares(residuals, x0=x0, args=(fit_times, fit_signal))

    # print('cost:', residuals(result['x'], fit_times, fit_signal))

    return result


# For time-dependent model
# =========================
def get_ocv_index(times, step_times, step_sizes, input_signal, samples_per_step=1, input_rthresh=0.05):
    step_index = get_step_indices_from_step_times(times, step_times)

    # Get index of sample(s) immediately before each step - should be nearly fully relaxed
    start_indices = step_index - samples_per_step
    end_indices = step_index

    # Get pre-step current
    input_prestep = [np.mean(input_signal[start_index:end_index])
                     for start_index, end_index in zip(start_indices, end_indices)]

    # Identify pre-step samples that are at OCV
    input_thresh = np.mean(np.abs(step_sizes)) * input_rthresh
    ocv_step_index = np.where(np.abs(input_prestep) < input_thresh)

    # Get index of samples that are representative of OCV state
    ocv_index = np.concatenate([np.arange(start_indices[i], end_indices[i], dtype=int)
                                for i in ocv_step_index[0]])

    return ocv_index
