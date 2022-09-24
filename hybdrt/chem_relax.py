import numpy as np
from scipy.optimize import minimize_scalar

from .models.drt1d import DRT
from .preprocessing import get_step_info, generate_model_signal


def identify_response_step(times, response_signal, offset_step_time):
    # Estimate step time from observed response
    # Start with large rthresh for step identification. Decrease as necessary to identify step
    rthresh = 20
    while True:
        try:
            step_times, step_sizes = get_step_info(times, response_signal, allow_consecutive=False,
                                                   offset_step_times=offset_step_time, rthresh=rthresh)
            break
        except IndexError:
            rthresh /= 2

    # Only keep first step
    step_time = step_times[0]
    step_size = step_sizes[0]

    return step_time, step_size


def get_downsample_kwargs(times, step_time, max_num_samples, prestep_samples=5, spacing='log'):
    """
    Get index of samples to use for fit given limited number of samples
    :param times:
    :param step_times:
    :param max_num_samples:
    :param int prestep_samples: Number of samples prior to first step to include. Default: 5
    :param str spacing: 'log' or 'linear'
    :return:
    """
    # Since we are more concerned with the end value and less concerned with short-timescale relaxations,
    # use linearly spaced samples rather than logarithmically spaced samples.
    poststep_samples = max_num_samples - prestep_samples

    if spacing == 'log':
        min_tdelta = times[times > step_time][0] - step_time
        max_tdelta = times[times > step_time][-1] - step_time
        ideal_times = np.logspace(np.log10(min_tdelta), np.log10(max_tdelta), poststep_samples - 1)
        ideal_times = np.concatenate([[0], ideal_times])  # include 0
    elif spacing == 'linear':
        min_tdelta = times[times >= step_time][0] - step_time
        max_tdelta = times[times >= step_time][-1] - step_time
        ideal_times = np.linspace(min_tdelta, max_tdelta, poststep_samples)
    else:
        raise ValueError(f"Invalid spacing option {spacing}. Options are 'log', 'linear'")

    return {'prestep_samples': prestep_samples, 'ideal_times': ideal_times}


def optimize_step_time(times, response_signal, downsample=True, max_num_samples=50, ridge_kw={}, **kw):
    # Estimate step time from response
    rs_time, rs_size = identify_response_step(times, response_signal, True)

    drt = DRT(tau_basis_type='Cole-Cole', tau_epsilon=0.995, time_precision=3)

    def cost(step_time):

        # Set basis_tau based on step_time and times
        drt.basis_tau = drt.get_tau_from_times(times, [step_time], ppd=50)

        # Create dummy input signal
        # Set the dummy step size with same sign as response step
        # This ensures that the relaxation can be fitted with a non-negative DRT
        dummy_step_sizes = [1.0 * np.sign(rs_size)]
        dummy_input_signal = generate_model_signal(times, [step_time], dummy_step_sizes, None, 'ideal')

        # Generate kwargs to limit number of data points used in fit
        if downsample:
            ds_kwargs = get_downsample_kwargs(times, step_time, max_num_samples)
        else:
            ds_kwargs = {}

        # Fit DRT
        ridge_defaults = dict(hyper_l2_lambda=True, nonneg=True, l2_lambda_0=1e-20, hl_l2_beta=2.5)
        ridge_defaults.update(ridge_kw)
        drt.ridge_fit(times, dummy_input_signal, response_signal, step_times=[step_time],
                      downsample=downsample, downsample_kw=ds_kwargs,
                      **ridge_defaults)

        response_pred = drt.predict_response()

        # Get residuals
        resid = response_pred - drt.raw_response_signal

        print(step_time, np.sum(resid**2))

        return np.sum(resid ** 2)

    # result = least_squares(resid, [rs_time], **kw)
    result = minimize_scalar(cost, **kw)

    return result





