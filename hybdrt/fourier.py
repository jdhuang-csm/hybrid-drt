import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from hybdrt.utils.array import is_uniform


# Function for extracting impedance from chrono data using Fourier transform
# Intended for comparison only
def fft_impedance(times, i_signal, v_signal, order=1, exclude_zero=True, window=None, window_kwargs=None,
                  smooth=False, smooth_sigma=1):
    # Check if times are uniformly spaced
    if is_uniform(times):
        # Get sampling interval
        t_sample = np.mean(np.diff(times))
    else:
        raise ValueError('times must be uniformly spaced for Fourier extraction of impedance')

    # Smooth prior to FFT
    if smooth:
        i_signal = gaussian_filter(i_signal, sigma=smooth_sigma)
        v_signal = gaussian_filter(v_signal, sigma=smooth_sigma)

    # Get numerical derivatives of i and v
    di_dt = np.diff(i_signal, n=order)
    dv_dt = np.diff(v_signal, n=order)

    # Apply window function
    if window is not None:
        try:
            if window_kwargs is None:
                window_kwargs = {}
            window = getattr(np, window)(len(di_dt), **window_kwargs)
            di_dt = di_dt * window
            dv_dt = dv_dt * window
        except AttributeError:
            raise ValueError(f'Invalid window {window}. window argument must be a numpy window function')

    # Transform i and v derivatives
    i_fft = fft.rfft(di_dt)
    v_fft = fft.rfft(dv_dt)

    # Calculate impedance
    z_fft = v_fft / i_fft

    # Get Fourier transform frequencies
    frequencies = fft.rfftfreq(len(di_dt), d=t_sample)

    # Exclude zero frequency
    if exclude_zero:
        frequencies = frequencies[1:]
        z_fft = z_fft[1:]

    return frequencies, z_fft

