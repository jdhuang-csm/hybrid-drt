import numpy as np
from scipy.stats import iqr
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
# import time

from .. import preprocessing as pp


def estimate_background(x_meas, y_meas, y_pred, gp=None, kernel_type='gaussian',
                        length_scale_bounds=(0.01, 10), periodicity_bounds=(1e-3, 1e3), noise_level_bounds=(0.1, 10),
                        kernel_size=1, n_restarts=1, kernel_scale_factor=1):
    # Get residuals
    y_err = y_meas - y_pred

    # Set up GP
    if gp is None:
        # Constraint noise level to within a factor of 10 of the apparent error variance
        kernel = WhiteKernel(noise_level=1, noise_level_bounds=noise_level_bounds)

        if kernel_type == 'gaussian':
            # Initialize each RBF at a different length scale
            length_scale_splits = np.logspace(np.log10(length_scale_bounds[0]), np.log10(length_scale_bounds[1]),
                                              kernel_size + 1)
            for i in range(kernel_size):
                # Initialize length scales at log-uniform intervals
                med_length_scale = (length_scale_splits[i] * length_scale_splits[i + 1]) ** 0.5
                kernel += 1.0 * RBF(length_scale_bounds=length_scale_bounds,
                                    length_scale=med_length_scale)
        elif kernel_type == 'periodic':
            kernel += 1.0 * ExpSineSquared(periodicity_bounds=periodicity_bounds)
        elif kernel_type == 'locper':
            kernel += 1.0 * RBF(length_scale_bounds=length_scale_bounds) \
                      * ExpSineSquared(periodicity_bounds=periodicity_bounds)
        else:
            raise ValueError(f"Invalid kernel_type {kernel_type}. Options: 'gaussian', 'periodic', 'locper'")

        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=n_restarts)
        # print(gp.kernel)

    # Format x for GP fit
    x_mat = x_meas[:, None]

    # Fit GP to residuals
    gp.fit(x_mat, y_err)

    if kernel_scale_factor != 1:
        kp = gp.kernel_.get_params()
        kernel_names = [k for k in kp.keys() if k.find('_') == -1]
        # Increase the noise level and decrease the cov magnitude by kernel_scale_factor
        for key in kernel_names:
            if isinstance(kp[f'{key}'], WhiteKernel):
                gp.kernel_.set_params(**{f'{key}__noise_level': kp[f'{key}__noise_level'] / kernel_scale_factor})
            else:
                gp.kernel_.set_params(
                    **{f'{key}__k1__constant_value': kp[f'{key}__k1__constant_value'] * kernel_scale_factor})
        # Re-fit with fixed kernel
        gp.optimizer = None
        gp.kernel = gp.kernel_
        gp.fit(x_mat, y_err)

    # Get GP signal
    y_bkg = gp.predict(x_mat)

    return gp, y_bkg


def estimate_chrono_background(drt, times, i_signal, v_signal, max_iter=1, gp=None, kernel_type='gaussian',
                               length_scale_bounds=(0.01, 10), periodicity_bounds=(1e-3, 1e3),
                               noise_level_bounds=(0.1, 10),
                               kernel_size=1, n_restarts=1, kernel_scale_factor=1, y_err_thresh=1e-3,
                               linear_downsample=True, linear_sample_interval=None,
                               fit_kw=None):
    if fit_kw is None:
        fit_kw = {}

    # Copy data
    i_signal = i_signal.copy()
    v_signal = v_signal.copy()

    # Initialize background estimate
    y_bkg = None

    sample_index = None

    gps = []
    for i in range(max_iter):
        # Fit data
        drt.fit_chrono(times, i_signal, v_signal, **fit_kw)

        x_meas = drt.get_fit_times()
        y_pred = drt.predict_response(times=x_meas)
        y_meas = drt.raw_response_signal.copy()

        if y_bkg is None:
            y_bkg = np.zeros(len(x_meas))

        if linear_downsample:
            if sample_index is None:
                # Get linearly spaced times. Only need to do this on the first iteration
                if linear_sample_interval is None:
                    # min_t_sample = np.min(np.diff(times))
                    # max_t_sample = np.max(np.diff)
                    linear_sample_interval = 0.05
                lin_times = np.arange(drt.get_fit_times()[0], drt.get_fit_times()[-1] + 1e-8, linear_sample_interval)
                x_gp, y_pred_gp, y_meas_gp, sample_index = \
                    pp.downsample_data(x_meas, y_pred, y_meas, target_times=lin_times, stepwise_sample_times=False,
                                       method='match', antialiased=False)
                print('linear downsample size:', len(x_gp))
            else:
                x_gp = x_meas[sample_index]
                y_pred_gp = y_pred[sample_index]
                y_meas_gp = y_meas[sample_index]
        else:
            x_gp = x_meas
            y_pred_gp = y_pred
            y_meas_gp = y_meas

        # Get IQR
        y_iqr = iqr(y_meas)

        # Estimate background from fit
        gp_i, y_bkg_i = estimate_background(x_gp, y_meas_gp, y_pred_gp, gp=gp, kernel_type=kernel_type,
                                            length_scale_bounds=length_scale_bounds,
                                            periodicity_bounds=periodicity_bounds,
                                            noise_level_bounds=noise_level_bounds,
                                            kernel_size=kernel_size, n_restarts=n_restarts,
                                            kernel_scale_factor=kernel_scale_factor)

        gps.append(gp_i)

        # If data was downsampled, need to predict background for all times
        if linear_downsample:
            # ts = time.time()
            # Fit to the full dataset with kernel fixed at previously optimized state
            gp_i.optimizer = None
            gp_i.kernel = gp_i.kernel_
            gp_i.fit(x_meas[:, None], y_meas - y_pred)
            y_bkg_i = gp_i.predict(x_meas[:, None])
            # print('Full sample fit/predict time: {:.1f} s'.format(time.time() - ts))

        # Add signal to background estimate
        y_bkg += y_bkg_i

        # Subtract background from measured values
        y_meas = y_meas - y_bkg_i

        if drt.chrono_mode == 'galv':
            v_signal[drt.sample_index] -= y_bkg_i
        else:
            i_signal[drt.sample_index] -= y_bkg_i

        # If median residual magnitude is below the threshold, stop
        if np.median(np.abs(y_meas - y_pred)) <= y_iqr * y_err_thresh:
            break

    return gps, y_bkg


def get_background_matrix(gps, X_pred, y_drt=None, corr_power=0):
    """
    Matrix for background estimation. mat @ resid gives background
    """
    bkg_mat = 0
    for gp in gps:
        K_trans = gp.kernel_(X_pred, gp.X_train_)
        K = gp.kernel_(gp.X_train_)
        # Since we only need the optimized kernel, not the training data, use X_pred to construct the matrix
        # K_trans = gp.kernel_(X_pred, X_pred)
        # K = gp.kernel_(X_pred)
        K_inv = np.linalg.inv(K)
        bkg_mat += K_trans @ K_inv

    # Penalize the background for correlation with model response
    if y_drt is not None and corr_power != 0:
        # Get correlation between background functions and model response
        bkg_y = np.hstack((bkg_mat, y_drt[:, None]))
        cor = np.corrcoef(bkg_y, rowvar=False)
        # Last row: correlation between each column of bkg_mat and y_drt
        cross_cor = np.abs(cor[-1, :-1])
        factor = (1 - cross_cor)
        # print(np.max(factor))
        # factor /= np.max(factor)

        # print(cross_cor.shape)
        # print(cross_cor_norm)
        # Multiply each row of bkg_mat by 1 minus its correlation with y_drt
        bkg_mat = bkg_mat @ np.diag(factor ** corr_power)

    return bkg_mat
