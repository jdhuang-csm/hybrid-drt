import numpy as np
from scipy.optimize import least_squares


def rbf(x, r, mu, sigma, order=0):
    y = (x - mu) / sigma
    f = np.exp(-0.5 * (y ** 2))
    if order == 0:
        return r * f
    elif order == 1:
        return -r * f * y
    elif order == 2:
        return r * f * (y ** 2 - 1 / sigma)
    else:
        raise ValueError(f'Invalid order {order}')


def sheet_func_2d(tau_mesh, r_vec, lt_vec, sigma_vec, order=0):
    return rbf(tau_mesh, r_vec[:, None], lt_vec[:, None], sigma_vec[:, None], order=order)


def eval_sheets_2d(tau_mesh, r_mat, lt_mat, sigma_mat, order=0):
    vals = [sheet_func_2d(tau_mesh, r_mat[i], lt_mat[i], sigma_mat[i], order=order) for i in range(r_mat.shape[0])]
    return np.sum(vals, axis=0)


def residuals(y, tau_mesh, r_mat, lt_mat, sigma_mat, order=0):
    y_hat = eval_sheets_2d(tau_mesh, r_mat, lt_mat, sigma_mat, order=order)
    return (y_hat - y).flatten()


def optimize_sheets(y, tau_mesh, r0, lt0, sigma0, order=0):

    x0 = np.concatenate([r0.flatten(), lt0.flatten(), sigma0.flatten()])
    split_len = len(r0.flatten())
    mat_shape = r0.shape

    def resid_func(x):
        r_mat = x[:split_len].reshape(mat_shape)
        lt_mat = x[split_len:2 * split_len].reshape(mat_shape)
        sigma_mat = x[2 * split_len:].reshape(mat_shape)
        return residuals(y, tau_mesh, r_mat, lt_mat, sigma_mat, order=order)

    return least_squares(resid_func, x0, method='trf')

