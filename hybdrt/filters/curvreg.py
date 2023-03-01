# Implements curvature regularizing filters as described in https://ieeexplore.ieee.org/document/7835193
import numpy as np


def decompose_domain(img):
    nr, nc = img.shape

    t_rows = np.arange(0, nr, 2)
    c_rows = np.arange(1, nr, 2)

    bt_cols = np.arange(0, nc, 2)
    wt_cols = np.arange(1, nc, 2)

    wc_cols = np.arange(0, nc, 2)
    bc_cols = np.arange(1, nc, 2)

    bt_index = np.meshgrid(t_rows, bt_cols)
    wt_index = np.meshgrid(t_rows, wt_cols)
    bc_index = np.meshgrid(c_rows, bc_cols)
    wc_index = np.meshgrid(c_rows, wc_cols)

    return bt_index, wt_index, bc_index, wc_index


def min_projection_distance(u, domain_index, curv_type='gc'):
    i, j = domain_index
    u_ij = u[i, j]

    if curv_type == 'gc':
        # Gaussian curvature
        d1 = 0.5 * (u[i - 1, j] + u[i + 1, j]) - u_ij

        d2 = 0.5 * (u[i, j - 1] + u[i, j + 1]) - u_ij

        d3 = 0.5 * (u[i - 1, j - 1] + u[i + 1, j + 1]) - u_ij

        d4 = 0.5 * (u[i - 1, j + 1] + u[i + 1, j - 1]) - u_ij

        d5 = u[i - 1, j] + u[i, j - 1] - u[i - 1, j - 1] - u_ij

        d6 = u[i - 1, j] + u[i, j + 1] - u[i - 1, j + 1] - u_ij

        d7 = u[i, j - 1] + u[i + 1, j] - u[i + 1, j - 1] - u_ij

        d8 = u[i, j + 1] + u[i + 1, j] - u[i + 1, j + 1] - u_ij

        distances = np.stack([d1, d2, d3, d4, d5, d6, d7, d8], axis=0)
    elif curv_type == 'mc':
        # Mean curvature
        d1 = (5 / 16) * (u[i - 1, j] + u[i + 1, j]) \
             + (5 / 8) * u[i, j + 1] \
             - (1 / 8) * (u[i - 1, j + 1] + u[i + 1, j + 1]) \
             - u_ij
        d2 = (5 / 16) * (u[i - 1, j] + u[i + 1, j]) \
             + (5 / 8) * u[i, j - 1] \
             - (1 / 8) * (u[i - 1, j - 1] + u[i + 1, j - 1]) \
             - u_ij
        d3 = (5 / 16) * (u[i, j - 1] + u[i, j + 1]) \
             + (5 / 8) * u[i - 1, j] \
             - (1 / 8) * (u[i - 1, j - 1] + u[i - 1, j + 1]) \
             - u_ij
        d4 = (5 / 16) * (u[i, j - 1] + u[i, j + 1]) \
             + (5 / 8) * u[i + 1, j] \
             - (1 / 8) * (u[i + 1, j - 1] + u[i + 1, j + 1]) \
             - u_ij
        distances = np.stack([d1, d2, d3, d4], axis=0)
    else:
        raise ValueError(f'Invalid curv_type {curv_type}')

    min_index = np.argmin(np.abs(distances), axis=0)

    return np.take_along_axis(distances, np.expand_dims(min_index, axis=0), axis=0)[0]


def pad_image(img, mode, cval):
    img_pad = np.empty((img.shape[0] + 2, img.shape[1] + 2), dtype=img.dtype)
    img_pad[1:-1, 1:-1] = img.copy()

    if mode == 'reflect':
        img_pad[0] = img_pad[2].copy()
        img_pad[-1] = img_pad[-3].copy()
        img_pad[:, 0] = img_pad[:, 2].copy()
        img_pad[:, -1] = img_pad[:, -3].copy()
    elif mode == 'nearest':
        img_pad[0] = img_pad[1].copy()
        img_pad[-1] = img_pad[-2].copy()
        img_pad[:, 0] = img_pad[:, 1].copy()
        img_pad[:, -1] = img_pad[:, -2].copy()
    elif mode == 'wrap':
        img_pad[0] = img_pad[-2].copy()
        img_pad[-1] = img_pad[1].copy()
        img_pad[:, 0] = img_pad[:, -2].copy()
        img_pad[:, -1] = img_pad[:, 1].copy()
    elif mode == 'constant':
        img_pad[0] = cval
        img_pad[-1] = cval
        img_pad[:, 0] = cval
        img_pad[:, -1] = cval
    else:
        raise ValueError(f'Invalid mode {mode}')

    # Fill in corners
    img_pad[0, 0] = 0.5 * (img_pad[0, 1] + img_pad[1, 0])
    img_pad[-1, 0] = 0.5 * (img_pad[-1, 1] + img_pad[-2, 0])
    img_pad[0, -1] = 0.5 * (img_pad[0, -2] + img_pad[1, -1])
    img_pad[-1, -1] = 0.5 * (img_pad[-1, -2] + img_pad[-2, -1])

    return img_pad


def cr_filter(img, n_iter=10, curv_type='gc', mode='reflect', cval=0.0):
    domain_indices = decompose_domain(img)

    u = pad_image(img, mode, cval)
    for i in range(n_iter):
        for domain_index in domain_indices:
            padded_index = [domain_index[0] + 1, domain_index[1] + 1]
            du = min_projection_distance(u, padded_index, curv_type=curv_type)
            # print(np.max(np.abs(du)))
            # print(du.shape)
            u[padded_index] = u[padded_index] + du

    return u[1:-1, 1:-1]
