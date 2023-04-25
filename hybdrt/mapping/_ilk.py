# Copyright (C) 2019, the scikit-image team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#  3. Neither the name of skimage nor the names of its contributors may be
#     used to endorse or promote products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Functions in this module are adapted from scikit-image (skimage.registration._optical_flow) as of Feb. 2023.

from functools import partial
from itertools import combinations_with_replacement

import numpy as np
from scipy import ndimage as ndi

from skimage._shared.filters import gaussian as gaussian_filter
from skimage._shared.utils import _supported_float_type
from skimage.transform import warp
from skimage.registration._optical_flow_utils import coarse_to_fine, get_warp_points

from hybdrt.filters._filters import masked_filter, rms_filter


def error_weights(error_image, prev_weights, rms_size):
    err_rms = masked_filter(error_image, prev_weights, rms_filter, size=rms_size, empty=True)
    weights = np.exp(-(error_image / (4 * err_rms + 0.1 * np.std(error_image))) ** 6)

    return (weights * prev_weights) ** 0.5


def _partial_ilk(reference_image, moving_image, flow0, flow_axes, radius, num_warp, gaussian, sigma,
                 prefilter, weights, update_weights, intensity_flow):
    """Iterative Lucas-Kanade (iLK) solver for optical flow estimation.
    Modified for flow constrained to a subset of axes and pixel weighting
    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    flow0 : ndarray, shape (reference_image.ndim, M, N[, P[, ...]])
        Initialization for the vector field.
    flow_axes : tuple
        Axes along which to estimate flow.
    radius : tuple or int
        Radius of the window considered around each pixel.
    num_warp : int
        Number of times moving_image is warped.
    gaussian : bool
        if True, a gaussian kernel is used for the local
        integration. Otherwise, a uniform kernel is used.
    prefilter : bool
        Whether to prefilter the estimated optical flow before each
        image warp. This helps to remove potential outliers.
    Returns
    -------
    flow : ndarray, shape ((reference_image.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.
        :param weights:
    """
    dtype = reference_image.dtype
    img_ndim = reference_image.ndim
    flow_ndim = len(flow_axes)

    if intensity_flow:
        flow_ndim += 1

    # Format size tuple
    if np.isscalar(radius):
        size = img_ndim * (2 * radius + 1,)
        radius = img_ndim * (radius, )
    else:
        size = tuple(2 * np.array(radius) + 1)

    if gaussian:
        if sigma is None:
            sigma = tuple(np.array(radius).astype(float) / 2)
        if weights is None:
            filter_func = partial(gaussian_filter, sigma=sigma, mode='mirror')
        else:
            filter_func = partial(masked_filter, mask=weights, sigma=sigma, mode='mirror',
                                  filter_func=gaussian_filter)
    else:
        if weights is None:
            filter_func = partial(ndi.uniform_filter, size=size,
                                  mode='mirror')
        else:
            filter_func = partial(masked_filter, mask=weights, size=size, mode='mirror',
                                  filter_func=ndi.uniform_filter)

    # Initialize flow from flow0
    flow = flow0
    partial_flow = np.empty((flow_ndim, *reference_image.shape))
    for i, ax in enumerate(flow_axes):
        partial_flow[i] = flow[ax]

    # # Initialize weights
    # if weights is None:
    #     weights = np.ones_like(reference_image)

    # For each pixel location (i, j), the optical flow X = flow[:, i, j]
    # is the solution of the ndim x ndim linear system
    # A[i, j] * X = b[i, j]
    if flow_ndim == 1:
        A, b = None, None
    else:
        A = np.zeros(reference_image.shape + (flow_ndim, flow_ndim), dtype=dtype)
        b = np.zeros(reference_image.shape + (flow_ndim,), dtype=dtype)

    grid = np.meshgrid(*[np.arange(n, dtype=dtype)
                         for n in reference_image.shape],
                       indexing='ij', sparse=True)

    for _ in range(num_warp):
        if prefilter:
            # flow = ndi.median_filter(flow, (1,) + img_ndim * (3,))
            partial_flow = ndi.median_filter(partial_flow, (1,) + img_ndim * (3,))

            for i, ax in enumerate(flow_axes):
                flow[ax] = partial_flow[i]

        # # Pad flow with zeros along static axes for warp
        # flow_pad = np.zeros((img_ndim, *reference_image.shape))
        # for i, ax in enumerate(flow_axes):
        #     flow_pad[ax] = flow[i]

        # for i, ax in enumerate(flow_axes):
        #     partial_flow[i] = flow[ax]

        moving_image_warp = warp(moving_image, get_warp_points(grid, flow),
                                 mode='edge')

        # Calculate gradient for flow axes only
        grads = np.gradient(moving_image_warp, axis=flow_axes)  # flow_ndim x img_shape
        if intensity_flow:
            if flow_ndim == 2:
                # i_flow_scale = np.median(np.abs(grads))
                # print('grad scale:', i_flow_scale)
                grad = np.stack([grads, np.ones_like(grads)], axis=0)
            else:
                grad = np.stack(grads + np.ones_like(grads[0]), axis=0)
        else:
            grad = np.stack(grads, axis=0)  # flow_ndim x img_shape
        # print(flow_ndim, grad.shape)
        error_image = ((grad * partial_flow).sum(axis=0)
                       + reference_image - moving_image_warp)

        # Local linear systems creation
        if flow_ndim == 1:
            grad = grad
            A = filter_func(grad * grad)
            b = filter_func(grad * error_image)  # * weights)

            # Don't consider badly conditioned linear systems
            idx = np.abs(A) < 1e-14
            A[idx] = 1
            b[idx] = 0

            partial_flow = np.expand_dims(b / A, 0)

        else:
            # grad = grad * np.expand_dims(weights, 0)
            for i, j in combinations_with_replacement(range(flow_ndim), 2):
                A[..., i, j] = A[..., j, i] = filter_func(grad[i] * grad[j])

            for i in range(flow_ndim):
                b[..., i] = filter_func(grad[i] * error_image)  # * weights

            # Don't consider badly conditioned linear systems
            idx = abs(np.linalg.det(A)) < 1e-14
            A[idx] = np.eye(flow_ndim, dtype=dtype)
            b[idx] = 0

            # Solve the local linear systems
            partial_flow = np.moveaxis(np.linalg.solve(A, b), img_ndim, 0)

        for i, ax in enumerate(flow_axes):
            flow[ax] = partial_flow[i]

        if update_weights:
            weights = error_weights(error_image, weights, size)
            print('min weight:', np.min(weights))

    if intensity_flow:
        return np.append(flow, partial_flow[-1:], axis=0)
    else:
        return flow


def partial_flow_ilk(reference_image, moving_image, *,
                     flow_axes, radius=7, sigma=None, num_warp=10, gaussian=False,
                     prefilter=False, weights=None, update_weights=False, intensity_flow=False,
                     dtype=np.float32):
    """Coarse to fine optical flow estimator.
    The iterative Lucas-Kanade (iLK) solver is applied at each level
    of the image pyramid. iLK [1]_ is a fast and robust alternative to
    TVL1 algorithm although less accurate for rendering flat surfaces
    and object boundaries (see [2]_).
    Parameters
    ----------
    reference_image : ndarray, shape (M, N[, P[, ...]])
        The first gray scale image of the sequence.
    moving_image : ndarray, shape (M, N[, P[, ...]])
        The second gray scale image of the sequence.
    flow_axes: tuple
        Axes along which to estimate flow.
    radius : int, optional
        Radius of the window considered around each pixel.
    num_warp : int, optional
        Number of times moving_image is warped.
    gaussian : bool, optional
        If True, a Gaussian kernel is used for the local
        integration. Otherwise, a uniform kernel is used.
    prefilter : bool, optional
        Whether to prefilter the estimated optical flow before each
        image warp. When True, a median filter with window size 3
        along each axis is applied. This helps to remove potential
        outliers.
    dtype : dtype, optional
        Output data type: must be floating point. Single precision
        provides good results and saves memory usage and computation
        time compared to double precision.
    Returns
    -------
    flow : ndarray, shape ((reference_image.ndim, M, N[, P[, ...]])
        The estimated optical flow components for each axis.
    Notes
    -----
    - The implemented algorithm is described in **Table2** of [1]_.
    - Color images are not supported.
    References
    ----------
    .. [1] Le Besnerais, G., & Champagnat, F. (2005, September). Dense
       optical flow by iterative local window registration. In IEEE
       International Conference on Image Processing 2005 (Vol. 1,
       pp. I-137). IEEE. :DOI:`10.1109/ICIP.2005.1529706`
    .. [2] Plyer, A., Le Besnerais, G., & Champagnat,
       F. (2016). Massively parallel Lucas Kanade optical flow for
       real-time video processing applications. Journal of Real-Time
       Image Processing, 11(4), 713-730. :DOI:`10.1007/s11554-014-0423-0`
    """

    solver = partial(_partial_ilk, flow_axes=flow_axes, radius=radius, sigma=sigma,
                     num_warp=num_warp, gaussian=gaussian,
                     prefilter=prefilter, weights=weights, update_weights=update_weights,
                     intensity_flow=intensity_flow
                     )

    if np.dtype(dtype) != _supported_float_type(dtype):
        msg = f"dtype={dtype} is not supported. Try 'float32' or 'float64.'"
        raise ValueError(msg)

    return coarse_to_fine(reference_image, moving_image, solver, dtype=dtype)
