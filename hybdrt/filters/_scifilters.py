# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Functions in this module were adapted from scipy.ndimage._filters (Jan. 2023)
import numbers
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.filters import correlate1d, gaussian_filter1d


# "Empty" Gaussian filter
# -----------------------------------------------------------
def _empty_gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel with the central weight set to zero
    Adapted from scipy.ndimage._filters._empty_gaussian_kernel1d
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x[x == 0] = 0
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        exponent_range = np.arange(order + 1)
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order) / -sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def empty_gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                            mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """
    1-D "empty" Gaussian filter. Adapted from scipy.ndimage.gaussian_filter1d
    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int, optional
        Radius of the Gaussian kernel. If specified, the size of
        the kernel will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.
    Returns
    -------
    gaussian_filter1d : ndarray
    Notes
    -----
    The Gaussian kernel will have size ``2*radius + 1`` where
    ``radius = round(truncate * sigma)``.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _empty_gaussian_kernel1d(sigma, order, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def empty_gaussian_filter(input, sigma, order=0, output=None,
                          mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """
    Multidimensional "empty" Gaussian filter. Adapted from scipy.ndimage.gaussian_filter
    Applies a Gaussian filter with zero weight given to the central pixel
    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    radius : None or int or sequence of ints, optional
        Radius of the Gaussian kernel. The radius are given for each axis
        as a sequence, or as a single number, in which case it is equal
        for all axes. If specified, the size of the kernel along each axis
        will be ``2*radius + 1``, and `truncate` is ignored.
        Default is None.
    Returns
    -------
    gaussian_filter : ndarray
        Returned array of same shape as `input`.
    Notes
    -----
    The multidimensional filter is implemented as a sequence of
    1-D convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.
    The Gaussian kernel will have size ``2*radius + 1`` along each axis
    where ``radius = round(truncate * sigma)``.
    """
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    radiuses = _ni_support._normalize_sequence(radius, input.ndim)
    axes = list(range(input.ndim))
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode, radius in axes:
            empty_gaussian_filter1d(input, sigma, axis, order, output,
                                    mode, cval, truncate, radius=radius)
            input = output
    else:
        output[...] = input[...]
    return output


# 1d laplacian filters
# -----------------------
def generic_laplace1d(input, derivative2, axis=-1, output=None, mode="reflect",
                      cval=0.0,
                      extra_arguments=(),
                      extra_keywords=None):
    """
    1-D Laplace filter using a provided second derivative function.
    Parameters
    ----------
    %(input)s
    derivative2 : callable
        Callable with the following signature::
            derivative2(input, axis, output, mode, cval,
                        *extra_arguments, **extra_keywords)
        See `extra_arguments`, `extra_keywords` below.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    %(extra_keywords)s
    %(extra_arguments)s
    """
    if extra_keywords is None:
        extra_keywords = {}
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)

    derivative2(input, axis, output, mode, cval,
                *extra_arguments, **extra_keywords)

    return output


def laplace1d(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """1-D Laplace filter based on approximate second derivatives.
    Parameters
    ----------
    %(input)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    """

    def derivative2(input, axis, output, mode, cval):
        return correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)

    return generic_laplace1d(input, derivative2, axis, output, mode, cval)


def gaussian_laplace1d(input, sigma, axis=-1, output=None, mode="reflect",
                       cval=0.0, **kwargs):
    """Multidimensional Laplace filter using Gaussian second derivatives.
    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        The standard deviations of the Gaussian filter are given for
        each axis as a sequence, or as a single number, in which case
        it is equal for all axes.
    %(output)s
    %(mode_multiple)s
    %(cval)s
    Extra keyword arguments will be passed to gaussian_filter().
    """
    input = np.asarray(input)

    def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
        order = 2
        return gaussian_filter1d(input, sigma, axis, order, output, mode, cval,
                                 **kwargs)

    return generic_laplace1d(input, derivative2, axis, output, mode, cval,
                             extra_arguments=(sigma,),
                             extra_keywords=kwargs)
