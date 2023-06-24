import cvxopt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from mitlef.pade_approx import create_approx_func

from .. import utils


def fit_basis_functions(x, f, basis_x, basis_type, epsilon=None, l1_lambda=0, l2_lambda=0, nonneg=False,
                        fit_intercept=True):
    """
    Fit basis functions to signal f(x)
    :param ndarray x: independent variable
    :param ndarray f: dependent variable
    :param ndarray basis_x: x values at which basis functions are centered
    :param str basis_type: type of basis function. Options: 'gaussian', 'Cole-Cole', 'step'
    :param float epsilon: shape parameter for basis functions
    :return:
    """
    utils.validation.check_basis_type(basis_type)
    # Select default epsilon value if not provided
    if epsilon is None:
        if basis_type == 'gaussian':
            epsilon = 1 / np.mean(np.diff(basis_x))
        elif basis_type == 'Cole-Cole':
            epsilon = 0.96

    # Make epsilon list if single value provided
    if np.shape(epsilon) == ():
        epsilon = [epsilon] * len(basis_x)

    phi_basis = get_basis_func(basis_type)

    if fit_intercept:
        A = np.zeros((len(x), len(basis_x) + 1))
        A[:, 0] = 1  # intercept
        coef_start_index = 1
    else:
        A = np.zeros((len(x), len(basis_x)))
        coef_start_index = 0

    for k in range(len(basis_x)):
        A[:, k + coef_start_index] = phi_basis(x - basis_x[k], epsilon[k])

    # L2 penalty matrix
    L = np.eye(A.shape[1]) * l2_lambda

    # L1 penalty vector
    l1v = np.ones(A.shape[1]) * l1_lambda

    P = cvxopt.matrix((A.T @ A + L).T)
    q = cvxopt.matrix((-f.T @ A + l1v).T)
    G = cvxopt.matrix(-np.eye(A.shape[1]))
    if nonneg:
        h = cvxopt.matrix(np.zeros(A.shape[1]))
    else:
        h = cvxopt.matrix(10 * np.ones(A.shape[1]))
    cvx_result = cvxopt.solvers.qp(P, q, G, h)
    coef = np.array(list(cvx_result['x']))

    # If intercept not fitted, pad coefficients with leading zero to indicate zero intercept
    if not fit_intercept:
        coef = np.concatenate(([0], coef))

    return coef


def evaluate_basis_fit(coef, eval_x, basis_x, basis_type, epsilon):
    utils.validation.check_basis_type(basis_type)
    phi_basis = get_basis_func(basis_type)

    # Make epsilon array if single value provided
    if np.shape(epsilon) == ():
        epsilon = [epsilon] * len(basis_x)

    A = np.zeros((len(np.atleast_1d(eval_x)), len(basis_x) + 1))
    A[:, 0] = 1  # intercept
    for k in range(len(basis_x)):
        A[:, k + 1] = phi_basis(eval_x - basis_x[k], epsilon[k])

    return A @ coef


def get_basis_func(basis_type, zga_params=None):
    """
    Get basis function
    :param str basis_type: type of basis function. Options: 'gaussian', 'Cole-Cole', 'Zic'
    :return: function which takes args y and epsilon and returns the value of the basis function
    """
    utils.validation.check_basis_type(basis_type)

    if basis_type == 'gaussian':
        def phi(y, epsilon):
            return np.exp(-(epsilon * y) ** 2)  # (epsilon / np.sqrt(np.pi)) * np.exp(-(epsilon * y) ** 2)
    elif basis_type == 'beta':
        def phi(y):
            return np.abs(y) * (1 - np.abs(y))
    elif basis_type == 'beta-rbf':
        f = get_basis_func('gaussian')
        g = get_basis_func('beta')

        def phi(y, mu, epsilon):
            return f(y - mu, epsilon) * g(y)
    elif basis_type == 'Cole-Cole':
        def phi(y, epsilon):
            return (1 / (2 * np.pi)) * np.sin((1 - epsilon) * np.pi) / (
                    np.cosh(epsilon * y) - np.cos((1 - epsilon) * np.pi))
    elif basis_type == 'zga':
        # ZARC approximation of Gaussian RBF
        y_basis, coef, eps_zga = zga_params
        phi_zarc = get_basis_func('Cole-Cole')

        def phi(y, epsilon):
            # Sum all ZARCs used to approximate Gaussian RBF
            # epsilon ignored - included for compatibility only
            f_out = np.array([x_i * phi_zarc(y + y_i, eps_zga) for x_i, y_i in zip(coef[1:], y_basis)])
            return np.sum(f_out, axis=0)
    elif basis_type == 'step':
        def phi(y, epsilon=None):
            """epsilon is ignored - included only for compatibility"""
            return utils.array.unit_step(y)
    elif basis_type == 'delta':
        def phi(y, epsilon):
            if np.isscalar(y):
                if y == 0:
                    return 1
                else:
                    return 0
            else:
                out = np.zeros_like(y, dtype=np.float_)
                out[y == 0] = 1
                return out
    elif basis_type == 'Zic':
        def phi(y, epsilon=None):
            """epsilon is ignored - included only for compatibility"""
            return 2 * np.exp(y) / (1 + np.exp(2 * y))
    elif basis_type == 'ramp':
        # Ramp with constant slope starting at zero
        def phi(y, epsilon):
            # epsilon is slope
            if np.shape(y) == ():
                if y > 0:
                    out = y * epsilon
                else:
                    out = 0
            else:
                out = np.zeros(y.shape)
                out[y > 0] = y * epsilon
            return out
    elif basis_type == 'bounded_ramp':
        # ramp from zero to one with width epsilon
        def phi(y, epsilon):
            width = 1 / epsilon
            if np.shape(y) == ():
                if y <= 0:
                    out = 0
                elif 0 < y < width:
                    out = y * epsilon
                else:
                    out = 1
            else:
                out = np.zeros(y.shape)
                out[(y > 0) & (y < width)] = y * epsilon
                out[y >= width] = 1
            return out
    elif basis_type == 'pwl':
        # Piecewise linear
        def phi(y, epsilon):
            half_width = 1 / epsilon
            if np.shape(y) == ():
                if -half_width < y < half_width:
                    out = (half_width - abs(y)) / half_width
                else:
                    out = 0
            else:
                out = np.zeros(y.shape)
                index = np.where((y > -half_width) & (y < half_width))
                out[index] = (half_width - np.abs(y[index])) / half_width
            return out
    elif basis_type == 'pwl_transformed':
        # Piecewise linear with delta transform
        def phi(y, epsilon):
            half_width = 1 / epsilon
            if np.shape(y) == ():
                if y < - half_width:
                    out = 0
                elif -half_width < y < 0:
                    out = (half_width - abs(y)) / half_width
                else:
                    out = 1
            else:
                out = np.zeros(y.shape)
                rise_index = np.where((y > -half_width) & (y < 0))
                flat_index = np.where(y > 0)
                out[rise_index] = (half_width - np.abs(y[rise_index])) / half_width
                out[flat_index] = 1
            return out
    else:
        phi = None

    return phi


def get_basis_func_derivative(basis_type, order, zga_params=None):
    """
    Get function for evaluating derivative of basis function
    :param basis_type:
    :param order:
    :return:
    """
    utils.validation.check_basis_type(basis_type)

    func = None

    if order == 0:
        func = get_basis_func(basis_type, zga_params)
    elif basis_type == 'gaussian':
        phi = get_basis_func(basis_type)
        if order == 1:
            def func(y, epsilon):
                "Derivative of Gaussian RBF"
                return -2 * epsilon ** 2 * y * phi(y, epsilon)  # * (epsilon / np.sqrt(np.pi))
        elif order == 2:
            def func(y, epsilon):
                "2nd derivative of Gaussian RBF"
                return (-2 * epsilon ** 2 + 4 * epsilon ** 4 * y ** 2) * phi(y, epsilon)
                # * (epsilon / np.sqrt(np.pi))
        elif order == 3:
            def func(y, epsilon):
                "3rd derivative of Gaussian RBF"
                return (12 * epsilon ** 4 * y - 8 * epsilon ** 6 * y ** 3) * phi(y, epsilon)
                # * (epsilon / np.sqrt(np.pi))
    elif basis_type == 'beta':
        if order == 1:
            def func(y):
                return (y / np.abs(y)) - 2 * y
        elif order == 2:
            def func(y):
                return -2 * np.ones_like(y)
    elif basis_type == 'beta-rbf':
        f = get_basis_func('gaussian')
        fx = get_basis_func_derivative('gaussian', order=1)
        g = get_basis_func('beta')
        gx = get_basis_func_derivative('beta', order=1)
        if order == 1:
            def func(y, mu, epsilon):
                return f(y - mu, epsilon) * gx(y) + fx(y - mu, epsilon) * g(y)
        elif order == 2:
            fxx = get_basis_func_derivative('gaussian', order=2)
            gxx = get_basis_func_derivative('beta', order=2)

            def func(y, mu, epsilon):
                return gxx(y) * f(y - mu, epsilon) + 2 * gx(y) * fx(y - mu, epsilon) + fxx(y - mu, epsilon) * g(y)
    elif basis_type == 'Cole-Cole':
        if order == 1:
            def func(y, epsilon):
                nume = -np.sin((1 - epsilon) * np.pi) * np.sinh(epsilon * y) * epsilon
                deno = 2 * np.pi * (np.cosh(epsilon * y) - np.cos((1 - epsilon) * np.pi)) ** 2
                return nume / deno
        elif order == 2:
            def func(y, epsilon):
                nume = epsilon ** 2 * np.sin((1 - epsilon) * np.pi) * (
                        2 * (np.sinh(epsilon * y)) ** 2 - (
                        np.cosh(epsilon * y) - np.cos((1 - epsilon) * np.pi)) * np.cosh(epsilon * y)
                )
                deno = 2 * np.pi * (np.cosh(epsilon * y) - np.cos((1 - epsilon) * np.pi)) ** 3
                return nume / deno
    elif basis_type == 'zga':
        y_basis, coef, eps_zga = zga_params
        f_zarc = get_basis_func_derivative('Cole-Cole', order, zga_params)

        def func(y, epsilon):
            # Sum all ZARCs used to approximate Gaussian RBF
            # epsilon ignored - included for compatibility only
            f_out = np.array([x_i * f_zarc(y + y_i, eps_zga) for x_i, y_i in zip(coef[1:], y_basis)])
            return np.sum(f_out, axis=0)
    elif basis_type == 'pwl':
        # Not differentiable - use discrete differences
        phi = get_basis_func(basis_type)

        def discrete_diff(func, y, epsilon):
            dy = epsilon / 5
            return (func(y + dy, epsilon) - func(y - dy, epsilon)) / (2 * dy)

        if order == 1:
            def func(y, epsilon):
                return discrete_diff(phi, y, epsilon)
        elif order == 2:
            def func(y, epsilon):
                def dfdy(y, epsilon):
                    return discrete_diff(phi, y, epsilon)

                return discrete_diff(dfdy, y, epsilon)

    # Check if function defined
    if func is None:
        raise ValueError(f'Derivative of order {order} not implemented for basis type {basis_type}')

    return func


def get_basis_func_integral(basis_type, zga_params=None):
    """
    Get function for evaluating derivative of basis function
    :param basis_type:
    :param order:
    :return:
    """
    utils.validation.check_basis_type(basis_type)

    if basis_type == 'gaussian':
        def phi(y, epsilon):
            return (np.pi ** 0.5 / (2 * epsilon)) * (1 + erf(epsilon * y))
    elif basis_type == 'delta':
        def phi(y, epsilon):
            return utils.array.unit_step(y)
    else:
        # TODO: integrals for other basis types
        raise ValueError(f'Basis func integral not yet implemented for basis_type {basis_type}')

    return phi


def get_integrated_derivative_func(basis_type='gaussian', order=1, indefinite=False):
    """
    Create function for integrated derivative matrix

    Parameters:
    -----------
    basis : string, optional (default: 'gaussian')
        Basis function used to approximate DRT
    order : int, optional (default: 1)
        Order of DRT derivative for ridge penalty
    indefinite : bool
        If True, return the indefinite integral function.
        If False, return the definite integral function from -inf to inf
    """
    utils.validation.check_basis_type(basis_type)
    if basis_type != 'gaussian':
        raise ValueError('Integrated derivative matrix only implemented for gaussian basis function')
    if basis_type == 'gaussian':
        if indefinite:
            if order == 0:
                def func(x, x_n, x_m, epsilon):
                    a = epsilon * (x_m - x_n)
                    b = epsilon * (x_m + x_n - 2 * x)
                    out = erf(b / np.sqrt(2))
                    out *= -np.sqrt(np.pi / 8) * epsilon ** -1 * np.exp(-0.5 * a ** 2)
                    return out
            elif order == 1:
                def func(x, x_n, x_m, epsilon):
                    a = epsilon * (x_m - x_n)
                    b = epsilon * (x_m + x_n - 2 * x)
                    # out = -2 * b * np.exp(2 * epsilon ** 2 * x * (x_m + x_n))
                    # out += -np.sqrt(2 * np.pi) * (a ** 2 - 1) * \
                    #     np.exp(0.5 * epsilon ** 2 * ((x_m + x_n) ** 2 + 4 * x ** 2)) * erf(b / np.sqrt(2))
                    # out *= -0.25 * epsilon * np.exp(-epsilon ** 2 * (x_m ** 2 + x_n ** 2 + 2 * x ** 2))
                    out = b * np.exp(epsilon ** 2 * (2 * x * (x_m + x_n) - (x_m ** 2 + x_n ** 2 + 2 * x ** 2)))
                    out += 0.5 * np.sqrt(2 * np.pi) * (a ** 2 - 1) * \
                           np.exp(epsilon ** 2 * (
                                   0.5 * ((x_m + x_n) ** 2 + 4 * x ** 2) - (x_m ** 2 + x_n ** 2 + 2 * x ** 2)
                           )) * erf(b / np.sqrt(2))
                    out *= 0.5 * epsilon
                    return out
            elif order == 2:
                def func(x, x_n, x_m, epsilon):
                    a = epsilon * (x_m - x_n)
                    b = epsilon * (x_m + x_n - 2 * x)
                    # out = -2 * b * np.exp(2 * epsilon ** 2 * x * (x_m + x_n)) * \
                    #     (3 * a ** 2 - 2 * epsilon ** 2 * ((x - x_m) ** 2 + (x - x_n) ** 2) + 1)
                    # out += -np.sqrt(2 * np.pi) * (a ** 4 - 6 * a ** 2 + 3) * \
                    #     np.exp(0.5 * epsilon ** 2 * ((x_m + x_n) ** 2 + 4 * x ** 2)) * erf(b / np.sqrt(2))
                    # out *= 0.25 * epsilon ** 3 * np.exp(-epsilon ** 2 * (x_m ** 2 + x_n ** 2 + 2 * x ** 2))
                    out = 2 * b * np.exp(epsilon ** 2 * (2 * x * (x_m + x_n) - (x_m ** 2 + x_n ** 2 + 2 * x ** 2))) * \
                          (3 * a ** 2 - 2 * epsilon ** 2 * ((x - x_m) ** 2 + (x - x_n) ** 2) + 1)
                    out += np.sqrt(2 * np.pi) * (a ** 4 - 6 * a ** 2 + 3) * \
                           np.exp(epsilon ** 2 * (0.5 * ((x_m + x_n) ** 2 + 4 * x ** 2) - (x_m ** 2 + x_n ** 2 + 2 * x ** 2))) * erf(b / np.sqrt(2))
                    out *= -0.25 * epsilon ** 3
                    return out
        else:
            if order == 0:
                def func(x_n, x_m, epsilon):
                    a = epsilon * (x_m - x_n)  # epsilon * np.log(l_m / l_n)
                    return (np.pi / 2) ** 0.5 * epsilon ** (-1) * np.exp(-(a ** 2 / 2))  # * (epsilon / np.sqrt(np.pi))
            elif order == 1:
                def func(x_n, x_m, epsilon):
                    a = epsilon * (x_m - x_n)  # epsilon * np.log(l_m / l_n)
                    return -(np.pi / 2) ** 0.5 * epsilon * (-1 + a ** 2) * np.exp(
                        -(a ** 2 / 2))  # * (epsilon / np.sqrt(np.pi))
            elif order == 2:
                def func(x_n, x_m, epsilon):
                    a = epsilon * (x_m - x_n)
                    return (np.pi / 2) ** 0.5 * epsilon ** 3 * (3 - 6 * a ** 2 + a ** 4) * np.exp(-(a ** 2 / 2)) \
                        # * (epsilon / np.sqrt(np.pi))
            elif order == 3:
                def func(x_n, x_m, epsilon):
                    a = epsilon * (x_m - x_n)
                    return -(np.pi / 2) ** 0.5 * epsilon ** 5 * (-15 + (45 * a ** 2) - (15 * a ** 4) + (a ** 6)) \
                           * np.exp(-(a ** 2 / 2))
            else:
                raise ValueError(f'Invalid order {order}. Order must be between 0 and 2')

    return func


def get_basis_func_area(basis_type, epsilon, zga_params=None):
    """
    Return total area of basis function
    :param basis_type:
    :param epsilon:
    :param zga_params:
    :return:
    """
    utils.validation.check_basis_type(basis_type)

    if basis_type == 'gaussian':
        area = np.sqrt(np.pi) / epsilon
    elif basis_type in ('Cole-Cole', 'delta'):
        area = 1
    elif basis_type == 'zga':
        # ZGA consists of N Cole-Cole elements. Area is simply number of elements
        area = len(zga_params[0])
    elif basis_type == 'pwl':
        area = 1 / epsilon
    else:
        raise ValueError(f'Area undefined for basis_type {basis_type}')

    return area


def get_basis_approx_params(exact_basis_type, approx_basis_type, exact_func_epsilon, approx_func_epsilon, num_bases=21,
                            basis_extent=2, curvature_penalty=None, nonneg=False):
    """
    Construct an approximation of the exact_basis_type basis function using a finite number of approx_basis_type basis
    functions
    :param str exact_basis_type: basis function type to approximate
    :param str approx_basis_type: basis function type to use to approximate exact_basis_type
    :param float exact_func_epsilon: shape parameter of the exact basis function
    :param float approx_func_epsilon: shape parameter of the approximating basis functions. If None, optimize the shape
    parameter
    :param int num_bases: number of approx_basis_type basis functions to use to approximate exact_basis_type
    :param float curvature_penalty: strength of curvature penalty to apply when optimizing the shape parameter of the
    approximating function. If None, determine from exact_func_epsilon. Only used when approx_func_epsilon is None
    :return:
    """
    if exact_basis_type == 'gaussian':
        # Set basis and evaluation range based on inverse length scale
        x_basis = np.linspace(-basis_extent / exact_func_epsilon, basis_extent / exact_func_epsilon, num_bases)
        x_eval = np.linspace(-10 / exact_func_epsilon, 10 / exact_func_epsilon, 2000)
        print('x_basis range:', np.min(x_basis), np.max(x_basis))
        print('x_eval range:', np.min(x_eval), np.max(x_eval))
        # Evaluate exact function
        phi_exact = get_basis_func(exact_basis_type)
        f_exact = phi_exact(x_eval, exact_func_epsilon)
        # Set curvature penalty based on inverse length scale
        if curvature_penalty is None:
            curvature_penalty = 1e-2 / exact_func_epsilon ** 2
    else:
        raise ValueError('Basis function approximation only implemented for Gaussian RBF')
    if approx_func_epsilon is not None:
        # Optimize with fixed epsilon for approximation function
        coef = fit_basis_functions(x_eval, f_exact, x_basis, approx_basis_type, approx_func_epsilon, nonneg=nonneg,
                                   fit_intercept=False)
        epsilon = approx_func_epsilon
    else:
        # Optimize single epsilon value and coefficients
        def resid(epsilon):
            coef = fit_basis_functions(x_eval, f_exact, x_basis, approx_basis_type, epsilon[0], nonneg=nonneg,
                                       fit_intercept=False)
            f_hat = evaluate_basis_fit(coef, x_eval, x_basis, approx_basis_type, epsilon[0])
            # Get curvature evaluation matrix for smoothing
            p2 = construct_func_eval_matrix(x_basis, x_basis, approx_basis_type, epsilon[0], 2)
            l2 = curvature_penalty * (p2 @ coef[1:])
            return np.concatenate((f_hat - f_exact, l2))

        result = least_squares(resid, [0.95], bounds=(0, 1))
        # result = minimize_scalar(cost, bounds=(0, 1), method='bounded')
        epsilon = result['x'][0]

        # With optimal epsilon selected, get coefficients with fixed epsilon value
        coef = fit_basis_functions(x_eval, f_exact, x_basis, approx_basis_type, epsilon, nonneg=nonneg,
                                   fit_intercept=False)

    return x_basis, coef, epsilon


def construct_func_eval_matrix(basis_grid, eval_grid=None, basis_type='gaussian', epsilon=1, order=1,
                               zga_params=None):
    """
    Construct matrix em such that em@x produces a vector of function values evaluated at eval_grid
    :param ndarray basis_grid: array of basis values
    :param ndarray eval_grid: array of values at which to evaluate derivative
    :param str basis_type: type of basis function. Options: 'gaussian', 'Zic'
    :param epsilon: shape parameter for basis function
    :param order: order of derivative to calculate. Can be int, list, or float. If list, entries indicate relative
    weights of 0th, 1st, and 2nd derivatives, respectively. If float, calculate weighted mixture of nearest integer
    orders.
    :return: ndarray of derivative values, same size as eval_tau
    """
    utils.validation.check_basis_type(basis_type)

    if eval_grid is None:
        # if no time constants given, assume collocated with measurement times
        eval_grid = basis_grid.copy()

    # Get function to evaluate
    func = get_basis_func_derivative(basis_type, order, zga_params)

    # Evaulate function over mesh
    xx_basis, xx_eval = np.meshgrid(basis_grid, eval_grid)
    em = func(xx_eval - xx_basis, epsilon)

    return em


def get_impedance_func(part, basis_type='gaussian', zga_params=None):
    """
    Create integrand function for A matrix

    Parameters:
    -----------
    part : string
        Part of impedance for which to generate function ('real' or 'imag')
    basis : string, optional (default: 'gaussian')
        basis function
    """
    # TODO: would be more efficient to return a complex function and then just get real and imag parts from result
    utils.validation.check_basis_type(basis_type)

    func = None
    if basis_type == 'Cole-Cole':
        # Special case - analytical expression available
        if part == 'real':
            def func(w_n, t_m, epsilon):
                return np.real(1 / (1 + (1j * w_n * t_m) ** epsilon))
        elif part == 'imag':
            def func(w_n, t_m, epsilon):
                return np.imag(1 / (1 + (1j * w_n * t_m) ** epsilon))
    elif basis_type == 'delta':
        # Special case (RC) - analytical expression available
        # Include epsilon for compatibility only
        if part == 'real':
            def func(w_n, t_m, epsilon):
                return 1 / (1 + (w_n * t_m) ** 2)
        else:
            def func(w_n, t_m, epsilon):
                return -1j * w_n * t_m / (1 + (w_n * t_m) ** 2)
    elif basis_type == 'zga':
        # ZARC Gaussian approximation
        # Special case - analytical expression available
        y_basis, coef, eps_zga = zga_params
        f_zarc = get_impedance_func(part, 'Cole-Cole')

        def func(w_n, t_m, epsilon):
            # Sum responses of all ZARCs used to approximate Gaussian RBF
            # epsilon ignored - included for compatibility only
            f_out = np.array([x_i * f_zarc(w_n, t_m * np.exp(y_i), eps_zga) for x_i, y_i in zip(coef[1:], y_basis)])
            return np.sum(f_out, axis=0)
    else:
        # Otherwise - get function to integrate numerically
        basis_func = get_basis_func(basis_type)

        # y = ln (tau/tau_m)
        if part == 'real':
            def func(y, w_n, t_m, epsilon):
                return basis_func(y, epsilon) / (1 + np.exp(2 * (y + np.log(w_n * t_m))))
        elif part == 'imag':
            def func(y, w_n, t_m, epsilon):
                return -basis_func(y, epsilon) * np.exp(y) * w_n * t_m / (1 + np.exp(2 * (y + np.log(w_n * t_m))))
        else:
            raise ValueError(f'Invalid part {part}. Options: real, imag')

    return func


def get_response_func(basis_type, op_mode, step_model, zga_params=None):
    """
    Get integrand function for A matrix
    :param str basis_type: type of basis function. Options: 'gaussian', 'Cole-Cole', 'Zic'
    :param str op_mode: Operation mode ('galv' or 'pot')
    :param str step_model: model for signal step to use. Options: 'ideal', 'expdecay'
    :return: integrand function
    """
    utils.validation.check_ctrl_mode(op_mode)
    utils.validation.check_step_model(step_model)
    f_basis = get_basis_func(basis_type, zga_params)

    func = None
    if op_mode == 'galv':
        if step_model == 'ideal':
            if basis_type == 'Cole-Cole':
                # Special case - analytical expression available
                def func(tau_m, t_n, epsilon, ml_func):
                    if ml_func is None:
                        ml_func = create_approx_func(epsilon, epsilon + 1)
                    return (t_n / tau_m) ** epsilon * ml_func(-(t_n / tau_m) ** epsilon)
            elif basis_type == 'delta':
                # Special case (RC) - analytical expression available
                def func(tau_m, t_n):
                    return 1 - np.exp(-t_n / tau_m)
            elif basis_type == 'zga':
                # Special case - analytical expression available
                y_basis, coef, eps_zga = zga_params
                f_zarc = get_response_func('Cole-Cole', op_mode, step_model)

                def func(tau_m, t_n, epsilon, ml_func):
                    # Sum responses of all ZARCs used to approximate Gaussian RBF
                    # epsilon ignored - included for compatibility only
                    f_out = np.array(
                        [x_i * f_zarc(tau_m * np.exp(y_i), t_n, eps_zga, ml_func) for x_i, y_i in
                         zip(coef[1:], y_basis)]
                    )
                    return np.sum(f_out, axis=0)
            else:
                def func(y, tau_m, t_n, epsilon, tau_rise):
                    # tau_rise unused - included for compatibility only
                    return f_basis(y, epsilon) * (1 - np.exp(-t_n / (tau_m * np.exp(y))))
        elif step_model == 'expdecay':
            if basis_type == 'delta':
                def func(tau_m, t_n, tau_rise):
                    return (
                            1 - np.exp(-t_n / tau_m)
                            + (tau_rise / (tau_rise - tau_m)) * (
                                    np.exp(-t_n / tau_m) - np.exp(-t_n / tau_rise)
                            )
                    )
            else:
                def func(y, tau_m, t_n, epsilon, tau_rise):
                    # a = 1 / (1 - np.exp(y) * tau_m / tau_rise)
                    # return f_basis(y, epsilon) * (
                    #         1 + a * np.exp(-t_n / tau_rise) - (1 + a) * np.exp(-t_n / (tau_m * np.exp(y)))
                    # )
                    tau = np.exp(y) * tau_m
                    return f_basis(y, epsilon) * (
                            1 - np.exp(-t_n / tau)
                            + (tau_rise / (tau_rise - tau)) * (
                                    np.exp(-t_n / tau) - np.exp(-t_n / tau_rise)
                            )
                    )

    return func


# ---------------------------------------------
# Integral lookup for fast matrix construction
# ---------------------------------------------
def generate_impedance_lookup(basis_type, epsilon, grid_points=2000, zga_params=None):
    # The quantity that determines the value of the integral is omega_n * tau_m.
    # Create a lookup of z(omega * tau), then use to interpolate the integral value

    # Tested for gaussian basis, 1 <= epsilon <= 5
    # Define wt (omega * tau) interpolation ranges
    re_lim = 2.7
    im_lim = re_lim * 2  # imag function decays more slowly, need to go twice as far
    wt_re_grid = np.logspace(-re_lim, re_lim, grid_points)
    wt_im_grid = np.logspace(-im_lim, im_lim, grid_points)

    # ln(tau/tau_m) domain to integrate over
    y = np.linspace(-20, 20, 1000)

    # Get integrand functions
    z_re_func = get_impedance_func('real', basis_type, zga_params)
    z_im_func = get_impedance_func('imag', basis_type, zga_params)

    z_re_grid = np.array([np.trapz(z_re_func(y, wt, 1, epsilon), x=y) for wt in wt_re_grid])
    z_im_grid = np.array([np.trapz(z_im_func(y, wt, 1, epsilon), x=y) for wt in wt_im_grid])

    return (np.log(wt_re_grid), z_re_grid), (np.log(wt_im_grid), z_im_grid)


def generate_response_lookup(basis_type, op_mode, step_model, epsilon, grid_points=2000, tau_rise=None,
                             zga_params=None):
    # The quantity that determines the value of the integral is (time_n - step_time) / tau_m
    # Create a lookup of v((time_n - step_time) / tau_m), then use to interpolate the integral value

    # Tested for gaussian basis, galvanostatic, ideal step, 1 <= epsilon <= 5
    # Define time delta ((time_n - step_time) / tau_m) interpolation range
    td_grid = np.logspace(-6, 2, grid_points)

    # ln(tau/tau_m) domain to integrate over
    y = np.linspace(-20, 20, 1000)

    # Get integrand function
    response_func = get_response_func(basis_type, op_mode, step_model, zga_params)

    response_grid = np.array([np.trapz(response_func(y, 1, td, epsilon, tau_rise), x=y) for td in td_grid])

    return np.log(td_grid), response_grid
