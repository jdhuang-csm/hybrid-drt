import numpy as np
from scipy.integrate import quad
from scipy import linalg
from mitlef.pade_approx import create_approx_func
import warnings

from . import basis
from .basis import get_integrated_derivative_func
from .. import utils
from .. import preprocessing as pp


# Functions for use inside matrix construction functions
# ------------------------------------------------------
def get_response_derivative_func(basis_type, op_mode, step_model):
    """
    Get integrand function for derivative of response with respect to ln(t)
    Only used when dv_prior=True for ridge_fit. OBSOLETE
    :param str basis_type: type of basis function. Options: 'gaussian', 'Cole-Cole', 'Zic'
    :param str op_mode: Operation mode ('galv' or 'pot')
    :param str step_model: model for signal step to use. Options: 'ideal', 'expdecay'
    :return: integrand function
    """
    utils.validation.check_ctrl_mode(op_mode)
    f_basis = basis.get_basis_func(basis_type)

    if op_mode == 'galv':
        if step_model == 'ideal':
            raise ValueError('Need to derive integrand function for ideal step')
            # def func(y, tau_m, t_n, epsilon, tau_rise):
            #     # tau_rise unused - included for compatibility only
            #     return f_basis(y, epsilon) * (1 - np.exp(-t_n / (tau_m * np.exp(y))))
        elif step_model == 'expdecay':
            def func(y, tau_m, t_n, epsilon, tau_rise):
                a = 1 / (1 - np.exp(y) * tau_m / tau_rise)
                return f_basis(y, epsilon) * (
                        a * (-t_n / tau_rise) * np.exp(-t_n / tau_rise)
                        - (1 + a) * (-t_n / tau_m * np.exp(y)) * np.exp(-t_n / (tau_m * np.exp(y)))
                )

    return func


# Matrix construction
# -------------------
def construct_response_matrix(basis_tau, times, step_model, step_times, step_sizes, basis_type='gaussian',
                              epsilon=0.975,
                              tau_rise=None, op_mode='galv', integrate_method='trapz', integrate_points=1000,
                              zga_params=None, interpolate_grids=None):
    """
    Construct matrix for time response calculation
    :param ndarray basis_tau: array of basis time constants
    :param ndarray times: array of measurement times
    :param str step_model: model for signal step to use. Options: 'ideal', 'expdecay'
    :param ndarray step_times: array of step times
    :param ndarray step_sizes: array of step amplitudes
    :param basis_type:
    :param epsilon:
    :param tau_rise:
    :param str op_mode: Operation mode ('galv' or 'pot')
    :param str integrate_method: method for numerical evaluation of integrals. Options: 'trapz', 'quad'
    :param integrate_points:
    :return: matrix A, such that A@x gives the response
    """
    utils.validation.check_step_model(step_model)
    utils.validation.check_ctrl_mode(op_mode)
    utils.validation.check_basis_type(basis_type)

    # if derivative:
    #     if not (chrono_mode == 'galv' and basis_type == 'gaussian'):
    #         raise ValueError('Time derivative function only available for galvanostatic mode and gaussian basis')

    A_layered = np.zeros([len(step_times), len(times), len(basis_tau)])

    if tau_rise is None:
        tau_rise = np.zeros(len(step_times))

    # Get Mittag-Leffler approximation function for Cole-Cole or zga basis
    if basis_type in ('Cole-Cole', 'zga'):
        if basis_type == 'Cole-Cole':
            eps_ml = epsilon
        else:
            eps_ml = zga_params[2]

        if not (0 < eps_ml < 1):
            raise ValueError('Epsilon must be between 0 and 1 for Cole-Cole basis function')

        ml_func = create_approx_func(eps_ml, eps_ml + 1)

    # Get response function
    # if derivative:
    #     func = get_response_derivative_func(basis_type, chrono_mode, step_model)
    if integrate_method == 'interp':
        if interpolate_grids is None:
            raise ValueError("interpolate_grids must be provided for integrate_method 'interp'")
        log_td_grid, response_grid = interpolate_grids
        func = None
    else:
        log_td_grid, response_grid = None, None
        func = basis.get_response_func(basis_type, op_mode, step_model, zga_params)

    for k in range(len(step_times)):
        st = step_times[k]
        sa = step_sizes[k]
        # Only perform calculation if times exist after step time
        if np.sum(times > st) > 0:
            if op_mode == 'galv':
                if basis_type in ('Cole-Cole', 'zga'):
                    # Special case - analytical solution available
                    if step_model != 'ideal':
                        raise ValueError('Non-ideal step_type not supported for Cole-Cole basis function')
                    # All times before step are unaffected
                    tau_mesh, t_mesh = np.meshgrid(basis_tau, times[times > st])
                    # Don't allow true zeros in t_delta - violates domain of ML approximation
                    # Set to very small values instead
                    # t_delta_mesh = np.maximum(t_mesh - st, 1e-10)
                    A_layered[k, times > st, :] = sa * func(tau_mesh, t_mesh - st, epsilon, ml_func)
                elif basis_type == 'delta':
                    if step_model != 'ideal':
                        raise ValueError('Non-ideal step_type not supported for delta basis function')
                    tau_mesh, t_mesh = np.meshgrid(basis_tau, times[times > st])
                    A_layered[k, times > st, :] = sa * func(tau_mesh, t_mesh - st)
                else:
                    # all times before step are unaffected
                    # calculate integrals for times after step
                    if integrate_method == 'trapz':
                        y = np.linspace(-20, 20, integrate_points)
                        A_layered[k, times > st] = [
                            [np.trapz(func(y, tau_m, t_n - st, epsilon, tau_rise[k]), x=y) * sa for tau_m in basis_tau]
                            for t_n in times[times > st]
                        ]
                    elif integrate_method == 'quad':
                        A_layered[k, times > st] = [
                            [quad(func, -20, 20, args=(tau_m, t_n, epsilon, tau_rise[k]), epsabs=1e-4)[0] * sa
                             for tau_m in basis_tau]
                            for t_n in times[times > st]
                        ]
                    elif integrate_method == 'interp':
                        A_layered[k, times > st] = [
                            [np.interp(np.log((t_n - st) / basis_tau), log_td_grid, response_grid) * sa
                             for t_n in times[times > st]]
                        ]

            elif op_mode == 'pot':
                # Basis function is delta function
                mtau, mtimes = np.meshgrid(basis_tau, times)
                A_layered[k] = np.exp(-(mtimes - st) / mtau) * utils.array.unit_step(mtimes, st) * sa
                A_layered[k] = np.nan_to_num(A_layered[k], nan=0)

    A = np.sum(A_layered, axis=0)

    return A, A_layered


def construct_integrated_derivative_matrix(basis_grid, basis_type='gaussian', order=1, epsilon=1, zga_params=None,
                                           integration_limits=None):
    """
    Construct matrix for calculation of DRT ridge penalty.
    x^T@M@x gives integral of squared derivative of DRT over all ln(tau)

    Parameters:
    -----------
    frequencies : array
        Frequencies at which basis functions are centered
    basis : string, optional (default: 'gaussian')
        Basis function used to approximate DRT
    order : int, optional (default: 1)
        Order of derivative to penalize
    epsilon : float, optional (default: 1)
        Shape parameter for chosen basis function
    """
    utils.validation.check_basis_type(basis_type)

    if integration_limits is None:
        if basis_type == 'gaussian':
            if type(order) == list:
                f0, f1, f2 = order
                func0 = get_integrated_derivative_func(basis_type, 0)
                func1 = get_integrated_derivative_func(basis_type, 1)
                func2 = get_integrated_derivative_func(basis_type, 2)

                def func(x_n, x_m, epsilon):
                    return f0 * func0(x_n, x_m, epsilon) + f1 * func1(x_n, x_m, epsilon) + f2 * func2(x_n, x_m, epsilon)

            else:
                func = get_integrated_derivative_func(basis_type, order)

            if utils.array.is_uniform(basis_grid):
                # Matrix is symmetric Toeplitz if basis_type==gaussian and basis_eig is log-uniform.
                # Only need to calculate 1st column
                # May not apply to other basis functions
                x_0 = basis_grid[0]

                c = [func(x_n, x_0, epsilon) for x_n in basis_grid]
                # r = [quad(func,limits[0],limits[1],args=(w_0,t_m,epsilon),epsabs=1e-4)[0] for t_m in 1/omega]
                # if r[0]!=c[0]:
                # raise Exception('First entries of first row and column are not equal')
                M = linalg.toeplitz(c)
            else:
                # need to calculate all entries
                M = np.empty((len(basis_grid), len(basis_grid)))
                for n, x_n in enumerate(basis_grid):
                    M[n, :] = [func(x_n, x_m, epsilon) for x_m in basis_grid]
        elif basis_type == 'delta':
            # Make discrete differentiation matrix
            if order == 0:
                M = np.eye(len(basis_grid))
            elif order == 1:
                L = np.eye(len(basis_grid))
                np.fill_diagonal(L[1:, :-1], -1)
                L[0, 0] = 0
                L[1:, 1:] /= np.diff(basis_grid)[:, None]
                M = L.T @ L
            elif order == 2:
                L = np.eye(len(basis_grid)) * 2
                L[0, 0] = 1
                L[-1, -1] = 1
                np.fill_diagonal(L[1:, :-1], -1)
                np.fill_diagonal(L[:-1, 1:], -1)
                L[1:, 1:] /= np.diff(basis_grid)[:, None]
                L[:-1, :-1] /= np.diff(basis_grid)[:, None]
                M = L.T @ L
        else:
            # Get discrete evaluation matrix (em @ x = vector of function values)
            em = basis.construct_func_eval_matrix(basis_grid, None, basis_type, epsilon, order, zga_params)

            # x.T @ em @ em @ x = L2 norm of vector of function values
            M = em @ em

            # Multiply by grid spacing to approximate integral
            grid_space = np.mean(np.abs(np.diff(basis_grid)))
            M *= grid_space
    else:
        func = get_integrated_derivative_func(basis_type, order, indefinite=True)
        a, b = integration_limits
        xx_i, xx_j = np.meshgrid(basis_grid, basis_grid)
        M = func(b, xx_i, xx_j, epsilon) - func(a, xx_i, xx_j, epsilon)

    return M


def construct_impedance_matrix(frequencies, part, tau=None, basis_type='gaussian', epsilon=1, frequency_precision=10,
                               integrate_method='trapz', integrate_points=1000, zga_params=None,
                               interpolate_grids=None):
    """
    Construct A matrix for DRT. A' and A'' matrices transform DRT coefficients to real
    and imaginary impedance values, respectively, for given frequencies.
    :param ndarray frequencies: Frequencies
    :param string part : Part of impedance for which to construct matrix ('real' or 'imag')
    :param ndarray tau : Time constants at which to center basis functions. If None, use time constants
        corresponding to frequencies, i.e. tau=1/(2*pi*frequencies)
    :param str basis_type : Basis function to use to approximate DRT
    :param float epsilon : Shape parameter for chosen basis function
    :param str integrate_method: Method to use for numerical integration. Options: 'trapz', 'quad'
    :param int integrate_points: Number of points to use for trapezoidal integration
    """
    omega = frequencies * 2 * np.pi

    # check if tau is inverse of omega
    if tau is None:
        tau = 1 / omega
        tau_eq_omega = True
    elif len(tau) == len(omega):
        if utils.validation.check_equality(utils.array.rel_round(tau, frequency_precision),
                                utils.array.rel_round(1 / omega, frequency_precision)):
            tau_eq_omega = True
        else:
            tau_eq_omega = False
    else:
        tau_eq_omega = False

    # check if omega is subset of inverse tau
    # find index where first frequency matches tau
    match = utils.array.rel_round(1 / omega[0], frequency_precision) == utils.array.rel_round(tau, frequency_precision)
    if np.sum(match) == 1:
        start_idx = np.where(match == True)[0][0]
        # if tau vector starting at start_idx matches omega, omega is a subset of tau
        if utils.validation.check_equality(
                utils.array.rel_round(tau[start_idx:start_idx + len(omega)], frequency_precision),
                utils.array.rel_round(1 / omega, frequency_precision)
        ):
            tau_freq_subset = True
        else:
            tau_freq_subset = False
    elif np.sum(match) == 0:
        # if no tau corresponds to first omega, omega is not a subset of tau
        tau_freq_subset = False
    else:
        # if more than one match, must be duplicates in tau
        raise Exception('Repeated tau values')

    if not tau_freq_subset:
        # check if tau is subset of inverse omega
        # find index where first frequency matches tau
        match = utils.array.rel_round(1 / omega, frequency_precision) \
                == utils.array.rel_round(tau[0], frequency_precision)
        if np.sum(match) == 1:
            start_idx = np.where(match)[0][0]
            # if omega vector starting at start_idx matches tau, tau is a subset of omega
            if utils.validation.check_equality(
                    utils.array.rel_round(omega[start_idx:start_idx + len(tau)], frequency_precision),
                    utils.array.rel_round(1 / tau, frequency_precision)
            ):
                tau_freq_subset = True
            else:
                tau_freq_subset = False
        elif np.sum(match) == 0:
            # if no omega corresponds to first tau, tau is not a subset of omega
            tau_freq_subset = False
        else:
            # if more than one match, must be duplicates in omega
            raise Exception('Repeated omega values')

    # Determine if A is a Toeplitz matrix
    # Note that when there is simultaneous charge transfer, the matrix is never a Toeplitz matrix
    # because the integrand can no longer be written in terms of w_n*t_m only
    if utils.array.is_log_uniform(frequencies):
        if tau_eq_omega:
            is_toeplitz = True
        elif tau_freq_subset and utils.array.is_log_uniform(tau):
            is_toeplitz = True
        else:
            is_toeplitz = False
    else:
        is_toeplitz = False
    # print('toeplitz', is_toeplitz)

    # print(part,'is toeplitz',is_toeplitz)
    # print(part,'freq subset',tau_freq_subset)

    if integrate_method == 'interp':
        if interpolate_grids is None:
            raise ValueError("interpolate_grids must be provided to use integrate_method 'interp'")
        log_wt_grid, z_grid = interpolate_grids
        func = None
    else:
        # get function to integrate
        func = basis.get_impedance_func(part, basis_type, zga_params)

    if basis_type in ('Cole-Cole', 'zga', 'delta'):
        # Special case - analytical expression available
        if is_toeplitz:
            # only need to calculate 1st row and column
            w_0 = omega[0]
            t_0 = tau[0]

            r = func(w_0, tau, epsilon)
            c = func(omega, t_0, epsilon)

            if r[0] != c[0]:
                print(r[0], c[0])
                raise Exception('First entries of first row and column are not equal')

            A = linalg.toeplitz(c, r)
        else:
            # need to calculate all entries
            tt, ww = np.meshgrid(tau, omega)

            A = func(ww, tt, epsilon)
    else:
        # Must numerically integrate
        if integrate_method == 'quad':
            if part == 'real':
                quad_limits = (-np.inf, np.inf)
            elif part == 'imag':
                # scipy.integrate.quad is unstable for imag func with infinite limits
                quad_limits = (-20, 20)
            else:
                raise ValueError(f'Invalid part {part}. Options: real, imag')

        if is_toeplitz:  # is_loguniform(frequencies) and tau_eq_omega:
            # only need to calculate 1st row and column
            w_0 = omega[0]
            t_0 = tau[0]

            if integrate_method == 'quad':
                c = [quad(func, quad_limits[0], quad_limits[1], args=(w_n, t_0, epsilon), epsabs=1e-4)[0] for w_n in omega]
                r = [quad(func, quad_limits[0], quad_limits[1], args=(w_0, t_m, epsilon), epsabs=1e-4)[0] for t_m in tau]
            elif integrate_method == 'trapz':
                y = np.linspace(-20, 20, integrate_points)
                c = [np.trapz(func(y, w_n, t_0, epsilon), x=y) for w_n in omega]
                r = [np.trapz(func(y, w_0, t_m, epsilon), x=y) for t_m in tau]
            elif integrate_method == 'interp':
                c = np.interp(np.log(omega * t_0), log_wt_grid, z_grid)
                r = np.interp(np.log(w_0 * tau), log_wt_grid, z_grid)

            if r[0] != c[0]:
                print(r[0], c[0])
                raise Exception('First entries of first row and column are not equal')
            A = linalg.toeplitz(c, r)
        else:
            # need to calculate all entries
            A = np.empty((len(frequencies), len(tau)))
            for n, w_n in enumerate(omega):
                if integrate_method == 'quad':
                    A[n, :] = [quad(func, quad_limits[0], quad_limits[1], args=(w_n, t_m, epsilon), epsabs=1e-4)[0] for t_m
                               in tau]
                elif integrate_method == 'trapz':
                    y = np.linspace(-20, 20, integrate_points)
                    A[n, :] = [np.trapz(func(y, w_n, t_m, epsilon), x=y) for t_m in tau]
                elif integrate_method == 'interp':
                    A[n, :] = np.interp(np.log(w_n * tau), log_wt_grid, z_grid)

    return A


# inductance response vector
def construct_inductance_response_vector(times, step_model, step_times, step_sizes, tau_rise,
                                         op_mode='galv'):
    utils.validation.check_step_model(step_model)
    utils.validation.check_ctrl_mode(op_mode)

    irv = np.zeros(len(times))

    # If using expdecay step model, fill in values. Otherwise, return vector of zeros
    if step_model == 'expdecay':
        for k in range(len(step_times)):
            st = step_times[k]
            sa = step_sizes[k]
            tr = tau_rise[k]
            if op_mode == 'galv':
                irv[times >= st] += (sa / tr) * np.exp(-(times[times >= st] - st) / tr)
            else:
                raise ValueError('Inductance response vector not implemented for potentiostatic mode')

    return irv


def construct_ohmic_response_vector(times, step_model, step_times, step_sizes, tau_rise, input_signal, smooth,
                                    op_mode='galv'):
    utils.validation.check_step_model(step_model)
    utils.validation.check_ctrl_mode(op_mode)

    # print('unprocessed input_signal:', input_signal)
    if smooth:
        # Use ideal input signal from identified steps rather than noisy measured input signal
        input_signal = pp.generate_model_signal(times, step_times, step_sizes, tau_rise, step_model)
        # print('smoothed input signal:', input_signal)
    else:
        # Get delta from starting value
        # DIFFERENTIAL R_inf should respond only to changes in current
        prestep_signal = input_signal[times < step_times[0]]
        input_signal = input_signal - np.mean(prestep_signal)
    # print('processed input_signal:', input_signal)

    if op_mode == 'galv':
        rv = input_signal
    else:
        raise ValueError('Ohmic response vector not implemented for potentiostatic mode')

    return rv


def construct_capacitance_response_vector(times, step_model, step_times, step_sizes, tau_rise,
                                          op_mode='galv'):
    utils.validation.check_step_model(step_model)
    utils.validation.check_ctrl_mode(op_mode)

    crv = np.zeros(len(times))

    if step_model == 'ideal':
        for k in range(len(step_times)):
            st = step_times[k]
            sa = step_sizes[k]
            if op_mode == 'galv':
                crv[times >= st] += sa * (times[times >= st] - st)
            else:
                raise ValueError('Capacitance response vector not implemented for potentiostatic mode')
    else:
        raise ValueError('Capacitance response not implemented for non-ideal steps')
        # TODO: implement for non-ideal step

    return crv


def construct_inductance_impedance_vector(frequencies):
    return 1j * 2 * np.pi * frequencies


def construct_capacitance_impedance_vector(frequencies):
    return 1 / (1j * 2 * np.pi * frequencies)


# ============================
# Variance estimation matrices
# ============================
def construct_chrono_var_matrix(times, step_times, vmm_epsilon, error_structure=None):

    if error_structure is None:
        # Flexible error structure

        # Get transformed times
        # fwd_trans, rev_trans = get_time_transforms(times, step_times)
        # tt = fwd_trans(times)

        # vmm = construct_func_eval_matrix(tt, epsilon=vmm_epsilon, order=0)

        # by time
        vmm = basis.construct_func_eval_matrix(times, epsilon=vmm_epsilon, order=0)
        # By index
        # vmm = basis.construct_func_eval_matrix(np.arange(len(times)), epsilon=vmm_epsilon, order=0)
    elif error_structure == 'uniform':
        # Uniform errors
        vmm = np.ones((len(times), len(times)))

    # Normalize each row such that weights sum to 1
    vm_rowsum = np.sum(vmm, axis=1)
    vmm /= vm_rowsum[:, None]

    return vmm


def construct_eis_var_matrix(frequencies, vmm_epsilon, reim_cor, error_structure):
    n = len(frequencies)
    vmm = np.zeros((2 * n, 2 * n))

    # Construct main averaging matrix
    if error_structure is None:
        vmm_main = basis.construct_func_eval_matrix(np.log(frequencies), epsilon=vmm_epsilon, order=0)
    elif error_structure == 'uniform':
        vmm_main = np.ones((n, n))

    # Diagonals: re-re and im-im averaging
    vmm[:n, :n] = vmm_main
    vmm[n:, n:] = vmm_main

    # Off-diagonals: re-im and im-re averaging
    vmm[n:, :n] = vmm_main * reim_cor
    vmm[:n, n:] = vmm_main * reim_cor

    # Normalize each row such that weights sum to 1
    vm_rowsum = np.sum(vmm, axis=1)  # Normalize each row
    vmm /= vm_rowsum[:, None]

    return vmm


# ======================
# Matrix inversion
# ======================
def invert_psd(a, use_cholesky):
    """
    Invert positive semidefinite matrix
    :param a: matrix to invert
    :param use_cholesky: whether to use Cholesky factorization
    :return:
    """
    try:
        if use_cholesky:
            try:
                c = linalg.inv(linalg.cholesky(a))
                a_inv = c.T @ c
            except linalg.LinAlgError:
                a_inv = linalg.inv(a)
        else:
            a_inv = linalg.inv(a)
        return a_inv
    except linalg.LinAlgError as err:
        warnings.warn(f'Matrix inversion failed with error: \n{err}')
        return None
