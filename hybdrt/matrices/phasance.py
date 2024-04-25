import numpy as np
from scipy.special import gamma, erf


def unit_phasor_impedance(omega, nu):
    return (1j * omega) ** nu


def unit_phasor_voltage(t, nu):
    return t ** -nu / gamma(-nu + 1)


def get_nu_limits(nu_m):
    a = np.minimum(0, np.sign(nu_m))
    b = np.maximum(0, np.sign(nu_m))
    return a, b


def get_phasor_impedance_integral_func(basis_type, normalize=False):
    if basis_type == 'gaussian':
        if normalize:
            def func(nu, omega, nu_m, tau_c, epsilon):
                jwt = 1j * omega * tau_c
                out = 0.5 * np.sqrt(np.pi) * unit_phasor_impedance(omega * tau_c, nu_m) / epsilon
                out *= jwt ** (np.log(jwt) / (4 * epsilon ** 2))
                out *= erf(epsilon * (nu - nu_m) - np.log(jwt) / (2 * epsilon))
                return out
        else:
            def func(nu, omega, nu_m, epsilon):
                out = 0.5 * np.sqrt(np.pi) * unit_phasor_impedance(omega, nu_m) / epsilon
                out *= (1j * omega) ** (np.log(1j * omega) / (4 * epsilon ** 2))
                out *= erf(epsilon * (nu - nu_m) - np.log(1j * omega) / (2 * epsilon))
                return out
    else:
        raise ValueError(f'basis_type {basis_type} is not supported for phasance')

    return func


def get_phasor_response_integral_func(basis_type, normalize=False):
    if basis_type == 'gaussian':
        if normalize:
            def func(nu, t, nu_m, tau_c, epsilon):
                tt = t / tau_c
                out = 0.5 * np.sqrt(np.pi) * unit_phasor_voltage(tt, nu_m) / epsilon
                out *= tt ** (np.log(tt) / (4 * epsilon ** 2))
                out *= erf(epsilon * (nu - nu_m) + np.log(tt) / (2 * epsilon))
                return out
        else:
            def func(nu, t, nu_m, epsilon):
                out = 0.5 * np.sqrt(np.pi) * unit_phasor_voltage(t, nu_m) / epsilon
                out *= t ** (np.log(t) / (4 * epsilon ** 2))
                out *= erf(epsilon * (nu - nu_m) + np.log(t) / (2 * epsilon))
                return out
    else:
        raise ValueError(f'basis_type {basis_type} is not supported for phasance')

    return func


def get_phasor_impedance_func(basis_type, normalize=False):
    if basis_type == 'delta':
        return unit_phasor_impedance
    elif basis_type == 'gaussian':
        f_int = get_phasor_impedance_integral_func(basis_type, normalize=normalize)
        if normalize:
            def func(omega, nu_m, tau_c, epsilon):
                # Get integration limits
                a, b = get_nu_limits(nu_m)
                return f_int(b, omega, nu_m, tau_c, epsilon) - f_int(a, omega, nu_m, tau_c, epsilon)
        else:
            def func(omega, nu_m, epsilon):
                # Get integration limits
                a, b = get_nu_limits(nu_m)
                return f_int(b, omega, nu_m, epsilon) - f_int(a, omega, nu_m, epsilon)

    else:
        raise ValueError(f'basis_type {basis_type} is not supported for phasances')

    return func


def get_phasor_response_func(basis_type, op_mode='galv', step_model='ideal', normalize=False):
    if op_mode == 'galv' and step_model == 'ideal':
        if basis_type == 'delta':
            return unit_phasor_voltage
        elif basis_type == 'gaussian':
            f_int = get_phasor_response_integral_func(basis_type, normalize=normalize)
            if normalize:
                def func(t, nu_m, tau_c, epsilon):
                    # Get integration limits
                    a, b = get_nu_limits(nu_m)
                    return f_int(b, t, nu_m, tau_c, epsilon) - f_int(a, t, nu_m, tau_c, epsilon)
            else:
                def func(t, nu_m, epsilon):
                    # Get integration limits
                    a, b = get_nu_limits(nu_m)
                    return f_int(b, t, nu_m, epsilon) - f_int(a, t, nu_m, epsilon)
        else:
            raise ValueError(f'basis_type {basis_type} is not supported for phasances')

        return func
    else:
        raise ValueError("Phasance response is only supported for op_mode='galv' and step_model='ideal'. "
                         f"Received op_mode {op_mode}, step_model {step_model}")


def construct_phasor_z_matrix(frequencies, basis_nu, nu_basis_type, nu_epsilon, normalize=False, tau_c=None):
    omega = 2 * np.pi * frequencies
    nn, ww = np.meshgrid(basis_nu, omega)
    if nu_basis_type == 'delta':
        return unit_phasor_impedance(ww, nn)
    else:
        func = get_phasor_impedance_func(nu_basis_type, normalize=normalize)
        if normalize:
            return func(ww, nn, tau_c, nu_epsilon)
        else:
            return func(ww, nn, nu_epsilon)


def construct_phasor_v_matrix(times, basis_nu, nu_basis_type, nu_epsilon, step_model, step_times, step_sizes,
                              op_mode='galv', normalize=False, tau_c=None):

    rm_layered = np.zeros((len(step_times), len(times), len(basis_nu)))

    for k in range(len(step_times)):
        st = step_times[k]
        sa = step_sizes[k]
        if op_mode == 'galv':
            func = get_phasor_response_func(nu_basis_type, op_mode, step_model, normalize=normalize)
            nn, tt = np.meshgrid(basis_nu, times[times > st] - st)
            if nu_basis_type == 'delta':
                rm_layered[k, times > st] = sa * func(tt, nn)
            else:
                if normalize:
                    rm_layered[k, times > st] = sa * func(tt, nn, tau_c, nu_epsilon)
                else:
                    rm_layered[k, times > st] = sa * func(tt, nn, nu_epsilon)
        else:
            raise ValueError('phasor response is only implemented for galvanostatic mode')

    rm = np.sum(rm_layered, axis=0)

    return rm, rm_layered


# def phasor_scale_vector(nu, basis_tau, aggregate='max'):
#     """
#     Get vector for scaling phasor coefficients
#     :param nu_basis_type:
#     :param nu:
#     :param basis_tau:
#     :return:
#     """
#     freq = 1 / (2 * np.pi * basis_tau)
#
#     zm = construct_phasor_z_matrix(freq, nu, 'delta', None, normalize=False)
#
#     # scale_vector = 1 / np.exp(getattr(np, aggregate)(np.log(np.abs(zm)), axis=0))
#     scale_vector = 1 / getattr(np, aggregate)(np.abs(zm), axis=0)
#
#     return scale_vector


def phasor_scale_vector(nu, basis_tau, quantiles=(0.25, 0.75)):
    """
    Get vector for scaling phasor coefficients
    :param nu:
    :param basis_tau:
    :return:
    """
    lt = np.log(basis_tau)
    lt_max = np.max(lt)
    lt_min = np.min(lt)
    lt_range = lt_max - lt_min
    tau_q1 = np.exp(lt_min + quantiles[0] * lt_range)
    tau_q3 = np.exp(lt_min + quantiles[1] * lt_range)
    # print('phasor_scale_vector tau q1, q3: ({:.2e}, {:.2e})'.format(tau_q1, tau_q3))

    scale_vector = np.empty(len(nu))
    scale_vector[nu <= 0] = tau_q3 ** nu[nu <= 0]
    scale_vector[nu > 0] = tau_q1 ** nu[nu > 0]

    return scale_vector


# def integrate_dop(x_dop, nu, nu_basis_type, tau_neg, tau_pos):

