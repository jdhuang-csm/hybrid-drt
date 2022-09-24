import time
import warnings

import numpy as np

from .. import utils, preprocessing as pp
from .basis import evaluate_basis_fit, get_basis_func, get_basis_func_derivative, get_basis_func_integral, \
    construct_func_eval_matrix
from .mat1d import construct_integrated_derivative_matrix, \
    construct_impedance_matrix


def get_2d_basis_func(basis_type_1, basis_type_2):
    """

    :param basis_type_1:
    :param basis_type_2:
    :return:
    """
    phi_1 = get_basis_func(basis_type_1)
    phi_2 = get_basis_func(basis_type_2)

    def phi(y1, y2, epsilon_1, epsilon_2):
        return phi_1(y1, epsilon_1) * phi_2(y2, epsilon_2)

    return phi


def get_2d_basis_grid(basis_tau, basis_psi):
    """
    Construct tau and psi grids
    Ordering: (tau_1, psi_1), (tau_2, psi_1), ..., (tau_M, psi_1), (tau_1, psi_2), ...
    """
    # Loop through tau values P times
    tau_grid = np.repeat(basis_tau, len(basis_psi)).reshape(len(basis_tau), len(basis_psi)).T.flatten()
    # print('tau_grid:', tau_grid)
    # Repeat each psi value M times
    psi_grid = np.repeat(basis_psi, len(basis_tau))
    # print('psi_grid:', psi_grid)

    return tau_grid, psi_grid


def get_2d_mapped_basis_func(tau_basis_type, psi_basis_type, time_basis_type, basis_times, psi_map_coef):
    """
    Get psi basis function mapped to time domain: phi(tau, psi) -> phi_tilde(tau, t).
    For visualization and validation purposes only - not used in get_2d_response_func for efficiency reasons.
    :param str tau_basis_type: basis type for tau
    :param str psi_basis_type: basis type for psi
    :param str time_basis_type: basis type for time
    :param ndarray basis_times: array of basis times
    :param ndarray psi_map_coef: array of coefficients for mapping psi to time domain
    :return:
    """
    phi_basis = get_2d_basis_func(tau_basis_type, psi_basis_type)
    if time_basis_type == 'step':
        # Create mapped basis function: phi(tau, psi) -> phi_tilde(tau, t)
        def phi_tilde(t, y1, psi_p, epsilon_1, epsilon_2, time_epsilon):
            """
            Calculate basis function phi as a function of time
            :param float t: time
            :param float y1: ln(tau/tau_m)
            :param float psi_p: psi basis value
            :param float epsilon_1: epsilon for tau dimension
            :param float epsilon_2: epsilon for psi dimension
            :param float time_epsilon: epsilon for time dimension
            :return:
            """
            # psi_hat = np.array([unit_step(t, t_k) for t_k in basis_times]).T @ psi_map_coef
            psi_hat = evaluate_basis_fit(psi_map_coef, t, basis_times, time_basis_type, time_epsilon)
            return phi_basis(y1, psi_hat - psi_p, epsilon_1, epsilon_2)

    return phi_tilde


def get_2d_response_func(tau_basis_type, psi_basis_type, independent_measurements, psi_static, psi_is_time, psi_is_i,
                         time_basis_type, basis_times, time_epsilon, psi_map_coef,
                         op_mode, step_model, vectorization_level, separate_basis_times=False):
    """
    Get integrand function for 2d response matrix
    :param str tau_basis_type: type of basis function for tau. Options: 'gaussian', 'Cole-Cole', 'Zic'
    :param str psi_basis_type: type of basis function for psi. Options: 'gaussian'
    :param bool independent_measurements: if True, measurements are independent and unaffected by each other.
    If False, later measurements are affected by earlier measurements.
    :param bool psi_static: if True, psi is static within each measurement.
    If False, psi changes over time within each measurement.
    :param bool psi_is_time: if True, psi is time. If False, psi represents some other quantity.
    :param str time_basis_type: type of basis function for time. Only used when psi must be mapped to time.
    :param ndarray basis_times: array of basis times. Only used when psi must be mapped to time.
    :param ndarray psi_map_coef: coefficients of basis function fit of psi(t). Only used when psi must be mapped to
    time.
    :param str op_mode: Operation mode ('galvanostatic' or 'potentiostatic')
    :param str step_model: model for signal step to use. Options: 'ideal', 'expdecay'
    :return: integrand function
    """
    utils.check_op_mode(op_mode)
    utils.check_basis_type(tau_basis_type)
    utils.check_basis_type(psi_basis_type)
    # check_basis_type(time_basis_type)

    if op_mode != 'galvanostatic':
        raise ValueError('get_2d_response_func only implemented for galvanostatic mode')

    phi_basis = get_2d_basis_func(tau_basis_type, psi_basis_type)

    if psi_is_time:
        pass
        # Complete later
    else:
        if independent_measurements and psi_static:
            # Case 1
            # Get function for time response of ideal RC element
            if step_model == 'ideal':
                def rc_func(y, tau_m, t_n, st, tau_rise):
                    """
                    Voltage response of ideal RC element to ideal current step
                    :param float y: ln(tau/tau_m)
                    :param float tau_m: basis time constant
                    :param flaot t_n: measurement time
                    :param float st: step time
                    :param flaot tau_rise: Unused - included for compatibility only
                    :return:
                    """
                    # tau_rise unused - included for compatibility only
                    return 1 - np.exp(-(t_n - st) / (tau_m * np.exp(y)))
            elif step_model == 'expdecay':
                def rc_func(y, tau_m, t_n, st, tau_rise):
                    """
                    Voltage response of ideal RC element to finite current step
                    :param float y: ln(tau/tau_m)
                    :param float tau_m: basis time constant
                    :param flaot t_n: measurement time
                    :param float st: step time
                    :param flaot tau_rise: signal rise time
                    :return:
                    """
                    a = 1 / (1 - np.exp(y) * tau_m / tau_rise)
                    return 1 + a * np.exp(-(t_n - st) / tau_rise) - (1 + a) * np.exp(
                        -(t_n - st) / (tau_m * np.exp(y)))

            def func(y1, y2, tau_m, psi_p, t_n, st, epsilon_1, epsilon_2, tau_rise):
                """
                Response of basis function mp to integrate over y1
                :param y1: ln(tau/tau_m)
                :param y2: psi - psi_p
                :param tau_m: basis time constant
                :param psi_p: unused - included for compatibility only
                :param t_n: measurement time
                :param st: step time
                :param epsilon_1: epsilon for tau basis
                :param epsilon_2: epsilon for psi basis
                :param tau_rise: signal rise time. Only used for expdecay step_model
                :return:
                """
                return phi_basis(y1, y2, epsilon_1, epsilon_2) * rc_func(y1, tau_m, t_n, st, tau_rise)
        elif not independent_measurements:
            # =========================================================
            # Case 2 - dependent measurements and/or time-dependent psi
            # =========================================================
            if psi_is_i:
                # ======================================
                # Case 2b - Special case when psi == i
                # ======================================
                if step_model == 'ideal':
                    # Get function for time response of ideal RC element
                    def rc_func(y, tau_m, t_n, st, tau_rise):
                        """
                        Voltage response of ideal RC element to ideal current step
                        :param float y: ln(tau/tau_m)
                        :param float tau_m: basis time constant
                        :param flaot t_n: measurement time
                        :param float st: step time
                        :param flaot tau_rise: Unused - included for compatibility only
                        :return:
                        """
                        # tau_rise unused - included for compatibility only
                        return 1 - np.exp(-(t_n - st) / (tau_m * np.exp(y)))

                    # Get separate basis functions for tau and psi
                    # phi_psi must be integrated over psi (current)
                    phi_psi_int = get_basis_func_integral(psi_basis_type)
                    phi_tau = get_basis_func(tau_basis_type)

                    def func(y1, tau_m, psi_p, i_0, i_1, t_n, st, epsilon_1, epsilon_2, tau_rise):
                        return phi_tau(y1, epsilon_1) \
                               * (phi_psi_int(i_1 - psi_p, epsilon_2) - phi_psi_int(i_0 - psi_p, epsilon_2)) \
                               * rc_func(y1, tau_m, t_n, st, tau_rise)

                elif step_model == 'expdecay':
                    raise ValueError('Response func not implemented for case psi_is_i==True '
                                     'with expdecay step model')

            else:
                # ======================================
                # Case 2a - General case
                # ======================================
                if step_model == 'ideal':
                    # Get function for time response of ideal RC element
                    if vectorization_level == '1d':
                        def rc_func(y, tau_m, t_meas, t_k, st, tau_rise):
                            """
                            Voltage response of ideal RC element to ideal current step
                            :param ndarray y: ln(tau/tau_m)
                            :param float tau_m: basis time constant
                            :param flaot t_k: basis function start time
                            :param flaot t_meas: measurement time
                            :param float st: current step time
                            :param flaot tau_rise: Unused - included for compatibility only
                            :return:
                            """
                            if type(t_k) == np.ndarray:
                                start_time = np.maximum(t_k, st)
                            else:
                                start_time = max(t_k, st)
                            # if t_meas >= start_time:
                            return 1 - np.exp(-(t_meas - start_time) / (tau_m * np.exp(y)))
                            # else:
                            #     return 0
                    elif vectorization_level in ('2d', '3d'):
                        def rc_func(y, tau_m, t_meas, t_k, st, tau_rise):
                            """
                            Voltage response of ideal RC element to ideal current step
                            :param ndarray y: ln(tau/tau_m)
                            :param float tau_m: basis time constant
                            :param flaot t_k: basis function start time
                            :param flaot t_meas: measurement time
                            :param float st: current step time
                            :param flaot tau_rise: Unused - included for compatibility only
                            :return:
                            """
                            start_time = np.maximum(t_k, st)
                            return 1 - np.exp(-(t_meas - start_time) / (tau_m * np.exp(y)))

                elif step_model == 'expdecay':
                    def rc_func(y, tau_m, t_nume, t_deno, st, tau_rise):
                        """
                        Voltage response of ideal RC element to finite current step
                        :param ndarray y: ln(tau/tau_m)
                        :param float tau_m: basis time constant
                        :param flaot t_nume: time for numerator - integration limit
                        :param flaot t_deno: time for denominator - measurement time
                        :param float st: step time
                        :param flaot tau_rise: signal rise time
                        :return:
                        """
                        a = 1 / (1 - np.exp(y) * tau_m / tau_rise)
                        return np.exp((t_nume - t_deno) / (tau_m * np.exp(y))) \
                               + a * np.exp(
                            (t_nume - t_deno) / (tau_m * np.exp(y)) - (t_nume - t_deno) / tau_rise
                        )
                        # if t_deno >= st:
                        #     a = 1 / (1 - np.exp(y) * tau_m / tau_rise)
                        #     return np.exp((t_nume - t_deno) / (tau_m * np.exp(y))) \
                        #            + a * np.exp(
                        #         (t_nume - t_deno) / (tau_m * np.exp(y)) - (t_nume - t_deno) / tau_rise
                        #     )
                        # else:
                        #     return np.zeros_like(y)

                if time_basis_type == 'step':
                    phi_basis = get_2d_basis_func(tau_basis_type, psi_basis_type)

                    # Create array of next basis times for convenience
                    next_basis_time = np.zeros(len(basis_times))
                    next_basis_time[:-1] = basis_times[1:]
                    next_basis_time[-1] = np.inf

                    # Pre-calculate psi_hat at basis times
                    psi_eval_times = np.concatenate(([-np.inf], basis_times))
                    psi_hat_basis = evaluate_basis_fit(psi_map_coef, psi_eval_times,
                                                       basis_times, time_basis_type, time_epsilon)

                    if vectorization_level == '1d':
                        # def func(y1, y2, tau_m, psi_p, t_n, st, epsilon_1, epsilon_2, tau_rise):
                        #     """
                        #     Response of basis function mp to integrate over y1
                        #     :param ndarray y1: ln(tau/tau_m)
                        #     :param y2: unused, included for compatibility only
                        #     :param float tau_m: basis time constant
                        #     :param float psi_p: basis psi value
                        #     :param float t_n: measurement time
                        #     :param float st: step time
                        #     :param float epsilon_1: epsilon for tau basis
                        #     :param float epsilon_2: epsilon for psi basis
                        #     :param float tau_rise: signal rise time. Only used for expdecay step_model
                        #     :return:
                        #     """
                        #     # Get end time for each segment
                        #     end_time = np.array([min(t_n, nbt) for nbt in next_basis_time])
                        #
                        #     f_k = np.array(
                        #         [phi_basis(y1, psi_hat_basis[k] - psi_p, epsilon_1, epsilon_2)
                        #          * (
                        #                  rc_func(y1, tau_m, end_time[k], t_n, st, tau_rise)
                        #                  - rc_func(y1, tau_m, basis_times[k], t_n, st, tau_rise)
                        #          )
                        #          for k in range(len(basis_times)) if t_n >= basis_times[k]]
                        #     )
                        #
                        #     return np.sum(f_k, axis=0)  # / np.exp((t_n - st) / tau)

                        if separate_basis_times:
                            # Keep responses to different basis times separate - this allows us to use step ratios to
                            # quickly calculate responses to later steps
                            def func(y1, y2, tau_m, psi_p, t_n, st, epsilon_1, epsilon_2, tau_rise):
                                """
                                Response of basis function mp to integrate over y1
                                :param ndarray y1: ln(tau/tau_m)
                                :param y2: unused, included for compatibility only
                                :param float tau_m: basis time constant
                                :param float psi_p: basis psi value
                                :param float t_n: measurement time
                                :param float st: step time
                                :param float epsilon_1: epsilon for tau basis
                                :param float epsilon_2: epsilon for psi basis
                                :param float tau_rise: signal rise time. Only used for expdecay step_model
                                :return:
                                """
                                # Evaluate phi_mp at each basis time
                                phi_mp_basis = np.array([phi_basis(y1, psi_hat_basis[k] - psi_p, epsilon_1, epsilon_2)
                                                         for k in range(len(psi_hat_basis))])

                                # Get steps in phi_mp
                                phi_mp_steps = np.diff(phi_mp_basis, axis=0)

                                phi_mag = np.concatenate(([phi_mp_basis[0]], phi_mp_steps))

                                # Sum all basis function steps that occurred at or before measurement time
                                f_k = np.zeros((len(psi_eval_times), len(y1)))
                                # print(f_k.shape)
                                y1_mesh, pet_mesh = np.meshgrid(y1, psi_eval_times)
                                # print(y1_mesh.shape)

                                f_k[t_n >= pet_mesh] = np.array(
                                    phi_mag[t_n >= pet_mesh] * rc_func(
                                        y1_mesh[t_n >= pet_mesh], tau_m, t_n, pet_mesh[t_n >= pet_mesh], st, tau_rise
                                    )
                                )

                                return f_k
                        else:
                            # Sum responses to separate basis steps - uses less memory, but doesn't allow us to use
                            # step ratio calculation (unless delta transform is used)
                            def func(y1, y2, tau_m, psi_p, t_n, st, epsilon_1, epsilon_2, tau_rise):
                                """
                                Response of basis function mp to integrate over y1
                                :param ndarray y1: ln(tau/tau_m)
                                :param y2: unused, included for compatibility only
                                :param float tau_m: basis time constant
                                :param float psi_p: basis psi value
                                :param float t_n: measurement time
                                :param float st: step time
                                :param float epsilon_1: epsilon for tau basis
                                :param float epsilon_2: epsilon for psi basis
                                :param float tau_rise: signal rise time. Only used for expdecay step_model
                                :return:
                                """
                                # Evaluate phi_mp at each basis time
                                phi_mp_basis = np.array([phi_basis(y1, psi_hat_basis[k] - psi_p, epsilon_1, epsilon_2)
                                                for k in range(len(psi_hat_basis))])

                                # Get steps in phi_mp
                                phi_mp_steps = np.diff(phi_mp_basis, axis=0)

                                phi_mag = np.concatenate(([phi_mp_basis[0]], phi_mp_steps))

                                # Sum all basis function steps that occurred at or before measurement time
                                f_k = np.array(
                                    [phi_mag[k] * rc_func(y1, tau_m, t_n, psi_eval_times[k], st, tau_rise)
                                     for k in range(len(psi_eval_times)) if t_n >= psi_eval_times[k]]
                                )

                                return np.sum(f_k, axis=0)

                    elif vectorization_level == '2d':
                        # TODO: update 2d vectorized function
                        def func(y1, y2, tau_m, psi_p, t_n, st, epsilon_1, epsilon_2, tau_rise, q_calc_index):
                            """
                            Response of basis function mp to integrate over y1
                            :param ndarray y1: ln(tau/tau_m)
                            :param y2: unused, included for compatibility only
                            :param array tau_m: basis time constant
                            :param array psi_p: basis psi value
                            :param float t_n: measurement time
                            :param float st: step time
                            :param float epsilon_1: epsilon for tau basis
                            :param float epsilon_2: epsilon for psi basis
                            :param float tau_rise: signal rise time. Only used for expdecay step_model
                            :param ndarray q_calc_index: index indicating for which q indices the function should be
                            calculated
                            :return:
                            """
                            M = len(y1)  # integration dimension
                            Q = len(tau_m)  # same as len(psi_p)
                            K = len(basis_times)

                            # Create K x M x Q arrays
                            # -----------------------
                            # Get end time for each segment
                            nbt_kmq = np.tile(next_basis_time, (Q, M, 1)).T
                            end_time_kmq = np.minimum(nbt_kmq, t_n)

                            # Make basis_times array
                            bt_kmq = np.tile(basis_times, (Q, M, 1)).T

                            # Make psi_hat_basis array
                            phb_kmq = np.tile(psi_hat_basis, (Q, M, 1)).T

                            # Create y1, tau/psi mesh
                            tt, yy1 = np.meshgrid(tau_m, y1)
                            pp, yy1 = np.meshgrid(psi_p, y1)

                            # repeat mesh along k-axis
                            tau_kmq = np.tile(tt, (K, 1, 1))
                            psi_kmq = np.tile(pp, (K, 1, 1))
                            y1_kmq = np.tile(yy1, (K, 1, 1))

                            f_kmq = np.zeros((K, M, Q))

                            # Broadcast q_calc_index into kmq array
                            q_calc_flag = np.zeros_like(f_kmq, dtype=bool)
                            q_calc_flag[:, :, q_calc_index] = True

                            calc_index = np.where((t_n >= bt_kmq) & (q_calc_flag == True))

                            f_kmq[calc_index] = \
                                phi_basis(y1_kmq[calc_index], (phb_kmq - psi_kmq)[calc_index], epsilon_1, epsilon_2) \
                                * (
                                        rc_func(y1_kmq[calc_index], tau_kmq[calc_index], end_time_kmq[calc_index], t_n,
                                                st, tau_rise)
                                        - rc_func(y1_kmq[calc_index], tau_kmq[calc_index], bt_kmq[calc_index], t_n, st,
                                                  tau_rise)
                                )

                            return np.sum(f_kmq, axis=0)

                    elif vectorization_level == '3d':
                        # def func(y1, y2, tau_grid, psi_grid, times, st, epsilon_1, epsilon_2, tau_rise, nq_calc_index):
                        #     """
                        #     Response of basis function mp to integrate over y1
                        #     :param ndarray y1: ln(tau/tau_m)
                        #     :param ndarray y2: unused, included for compatibility only
                        #     :param ndarray tau_grid: basis time constant
                        #     :param ndarray psi_grid: basis psi value
                        #     :param ndarray times: measurement time
                        #     :param float st: step time
                        #     :param float epsilon_1: epsilon for tau basis
                        #     :param float epsilon_2: epsilon for psi basis
                        #     :param float tau_rise: signal rise time. Only used for expdecay step_model
                        #     :return:
                        #     """
                        #
                        #     M = len(y1)  # integration dimension
                        #     Q = len(tau_grid)  # same as len(psi_p)
                        #     K = len(basis_times)  # sum dimension
                        #     N = len(times)
                        #
                        #     # Create K x M x N x Q arrays
                        #     # -----------------------
                        #     # Tile k-vectors into arrays
                        #     nbt_kmnq = np.tile(next_basis_time, (Q, N, M, 1)).T
                        #     bt_kmnq = np.tile(basis_times, (Q, N, M, 1)).T
                        #     phb_kmnq = np.tile(psi_hat_basis, (Q, N, M, 1)).T
                        #
                        #     # Create y1, times, tau/psi mesh
                        #     t_mesh, y1_mesh, tau_mesh = np.meshgrid(times, y1, tau_grid)
                        #     t_mesh, y1_mesh, psi_mesh = np.meshgrid(times, y1, psi_grid)
                        #     # print(t_mesh.shape)
                        #
                        #     # repeat mesh along k-axis
                        #     tau_kmnq = np.tile(tau_mesh, (K, 1, 1, 1))
                        #     psi_kmnq = np.tile(psi_mesh, (K, 1, 1, 1))
                        #     y1_kmnq = np.tile(y1_mesh, (K, 1, 1, 1))
                        #     t_kmnq = np.tile(t_mesh, (K, 1, 1, 1))
                        #
                        #     # Get end time for each segment
                        #     end_time_kmnq = np.minimum(nbt_kmnq, t_kmnq)
                        #
                        #     f_kmnq = np.zeros((K, M, N, Q))
                        #
                        #     # Broadcast q_calc_index into kmnq array
                        #     if nq_calc_index is not None:
                        #         n_index, q_index = nq_calc_index
                        #         nq_calc_flag = np.zeros_like(f_kmnq, dtype=bool)
                        #         nq_calc_flag[:, :, n_index, q_index] = True
                        #     else:
                        #         nq_calc_flag = np.ones_like(f_kmnq, dtype=bool)
                        #
                        #     calc_index = np.where((t_kmnq >= bt_kmnq) & (nq_calc_flag == True))
                        #     f_kmnq[calc_index] = \
                        #         phi_basis(y1_kmnq[calc_index], (phb_kmnq - psi_kmnq)[calc_index], epsilon_1,
                        #                   epsilon_2) \
                        #         * (
                        #                 rc_func(y1_kmnq[calc_index], tau_kmnq[calc_index], end_time_kmnq[calc_index],
                        #                         t_kmnq[calc_index], st, tau_rise)
                        #                 - rc_func(y1_kmnq[calc_index], tau_kmnq[calc_index], bt_kmnq[calc_index],
                        #                           t_kmnq[calc_index], st, tau_rise)
                        #         )
                        #
                        #     return np.sum(f_kmnq, axis=0)
                        def func(y1, y2, tau_grid, psi_grid, times, st, epsilon_1, epsilon_2, tau_rise, nq_calc_index):
                            """
                            Response of basis function mp to integrate over y1
                            :param ndarray y1: ln(tau/tau_m)
                            :param ndarray y2: unused, included for compatibility only
                            :param ndarray tau_grid: basis time constant
                            :param ndarray psi_grid: basis psi value
                            :param ndarray times: measurement time
                            :param float st: step time
                            :param float epsilon_1: epsilon for tau basis
                            :param float epsilon_2: epsilon for psi basis
                            :param float tau_rise: signal rise time. Only used for expdecay step_model
                            :return:
                            """

                            M = len(y1)  # integration dimension
                            Q = len(tau_grid)  # same as len(psi_p)
                            K = len(psi_eval_times)  # sum dimension
                            N = len(times)

                            # Create K x M x N x Q arrays
                            # -----------------------
                            # Tile k-vectors into arrays
                            # nbt_kmnq = np.tile(next_basis_time, (Q, N, M, 1)).T
                            pet_kmnq = np.tile(psi_eval_times, (Q, N, M, 1)).T

                            # Create y1, times, tau/psi mesh
                            t_mesh, y1_mesh, tau_mesh = np.meshgrid(times, y1, tau_grid)
                            t_mesh, y1_mesh, psi_mesh = np.meshgrid(times, y1, psi_grid)
                            # print(t_mesh.shape)

                            # repeat mesh along k-axis
                            tau_kmnq = np.tile(tau_mesh, (K, 1, 1, 1))
                            psi_kmnq = np.tile(psi_mesh, (K, 1, 1, 1))
                            y1_kmnq = np.tile(y1_mesh, (K, 1, 1, 1))
                            t_kmnq = np.tile(t_mesh, (K, 1, 1, 1))

                            # Get end time for each segment
                            # end_time_kmnq = np.minimum(nbt_kmnq, t_kmnq)

                            f_kmnq = np.zeros((K, M, N, Q))

                            # Evaluate phi_mp at each basis time (and -inf time, which requires K + 1 layers)
                            # y1_k1mnq = np.tile(y1_mesh, (K + 1, 1, 1, 1))
                            # psi_k1mnq = np.tile(psi_mesh, (K + 1, 1, 1, 1))
                            phb_kmnq = np.tile(psi_hat_basis, (Q, N, M, 1)).T
                            phi_kmnq = phi_basis(y1_kmnq, phb_kmnq - psi_kmnq, epsilon_1, epsilon_2)

                            # Get steps in phi_mp
                            dphi_kmnq = np.zeros(phi_kmnq.shape)
                            dphi_kmnq[0] = phi_kmnq[0]
                            dphi_kmnq[1:] = np.diff(phi_kmnq, axis=0)

                            # Broadcast q_calc_index into kmnq array
                            if nq_calc_index is not None:
                                n_index, q_index = nq_calc_index
                                nq_calc_flag = np.zeros_like(f_kmnq, dtype=bool)
                                nq_calc_flag[:, :, n_index, q_index] = True
                            else:
                                nq_calc_flag = np.ones_like(f_kmnq, dtype=bool)

                            calc_index = np.where((t_kmnq >= pet_kmnq) & (nq_calc_flag == True))
                            f_kmnq[calc_index] = \
                                dphi_kmnq[calc_index] \
                                * rc_func(y1_kmnq[calc_index], tau_kmnq[calc_index], t_kmnq[calc_index],
                                          pet_kmnq[calc_index], st, tau_rise)

                            return np.sum(f_kmnq, axis=0)

    return func


def construct_2d_response_matrix(basis_tau, times, psi_exp, basis_psi,  # measurement_start_indices,
                                 independent_measurements, psi_static, psi_is_time, psi_is_i,
                                 step_model, step_times, step_sizes,
                                 tau_basis_type='gaussian', tau_epsilon=0.975, psi_basis_type='gaussian', psi_epsilon=1,
                                 time_basis_type=None, basis_times=None, time_epsilon=None, psi_map_coef=None,
                                 tau_rise=None, op_mode='galvanostatic',
                                 integrate_method='trapz', integrate_points=250, rtol=1e-5, vectorization_level='1d',
                                 use_delta_transform=False, correct_final_psi=False, use_step_ratios=True
                                 ):
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
    :param str op_mode: Operation mode ('galvanostatic' or 'potentiostatic')
    :param str integrate_method: method for numerical evaluation of integrals. Options: 'trapz', 'quad'
    :param integrate_points:
    :param bool use_delta_transform: if True, use the change of variables xt_mp = x_mp - x_m(p-1) to speed up matrix
    calculation (if it can be applied)
    :param bool correct_final_psi: if True, when using delta transform, recalculate values for final basis_psi so that
    they match the results without the delta transform. If False, do not recalculate; in this case, the final basis
    function in psi extends to infinity.
    Only applies when use_delta_transform=True.
    :param bool use_step_ratios: if True, reuse responses to first step to avoid redundant calculations for responses to
    subsequent steps. Incompatible with use_delta_transform until issue with phi_mag==0 is fixed
    :return: matrix A, such that A@x gives the response
    """
    utils.check_step_model(step_model)
    utils.check_op_mode(op_mode)
    utils.check_basis_type(tau_basis_type)
    utils.check_basis_type(psi_basis_type)

    if op_mode != 'galvanostatic':
        raise ValueError('construct_2d_response_matrix is only implemented for galvanostatic mode')

    A_layered = np.zeros([len(step_times), len(times), len(basis_tau) * len(basis_psi)])

    if tau_rise is None:
        tau_rise = np.zeros(len(step_times))

    # Check if basis_times are collocated with step_times
    if utils.check_equality(utils.rel_round(basis_times, 10), utils.rel_round(step_times, 10)):
        basis_times_are_step_times = True
    else:
        basis_times_are_step_times = False

    # Check if step ratios can be used to speed up matrix calculation
    if use_step_ratios:
        if not basis_times_are_step_times:
            use_step_ratios = False

    # Check if delta transform is available to speed up matrix calculation
    if use_delta_transform:
        use_delta_transform = is_delta_transform_available(basis_psi, psi_epsilon, psi_basis_type, psi_map_coef)

    if use_delta_transform and use_step_ratios:
        warnings.warn('Step ratios are not compatible with delta transform. Since step ratios typically provide greater'
                      ' efficiency, calculation will proceed with step ratios.')

    if use_delta_transform:
        print('Delta transform available')
        # Check monotonicity and direction of psi_hat
        # Check psi_map_coef for monotonicity rather than evaluating psi_hat at all times
        # First coefficient is intercept - doesn't have to have same sign as subsequent coefs
        if utils.is_monotonic_ascending(np.cumsum(-psi_map_coef)) and utils.is_monotonic_ascending(-basis_psi):
            # if psi is monotonically decreasing, must invert psi, basis_psi, and psi_map_coef
            psi_exp = psi_exp * (-1)
            basis_psi = basis_psi * (-1)
            psi_map_coef = psi_map_coef * (-1)

        # Select transformed basis type
        if psi_basis_type == 'pw_linear':
            transformed_basis_type = 'pwl_transformed'
        else:
            raise ValueError('Delta transform can only be used for when psi_basis_type == pw_linear')

    # Track number of entries calculated
    num_calculated = 0

    # # Get sample period
    # t_sample = np.min(np.diff(times))

    # Calculate response to each signal step
    for k in range(len(step_times)):
        st = step_times[k]
        sa = step_sizes[k]
        # Get time of first post-step sample
        first_sample_time = times[times > st][0]

        # all times before step are unaffected
        A_layered[k, times < st, :] = 0

        # calculate integrals for times after step
        if psi_is_time:
            pass
        else:
            # if independent_measurements and psi_static:
            if integrate_method == 'trapz':
                y1 = np.linspace(-5, 5, integrate_points)

                # Define tau and psi grids
                # Ordering: (tau_1, psi_1), (tau_2, psi_1), ..., (tau_M, psi_1), (tau_1, psi_2), ...
                tau_grid, psi_grid = get_2d_basis_grid(basis_tau, basis_psi)

                if not independent_measurements:
                    # ==========================================================
                    # Case 2 - dependent measurements and/or time-dependent psi
                    # ==========================================================
                    if psi_is_i:
                        # ======================================
                        # Case 2b - Special case when psi == i
                        # ======================================
                        # Get start and end current for this step
                        i_1 = evaluate_basis_fit(psi_map_coef, st, basis_times, time_basis_type, time_epsilon)
                        i_0 = i_1 - sa
                        print(i_0, i_1)

                        # Get function to integrate
                        func = get_2d_response_func(tau_basis_type, psi_basis_type, independent_measurements,
                                                    psi_static,
                                                      psi_is_time, psi_is_i, time_basis_type, basis_times,
                                                      time_epsilon,
                                                      psi_map_coef,
                                                      op_mode, step_model, '1d')

                        # Check magnitude of function at y1 = 0
                        start = time.time()
                        tau_mesh, time_mesh = np.meshgrid(tau_grid, times[times >= st])
                        psi_mesh, time_mesh = np.meshgrid(psi_grid, times[times >= st])
                        func_mag = np.zeros_like(A_layered[k])
                        func_mag[times >= st] = np.abs(np.nan_to_num(
                            func(0, tau_mesh, psi_mesh, i_0, i_1, time_mesh, st, tau_epsilon, psi_epsilon, tau_rise[k]),
                            0)
                        )
                        max_mag = np.max(func_mag)
                        post_step = func_mag[times >= st]
                        print('func_mag eval time: {:.2f}'.format(time.time() - start))
                        print('Num after step:', post_step.shape[0] * post_step.shape[1])
                        print('Num below rtol:', len(post_step[np.where(post_step < rtol * max_mag)]))
                        print('Frac below rtol: {:.3f}'.format(
                            len(post_step[np.where(post_step < rtol * max_mag)]) / (post_step.shape[0] * post_step.shape[1])
                        ))

                        # For all vectorization options, only perform calculation if:
                        # (1) time >= step time and
                        # (2) function magnitude at y1=0 exceeds rtol
                        if vectorization_level == '1d':
                            basis_tau_mesh, psi_exp_mesh = np.meshgrid(tau_grid, psi_exp)
                            basis_psi_mesh, time_mesh = np.meshgrid(psi_grid, times)
                            # identify entries to calculate
                            # Only calculate if func_mag exceeds rtol AND
                            # (
                            #   time constant is not fully relaxed
                            #   OR time is first post-step sample time*
                            # )
                            # *Need to make sure we calculate at least 1 entry for very short time constants
                            calc_index = np.where(
                                (func_mag > max_mag * rtol) &
                                (
                                    ((time_mesh - st) / basis_tau_mesh <= 11.5) |
                                    (time_mesh == first_sample_time)
                                )
                            )
                            fill_index = np.where(
                                (func_mag > max_mag * rtol) &
                                ((time_mesh - st) / basis_tau_mesh > 11.5) &
                                (time_mesh > first_sample_time)
                            )
                            print('Num to calculate:', len(calc_index[0]))
                            print('Num to fill:', len(fill_index[0]))
                            num_calculated += len(calc_index[0])

                            A_layered[k, calc_index[0], calc_index[1]] = [
                                np.trapz(
                                    np.nan_to_num(
                                        func(y1, tau_mp, psi_mp, i_0, i_1, t_n, st,
                                             tau_epsilon, psi_epsilon, tau_rise[k]),
                                        0),
                                    x=y1)
                                for tau_mp, psi_mp, t_n in
                                zip(basis_tau_mesh[calc_index], basis_psi_mesh[calc_index], time_mesh[calc_index])
                            ]

                            # Fill in end values for fully relaxed time constants
                            # Each basis_psi value will have a different end value
                            # I *think* different basis_tau values shouldn't have different end values for same basis_psi
                            start = time.time()
                            for col_index in np.unique(fill_index[1]):
                                col = A_layered[k, :, col_index].flatten()
                                nonzero_index = np.nonzero(col)[0]
                                # if len(nonzero_index) > 0:
                                # Get final value in column and fill to all subsequent values
                                col_end_index = nonzero_index[-1]
                                # print(col_end_index)
                                A_layered[k, col_end_index + 1:, col_index] = col[col_end_index]
                                # else:
                                #     # If all values are zero, this tau/psi combination has no time-domain response
                                #     # (may happen
                                #     pass

                            print('fill time: {:.2f}'.format(time.time() - start))


                        # elif vectorization_level == '2d':
                        #     row_index, col_index = np.where(func_mag > rtol)
                        #     row_calc_indices = np.unique(row_index)
                        #     col_calc_indices = [col_index[row_index == ri] for ri in row_calc_indices]
                        #     A_layered[k, row_calc_indices] = [
                        #         np.trapz(
                        #             np.nan_to_num(
                        #                 func(y1, p_n - psi_grid, tau_grid, psi_grid, t_n, st, tau_epsilon, psi_epsilon,
                        #                      tau_rise[k], cci_n),
                        #                 0),
                        #             x=y1, axis=0) * sa
                        #         for t_n, p_n, cci_n in zip(times[row_calc_indices], psi_exp[row_calc_indices],
                        #                                    col_calc_indices)
                        #     ]
                        # elif vectorization_level == '3d':
                        #     # Limit input to times after step time - array creation and summation is still costly even if
                        #     # func is not evaluated
                        #     calc_index = np.where(post_step > rtol)
                        #
                        #     A_layered[k, times >= st] = np.trapz(
                        #         np.nan_to_num(
                        #             func(y1, None, tau_grid, psi_grid, times[times >= st],
                        #                  st, tau_epsilon, psi_epsilon, tau_rise[k], calc_index),
                        #             0),
                        #         x=y1, axis=0) * sa

                        print(k, st, sa, np.sum(A_layered[k]))
                    else:
                        # ======================================
                        # General case (2a)
                        # ======================================
                        if use_delta_transform:
                            # -----------------------------------------
                            # Delta transform implementation
                            # -----------------------------------------
                            # TODO: make this work with step ratios. See note in calculate_step_ratio for why this
                            #  doesn't work currently
                            # Use fully vectorized function to check magnitude of function
                            check_func = get_2d_response_func(
                                tau_basis_type, transformed_basis_type, independent_measurements, psi_static, psi_is_time,
                                psi_is_i,
                                time_basis_type, basis_times, time_epsilon, psi_map_coef, op_mode, step_model, '3d'
                            )

                            start = time.time()
                            func_mag = np.zeros_like(A_layered[k])
                            func_mag[times >= st] = np.nan_to_num(
                                check_func([0], None, tau_grid, psi_grid, times[times >= st], st,
                                           tau_epsilon, psi_epsilon, tau_rise[k], None),
                                0)
                            post_step = func_mag[times >= st]
                            print('func_mag eval time: {:.2f}'.format(time.time() - start))
                            print('Num after step:', post_step.shape[0] * post_step.shape[1])
                            print('Num below rtol:', len(post_step[np.where(post_step < rtol)]))
                            print('Frac below rtol: {:.3f}'.format(
                                len(post_step[np.where(post_step < rtol)]) / (post_step.shape[0] * post_step.shape[1])
                            ))

                            # Get function at desired vectorization level for full calculation
                            # If using step ratios, must keep responses to different basis times separated
                            func = get_2d_response_func(
                                tau_basis_type, transformed_basis_type, independent_measurements, psi_static, psi_is_time,
                                psi_is_i,
                                time_basis_type, basis_times, time_epsilon, psi_map_coef, op_mode, step_model,
                                vectorization_level, separate_basis_times=use_step_ratios
                            )
                            if use_step_ratios:
                                end_calc_func = get_2d_response_func(
                                    tau_basis_type, transformed_basis_type, independent_measurements, psi_static, psi_is_time,
                                    psi_is_i,
                                    time_basis_type, basis_times, time_epsilon, psi_map_coef, op_mode, step_model,
                                    vectorization_level, separate_basis_times=False
                                )
                            else:
                                end_calc_func = func
                            # Identify basis_psi values that will have identical responses to this input signal step
                            # Get psi_hat at step time
                            psi_hat_step = evaluate_basis_fit(psi_map_coef, st, basis_times, time_basis_type, time_epsilon)
                            # Get psi basis function values
                            phi_transformed = get_basis_func(transformed_basis_type)
                            phi_p_vals = phi_transformed(psi_hat_step - basis_psi, psi_epsilon)
                            print('phi_p_vals:', phi_p_vals)
                            # Identify basis_psi values that are in full effect (must be fully on to gauarantee that they
                            # won't change later, which would elicit a different response)
                            psi_full_index = np.where(np.abs(phi_p_vals - 1) <= rtol)[0]
                            # psi_full_index = []
                            # Choose first basis_psi value to use for calculations, then duplicate to remaining basis_psi
                            if len(psi_full_index) > 0:
                                psi_lead_index = psi_full_index[0]
                                psi_twin_index = psi_full_index[1:]
                                print('lead, twins:', psi_lead_index, psi_twin_index)
                                # Convert psi indices to matrix column indices
                                if len(psi_twin_index) > 0:
                                    col_twin_index = np.concatenate(
                                        [np.arange(0, len(basis_tau), dtype=int) + p * len(basis_tau) for p in psi_twin_index]
                                    )
                                else:
                                    col_twin_index = []
                                print(col_twin_index)
                            else:
                                psi_lead_index = None
                                psi_twin_index = []
                                col_twin_index = []

                            if vectorization_level == '1d':
                                basis_tau_mesh, psi_exp_mesh = np.meshgrid(tau_grid, psi_exp)
                                basis_psi_mesh, time_mesh = np.meshgrid(psi_grid, times)
                                # Only calculate entries for which:
                                # (1) func_mag > rtol AND
                                # (2) func_end_value - func_mag > rtol (time constant is not fully relaxed)
                                func_end_value = end_calc_func(0, 0, 1e-6, -np.inf, 1e6, 0, tau_epsilon, psi_epsilon, tau_rise[k])
                                print('func_end_value:', func_end_value)
                                calc_index = np.where((func_mag > rtol) & (func_end_value - func_mag > rtol))
                                non_twin_index = np.argwhere(~np.isin(calc_index[1], col_twin_index))
                                row_index = calc_index[0][non_twin_index].flatten()
                                col_index = calc_index[1][non_twin_index].flatten()
                                calc_index = (row_index, col_index)
                                print('Num to calculate:', len(row_index))
                                num_calculated += len(row_index)
                                if use_step_ratios:
                                    f_mp_int = np.zeros((len(times), len(tau_grid), len(basis_times) + 1))
                                    f_mp_int[calc_index[0], calc_index[1]] = [
                                        np.trapz(
                                            np.nan_to_num(
                                                    func(y1, p_n - psi_mp, tau_mp, psi_mp, t_n, st, tau_epsilon, psi_epsilon,
                                                         tau_rise[k]),
                                                    0), x=y1, axis=1
                                        )
                                        for tau_mp, psi_mp, t_n, p_n in zip(
                                            basis_tau_mesh[calc_index], basis_psi_mesh[calc_index], time_mesh[calc_index],
                                            psi_exp_mesh[calc_index]
                                        )
                                    ]
                                    print(f_mp_int.shape)

                                    # f_sum = np.sum(f_mp_int, axis=2)
                                    # print(f_sum.shape)
                                    A_layered[k] = np.sum(f_mp_int, axis=2) * sa
                                    # A_layered[k, calc_index[0], calc_index[1]] = [
                                    #     sa * np.trapz(f_sum[ri, ci, :], x=y1) for ri, ci in zip(calc_index[0], calc_index[1])
                                    # ]
                                else:
                                    A_layered[k, row_index, col_index] = [
                                        np.trapz(
                                            np.nan_to_num(
                                                func(y1, p_n - psi_mp, tau_mp, psi_mp, t_n, st, tau_epsilon, psi_epsilon,
                                                     tau_rise[k]),
                                                0),
                                            x=y1) * sa
                                        for tau_mp, psi_mp, t_n, p_n in zip(basis_tau_mesh[calc_index], basis_psi_mesh[calc_index],
                                                                            time_mesh[calc_index], psi_exp_mesh[calc_index])
                                    ]

                                # Fill in twin values from psi_lead_index
                                for p in psi_twin_index:
                                    A_layered[k, :, p * len(basis_tau): (p + 1) * len(basis_tau)] = \
                                        A_layered[
                                            k, :, psi_lead_index * len(basis_tau): (psi_lead_index + 1) * len(basis_tau)
                                        ]

                                # Fill in constant value for fully relaxed time constants
                                # fill_index = np.where((func_mag > rtol) & (time_mesh - st > 10 * basis_tau_mesh))
                                fill_index = np.where(func_end_value - func_mag <= rtol)
                                print('Num to fill:', len(fill_index[0]))
                                int_func_end_value = np.trapz(
                                    end_calc_func(y1, 0, 1e-6, -np.inf, 1e6, 0, tau_epsilon, psi_epsilon, tau_rise[k]),
                                    x=y1
                                )
                                print('integrated end value:', int_func_end_value * sa)
                                A_layered[k, fill_index[0], fill_index[1]] = int_func_end_value * sa
                        else:
                            # -----------------------------------------
                            # Regular (untransformed) implementation
                            # -----------------------------------------
                            # TODO: make step_ratios robust when phi_mag==0. See note in calculate_step_ratios
                            if k > 0 and basis_times_are_step_times and use_step_ratios:
                                # We can reuse response to the first step for subsequent steps
                                start = time.time()
                                func_mag = np.zeros_like(A_layered[k])
                                func_mag[times >= st] = np.nan_to_num(
                                    check_func([0], None, tau_grid, psi_grid, times[times >= st], st,
                                               tau_epsilon, psi_epsilon, tau_rise[k], None),
                                    0)
                                post_step = func_mag[times >= st]
                                print('func_mag eval time: {:.2f}'.format(time.time() - start))
                                print('Num after step:', post_step.shape[0] * post_step.shape[1])
                                print('Num below rtol:', len(post_step[np.where(post_step < rtol)]))
                                print('Frac below rtol: {:.3f}'.format(
                                    len(post_step[np.where(post_step < rtol)]) / (post_step.shape[0] * post_step.shape[1])
                                ))

                                # Construct step_ratio matrix with same shape as f_mp_int[k:]
                                step_ratio = np.zeros((A_layered.shape[1], A_layered.shape[2], len(step_times) - k))
                                # For basis time corresponding to this step, calculate step ratio for each psi value
                                step_ratio[(times >= st), :, 0] = np.tile(
                                    [
                                        calculate_step_ratio(
                                            tau_basis_type, psi_basis_type, tau_epsilon, psi_epsilon, basis_times,
                                            time_basis_type, time_epsilon, psi_map_coef, psi_p, st
                                        ) for psi_p in psi_grid
                                    ], (len(times[(times >= st)]), 1)
                                )
                                # For subsequent basis times, step ratio is 1 - same response as first step
                                # (before applying step amplitude)
                                if k < len(step_times) - 1:
                                    step_ratio[(times >= st), :, 1:] = 1

                                # Broadcast step ratios to f_mp_int values
                                # Use k + 1 to account for concatenated -inf basis time in f_mp
                                A_layered[k] = np.sum(f_mp_int[:, :, k + 1:] * step_ratio, axis=2) * sa
                            else:
                                # Use fully vectorized function to check magnitude of function
                                check_func = get_2d_response_func(tau_basis_type, psi_basis_type, independent_measurements,
                                                                  psi_static,
                                                                  psi_is_time, psi_is_i, time_basis_type, basis_times,
                                                                  time_epsilon,
                                                                  psi_map_coef,
                                                                  op_mode, step_model, '3d')

                                start = time.time()
                                func_mag = np.zeros_like(A_layered[k])
                                func_mag[times >= st] = np.nan_to_num(
                                    check_func([0], None, tau_grid, psi_grid, times[times >= st], st,
                                               tau_epsilon, psi_epsilon, tau_rise[k], None),
                                    0)
                                post_step = func_mag[times >= st]
                                print('func_mag eval time: {:.2f}'.format(time.time() - start))
                                print('Num after step:', post_step.shape[0] * post_step.shape[1])
                                print('Num below rtol:', len(post_step[np.where(post_step < rtol)]))
                                print('Frac below rtol: {:.3f}'.format(
                                    len(post_step[np.where(post_step < rtol)]) / (post_step.shape[0] * post_step.shape[1])
                                ))

                                # Get function at desired vectorization level for full calculation
                                # If using step ratios, must keep responses to different basis times separated
                                func = get_2d_response_func(
                                    tau_basis_type, psi_basis_type, independent_measurements, psi_static, psi_is_time,
                                    psi_is_i,
                                    time_basis_type, basis_times, time_epsilon, psi_map_coef, op_mode, step_model,
                                    vectorization_level, separate_basis_times=use_step_ratios
                                )
                                # For all vectorization options, only perform calculation if:
                                # (1) time >= step time and
                                # (2) function magnitude at y1=0 exceeds rtol
                                if vectorization_level == '1d':
                                    calc_index = np.where(func_mag > rtol)
                                    print('Num to calculate:', len(calc_index[0]))
                                    num_calculated += len(calc_index[0])
                                    basis_tau_mesh, psi_exp_mesh = np.meshgrid(tau_grid, psi_exp)
                                    basis_psi_mesh, time_mesh = np.meshgrid(psi_grid, times)

                                    if use_step_ratios:
                                        # If using step ratios, must keep responses to different basis times separated
                                        f_mp_int = np.zeros((len(times), len(tau_grid), len(basis_times) + 1))
                                        f_mp_int[calc_index[0], calc_index[1]] = [
                                            np.trapz(
                                                np.nan_to_num(
                                                        func(y1, p_n - psi_mp, tau_mp, psi_mp, t_n, st, tau_epsilon, psi_epsilon,
                                                             tau_rise[k]),
                                                        0), x=y1, axis=1
                                            )
                                            for tau_mp, psi_mp, t_n, p_n in zip(
                                                basis_tau_mesh[calc_index], basis_psi_mesh[calc_index], time_mesh[calc_index],
                                                psi_exp_mesh[calc_index]
                                            )
                                        ]
                                        print(f_mp_int.shape)

                                        # f_sum = np.sum(f_mp_int, axis=2)
                                        # print(f_sum.shape)
                                        A_layered[k] = np.sum(f_mp_int, axis=2) * sa
                                        # A_layered[k, calc_index[0], calc_index[1]] = [
                                        #     sa * np.trapz(f_sum[ri, ci, :], x=y1) for ri, ci in zip(calc_index[0], calc_index[1])
                                        # ]
                                    else:
                                        A_layered[k, calc_index[0], calc_index[1]] = [
                                            np.trapz(
                                                np.nan_to_num(
                                                    func(y1, p_n - psi_mp, tau_mp, psi_mp, t_n, st, tau_epsilon, psi_epsilon,
                                                         tau_rise[k]),
                                                    0),
                                                x=y1) * sa
                                            for tau_mp, psi_mp, t_n, p_n in zip(basis_tau_mesh[calc_index], basis_psi_mesh[calc_index],
                                                                                time_mesh[calc_index], psi_exp_mesh[calc_index])
                                        ]
                                elif vectorization_level == '2d':
                                    row_index, col_index = np.where(func_mag > rtol)
                                    row_calc_indices = np.unique(row_index)
                                    col_calc_indices = [col_index[row_index == ri] for ri in row_calc_indices]
                                    A_layered[k, row_calc_indices] = [
                                        np.trapz(
                                            np.nan_to_num(
                                                func(y1, p_n - psi_grid, tau_grid, psi_grid, t_n, st, tau_epsilon, psi_epsilon,
                                                     tau_rise[k], cci_n),
                                                0),
                                            x=y1, axis=0) * sa
                                        for t_n, p_n, cci_n in zip(times[row_calc_indices], psi_exp[row_calc_indices],
                                                                   col_calc_indices)
                                    ]
                                elif vectorization_level == '3d':
                                    # Limit input to times after step time - array creation and summation is still costly even if
                                    # func is not evaluated
                                    calc_index = np.where(post_step > rtol)

                                    A_layered[k, times >= st] = np.trapz(
                                        np.nan_to_num(
                                            func(y1, None, tau_grid, psi_grid, times[times >= st],
                                                 st, tau_epsilon, psi_epsilon, tau_rise[k], calc_index),
                                            0),
                                        x=y1, axis=0) * sa

                                print(k, st, sa, np.sum(A_layered[k]))

    # ===============================
    # Perform inverse delta transform
    # ===============================
    if use_delta_transform:
        # For each basis function in psi, subtract next basis function to get original (untransformed) response
        for p in range(len(basis_psi) - 1):
            A_layered[:, :, p * len(basis_tau): (p + 1) * len(basis_tau)] = \
                A_layered[:, :, p * len(basis_tau): (p + 1) * len(basis_tau)] \
                - A_layered[:, :, (p + 1) * len(basis_tau): (p + 2) * len(basis_tau)]

        # Last basis function needs to be converted to regular pw_linear basis function
        # This only makes a difference for basis times at which psi_hat > psi_P
        # Find first basis time for which psi_hat > psi_P
        if correct_final_psi:
            psi_hat_basis = evaluate_basis_fit(psi_map_coef, basis_times, basis_times, time_basis_type, time_epsilon)
            recalc_bt = basis_times[psi_hat_basis > basis_psi[-1]]
            if len(recalc_bt) > 0:
                bt = np.min(recalc_bt)
                print(f'Recalculating response for last basis_psi value starting from basis time {bt}')
                psi_col_index = np.arange((len(basis_psi) - 1) * len(basis_tau), len(basis_psi) * len(basis_tau), dtype=int)
                for k in range(len(step_times)):
                    st = step_times[k]
                    sa = step_sizes[k]

                    # Get 3d vectorized func using regular (untransformed) basis function for filtering
                    check_func = get_2d_response_func(tau_basis_type, psi_basis_type, independent_measurements,
                                                      psi_static, psi_is_time, psi_is_i,
                                                      time_basis_type, basis_times, time_epsilon,
                                                      psi_map_coef,
                                                      op_mode, step_model, '3d')

                    func_mag = np.zeros_like(A_layered[k])
                    func_mag[np.ix_(times >= max(st, bt), psi_col_index)] = np.nan_to_num(
                        check_func([0], None, tau_grid[psi_col_index], psi_grid[psi_col_index], times[times >= max(st, bt)], st,
                                   tau_epsilon, psi_epsilon, tau_rise[k], None),
                        0)

                    # Get func using regular (untransformed) basis function for full calculation
                    func = get_2d_response_func(tau_basis_type, psi_basis_type, independent_measurements,
                                                psi_static, psi_is_time, psi_is_i,
                                                time_basis_type, basis_times, time_epsilon,
                                                psi_map_coef,
                                                op_mode, step_model, vectorization_level)
                    # For all vectorization options, only perform calculation if:
                    # (1) time >= step time and
                    # (2) function magnitude at y1=0 exceeds rtol
                    if vectorization_level == '1d':
                        calc_index = np.where(func_mag > rtol)
                        print('Num to recalculate:', len(calc_index[0]))
                        num_calculated += len(calc_index[0])
                        basis_tau_mesh, psi_exp_mesh = np.meshgrid(tau_grid, psi_exp)
                        basis_psi_mesh, time_mesh = np.meshgrid(psi_grid, times)
                        A_layered[k, calc_index[0], calc_index[1]] = [
                            np.trapz(
                                np.nan_to_num(
                                    func(y1, p_n - psi_mp, tau_mp, psi_mp, t_n, st, tau_epsilon, psi_epsilon,
                                         tau_rise[k]),
                                    0),
                                x=y1) * sa
                            for tau_mp, psi_mp, t_n, p_n in zip(basis_tau_mesh[calc_index], basis_psi_mesh[calc_index],
                                                                time_mesh[calc_index], psi_exp_mesh[calc_index])
                        ]

    A = np.sum(A_layered, axis=0)

    print('total number of entries calculated:', num_calculated)

    return A, A_layered


def calculate_step_ratio(tau_basis_type, psi_basis_type, tau_epsilon, psi_epsilon, basis_times, time_basis_type,
                         time_epsilon, psi_map_coef, psi_p, t_s):

    phi_basis = get_2d_basis_func(tau_basis_type, psi_basis_type)

    # Calculate psi_hat at basis times
    psi_eval_times = np.concatenate(([-np.inf], basis_times))
    # print(psi_eval_times)
    psi_hat_basis = evaluate_basis_fit(psi_map_coef, psi_eval_times,
                                       basis_times, time_basis_type, time_epsilon)
    # print(psi_hat_basis)

    # Evaluate phi_mp at each basis time
    phi_mp_basis = phi_basis(0, psi_hat_basis - psi_p, tau_epsilon, psi_epsilon)

    # Get steps in phi_mp
    phi_mp_steps = np.diff(phi_mp_basis, axis=0)

    phi_mag = np.concatenate(([phi_mp_basis[0]], phi_mp_steps))
    # print(phi_mag)

    # Sum all basis function steps that occurred at or before t_s
    f_k_0 = np.sum(
        [phi_mag[k] for k in range(len(psi_eval_times)) if t_s >= psi_eval_times[k]]
    )
    # print([phi_mag[k] for k in range(len(psi_eval_times)) if t_s >= psi_eval_times[k]])

    # Get basis function step at t_s
    f_k_s = phi_mag[psi_eval_times == t_s][0]

    # TODO: Make this work even if f_k_s == 0. This could happen if psi_hat remains constant from one step to the next.
    #  In this case, there will be no response to the first step, but there will still be a response to the later step.
    #  We can handle this either by (1) identifying cases where f_k_s==0 in construct_2d_response_matrix and then
    #  manually calculating those entries, or (2) storing the response to the first step BEFORE multiplying by phi_mag

    if f_k_s == 0:
        return 0
    else:
        return f_k_0 / f_k_s


def is_delta_transform_available(basis_psi, psi_epsilon, psi_basis_type, psi_map_coef):
    # Check monotonicity
    if (utils.is_monotonic(basis_psi) and utils.is_monotonic(np.cumsum(psi_map_coef))):
        # Check basis type
        if psi_basis_type == 'pw_linear':
            # Check for uniform spacing of basis_psi
            if utils.is_uniform(basis_psi):
                # Check that spacing is equal to 1 / epsilon
                psi_spacing = np.abs(np.mean(np.diff(basis_psi)))
                if utils.rel_round(psi_spacing, 5) == utils.rel_round(1 / psi_epsilon, 5):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False


def construct_2d_integrated_derivative_matrix(basis_tau, basis_psi, tau_basis_type, psi_basis_type,
                                              tau_epsilon, psi_epsilon, fxx_penalty, fyy_penalty, fxy_penalty,
                                              order):
    """
    Construct matrix for calculation of DRT ridge penalty.
    x^T@dm@x gives integral of squared derivative of DRT over all ln(tau)

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
    M = len(basis_tau)
    P = len(basis_psi)

    dm = np.zeros((M * P, M * P))

    # Fill in diagonal with penalty matrix for tau dimension
    dm_tau = construct_integrated_derivative_matrix(np.log(basis_tau), tau_basis_type, order, tau_epsilon)
    print('dm_tau max:', np.max(np.abs(dm_tau)))
    print('dm_tau sum:', np.sum(dm_tau))
    for p in range(P):
        dm[p * M: (p + 1) * M, p * M: (p + 1) * M] = dm_tau * fxx_penalty / P
    # print(np.sum(dm))

    # Fill in off-diagonals with penalty matrix for psi dimension
    dm_psi = construct_integrated_derivative_matrix(basis_psi, psi_basis_type, order, psi_epsilon)
    print('dm_psi max:', np.max(np.abs(dm_psi)))
    print('dm_psi sum:', np.sum(dm_psi))
    for m in range(M):
        dm[m: M * P: M, m: M * P: M] += dm_psi * fyy_penalty / M
        # print(np.max(np.abs(dm[m: M * P: M, m: M * P: M])))
    # print(np.sum(dm))

    if order == 2:
        # Add f_xy
        if tau_basis_type == 'gaussian' and psi_basis_type == 'gaussian':
            # Integral of fxy**2 can be evaluated analytically for 2d Gaussian RBF
            def f_xy_sq(x_m, x_n, y_q, y_r, epsilon_x, epsilon_y):
                """Returns double integral of f_xy(mq) * f_xy(nr)"""
                a_x = (epsilon_x * (x_m - x_n)) ** 2
                a_y = (epsilon_y * (y_q - y_r)) ** 2
                return (np.pi / 2) * epsilon_x * epsilon_y * (a_x - 1) * (a_y - 1) * np.exp(-0.5 * a_x - 0.5 * a_y)

            # Define tau and psi grids
            # Ordering: (tau_1, psi_1), (tau_2, psi_1), ..., (tau_M, psi_1), (tau_1, psi_2), ...
            x_grid, y_grid = get_2d_basis_grid(np.log(basis_tau), basis_psi)
            xx_m, xx_n = np.meshgrid(x_grid, x_grid)
            yy_q, yy_r = np.meshgrid(y_grid, y_grid)
            dm_xy = f_xy_sq(xx_m, xx_n, yy_q, yy_r, tau_epsilon, psi_epsilon)
        else:
            # Use discrete evaluation matrix
            em = construct_2d_func_eval_matrix(basis_tau, basis_psi, None, None, tau_basis_type, psi_basis_type,
                                               tau_epsilon, psi_epsilon, 'fxy')
            dm_xy = em @ em

        dm += dm_xy * fxy_penalty
        print('dm_xy max:', np.max(np.abs(dm_xy)))
        print('dm_xy sum:', np.sum(dm_xy))

    return dm


def construct_2d_func_eval_matrix(basis_tau, basis_psi, eval_tau, eval_psi, tau_basis_type, psi_basis_type,
                                  tau_epsilon, psi_epsilon, function):
    """
    Construct matrix L such that L@x gives the derivative of the DRT evaluated at eval_eig
    :param ndarray basis_tau: basis time constants
    :param ndarray eval_tau: time constants at which to evaluate derivative
    :param str basis_type: type of basis function. Options: 'gaussian', 'Zic'
    :param epsilon: shape parameter for basis function
    :param order: order of derivative to calculate. Can be int, list, or float. If list, entries indicate relative
    weights of 0th, 1st, and 2nd derivatives, respectively. If float, calculate weighted mixture of nearest integer
    orders.
    :return: ndarray of derivative values, same size as eval_tau
    """
    utils.check_basis_type(tau_basis_type)
    utils.check_basis_type(psi_basis_type)

    # Validate derivative string. x = tau, y = psi
    function_options = ['f', 'fx', 'fy', 'fxx', 'fyy', 'fxy']
    if function not in function_options:
        raise ValueError(f'Invalid function {function}. Options: {function_options}')

    # If no evaluation grid given, assume collocated with basis grid
    if eval_tau is None:
        eval_tau = basis_tau.copy()
    if eval_psi is None:
        eval_psi = basis_psi.copy()

    # Get appropriate derivative function
    if function == 'f':
        func = get_2d_basis_func(tau_basis_type, psi_basis_type)
    else:
        # Determine order of derivative in each dimension
        x_order = function.count('x')
        y_order = function.count('y')

        phi_x = get_basis_func_derivative(tau_basis_type, x_order)
        phi_y = get_basis_func_derivative(psi_basis_type, y_order)

        def func(x, y, epsilon_x, epsilon_y):
            return phi_x(x, epsilon_x) * phi_y(y, epsilon_y)

        # if function == 'fx':
        #     # df/dx
        #     def func(x, y, epsilon_x, epsilon_y):
        #         return -2 * epsilon_x ** 2 * x * phi(x, y, epsilon_x, epsilon_y)
        # elif function == 'fy':
        #     # df/dy
        #     def func(x, y, epsilon_x, epsilon_y):
        #         return -2 * epsilon_y ** 2 * y * phi(x, y, epsilon_x, epsilon_y)
        # elif function == 'fxx':
        #     # d2f/dx2
        #     def func(x, y, epsilon_x, epsilon_y):
        #         return (-2 * epsilon_x ** 2 + 4 * epsilon_x ** 4 * x ** 2) * phi(x, y, epsilon_x, epsilon_y)
        # elif function == 'fyy':
        #     # d2f/dy2
        #     def func(x, y, epsilon_x, epsilon_y):
        #         return (-2 * epsilon_y ** 2 + 4 * epsilon_y ** 4 * y ** 2) * phi(x, y, epsilon_x, epsilon_y)
        # elif function == 'fxy':
        #     # d2f/dxdy
        #     def func(x, y, epsilon_x, epsilon_y):
        #         return 4 * (epsilon_x * epsilon_y) ** 2 * x * y * phi(x, y, epsilon_x, epsilon_y)


    # Create mesh of basis and evaluation grids
    x_basis_grid, y_basis_grid = get_2d_basis_grid(np.log(basis_tau), basis_psi)
    x_eval_grid, y_eval_grid = get_2d_basis_grid(np.log(eval_tau), eval_psi)
    xx_m, xx_n = np.meshgrid(x_basis_grid, x_eval_grid)
    yy_q, yy_r = np.meshgrid(y_basis_grid, y_eval_grid)

    # Evaluate function at all mesh points
    dm = func(xx_n - xx_m, yy_r - yy_q, tau_epsilon, psi_epsilon)

    return dm


def construct_2d_inf_response_matrix(basis_psi, eval_psi, psi_basis_type, psi_epsilon, times, input_signal,
                                     step_times, step_sizes, tau_rise, step_model, smooth):
    """
    Construct matrix rm such that rm@x, where x is vector of R_inf coefficients corresponding to basis_psi,
    produces vector of R_inf (ohmic resistance) response values
    :param times:
    :param basis_psi:
    :param psi_basis_type:
    :param psi_epsilon:
    :param ndarray input_signal: measured input signal. Only used if smooth == False
    :param ndarray step_times: array of step times. Only used if smooth == True
    :param ndarray step_sizes: array of step sizes. Only used if smooth == True
    :param ndarray tau_rise: array of rise times for expdecay model. Only used if smooth == True
    :param str step_model: Type of input signal step model to use. Only used if smooth == True
    :param bool smooth: if True, construct smooth response matrix using model input signal rather than raw input signal
    :return:
    """
    # Matrix to evaluate R_inf. em @ x = R_inf(psi)
    em = construct_func_eval_matrix(basis_psi, eval_psi, psi_basis_type, psi_epsilon, 0)

    if smooth:
        # Use ideal input signal from identified steps rather than noisy measured input signal
        input_signal = pp.generate_model_signal(times, step_times, step_sizes, tau_rise, step_model)
    else:
        # Get delta from starting value
        # DIFFERENTIAL R_inf should respond only to changes in current
        prestep_signal = input_signal[times < step_times[0]]
        input_signal = input_signal - np.mean(prestep_signal)

    # Input signal matrix - tile input_signal vector
    im = np.tile(input_signal, (len(basis_psi), 1)).T

    return im * em


def construct_2d_impedance_matrix(frequencies, psi_exp, part, basis_tau, basis_psi,
                                  independent_measurements, psi_static, psi_is_time,
                                  tau_basis_type, tau_epsilon, psi_basis_type, psi_epsilon,
                                  integrate_method='trapz', zga_params=None):

    # Impedance function depends only on tau and freq
    # Approach:
    # 1: Get unique frequencies
    # 2: Calculate regular impedance matrix for unique frequencies and basis_tau
    # 2: Duplicate regular impedance matrix across psi and scale by psi basis func value

    # Get unique frequencies and index to map all frequencies to freq_unique
    freq_unique, inverse_index = np.unique(frequencies, return_inverse=True)

    # Construct reference matrix using freq_unique
    zm_ref = construct_impedance_matrix(freq_unique, part, basis_tau, basis_type=tau_basis_type, epsilon=tau_epsilon,
                                        frequency_precision=self.frequency_precision, integrate_method=integrate_method,
                                        zga_params=zga_params)

    # Define tau and psi grids
    # Ordering: (tau_1, psi_1), (tau_2, psi_1), ..., (tau_M, psi_1), (tau_1, psi_2), ...
    tau_grid, psi_grid = get_2d_basis_grid(basis_tau, basis_psi)

    # Copy reference matrix elements to corresponding frequencies, duplicating for each basis_psi value
    zm = np.tile(zm_ref[inverse_index], (1, len(basis_psi)))

    # Get psi basis function magnitude for each basis_psi at each psi_exp
    pm = construct_func_eval_matrix(psi_grid, psi_exp, basis_type=psi_basis_type, epsilon=psi_epsilon, order=0,
                                    zga_params=None)

    # Broadcast psi basis magnitude to impedance response matrix
    return zm * pm


# =========================
# For chemical relaxation
# =========================
def echem_response(times, x_mc, b_m, tau_m, tau_c, t_e, t_c):

    t_s = max(t_e, t_c)

    out = np.zeros(len(times))

    # Relaxing resistance contribution
    t_s_index = np.where(times >= t_s)
    out[t_s_index] = x_mc * (
        1 - np.exp(-(times[t_s_index] - t_s) / tau_m)
        - (tau_c / (tau_c - tau_m)) * (
            np.exp(-(times[t_s_index] - t_c) / tau_c)
            - np.exp(-(times[t_s_index] - t_s) / tau_m + (t_c - t_s) / tau_c)
        )
    )

    # Add constant resistance contribution
    t_e_index = np.where(times >= t_e)
    out[t_e_index] += b_m * (1 - np.exp(-(times[t_e_index] - t_e) / tau_m))

    return out