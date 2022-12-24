import numpy as np
from .validation import check_ctrl_mode


def get_time_transforms(times, step_times):

    # Get minimum sample period
    t_sample = np.min(np.diff(times))
    # print('t_sample', t_sample)

    # if step_times[0] > times[0]:
    #     start_times = np.concatenate([times[0:1], step_times])
    # else:
    start_times = np.array(step_times)

    trans_base = np.log(t_sample / 4)
    trans_offsets = np.log(start_times[1:] - start_times[:-1]) - trans_base
    trans_offsets = np.concatenate([[0], np.cumsum(trans_offsets)])

    def fwd_transform(t):
        t = np.atleast_1d(t)

        tt = np.zeros_like(t, dtype=float)

        # Prior to first step - linear
        tt[t < start_times[0]] = t[t < start_times[0]] - start_times[0]

        for i, start_time in enumerate(start_times):
            if i == len(start_times) - 1:
                end_time = np.inf
            else:
                end_time = start_times[i + 1]

            step_index = np.where((t >= start_time) & (t < end_time))
            if len(step_index[0]) > 0:
                # Get time since step. Don't allow time_deltas smaller than 1/2 sample period
                time_delta = np.maximum(t[step_index] - start_time, t_sample / 2)
                # Transformed time for segment starts at offset.
                # Subtract log(t_sample / 4) to ensure that segment starts after end of last segment
                tt[step_index] = trans_offsets[i] + np.log(time_delta) - trans_base

        return tt

    def rev_transform(tt):
        tt = np.atleast_1d(tt)

        t = np.zeros_like(tt, dtype=float)

        t[tt < trans_offsets[0]] = tt[tt < trans_offsets[0]] + start_times[0]

        for i, start_tt in enumerate(trans_offsets):
            if i == len(trans_offsets) - 1:
                end_tt = np.inf
            else:
                end_tt = trans_offsets[i + 1]

            step_index = np.where((tt >= start_tt) & (tt <= end_tt))
            if len(step_index[0]) > 0:
                # Reverse transform to time_delta
                time_delta = np.exp(tt[step_index] - start_tt + trans_base)
                # Add start time to recover original time
                t[step_index] = time_delta + start_times[i]

        return t

    return rev_transform, fwd_transform


def get_input_and_response(i_signal, v_signal, ctrl_mode):
    if ctrl_mode is not None:
        check_ctrl_mode(ctrl_mode)
        if ctrl_mode == 'galv':
            input_signal = i_signal
            response_signal = v_signal
        else:
            input_signal = v_signal
            response_signal = i_signal
    else:
        input_signal = None
        response_signal = None

    return input_signal, response_signal


def signals_to_tuple(times, input_signal, response_signal, ctrl_mode):
    if ctrl_mode is not None:
        check_ctrl_mode(ctrl_mode)
        if ctrl_mode == 'galv':
            # Input is current
            chrono_tuple = (times, input_signal, response_signal)
        else:
            # Input is voltage
            chrono_tuple = (times, response_signal, input_signal)
    else:
        chrono_tuple = None

    return chrono_tuple



# def transform_times(times, step_times):
#     """
#     Transform times for visualizing time after each step on a log scale
#     :param times:
#     :param step_times:
#     :return:
#     """
#     trans_times = times.copy()
#
#     # Get minimum sample period
#     t_sample = np.min(np.diff(times))
#
#     if step_times[0] > times[0]:
#         start_times = np.concatenate([times[0:1], step_times])
#     else:
#         start_times = step_times
#
#     offset = 0
#     offsets = np.zeros(len(start_times))
#     for i, start_time in enumerate(start_times):
#         if i == len(start_times) - 1:
#             end_time = np.inf
#         else:
#             end_time = start_times[i + 1]
#
#         step_index = np.where((times >= start_time) & (times < end_time))
#         # print(start_time, end_time, step_index)
#         # Get time since step. Don't allow time_deltas smaller than 1/2 sample period
#         time_delta = np.maximum(times[step_index] - start_time, t_sample / 2)
#         # Transformed time for segment starts at offset.
#         # Subtract log(t_sample / 4) to ensure that segment starts after end of last segment
#         trans_times[step_index] = offset + np.log(time_delta) - np.log(t_sample / 4)
#
#         # Increment offset such that next step will start at end of this step
#         offset = trans_times[step_index][-1]
#
#         # Store offset for inverse transform
#         if i < len(start_times) - 1:
#             offsets[i + 1] = offset
#
#     return trans_times, (times[0], t_sample, offsets)
#
#
# def inverse_time_transform(trans_times, step_times, transform_data):
#     """
#     Perform inverse of time transform to recover original times.
#     :param ndarray trans_times: transformed times
#     :param ndarray step_times: array of step times
#     :param tuple transform_data: Data from transform_times required for inverse transform
#     :return:
#     """
#     times = trans_times.copy()
#
#     # Extract necessary inversion data
#     t0, t_sample, offsets = transform_data
#
#     # offsets = np.concatenate([[0], offsets])
#     start_times = np.concatenate([[t0], step_times])
#
#     for i, start_tt in enumerate(offsets):
#         if i == len(offsets) - 1:
#             end_tt = np.inf
#         else:
#             end_tt = offsets[i + 1]
#
#         step_index = np.where((trans_times >= start_tt) & (trans_times <= end_tt))
#         # Reverse transform to time_delta
#         time_delta = np.exp(trans_times[step_index] - start_tt + np.log(t_sample / 4))
#         # Add start time to recover original time
#         times[step_index] = time_delta + start_times[i]
#
#     return times
