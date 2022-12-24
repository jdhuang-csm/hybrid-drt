import numpy as np
from . import chrono


def multiply_list(data_list, factor):
    if np.shape(factor) == ():
        return [data * factor for data in data_list]
    elif np.shape(factor) == np.shape(data_list):
        try:
            return [data * f for data, f in zip(data_list, factor)]
        except TypeError:
            return data_list
    else:
        raise ValueError('Shapes of data_list and factor must be the same')


def obs_vector_to_data_vector(vector, data_list, expand_factor=1):
    """
    Expand a vector of length num_obs to length of total number of points in data_list by repeating vector values
    :param ndarray vector: vector to expand
    :param list or ndarray data_list: data list or array
    :param int expand_factor:
    """
    if not len(vector) == len(data_list):
        raise ValueError('vector and data_list must have same length')
    else:
        data_vector = np.concatenate([[val] * get_data_tuple_length(data) * expand_factor
                                      for val, data in zip(vector, data_list)])
        return data_vector


def reshape_vector_to_data(vector, data_list, expand_factor=1):
    """
    Reshape flattened vector to list or array in shape of original data
    :param ndarray vector: flattened vector to reshape
    :param list or ndarray data_list: data list or array with desired shape
    """
    if type(data_list) == list:
        # Ragged data - use list
        vector_list = []
        start = 0
        for i, data in enumerate(data_list):
            num_i = get_data_tuple_length(data) * expand_factor
            vector_list.append(vector[start:start + num_i])
            start += num_i
        return vector_list
    elif type(data_list) == np.ndarray:
        return np.reshape(vector, (data_list.shape[0], data_list.shape[1] * expand_factor))


def get_data_obs_indices(data_list, expand_factor=1):
    obs_start_index = np.empty(len(data_list), dtype=int)
    obs_end_index = np.empty(len(data_list), dtype=int)
    start = 0
    for i, data in enumerate(data_list):
        obs_start_index[i] = start
        start += get_data_tuple_length(data) * expand_factor
        obs_end_index[i] = start
    return obs_start_index, obs_end_index


def get_data_tuple_item(data_tuple, tuple_index):
    # Convenience function for handling heterogeneous list of tuples and Nones
    if data_tuple is None:
        return None
    elif type(data_tuple) == tuple:
        return data_tuple[tuple_index]
    elif tuple_index == 0:
        # Data is single array, not tuple. Return whole array
        return data_tuple


def get_data_tuple_length(data_tuple):
    a = get_data_tuple_item(data_tuple, 0)
    if a is None:
        return 0
    else:
        return len(a)


def get_data_list_size(data_list):
    return np.sum([get_data_tuple_length(data) for data in data_list])


def get_sampled_chrono_data_list(chrono_data_list, sample_index_list):
    """
    Get list of downsampled chrono datasets
    """
    sample_data_list = []

    for i, data in enumerate(chrono_data_list):
        if get_data_tuple_item(data, 0) is not None:
            times, i_signal, v_signal = data
            sample_index = sample_index_list[i]

            # Get downsampled data
            sample_times = times[sample_index]
            sample_i = i_signal[sample_index]
            sample_v = v_signal[sample_index]

            sample_data_list.append((sample_times, sample_i, sample_v))
        else:
            sample_data_list.append(None)

    return sample_data_list


def data_list_to_vector(data_list, data_type, ctrl_mode):
    if data_type == 'eis':
        y_list = [get_data_tuple_item(data, 1) for data in data_list]
        y_list = [y for y in y_list if y is not None]
        if len(y_list) > 0:
            y_vector = np.concatenate([np.concatenate([y.real, y.imag]) for y in y_list])
        else:
            y_vector = []
    elif data_type == 'chrono':
        iv_list = [(get_data_tuple_item(data, 1), get_data_tuple_item(data, 2))
                   for data in data_list if get_data_tuple_length(data) > 0]
        y_list = [chrono.get_input_and_response(iv[0], iv[1], ctrl_mode) for iv in iv_list]
        if len(y_list) > 0:
            y_vector = np.concatenate(y_list)
        else:
            y_vector = []
    else:
        raise ValueError(f'Invalid data type {data_type}')

    return y_vector


def get_data_type(chrono_data, eis_data):
    if chrono_data is not None and eis_data is not None:
        return 'hybrid'
    elif chrono_data is not None:
        return 'chrono'
    elif eis_data is not None:
        return 'eis'
    else:
        raise ValueError('No data provided')

