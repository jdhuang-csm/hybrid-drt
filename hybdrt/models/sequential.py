import numpy as np
from copy import deepcopy

from .. import fileload as fl


def fit_sequence(drt, chrono_data_list, eis_data_list=None, **fit_kw):
    # Concatenate all chrono data to get timestamps right
    concat_chrono_df = fl.concatenate_chrono_data(chrono_data_list)

    if eis_data_list is None:
        eis_data_list = [None] * len(chrono_data_list)

    drt_list = []

    chrono_offset = 0
    v_projected = np.zeros(len(concat_chrono_df))
    for i in range(len(chrono_data_list)):
        print(f'Fitting dataset {i}...')
        print(chrono_offset)
        # Get eis data
        eis_data = eis_data_list[i]
        if eis_data is None:
            eis_tuple = None
        else:
            eis_tuple = fl.get_eis_tuple(eis_data)

        # Get chrono data from concatenated df
        raw_chrono_data = chrono_data_list[i]
        num_chrono = len(raw_chrono_data)
        chrono_data = concat_chrono_df[chrono_offset:chrono_offset + num_chrono]
        chrono_tuple = fl.get_chrono_tuple(chrono_data)

        # Subtract projected chrono response from chrono data
        times, i_sig, v_sig = chrono_tuple
        v_sig = v_sig - v_projected[chrono_offset:chrono_offset + num_chrono]
        chrono_tuple = (times, i_sig, v_sig)
        print(times[0], times[-1])

        # Fit and append to output list
        if eis_tuple is None:
            drt.fit_chrono(*chrono_tuple, **fit_kw)
        else:
            drt.fit_hybrid(*chrono_tuple, *eis_tuple, **fit_kw)

        drt_list.append(deepcopy(drt))

        # Project response to current excitation signal for all future times
        if i < len(chrono_data_list) - 1:
            t_pred = concat_chrono_df.loc[chrono_offset + num_chrono:, 'elapsed'].values
            print(t_pred[0])
            v_proj_i = drt.predict_response(concat_chrono_df.loc[chrono_offset + num_chrono:, 'elapsed'].values,
                                            v_baseline=0)
            v_projected[chrono_offset + num_chrono:] += v_proj_i

        # Update offset index
        chrono_offset += num_chrono

    return drt_list, v_projected
