import pandas as pd
from datetime import datetime, timedelta
import warnings
import numpy as np
from hybdrt.utils.eis import polar_from_complex
import calendar
import time
import re


def get_file_source(file):
    """Determine file source"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    # Determine	format
    header = txt.split('\n')[0]
    if header == 'EXPLAIN':
        source = 'gamry'
    elif header == 'ZPLOT2 ASCII':
        source = 'zplot'
    elif header == 'EC-Lab ASCII FILE':
        source = 'biologic'
    elif header == 'RelaxIS 3.0 Spectrum export':
        source = 'relaxis'
    else:
        source = None

    return source

def read_txt(file):
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    return txt


def check_source(file, source=None):
    known_sources = ['gamry', 'zplot', 'biologic', 'relaxis']

    if source is None:
        source = get_file_source(file)
        if source is None:
            raise ValueError('Could not identify file format. To read this file, '
                             'manually specify the file format by providing the source argument. '
                             'Recognized sources: {}'.format(', '.join(known_sources)))

    if source not in known_sources:
        raise ValueError('Unrecognized source {}. Recognized sources: {}'.format(source, ', '.join(known_sources)))

    return source


def get_custom_file_time(file):
    txt = read_txt(file)

    date_start = txt.find('DATE')
    date_end = txt[date_start:].find('\n') + date_start
    date_line = txt[date_start:date_end]
    date_str = date_line.split('\t')[2]

    time_start = txt.find('TIME')
    time_end = txt[time_start:].find('\n') + time_start
    time_line = txt[time_start:time_end]
    time_str = time_line.split('\t')[2]
    # Separate fractional seconds
    time_str, frac_seconds = time_str.split('.')

    dt_str = date_str + ' ' + time_str
    time_format_code = "%m/%d/%Y %H:%M:%S"
    file_time = time.strptime(dt_str, time_format_code)

    return float(calendar.timegm(file_time)) + float('0.' + frac_seconds)


def get_timestamp(file, source=None):
    """Get experiment start timestamp from file"""
    txt = read_txt(file)

    source = check_source(file, source)

    if source == 'gamry':
        try:
            date_start = txt.find('DATE')
            date_end = txt[date_start:].find('\n') + date_start
            date_line = txt[date_start:date_end]
            date = date_line.split('\t')[2]

            time_start = txt.find('TIME')
            time_end = txt[time_start:].find('\n') + time_start
            time_line = txt[time_start:time_end]
            time_txt = time_line.split('\t')[2]

            timestr = date + ' ' + time_txt
            dt = datetime.strptime(timestr, "%m/%d/%Y %H:%M:%S")
        except ValueError:
            time_sec = get_custom_file_time(file)
            dt = datetime.utcfromtimestamp(time_sec)

    elif source == 'zplot':
        date_start = txt.find('Date')
        date_end = txt[date_start:].find('\n') + date_start
        date_line = txt[date_start:date_end]
        date = date_line.split()[1]

        time_start = txt.find('Time')
        time_end = txt[time_start:].find('\n') + time_start
        time_line = txt[time_start:time_end]
        time_txt = time_line.split()[1]

        timestr = date + ' ' + time_txt
        dt = datetime.strptime(timestr, "%m-%d-%Y %H:%M:%S")

    return dt


def find_time_column(df):
    return np.intersect1d(['Time', 'T'], df.columns)[0]


def read_header(file):
    txt = read_txt(file)

    # find start of curve data
    table_index = txt.upper().find('\nCURVE\tTABLE')
    if table_index == -1:
        table_index = txt.upper().find('\nZCURVE\tTABLE')

    return txt[:table_index + 1]


def read_chrono(file, source=None, return_tuple=False):
    """
    Read chronopotentiometry data from Gamry .DTA file

    Args:
        file: file to read
    """
    txt = read_txt(file)

    source = check_source(file, source)

    if source == 'gamry':
        # find start of curve data
        cidx = txt.upper().find('\nCURVE\tTABLE') + 1
        skipfooter = 0

        if cidx == -1:
            # coudn't find OCV curve data in file
            # return empty dataframe
            return pd.DataFrame([])
        else:
            # preceding text
            pretxt = txt[:cidx]

            # ocv curve data
            ctable = txt[cidx:]
            # column headers are next line after ZCURVE TABLE line
            header_start = ctable.find('\n') + 1
            header_end = header_start + ctable[header_start:].find('\n')
            header = ctable[header_start:header_end].split('\t')
            # units are next line after column headers
            unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
            units = ctable[header_end + 1:unit_end].split('\t')
            # determine # of rows to skip by counting line breaks in preceding text
            skiprows = len(pretxt.split('\n')) + 2

            # if table is indented, ignore empty left column
            if header[0] == '':
                usecols = header[1:]
            else:
                usecols = header
            # read data to DataFrame
            data = pd.read_csv(file, sep='\t', skiprows=skiprows, skipfooter=skipfooter, header=None, names=header,
                               usecols=usecols, engine='python')

            # get timestamp
            try:
                dt = get_timestamp(file)
                time_col = find_time_column(data) # time col may be either 'T' or 'Time'
                # print(time_col, data[time_col])
                data['timestamp'] = [dt + timedelta(seconds=t) for t in data[time_col]]
            except ValueError:
                warnings.warn(f'Could not read timestamp from file {file}')
    else:
        raise ValueError(f'read_chrono is not implemented for source {source}')

    if return_tuple:
        data = get_chrono_tuple(data)

    return data


def concatenate_chrono_data(chrono_data_list, eis_data_list=None, trim_index=None, trim_time=None,
                            loop=False, print_progress=False):
    """Concatenate curve data from multiple files"""
    if type(chrono_data_list[0]) == pd.DataFrame:
        chrono_dfs = chrono_data_list
    else:
        # Assume data_list contains files
        if loop:
            chrono_dfs = []
            num_to_load = len(chrono_data_list)
            if print_progress:
                print(f'Loading {num_to_load} chrono files...')
            for i, file in enumerate(chrono_data_list):
                chrono_dfs.append(read_chrono(file))
                if print_progress and ((i + 1) % int(num_to_load / 5) == 0 or i == num_to_load - 1):
                    print(f'{i + 1} / {num_to_load}')
        else:
            chrono_dfs = [read_chrono(file) for file in chrono_data_list]

    if eis_data_list is not None:
        if type(eis_data_list[0]) == pd.DataFrame:
            eis_dfs = eis_data_list
        else:
            # Assume data_list contains files
            eis_dfs = [read_eis(file) for file in eis_data_list]

        # Keep consistent columns only
        chrono_dfs = [df.loc[:, ['Time', 'Im', 'Vf', 'timestamp']] for df in chrono_dfs]
        eis_dfs = [df.loc[:, ['Time', 'Idc', 'Vdc', 'timestamp']].rename({'Idc': 'Im', 'Vdc': 'Vf'}, axis=1)
                   for df in eis_dfs]

        dfs = chrono_dfs + eis_dfs
    else:
        dfs = chrono_dfs

    # start_times = [df['timestamp'][0] for df in dfs]
    # start_time = min(start_times)

    # Sort by first timestamp
    dfs = sorted(dfs, key=lambda x: x['timestamp'][0])
    start_time = dfs[0]['timestamp'][0]

    ts_func = lambda ts: (ts - start_time).dt.total_seconds()
    for i, df in enumerate(dfs):
        df['elapsed'] = ts_func(df['timestamp'])
        df['file_id'] = i

    # Trim beginning of each file
    if trim_index is not None:
        dfs = [df.loc[trim_index:] for df in dfs]
    if trim_time is not None:
        dfs = [df[df['T'] >= trim_time] for df in dfs]

    df_out = pd.concat(dfs, ignore_index=True)

    return df_out


def concatenate_eis_data(eis_data_list, loop=False, print_progress=False):
    """Concatenate EIS data from multiple files"""
    if type(eis_data_list[0]) == pd.DataFrame:
        dfs = eis_data_list
    else:
        # Assume data_list contains files
        if loop:
            dfs = []
            num_to_load = len(eis_data_list)
            if print_progress:
                print(f'Loading {num_to_load} EIS files...')
            for i, file in enumerate(eis_data_list):
                dfs.append(read_eis(file))
                if print_progress and ((i + 1) % int(num_to_load / 5) == 0 or i == num_to_load - 1):
                    print(f'{i + 1} / {num_to_load}')
        else:
            dfs = [read_eis(file) for file in eis_data_list]

    # Sort by first timestamp
    dfs = sorted(dfs, key=lambda x: x['timestamp'][0])
    start_time = dfs[0]['timestamp'][0]
    # start_times = [df['timestamp'][0] for df in dfs]
    # start_time = min(start_times)

    ts_func = lambda ts: (ts - start_time).dt.total_seconds()
    for i, df in enumerate(dfs):
        df['elapsed'] = ts_func(df['timestamp'])
        df['file_id'] = i

    # # Trim beginning of each file
    # if trim_index is not None:
    #     dfs = [df.loc[trim_index:] for df in dfs]
    # if trim_time is not None:
    #     dfs = [df[df['T'] >= trim_time] for df in dfs]

    df_out = pd.concat(dfs, ignore_index=True)

    return df_out


def read_eis(file, source=None, warn=True, return_tuple=False):
    """read EIS zcurve data from Gamry .DTA file"""
    txt = read_txt(file)

    source = check_source(file, source)

    if source == 'gamry':
        # find start of zcurve data
        zidx = txt.find('ZCURVE')
        # check for experiment aborted flag
        if txt.find('EXPERIMENTABORTED') > -1:
            skipfooter = len(txt[txt.find('EXPERIMENTABORTED'):].split('\n')) - 1
        else:
            skipfooter = 0

        # preceding text
        pretxt = txt[:zidx]

        # zcurve data
        ztable = txt[zidx:]
        # column headers are next line after ZCURVE TABLE line
        header_start = ztable.find('\n') + 1
        header_end = header_start + ztable[header_start:].find('\n')
        header = ztable[header_start:header_end].split('\t')
        # units are next line after column headers
        unit_end = header_end + 1 + ztable[header_end + 1:].find('\n')
        units = ztable[header_end + 1:unit_end].split('\t')
        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n')) + 2

        # if table is indented, ignore empty left column
        if header[0] == '':
            usecols = header[1:]
        else:
            usecols = header

        # if extra tab at end of data rows, add an extra column to header to match (for Igor data)
        first_data_row = ztable[unit_end + 1: unit_end + 1 + ztable[unit_end + 1:].find('\n')]
        if first_data_row.split('\t')[-1] == '':
            header = header + ['extra_tab']

        # read data to DataFrame
        # python engine required to use skipfooter
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols,
                           skipfooter=skipfooter, engine='python', encoding=None, encoding_errors='ignore')

        # add timestamp
        try:
            dt = get_timestamp(file)
            time_col = np.intersect1d(['Time', 'T'], data.columns)[
                0]  # EIS files in Repeating jv-EIS files have column named 'Time' instead of 'T'
            data['timestamp'] = [dt + timedelta(seconds=t) for t in data[time_col]]
        except Exception:
            if warn:
                warnings.warn(f'Reading timestamp failed for file {file}')

    elif source == 'zplot':
        # find start of zcurve data
        zidx = txt.find('End Comments')

        # preceding text
        pretxt = txt[:zidx]

        # z data
        ztable = txt[zidx:]
        # column headers are in line above "End Comments"
        header = pretxt.split('\n')[-2].strip().split('\t')

        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n'))

        # if table is indented, ignore empty left column
        if header[0] == '':
            usecols = header[1:]
        else:
            usecols = header

        # read data to DataFrame
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols)

        # rename to standard format
        rename = {"Z'(a)": "Zreal", "Z''(b)": "Zimag", "Freq(Hz)": "Freq"}
        data = data.rename(rename, axis=1)

        # calculate Zmod and Zphz
        Zmod, Zphz = polar_from_complex(data)
        data['Zmod'] = Zmod
        data['Zphz'] = Zphz

    elif source == 'biologic':
        # header_count_index = txt.find('Nb header lines :')
        f_index = txt.find('freq/Hz')

        # Identify separator - could be tab or comma
        sep = txt[f_index + 7]

        skiprows = len(txt[:f_index].split('\n')) - 1

        # read data to DataFrame
        data = pd.read_csv(file, sep=sep, skiprows=skiprows, encoding=None, encoding_errors='ignore')

        # Rename columns to standard names (matching Gamry format)
        rename = {
            'freq/Hz': 'Freq',
            'Re(Z)/Ohm': 'Zreal',
            '-Im(Z)/Ohm': 'Zimag',
            '|Z|/Ohm': 'Zmod',
            'Phase(Z)/deg': 'Zphz',
            'time/s': 'Time',
            '<Ewe>/V': 'Vdc',
            '<I>/mA': 'Idc',
            # 'Cs/F',
            # 'Cp/F',
            # 'cycle number',
            # 'I Range',
            # '|Ewe|/V',
            # '|I|/A',
            # 'Ns',
            # '(Q-Qo)/mA.h',
            # 'Re(Y)/Ohm-1',
            # 'Im(Y)/Ohm-1',
            # '|Y|/Ohm-1',
            # 'Phase(Y)/deg',
            # 'dq/mA.h'
        }

        data = data.rename(rename, axis=1)

        data['Zimag'] *= -1
    elif source == 'relaxis':
        # Find header line
        header_index = txt.find('\nData: ')
        # Skip next two rows of metadata
        skiprows = len(txt[:header_index].split('\n')) + 2
        
        # Get data headers and rename
        header_line = txt[header_index + 1:].split('\n')[0]
        header_items = header_line.split('\t')
        header = [h.replace('Data: ', '') for h in header_items]
        
        # Read table
        data = pd.read_csv(file, sep='\t', skiprows=skiprows, encoding=None, 
                           header=None, names=header,
                           encoding_errors='ignore')
        
        # Rename to standard fields
        rename = {
            "Frequency": "Freq", 
            "Z'": "Zreal", 
            "Z''": "Zimag", 
            "|Z|": "Zmod",
            "Theta (Z)": "Zphz"
        }
        
        data = data.rename(rename, axis=1)        
    else:
        raise ValueError('Unrecognized file format')

    if return_tuple:
        data = get_eis_tuple(data)

    return data


def get_eis_tuple(data, min_freq=None, max_freq=None):
    """Convenience function - get frequency and Z from EIS DataFrame"""
    if type(data) != pd.DataFrame:
        data = read_eis(data)

    freq = data['Freq'].values.copy()
    z = data['Zreal'].values.copy() + 1j * data['Zimag'].values.copy()

    if min_freq is not None:
        index = freq >= min_freq
        freq = freq[index]
        z = z[index]

    if max_freq is not None:
        index = freq <= max_freq
        freq = freq[index]
        z = z[index]

    return freq, z


def get_chrono_tuple(data, start_time=None, end_time=None):
    if type(data) != pd.DataFrame:
        data = read_chrono(data)

    if 'elapsed' in data.columns:
        time_col = 'elapsed'
    else:
        time_col = find_time_column(data)
    times = data[time_col].values.copy()
    i_signal = data['Im'].values.copy()
    v_signal = data['Vf'].values.copy()

    # Truncate to time range
    if start_time is not None:
        index = times >= start_time
        times = times[index]
        i_signal = i_signal[index]
        v_signal = v_signal[index]

    if end_time is not None:
        index = times <= end_time
        times = times[index]
        i_signal = i_signal[index]
        v_signal = v_signal[index]

    return times, i_signal, v_signal


def get_hybrid_tuple(chrono_data, eis_data, append_eis_iv=False,
                     start_time=None, end_time=None,
                     min_freq=None, max_freq=None):
    """
    Get data tuple for hybrid measurement
    :param chrono_data: chrono file path or DataFrame
    :param eis_data: EIS file path or DataFrame
    :param bool append_eis_iv: if True, extract DC I-V data from the EIS data
    and append it to the chrono data. Only valid if the EIS measurement was
    performed after the chrono measurement
    :return:
    """
    if type(chrono_data) != pd.DataFrame:
        chrono_data = read_chrono(chrono_data)

    if type(eis_data) != pd.DataFrame:
        eis_data = read_eis(eis_data)

    times, i_sig, v_sig = get_chrono_tuple(chrono_data, start_time=start_time, end_time=end_time)
    freq, z = get_eis_tuple(eis_data, min_freq=min_freq, max_freq=max_freq)

    if append_eis_iv:
        time_offset = get_time_offset(eis_data, chrono_data)
        if time_offset > 0:
            t_eis, i_eis, v_eis = iv_from_eis(eis_data)
            t_eis += time_offset
            times = np.concatenate([times, t_eis])
            i_sig = np.concatenate([i_sig, i_eis])
            v_sig = np.concatenate([v_sig, v_eis])

    return times, i_sig, v_sig, freq, z


def get_time_offset(df, df_ref):
    return (df.loc[0, 'timestamp'] - df_ref.loc[0, 'timestamp']).total_seconds()


def iv_from_eis(data):
    if type(data) != pd.DataFrame:
        data = read_eis(data)

    if 'elapsed' in data.columns:
        times = data['elapsed'].values
    else:
        times = data['Time'].values
    i_sig = data['Idc'].values
    v_sig = data['Vdc'].values

    return times, i_sig, v_sig


def read_notes(file, parse=True):
    with open(file) as f:
        txt = f.read()

    # Find start of notes block
    notes_start = txt.find('NOTES')
    notes_start += txt[notes_start:].find('\n') + 2

    # Notes block is indented
    regex = r"\n(?!\t)"
    match = re.search(regex, txt[notes_start:])
    notes = txt[notes_start:notes_start + match.start(0)]

    # Parse into dict
    if parse:
        notes = {entry.split('\t')[0]: entry.split('\t')[1] for entry in notes.split('\n\t')}

    return notes


def read_curve(file):
    """
    Read generic CURVE data from Gamry DTA file
    :param file:
    :return:
    """
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    # find start of curve data
    cidx = txt.find('CURVE\tTABLE')

    # preceding text
    pretxt = txt[:cidx]

    # curve data
    ctable = txt[cidx:]

    # column headers are next line after CURVE TABLE line
    header_start = ctable.find('\n') + 1
    header_end = header_start + ctable[header_start:].find('\n')
    header = ctable[header_start:header_end].split('\t')

    # # units are next line after column headers
    # unit_end = header_end + 1 + ctable[header_end + 1:].find('\n')
    # units = ctable[header_end + 1:unit_end].split('\t')

    # determine # of rows to skip by counting line breaks in preceding text
    skiprows = len(pretxt.split('\n')) + 2

    # if table is indented, ignore empty left column
    if header[0] == '':
        usecols = header[1:]
    else:
        usecols = header
    # read data to DataFrame
    data = pd.read_csv(file, sep='\t', skiprows=skiprows, header=None, names=header, usecols=usecols)

    return data
