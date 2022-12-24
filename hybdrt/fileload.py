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
    # determine	format
    if txt.split('\n')[0] == 'EXPLAIN':
        source = 'gamry'
    elif txt.split('\n')[0] == 'ZPLOT2 ASCII':
        source = 'zplot'

    return source


def get_custom_file_time(file):
    with open(file, 'r') as f:
        txt = f.read()

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


def get_timestamp(file):
    """Get experiment start timestamp from file"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

    source = get_file_source(file)

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


def read_chrono(file):
    """
    Read chronopotentiometry data from Gamry .DTA file

    Args:
        file: file to read
    """
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()

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

        return data


def concatenate_chrono_data(files, trim_index=None, trim_time=None):
    """Concatenate curve data from multiple files"""
    # sort files
    files = sorted(files, key=get_timestamp)

    dfs = [read_chrono(file) for file in files]
    start_times = [df['timestamp'][0] for df in dfs]
    start_time = min(start_times)

    ts_func = lambda ts: (ts - start_time).dt.total_seconds()
    for df in dfs:
        df['elapsed'] = ts_func(df['timestamp'])

    # Trim beginning of each file
    if trim_index is not None:
        dfs = [df.loc[trim_index:] for df in dfs]
    if trim_time is not None:
        dfs = [df[df['T'] >= trim_time] for df in dfs]

    df_out = pd.concat(dfs, ignore_index=True)

    return df_out


def concatenate_eis_data(files):
    """Concatenate curve data from multiple files"""
    # sort files
    files = sorted(files, key=get_timestamp)

    dfs = [read_eis(file) for file in files]
    start_times = [df['timestamp'][0] for df in dfs]
    start_time = min(start_times)

    ts_func = lambda ts: (ts - start_time).dt.total_seconds()
    for df in dfs:
        df['elapsed'] = ts_func(df['timestamp'])

    # # Trim beginning of each file
    # if trim_index is not None:
    #     dfs = [df.loc[trim_index:] for df in dfs]
    # if trim_time is not None:
    #     dfs = [df[df['T'] >= trim_time] for df in dfs]

    df_out = pd.concat(dfs)

    return df_out


def read_eis(file, warn=True):
    """read EIS zcurve data from Gamry .DTA file"""
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    source = get_file_source(file)

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

    return data


def get_eis_tuple(df, min_freq=None, max_freq=None):
    """Convenience function - get frequency and Z from EIS DataFrame"""
    freq = df['Freq'].values.copy()
    z = df['Zreal'].values.copy() + 1j * df['Zimag'].values.copy()

    if min_freq is not None:
        index = freq >= min_freq
        freq = freq[index]
        z = z[index]

    if max_freq is not None:
        index = freq <= max_freq
        freq = freq[index]
        z = z[index]

    return freq, z


def get_chrono_tuple(df, start_time=None, end_time=None):
    if 'elapsed' in df.columns:
        time_col = 'elapsed'
    else:
        time_col = find_time_column(df)
    times = df[time_col].values.copy()
    i_signal = df['Im'].values.copy()
    v_signal = df['Vf'].values.copy()

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
