import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import warnings
import numpy as np
from hybdrt.utils.eis import polar_from_complex
import calendar
import time
import re
from typing import Union, Optional
from pathlib import Path


FilePath = Union[Path, str]

# Try to import galvani to read MPR files
try:
    from galvani.BioLogic import MPRfile
except ImportError:
    _galvani_installed = False
else:
    _galvani_installed = True


def get_extension(file: FilePath):
    """Get file extension

    :param FilePath file: str or Path
    :return str: extension string
    """
    file = Path(file)
    return file.name.split('.')[-1]

    
def get_file_source(text: str):
    """Determine file source"""

    # Determine	format
    header = text.split('\n')[0]
    if header == 'EXPLAIN':
        source = 'gamry'
    elif header == 'ZPLOT2 ASCII':
        source = 'zplot'
    elif header == 'EC-Lab ASCII FILE':
        source = 'biologic'
    elif header.split(' ')[0] == 'RelaxIS':
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


# def check_source(file: FilePath, source: Optional[str] = None):
#     known_sources = ['gamry', 'zplot', 'biologic', 'relaxis']

#     if source is None:
#         source = get_file_source(read_txt(file))
#         if source is None:
#             raise ValueError('Could not identify file format. To read this file, '
#                              'manually specify the file format by providing the source argument. '
#                              'Recognized sources: {}'.format(', '.join(known_sources)))

#     if source not in known_sources:
#         raise ValueError('Unrecognized source {}. Recognized sources: {}'.format(source, ', '.join(known_sources)))

#     return source

_known_sources = ['gamry', 'zplot', 'biologic', 'relaxis']

def check_source(source):
    if source not in _known_sources:
        raise ValueError('Unrecognized data source {}. Recognized sources: {}'.format(source, ', '.join(_known_sources)))
    

def read_with_source(file: Union[Path, str], source: Optional[str] = None):
    text = read_txt(file)

    if source is None:
        source = get_file_source(text)
        if source is None:
            raise ValueError('Could not identify file format. To read this file, '
                             'manually specify the file format by providing the source argument. '
                             'Recognized sources: {}'.format(', '.join(_known_sources)))
    
    check_source(source)

    return text, source


def get_custom_file_time(file: Union[Path, str]) -> float:
    """Get timestamp from file generated by pygamry.

    :param Union[Path, str] file: Path to file.
    :return float: float indicating time
    """
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


def get_timestamp(file: Union[Path, str], source: Optional[str] = None) -> datetime:
    """Get experiment timestamp from file.

    :param Union[Path, str] file: Path to file.
    :param Optional[str] source: File source (e.g. gamry, biologic). 
        Defaults to None (auto-detect).
    :return datetime: datetime object
    """
    
    txt, source = read_with_source(file, source)

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
        
    elif source == 'biologic':
        find_str = 'Acquisition started on :'
        index = txt.find(find_str) + len(find_str)
        timestr = txt[index:].split('\n')[0].strip()
        dt = datetime.strptime(timestr, "%m/%d/%Y %H:%M:%S.%f")

    return dt


def _get_read_kwargs(
    text: str, 
    source: str, 
    data_start_str: Optional[str] = None,
    remove_blank: bool = True
    ):
    check_source(source)
    
    
    if source == 'gamry':
        data_index = text.upper().find(data_start_str) + 1
        
        # preceding text
        pretxt = text[:data_index]

        # data table text
        table_text = text[data_index:]
        
        # Column headers are in second line of table text
        header_start = table_text.find('\n') + 1
        header_end = header_start + table_text[header_start:].find('\n')
        names = table_text[header_start:header_end].split('\t')
        
        # units are next line after column headers
        unit_end = header_end + 1 + table_text[header_end + 1:].find('\n')
        units = table_text[header_end + 1:unit_end].split('\t')
        
        # Determine # of rows to skip by counting line breaks in preceding text
        # Skip 2 extra rows for units and header names
        skiprows = len(pretxt.split('\n')) + 2

        # # if table is indented, ignore empty left column
        # if names[0] == '':
        #     usecols = names[1:]
        # else:
        #     usecols = names
            
        # check for experiment aborted flag
        if text.find('EXPERIMENTABORTED') > -1:
            skipfooter = len(text[text.find('EXPERIMENTABORTED'):].split('\n')) - 1
        else:
            skipfooter = 0
            
        kwargs = dict(
            sep='\t', 
            skiprows=skiprows,
            skipfooter=skipfooter,
            header=None, 
            names=names,
            engine='python'
        )
    elif source == 'biologic':
        # Get number of header lines
        nh_str = 'Nb header lines :'
        nh_index = text.find(nh_str)
        if nh_index > 0:
            nh = int(text[nh_index + len(nh_str):].split('\n')[0].strip())
        else:
            # No header
            nh = 0
            
        # Identify separator - could be tab or comma
        header_row = text.split('\n')[nh - 1]
        if len(header_row.split('\t')) > 1:
            sep = '\t'
        else:
            sep = ','
            
        # Get header names
        names = header_row.split(sep)
            
        kwargs = dict(
            sep=sep,
            skiprows=nh,
            names=names,
        )
    elif source == 'zplot':
        # Find start of data
        data_index = text.find('End Comments')

        # preceding text
        pretxt = text[:data_index]

        # data table text
        table_text = text[data_index:]
        
        # column headers are in line above "End Comments"
        names = pretxt.split('\n')[-2].strip().split('\t')

        # determine # of rows to skip by counting line breaks in preceding text
        skiprows = len(pretxt.split('\n'))

        # # if table is indented, ignore empty left column
        # if names[0] == '':
        #     usecols = names[1:]
        # else:
        #     usecols = names
            
        kwargs = dict(
            sep='\t', 
            skiprows=skiprows, 
            header=None, 
            names=names
        )
    elif source == 'relaxis':
        # Find header line
        header_index = text.find('\nData: ')
        
        # Skip next two rows of metadata
        skiprows = len(text[:header_index].split('\n')) + 2
        
        # Get data headers and rename
        header_line = text[header_index + 1:].split('\n')[0]
        header_items = header_line.split('\t')
        header = [h.replace('Data: ', '') for h in header_items]
        
        kwargs = dict(
            sep='\t', 
            skiprows=skiprows, 
            header=None, 
            names=header
        )
    else:
        # No source given or detected. 
        # Assume a csv file and determine sep from first row.
        if len(text.split('\t')) > 1:
            sep = '\t'
        else:
            sep = None
            
        kwargs = dict(sep=sep)
        
    # Handle blank columns
    if 'names' in kwargs.keys():
        names = kwargs['names']
        for i, n in enumerate(names):
            if len(n) == 0:
                names[i] = f'blank{i}'
        usecols = [n for n in names if n.find('blank') == -1]
        kwargs['names'] = names
        if remove_blank:
            kwargs['usecols'] = usecols
    
    return kwargs


def read_mpr(file: FilePath):
    if not _galvani_installed:
        raise ModuleNotFoundError("The galvani package must be installed to read BioLogic .mpr files")
    
    file = Path(file)
    mpr = MPRfile(file.__str__())
    return mpr

        
def find_time_column(data: DataFrame, source: str):
    if source == 'gamry':        
        return np.intersect1d(['Time', 'T', 'time'], data.columns)[0]
    elif source == 'biologic':
        return 'time/s'
    


def read_header(file):
    txt = read_txt(file)

    # find start of curve data
    table_index = txt.upper().find('\nCURVE\tTABLE')
    if table_index == -1:
        table_index = txt.upper().find('\nZCURVE\tTABLE')

    return txt[:table_index + 1]


def read_generic(
    file: Union[Path, str], 
    source: Optional[str] = None,
    data_start_str: Optional[str] = None,
    with_timestamp: bool = True,
    **kwargs
    ) -> DataFrame:
    
    if get_extension(file) == 'mpr':
        mpr = read_mpr(file)
        data = pd.DataFrame(mpr.data)
    else:
        txt, source = read_with_source(file, source)

        # Set defaults
        read_kw = {
            'encoding': None,
            'encoding_errors': 'ignore',
            'engine': 'python'
        }
        
        # Get kwargs for reading based on header
        read_kw.update(_get_read_kwargs(txt, source, data_start_str))
        
        # Update with user-supplied kwargs
        read_kw.update(kwargs)
        
        # Read dataframe
        data = pd.read_csv(file, **read_kw)
    
        if with_timestamp:
            append_timestamp(file, data, source)
    
    return data


def append_timestamp(file: Union[Path, str], data: DataFrame, source: str, warn=True):
    # Get point-by-point timestamps
    try:
        dt = get_timestamp(file)
        time_col = find_time_column(data, source)
        data['timestamp'] = [dt + timedelta(seconds=t) for t in data[time_col]]
    except Exception as err:
        # Failed to get timestamp
        warnings.warn(f'Failed to get timestamp for file {Path(file).name} with error:\n{err}')
        raise err
    

def read_chrono(file, source=None, return_tuple=False, with_timestamp: bool = True):
    """
    Read chronopotentiometry data from Gamry .DTA file

    Args:
        file: file to read
    """
    if get_extension(file) == 'mpr':
        source = 'biologic'
        data_start_str = None
        with_timestamp = False
    else:
        txt, source = read_with_source(file, source)

        # Get kwargs for reading
        data_start_str = None
        if source == 'gamry':
            data_start_str = '\nCURVE\tTABLE'
            
    # Read into dataframe
    data = read_generic(file, source, data_start_str, with_timestamp)

    if return_tuple:
        data = get_chrono_tuple(data, source)

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


def read_eis(
    file, 
    source: Optional[str] = None, 
    warn: bool = True, 
    return_tuple: bool = False, 
    with_timestamp: bool = True,
    rename: bool = True
    ):

    """read EIS zcurve data from Gamry .DTA file"""
    
    if get_extension(file) == 'mpr':
        source = 'biologic'
        data_start_str = None
        with_timestamp = False
    else:
        txt, source = read_with_source(file, source)
        
        # Read into dataframe
        data_start_str = None
        if source == 'gamry':
            data_start_str = '\nZCURVE'
        
    data = read_generic(file, source, data_start_str, with_timestamp)
    
    # Rename to standard format
    if rename:
        if source == 'zplot':
            col_map = {"Z'(a)": "Zreal", "Z''(b)": "Zimag", "Freq(Hz)": "Freq"}
            data = data.rename(col_map, axis=1)

            # calculate Zmod and Zphz
            Zmod, Zphz = polar_from_complex(data)
            data['Zmod'] = Zmod
            data['Zphz'] = Zphz
        elif source == 'biologic':
            # Rename columns to standard names (matching Gamry format)
            col_map = {
                'freq/Hz': 'Freq',
                'Re(Z)/Ohm': 'Zreal',
                '-Im(Z)/Ohm': 'Zimag',
                '|Z|/Ohm': 'Zmod',
                'Phase(Z)/deg': 'Zphz',
                'time/s': 'Time',
                '<Ewe>/V': 'Vdc',
                '<I>/mA': 'Idc',
            }
            
            # Invert Zimag
            data['-Im(Z)/Ohm'] *= -1
        elif source == 'relaxis':
            # Rename to standard fields
            col_map = {
                "Frequency": "Freq", 
                "Z'": "Zreal", 
                "Z''": "Zimag", 
                "|Z|": "Zmod",
                "Theta (Z)": "Zphz"
            }
        else:
            col_map = {}
            
        data = data.rename(col_map, axis=1)
      
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


def get_chrono_tuple(
    data: Union[pd.DataFrame, Path, str], 
    source: Optional[str] = None,
    start_time: Optional[float] = None, 
    end_time: Optional[float] = None, 
    columns: Optional[list] = None):

    if type(data) != pd.DataFrame:
        data = read_chrono(data, source)
        
    if columns is None:
        # Determine columns from source
        if source is None or source == 'gamry':
            if 'elapsed' in data.columns:
                time_col = 'elapsed'
            else:
                time_col = find_time_column(data, source)
            columns = [time_col, 'Im', 'Vf'] 
        elif source == 'biologic':
            # TODO: handle units and possible column renaming
            columns = ['time/s', 'I/mA', 'Ewe/V']
            
    print(source, columns)
        
    tup = tuple([data[c].values.copy() for c in columns])
    
    # times = data[columns[0]].copy()
    # i_signal = data[columns[1]].values.copy()
    # v_signal = data[columns[2]].values.copy()

    # Truncate to time range
    if start_time is not None:
        index = tup[0] >= start_time
        tup = tuple([arr[index] for arr in tup])
        
    if end_time is not None:
        index = tup[0] <= end_time
        tup = tuple([arr[index] for arr in tup])
        # index = times <= end_time
        # times = times[index]
        # i_signal = i_signal[index]
        # v_signal = v_signal[index]

    return tup


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
