from pathlib import Path
from typing import Union, Optional, Tuple, List
import pandas as pd
from datetime import datetime
import warnings

from .core import (
    FileSource, detect_file_source, read_with_source, get_extension, extract_timestamp,
    detect_time_column
)
from .srcconvert import standardize_z_data, standardize_chrono_data, reader_kwarg_gen
from .datatypes import ZData, ChronoData
from .sources.eclab_mpr import read_mpr
from ..utils import units

FilePath = Union[str, Path]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def read_eis(
    file: FilePath,
    source: Optional[FileSource] = None,
    standardize: bool = True,
    as_dataframe: bool = False,
    with_timestamp: bool = False,
    return_source: bool = False,
) -> Union[ZData, Tuple[ZData, FileSource]]:
    """Read impedance spectroscopy data and normalize into ImpedanceData."""
    if source is None:
        source = detect_file_source(file)
    if source is None:
        raise ValueError(f"Could not detect source for {file}")

    # Convert to ZData: if standardize==True and as_dataframe==False
    convert = standardize and not as_dataframe
    
    # Only load timestamps into the dataframe if convert==False
    df, source = _read_generic(file, source, with_timestamp=(with_timestamp and not convert))

    if standardize:
        # standardize column names
        df = standardize_z_data(df, source=source)

    if not convert:
        # Leave as dataframe (unstandardized or partially standardized)
        data = df
    else:
        # Convert to ZData (standardized)
        if with_timestamp:
            ts = extract_timestamp(file, source)
        else:
            ts = None
        data = ZData.from_dataframe(df, timestamp=ts)
    

    return (data, source) if return_source else data


def read_chrono(
    file: FilePath,
    source: Optional[FileSource] = None,
    standardize: bool = True,
    as_dataframe: bool = False,
    with_timestamp: bool = False,
    return_source: bool = False,
) -> Union[ChronoData, Tuple[ChronoData, FileSource]]:
    """Read chrono/IV data and normalize into TimeSeriesData."""
    if source is None:
        source = detect_file_source(file)
    if source is None:
        raise ValueError(f"Could not detect source for {file}")
    
    # Convert to ChronoData: if standardize==True and as_dataframe==False
    convert = standardize and not as_dataframe

    df, source = _read_generic(file, source, with_timestamp=(with_timestamp and not convert))

    if standardize:
        # standardize column names
        df = standardize_chrono_data(df, source=source)
   
    if not convert:
        # Leave as dataframe (unstandardized or partially standardized)
        data = df
    else:
        # Convert to ChronoData (standardized)
        if with_timestamp:
            ts = extract_timestamp(file, source)
        else:
            ts = None
        data = ChronoData.from_dataframe(df, timestamp=ts)

    return (data, source) if return_source else data


# ---------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------

# def concat_impedance(files: List[FilePath], *, sort_by_time: bool = True, **kwargs) -> ImpedanceData:
#     datasets = [read_impedance(f, with_timestamp=True, **kwargs) for f in files]
#     freq, zr, zi, timestamps = [], [], [], []

#     for d in datasets:
#         freq.extend(d.frequency)
#         zr.extend(d.z_real)
#         zi.extend(d.z_imag)
#         if hasattr(d, "timestamps") and d.timestamps:
#             timestamps.extend(d.timestamps)

#     if sort_by_time and timestamps:
#         sorted_idx = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
#         freq = [freq[i] for i in sorted_idx]
#         zr = [zr[i] for i in sorted_idx]
#         zi = [zi[i] for i in sorted_idx]
#         timestamps = [timestamps[i] for i in sorted_idx]

#     result = ImpedanceData(freq, zr, zi)
#     if timestamps:
#         result.timestamps = timestamps
#     return result


# def concat_timeseries(files: List[FilePath], *, sort_by_time: bool = True, **kwargs) -> TimeSeriesData:
#     datasets = [read_timeseries(f, with_timestamp=True, **kwargs) for f in files]
#     t, i, v, timestamps, file_ids = [], [], [], [], []

#     for idx, d in enumerate(datasets):
#         t.extend(d.time)
#         i.extend(d.current)
#         v.extend(d.voltage)
#         if hasattr(d, "timestamps") and d.timestamps:
#             timestamps.extend(d.timestamps)
#         file_ids.extend([idx] * len(d.time))

#     if sort_by_time and timestamps:
#         sorted_idx = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
#         t = [t[i] for i in sorted_idx]
#         i = [i[i] for i in sorted_idx]
#         v = [v[i] for i in sorted_idx]
#         timestamps = [timestamps[i] for i in sorted_idx]
#         file_ids = [file_ids[i] for i in sorted_idx]

#     result = TimeSeriesData(t, i, v)
#     if timestamps:
#         result.timestamps = timestamps
#     result.file_ids = file_ids
#     return result


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
# def _get_read_kwargs(
#     text: str, 
#     source: FileSource, 
#     # data_start_str: Optional[str] = None,
#     remove_blank: bool = True
#     ):
    
#     if source == FileSource.GAMRY_DTA:
#         # TODO: need to test this. We may still need to specify ZCURVE for EIS data to skip past OCV data table
#         data_index = text.upper().find("CURVE\tTABLE") + 1
        
#         # preceding text
#         pretxt = text[:data_index]

#         # data table text
#         table_text = text[data_index:]
        
#         # Column headers are in second line of table text
#         header_start = table_text.find('\n') + 1
#         header_end = header_start + table_text[header_start:].find('\n')
#         names = table_text[header_start:header_end].split('\t')
        
#         # units are next line after column headers
#         unit_end = header_end + 1 + table_text[header_end + 1:].find('\n')
#         units = table_text[header_end + 1:unit_end].split('\t')
        
#         # Determine # of rows to skip by counting line breaks in preceding text
#         # Skip 2 extra rows for units and header names
#         skiprows = len(pretxt.split('\n')) + 2
            
#         # check for experiment aborted flag
#         if text.find('EXPERIMENTABORTED') > -1:
#             skipfooter = len(text[text.find('EXPERIMENTABORTED'):].split('\n')) - 1
#         else:
#             skipfooter = 0
            
#         kwargs = dict(
#             sep='\t', 
#             skiprows=skiprows,
#             skipfooter=skipfooter,
#             header=None, 
#             names=names,
#             engine='python'
#         )
#     elif source == FileSource.ECLAB_TXT:
#         # Get number of header lines
#         nh_str = 'Nb header lines :'
#         nh_index = text.find(nh_str)
#         if nh_index > 0:
#             nh = int(text[nh_index + len(nh_str):].split('\n')[0].strip())
#         else:
#             # No header
#             nh = 0
            
#         # Identify separator - could be tab or comma
#         header_row = text.split('\n')[nh - 1]
#         if len(header_row.split('\t')) > 1:
#             sep = '\t'
#         else:
#             sep = ','
            
#         # Get header names
#         names = header_row.split(sep)
            
#         kwargs = dict(
#             sep=sep,
#             skiprows=nh,
#             names=names,
#         )
#     elif source == FileSource.ZPLOT:
#         # Find start of data
#         data_index = text.find('End Comments')

#         # preceding text
#         pretxt = text[:data_index]

#         # data table text
#         table_text = text[data_index:]
        
#         # column headers are in line above "End Comments"
#         names = pretxt.split('\n')[-2].strip().split('\t')

#         # determine # of rows to skip by counting line breaks in preceding text
#         skiprows = len(pretxt.split('\n'))
            
#         kwargs = dict(
#             sep='\t', 
#             skiprows=skiprows, 
#             header=None, 
#             names=names
#         )
#     elif source == FileSource.RELAXIS:
#         # Find header line
#         header_index = text.find('\nData: ')
        
#         # Skip next two rows of metadata
#         skiprows = len(text[:header_index].split('\n')) + 2
        
#         # Get data headers and rename
#         header_line = text[header_index + 1:].split('\n')[0]
#         header_items = header_line.split('\t')
#         header = [h.replace('Data: ', '') for h in header_items]
        
#         kwargs = dict(
#             sep='\t', 
#             skiprows=skiprows, 
#             header=None, 
#             names=header
#         )
#     else:
#         # No source given or detected. 
#         # Assume a csv file and determine sep from first row.
#         if len(text.split('\t')) > 1:
#             sep = '\t'
#         else:
#             sep = None
            
#         kwargs = dict(sep=sep)
        
#     # Handle blank columns
#     if 'names' in kwargs.keys():
#         names = kwargs['names']
#         for i, n in enumerate(names):
#             if len(n) == 0:
#                 names[i] = f'blank{i}'
#         usecols = [n for n in names if n.find('blank') == -1]
#         kwargs['names'] = names
#         if remove_blank:
#             kwargs['usecols'] = usecols
            
#     # Set defaults
#     defaults = {
#             'encoding': None,
#             'encoding_errors': 'ignore',
#             'engine': 'python'
#         }
#     for k, v in defaults.items():
#         kwargs.setdefault(k, v)
    
#     return kwargs


def _read_generic(
    file: Union[Path, str], 
    source: Optional[str] = None,
    data_start_str: Optional[str] = None,
    with_timestamp: bool = True,
    **kwargs
    ) -> Tuple[pd.DataFrame, FileSource]:
    
    if get_extension(file).lower() == 'mpr':
        mpr = read_mpr(file, unscale=True)
        data = pd.DataFrame(mpr.data)
        source = FileSource.ECLAB_MPR
    else:
        txt, source = read_with_source(file, source)

        # Get kwargs for reading based on source and header
        read_kw, unit_kw = reader_kwarg_gen(source)(txt, source) #, data_start_str)
        
        # print(read_kw)
        
        # Update with user-supplied kwargs
        read_kw.update(kwargs)
        
        # Read into dataframe
        data = pd.read_csv(file, **read_kw)
    
        # Handle units as needed
        if len(unit_kw) > 0:
            data = unscale_data(data, unit_kw["unit_prefixes"], unit_kw["new_names"])

            
    if with_timestamp:
        timestamp = extract_timestamp(file, source)
        time_col = detect_time_column(list(data.columns), source)
        if all([timestamp, time_col]):
            data.timestamp = timestamp + data[time_col]
        else:
            warnings.warn(f"Could not load timestamps for file {file}")
    
    
    return data, source


def unscale_data(data: pd.DataFrame, prefixes, new_names):
    """Scale all fields in a dataframe to their base units (e.g., mV -> V).

    :param data: data array; must be a structured ndarray
    :type data: ndarray
    :return: unscaled data array
    :rtype: ndarray
    """
    # TODO: consider precision loss?
    if len(prefixes) != len(new_names) or len(prefixes) != len(data.columns):
        raise ValueError("Number of columns, prefixes, and new_names must be equal")  
    
    # Scale each field
    old_names = list(data.columns)
    scaled = data.copy()
    for i in range(len(new_names)):
        prefix = prefixes[i]
        if prefix is not None:
            # Remove prefix and return to raw scaling
            up = units.UnitPrefix(prefix)
            scaled[old_names[i]] = up.scaled_to_raw(data[old_names[i]])
            
    # Rename fields with new units
    scaled = scaled.rename(dict(zip(old_names, new_names)), axis=1)
    
    return scaled


