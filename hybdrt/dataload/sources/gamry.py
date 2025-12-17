from typing import Tuple
from ..core import FileSource

def get_read_kwargs(text: str, source: FileSource) -> Tuple[dict, dict]:
    # TODO: need to test this. We may still need to specify ZCURVE for EIS data to skip past OCV data table
    data_index = text.upper().find("CURVE\tTABLE") + 1
    
    # preceding text
    pretxt = text[:data_index]

    # data table text
    table_text = text[data_index:]
    
    # Column headers are in second line of table text
    header_start = table_text.find('\n') + 1
    header_end = header_start + table_text[header_start:].find('\n')
    names = table_text[header_start:header_end].strip().split('\t')
    
    # units are next line after column headers
    unit_end = header_end + 1 + table_text[header_end + 1:].find('\n')
    units = table_text[header_end + 1:unit_end].split('\t')
    
    # Determine # of rows to skip by counting line breaks in preceding text
    # Skip 2 extra rows for units and header names
    skiprows = len(pretxt.split('\n')) + 2
        
    # check for experiment aborted flag
    if text.find('EXPERIMENTABORTED') > -1:
        skipfooter = len(text[text.find('EXPERIMENTABORTED'):].split('\n')) - 1
    else:
        skipfooter = 0
        
    read_kw = dict(
        sep='\t', 
        skiprows=skiprows,
        skipfooter=skipfooter,
        header=None, 
        names=names,
        engine='python',
        encoding_errors='ignore'
    )
    
    unit_kw = {}
    
    return read_kw, unit_kw
    
    
Z_HEADER_MAP = {
    "Freq": "freq",
    "Zreal": "z_re",
    "Zimag": "z_im",
    "Zmod": "z_mod",
    "Zphz": "z_phase",
    "Time": "time",
    "Idc": "i",
    "Vdc": "v"
}

CHRONO_HEADER_MAP = {
    "Time": "time",
    "Im": "i",
    "Vf": "v"
}

INVERT_Z_IM = False
