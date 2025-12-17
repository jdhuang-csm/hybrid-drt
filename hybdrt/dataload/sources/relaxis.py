from typing import Tuple
from ..core import FileSource


def get_read_kwargs(text: str, source: FileSource) -> Tuple[dict, dict]:
    
    # Find header line
    header_index = text.find('\nData: ')
    
    # Skip header and next row of metadata
    skiprows = len(text[:header_index].split('\n')) + 2
    
    # Get data headers
    header_line = text[header_index + 1:].split('\n')[0]
    header = header_line.split('\t')
    
    # header = [h.replace('Data: ', '') for h in header_items]
    
    read_kw = dict(
        sep='\t', 
        skiprows=skiprows, 
        header=None, 
        names=header
    )
    
    unit_kw = {}
    
    return read_kw, unit_kw
        

Z_HEADER_MAP = {
    "Data: Frequency": 'freq',
    "Data: Z'": 'z_re',
    "Data: Z''": 'z_im',
    "Data: |Z|": 'z_mod',
    "Data: Theta (Z)": 'z_phase',
}

CHRONO_HEADER_MAP = {
    # "time/s": "time",
    # '<Ewe>/V': 'voltage',
    # '<I>/mA': 'current',
    # 'Ewe/V': 'voltage',
    # 'I/mA': 'current',
}

INVERT_Z_IM = False