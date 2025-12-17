from typing import Tuple
from ..core import FileSource


def get_read_kwargs(text: str, source: FileSource) -> Tuple[dict, dict]:
    # Find start of data
    data_index = text.find('End Comments')

    # preceding text
    pretxt = text[:data_index]
    
    # column headers are in line above "End Comments"
    names = pretxt.splitlines()[-2].strip().split('\t')

    # determine # of rows to skip by counting line breaks in preceding text
    skiprows = len(pretxt.splitlines())

    read_kw = dict(
        sep='\t', 
        skiprows=skiprows, 
        header=None, 
        names=names
    )
    
    unit_kw = {}
    
    return read_kw, unit_kw
        

Z_HEADER_MAP = {
    "Freq(Hz)": 'freq',
    "Z'(a)": 'z_re',
    "Z''(b)": 'z_im',
}

CHRONO_HEADER_MAP = {
    # "time/s": "time",
    # '<Ewe>/V': 'voltage',
    # '<I>/mA': 'current',
    # 'Ewe/V': 'voltage',
    # 'I/mA': 'current',
}

INVERT_Z_IM = False