from typing import Tuple
from ..core import FileSource
from .eclab import process_fieldnames


def get_read_kwargs(text: str, source: FileSource) -> Tuple[dict, dict]:
    # TODO: need to check mpt format
    if source == FileSource.ECLAB_TXT:
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
        
        # Handle unnamed columns
        names = [name if name.strip() != '' else f'unnamed_{i}/au' for i, name in enumerate(names)]
        
        # Get units and scale prefixes
        # stripped_names, unit_list = split_list(names, split_fieldname)
        # prefixes, base_units = split_list(unit_list, split_unit)
        prefixes, base_units, new_names = process_fieldnames(names)
        
        read_kw = dict(
            sep=sep,
            skiprows=nh,
            names=names,
            encoding_errors='ignore'
            )
        
        unit_kw = dict(   
            base_units=base_units,
            unit_prefixes=prefixes,
            new_names=new_names
        )
            
        return read_kw, unit_kw
        

        

# TODO: does df.rename break if columns in map are not in df? 
Z_HEADER_MAP = {
    'freq/Hz': 'freq',
    'Re(Z)/Ohm': 'z_re',
    '-Im(Z)/Ohm': 'z_im',
    '|Z|/Ohm': 'z_mod',
    'Phase(Z)/deg': 'z_phase',
    'time/s': 'time',
    # Averaged values
    '<I>/A': 'i',
    '<Ewe>/V': 'v',
    # Raw values
    'I/A': 'i',
    'Ewe/V': 'v',
}

CHRONO_HEADER_MAP = {
    "time/s": "time",
    # Averaged values
    '<Ewe>/V': 'v',
    '<I>/A': 'i',
    # Raw values
    'Ewe/V': 'v',
    'I/A': 'i',
}

INVERT_Z_IM = True