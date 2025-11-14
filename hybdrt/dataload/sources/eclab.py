from ..core import FileSource


def get_read_kwargs(text: str, source: FileSource):
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
            
        return dict(
            sep=sep,
            skiprows=nh,
            names=names,
        )
        

# TODO: does df.rename break if columns in map are not in df? 
Z_HEADER_MAP = {
    'freq/Hz': 'freq',
    'Re(Z)/Ohm': 'z_re',
    '-Im(Z)/Ohm': 'z_im',
    '|Z|/Ohm': 'modulus',
    'Phase(Z)/deg': 'phase',
    'time/s': 'time',
    # Averaged values
    '<Ewe>/V': 'voltage',
    '<I>/mA': 'current',
    # Raw values
    'Ewe/V': 'voltage',
    'I/mA': 'current',
}

CHRONO_HEADER_MAP = {
    "time/s": "time",
    '<Ewe>/V': 'voltage',
    '<I>/mA': 'current',
    'Ewe/V': 'voltage',
    'I/mA': 'current',
}

INVERT_Z_IM = True