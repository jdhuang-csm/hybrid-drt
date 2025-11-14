import numpy as np
from numpy import ndarray
from pathlib import Path
from galvani.BioLogic import MPRfile
import pandas as pd
from typing import Union, Callable, Tuple

from ..utils import units


def read_mpr(file: Union[str, Path], unscale: bool = False):
    file = Path(file)
    mpr = MPRfile(file.__str__())
    
    if unscale:
        # Convert all units to base units (remove m, k, mu, etc. scaling)
        mpr.data = unscale_data(mpr.data)
        
    return mpr


def split_list(x: list, split_func: Callable) -> Tuple[list]:
    """Split each entry of a list with split_func, then return separate
    lists of the split_func outputs.

    :param list x: _description_
    :param Callable split_func: _description_
    :return Tuple[List]: A tuple of lists. The ith list contains the
        ith split_func output for all entries of the input 
        list.
    """
    split = [split_func(xi) for xi in x]
    return tuple([[s[i] for s in split] for i in range(len(split[0]))])


def split_fieldname(fieldname: str):
    # Find last slash separator
    index = fieldname[::-1].find('/')
    if index == -1:
        return fieldname, None
    
    index = -(index + 1)
    name = fieldname[:index]
    unit = fieldname[index + 1:]
        
    return name, unit


def split_unit(unit: Union[str, None]):
    if unit is None:
        return None, None
    elif len(unit) > 1 and unit[0] in units._all_prefixes:
        prefix = unit[0]
        base_unit = unit[1:]
    else:
        prefix = None
        base_unit = unit
        
    return prefix, base_unit


def unscale_data(data: ndarray):
    # TODO: consider precision loss?
    # Determine prefixes and base units
    fieldnames = list(data.dtype.fields.keys())
    names, unit_list = split_list(fieldnames, split_fieldname)
    prefixes, base_units = split_list(unit_list, split_unit)
    
    # Check each field
    scaled = data.copy()
    new_fieldnames = fieldnames.copy()
    for i in range(len(names)):
        prefix = prefixes[i]
        if prefix is not None:
            # Remove prefix and return to raw scaling
            fieldname = fieldnames[i]
            up = units.UnitPrefix(prefix)
            scaled[fieldname] = up.scaled_to_raw(scaled[fieldname])
            new_fieldnames[i] = f'{names[i]}/{base_units[i]}'
            
    new_dtype = np.dtype(dict(zip(new_fieldnames, data.dtype.fields.values())))
    scaled.dtype = new_dtype
    return scaled