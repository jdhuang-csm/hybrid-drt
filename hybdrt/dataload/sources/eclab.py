from typing import Callable, Tuple, Union, List

from ...utils import units


def split_list(x: list, split_func: Callable) -> Tuple[list]:
    """Split each entry of a list with split_func, then return separate
    lists of the split_func outputs.

    :param list x: List to split.
    :param Callable split_func: Function with which to split each entry of x.
    :return Tuple[List]: A tuple of lists. The ith list contains the
        ith split_func output for all entries of the input 
        list.
    """
    split = [split_func(xi) for xi in x]
    return tuple([[s[i] for s in split] for i in range(len(split[0]))])


def split_fieldname(fieldname: str):
    """Split a fieldname into name and unit components.
    :param fieldname: Fieldname string (e.g., "voltage/mV")
    :type fieldname: str
    :return: name and unit components
    :rtype: Tuple[str, Union[str, None]]
    """
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
    elif len(unit) > 1 and unit[0] in units.ALL_PREFIXES:
        prefix = unit[0]
        base_unit = unit[1:]
    else:
        prefix = None
        base_unit = unit
        
    return prefix, base_unit

def process_fieldnames(fieldnames: List[str]):
    """Given a list of field names with arbitrary units, extract unit prefixes, base units, and new field names with base units

    :param fieldnames: _description_
    :type fieldnames: List[str]
    :return: _description_
    :rtype: _type_
    """
    names, unit_list = split_list(fieldnames, split_fieldname)
    prefixes, base_units = split_list(unit_list, split_unit)
    
    new_names = [f'{names[i]}/{base_units[i]}' for i in range(len(names))]
    
    return prefixes, base_units, new_names