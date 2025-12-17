import pandas as pd
from typing import Callable
from .sources import eclab_txt, gamry, relaxis, zplot
from .core import FileSource


def get_module(source: FileSource):
    if source.software == "GAMRY":
        return gamry
    if source.software == "ECLAB":
        return eclab_txt
    if source.software == "RELAXIS":
        return relaxis
    if source.software == "ZPLOT":
        return zplot
    
    # Could replace with getattr(sources, source.software.lower())
    

def reader_kwarg_gen(source: FileSource) -> Callable[[str, FileSource], dict]:
    """Get function to generate kwargs for reader

    :param source: _description_
    :type source: FileSource
    :return: Function to generate kwargs for read_csv based on file text
        and source
    :rtype: Callable[[str, FileSource]]
    """
    return get_module(source).get_read_kwargs
    
    
def standardize_z_data(data: pd.DataFrame, source: FileSource):
    module = get_module(source)
    data = data.rename(module.Z_HEADER_MAP, axis=1)
    
    if module.INVERT_Z_IM and "z_im" in list(data.columns):
        data["z_im"] *= -1
        
    return data


def standardize_chrono_data(data: pd.DataFrame, source: FileSource):
    module = get_module(source)
    data = data.rename(module.CHRONO_HEADER_MAP, axis=1)
        
    return data


# def load_kwarg_func(source: FileSource):
#     return get_module(source).get_read_kwargs