import numpy as np
import pandas as pd
from typing import Union, Optional, List
from pathlib import Path

from enum import StrEnum, auto



class FileSource(StrEnum):
    GAMRY_DTA = auto()
    ZPLOT = auto()
    ECLAB_TXT = auto()
    ECLAB_MPR = auto()
    ECLAB_MPT = auto()
    RELAXIS = auto()
    CUSTOM = auto()
    
# Mapping for known file extensions
EXT_SOURCE_MAP = {
    "dta": FileSource.GAMRY_DTA,
    "mpr": FileSource.ECLAB_MPR,
    "mpt": FileSource.ECLAB_MPT,
    "z": FileSource.ZPLOT,
}

# Mapping for known header tags
HEADER_SOURCE_MAP = {
    "EXPLAIN": FileSource.GAMRY_DTA,
    "EC-Lab ASCII FILE": FileSource.ECLAB_TXT,
    "ZPLOT2 ASCII": FileSource.ZPLOT,
    "RelaxIS": FileSource.RELAXIS
}
    
    
def _process_filesource(source: Union[str, FileSource]):
    # Convenience function to allow users to provide either string or enum input
    if isinstance(source, str):
        try:
            return FileSource[source.upper()]
        except KeyError:
            raise KeyError(f"Unexpected source name {source}. Known sources: {FileSource._member_names_}")
    elif isinstance(source, FileSource):
        return source
    raise TypeError(f"Unexpected source value {source}. Expected a string or FileSource member")


def read_txt(file: Union[Path, str]):
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    return txt


def get_extension(file: Union[Path, str]) -> str:
    """Return the file extension from a given file path.

    :param file: The file path or name.
    :type file: Union[pathlib.Path, str]
    :return: The file extension (without the leading dot).
    :rtype: str
    :raises ValueError: If the file has no extension.

    **Example**

    .. code-block:: python

        >>> get_extension("example.txt")
        'txt'
        >>> get_extension(Path("/home/user/archive.tar.gz"))
        'gz'
    """
    file = Path(file)
    parts = file.name.split(".")
    if len(parts) == 1:
        raise ValueError(f"No extension found for file: {file}")
    return parts[-1]


def detect_source_from_ext(file: Union[str, Path]) -> Union[FileSource, None]:
    """Detect the file source based on extension or content.

    :param file: Path to the data file.
    :type file: Union[str, pathlib.Path]
    :return: Detected file source.
    :rtype: FileSource
    """
    file = Path(file)

    # Guess from extension first
    ext = get_extension(file).lower()
    return EXT_SOURCE_MAP.get(ext)

    
def detect_source_from_text(text: str) -> Union[FileSource, None]:
    """Determine file source from contents"""
    # Get header line
    header = text.split('\n')[0]
    
    # Try exact match
    source = HEADER_SOURCE_MAP.get(header)
    
    if source is None:
        # Match first word only
        header = header.split(" ")[0]
        source = {k.split(" ")[0]: v for k, v in HEADER_SOURCE_MAP.items()}.get(header)
        
    return source

def detect_file_source(file: Union[str, Path]) -> Union[FileSource, None]:
    # First try using extension
    source = detect_source_from_ext(file)
    
    if source is None:
        # Try using header
        source = detect_source_from_text(read_txt(file))
        
    return source
    

class DataTuple(tuple):
    fields: List[str]
    
    def __new__(cls, data: Union[tuple, list]):
        if len(data) != len(cls.fields):
            raise ValueError(f"Expected {len(cls.fields)} fields, but received {len(data)}")
        obj = super().__new__(cls, tuple(data))
        
        return obj
        
    def as_array(self):
        return np.array(self).T
    
    def as_dataframe(self):
        return pd.DataFrame(self.as_array(), columns=self.fields)
    
    def _get_field(self, field: str):
        return self[self.fields.index(field)]
    
    
class ChronoTuple(DataTuple):
    fields = ["time", "current", "voltage"]
    
    @property
    def time(self):
        return self._get_field("time")
    
    @property
    def current(self):
        return self._get_field("current")
    
    @property
    def voltage(self):
        return self._get_field("voltage")
    
    
class ZTuple(DataTuple):
    fields = ["freq", "z"]
    
    @property
    def freq(self):
        return self._get_field("freq")
    
    @property
    def z(self):
        return self._get_field("z")
    
    def trim_freq(self, f_min: Optional[float] = None, f_max: Optional[float] = None) -> DataTuple:
        """Return a new ZTuple object limited to frequencies between f_min and f_max

        :param Optional[float] f_min: Minimum frequency. Defaults to None (no limit)
        :param Optional[float] f_max: Maximum frequency. Defaults to None (no limit)
        returns: ZTuple
        """
        f_min = -np.inf if f_min is None else f_min
        f_max = np.inf if f_max is None else f_max
        mask = (self.freq >= f_min) & (self.freq <= f_max)
        
        return ZTuple((self.freq[mask], self.z[mask]))


