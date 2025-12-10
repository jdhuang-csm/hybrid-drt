from pathlib import Path
from typing import Union, Optional, List
from enum import StrEnum, auto
from datetime import datetime
from .mpr import read_mpr

FilePath = Union[str, Path]


class FileSource(StrEnum):
    GAMRY_DTA = auto()
    ZPLOT = auto()
    ECLAB_TXT = auto()
    ECLAB_MPR = auto()
    ECLAB_MPT = auto()
    RELAXIS = auto()
    CUSTOM = auto()
    
    @property
    def software(self):
        return self.name.split("_")[0]


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
    "RelaxIS": FileSource.RELAXIS,
}


def get_extension(file: FilePath) -> str:
    """Return the file extension from a given file path."""
    file = Path(file)
    parts = file.name.split(".")
    if len(parts) == 1:
        raise ValueError(f"No extension found for file: {file}")
    return parts[-1]


def read_txt(file: FilePath) -> str:
    """Read a text file, falling back to latin1 if UTF-8 fails."""
    try:
        with open(file, "r") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file, "r", encoding="latin1") as f:
            return f.read()


def detect_source_from_ext(file: FilePath) -> Optional[FileSource]:
    ext = get_extension(file).lower()
    return EXT_SOURCE_MAP.get(ext)


def detect_source_from_text(text: str) -> Optional[FileSource]:
    header = text.split("\n")[0]
    source = HEADER_SOURCE_MAP.get(header)
    if source is None:
        header_word = header.split(" ")[0]
        source = {k.split(" ")[0]: v for k, v in HEADER_SOURCE_MAP.items()}.get(header_word)
    return source


def detect_file_source(file: FilePath) -> Optional[FileSource]:
    source = detect_source_from_ext(file)
    if source is None:
        source = detect_source_from_text(read_txt(file))
    return source

def read_with_source(file: FilePath, source: Optional[FileSource] = None):
    # Read text and return source. Avoid multiple file reads
    text = read_txt(file)
    if source is None:
        source = detect_source_from_ext(file)
        if source is None:
            source = detect_source_from_text(text)
            
    return text, source


# ------------------------------
# Timestamp extraction
# ------------------------------
def get_line(text: str, pattern: str) -> str:
    # Get first line of text that contains pattern
    start = text.find(pattern)
    end = text[start:].find('\n') + start
    return text[start:end]


def detect_time_column(columns: List[str], source: str):
    if source == FileSource.GAMRY_DTA or source is None:
        possible = ["time", "t"]
        return next(c for c in possible if c.lower() in columns)
    if source == FileSource.ECLAB_TXT:
        return 'time/s'
    
    
def extract_timestamp(file: FilePath, source: Optional[FileSource] = None) -> Optional[datetime]:
    """Get experiment timestamp from file.

    :param Union[Path, str] file: Path to file.
    :param Optional[str] source: File source (e.g. gamry, biologic). 
        Defaults to None (auto-detect).
    :return datetime: datetime object
    """
    if get_extension(file).lower() == "mpr":
        mpr = read_mpr(file)
        return mpr.timestamp
    else:
        txt, source = read_with_source(file, source)

        if source == FileSource.GAMRY_DTA:
            date_line = get_line(txt, "DATE")
            date = date_line.split('\t')[2]

            time_line = get_line(txt, "TIME")
            time_txt = time_line.split('\t')[2]

            return datetime.strptime(f"{date} {time_txt}", "%m/%d/%Y %H:%M:%S")
        
        elif source == FileSource.ZPLOT:
            date_line = get_line(txt, "Date")
            date = date_line.split()[1]

            time_line = get_line(txt, "Time")
            time_txt = time_line.split()[1]

            return datetime.strptime(f"{date} {time_txt}", "%m-%d-%Y %H:%M:%S")
            
        elif source == FileSource.ECLAB_TXT:
            find_str = 'Acquisition started on :'
            index = txt.find(find_str) + len(find_str)
            timestr = txt[index:].splitlines[0].strip()
            return datetime.strptime(timestr, "%m/%d/%Y %H:%M:%S.%f")
        
    return

