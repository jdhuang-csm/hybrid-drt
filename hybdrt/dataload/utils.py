import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import warnings
import numpy as np
from hybdrt.utils.eis import polar_from_complex
import calendar
import time
import re
from typing import Union, Optional
from pathlib import Path



def get_extension(file: Union[Path, str]) -> str:
    """Get file extension

    :param FilePath file: str or Path
    :return str: extension string
    """
    file = Path(file)
    return file.name.split('.')[-1]


def read_txt(file: Union[Path, str]):
    try:
        with open(file, 'r') as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin1') as f:
            txt = f.read()
    return txt

