import pandas as pd
import numpy as np
from numpy import ndarray
from typing import List, Union, Optional


class DataTuple(tuple):
    fields: List[str]
    
    def __new__(cls, data: Union[tuple, list]):
        if len(data) != len(cls.fields):
            raise ValueError(f"Expected {len(cls.fields)} fields, but received {len(data)}")
        obj = super().__new__(cls, tuple(data))
        
        # for field, field_data in zip(cls.fields, data):
        #     setattr(obj, field, field_data)
        
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