import numpy as np
from numpy import ndarray
from datetime import datetime
from typing import List, Optional
import pandas as pd
from enum import StrEnum, auto

from ..utils.eis import complex_from_polar
# class ImpedanceData:
#     """Class representing EIS data."""
#     def __init__(self, frequency: List[float], z_real: List[float], z_imag: List[float]):
#         self.frequency = frequency
#         self.z_real = z_real
#         self.z_imag = z_imag
#         self.timestamps: Optional[List] = None

class ImmittanceFields(StrEnum):
    @classmethod
    def cartesian_fields(cls):
        return cls.REAL.value, cls.IMAG.value
    
    @classmethod
    def polar_fields(cls):
        return cls.MODULUS.value, cls.PHASE.value
    
    
class ZFields(ImmittanceFields):
    FREQUENCY = "freq"
    REAL = "z_re"
    IMAG = "z_im"
    MODULUS = "z_mod"
    PHASE = "z_phase"
    TIME = "time"
    

class YFields(StrEnum):
    FREQUENCY = "freq"
    REAL = "y_re"
    IMAG = "y_im"
    MODULUS = "y_mod"
    PHASE = "y_phase"
    TIME = "time"
    
   


class BaseData(object):
    fields: List[str]
    
    def __init__(self, time: Optional[ndarray] = None, timestamp: Optional[datetime] = None):
        self.time = time
        self.timestamp = timestamp
        
    def as_tuple(self):
        return tuple(getattr(self, f) for f in self.fields)
        
    def as_array(self):
        return np.array(self.as_tuple()).T
    
    def as_dataframe(self):
        return pd.DataFrame.from_dict(dict(zip(self.fields, self.as_tuple())))
    
    @property
    def timestamps(self):
        if all([self.time, self.timestamp]):
            return self.timestamp + self.time
        
    
# TODO: generalize to immittance
class ImmittanceData(BaseData):
    field_enum: ImmittanceFields
    
    def __init__(self, freq: ndarray, x: ndarray, time: Optional[ndarray] = None, timestamp: Optional[datetime] = None):
        self.freq = freq
        self._x = x
        super().__init__(time=time, timestamp=timestamp)
   
    
    def trim_freq(self, f_min: Optional[float] = None, f_max: Optional[float] = None) -> BaseData:
        """Return a new ZTuple object limited to frequencies between f_min and f_max

        :param Optional[float] f_min: Minimum frequency. Defaults to None (no limit)
        :param Optional[float] f_max: Maximum frequency. Defaults to None (no limit)
        returns: ZTuple
        """
        f_min = -np.inf if f_min is None else f_min
        f_max = np.inf if f_max is None else f_max
        mask = (self.freq >= f_min) & (self.freq <= f_max)
        
        time_input = self.time[mask] if self.time is not None else None
            
        return ZData(self.freq[mask], self._x[mask], time=time_input, timestamp=self.timestamp)
    
    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, timestamp: Optional[datetime] = None):
        try:
            freq = data[cls.field_enum.FREQUENCY.value].values
        except KeyError:
            raise ValueError("Data must contain column freq")
        
        # Cartesian
        cart_cols = cls.field_enum.cartesian_fields()
        pol_cols = cls.field_enum.polar_fields()
        if all([x in data.columns for x in cart_cols]):
            z = data["z_re"].values + 1j * data["z_im"].values
        # Polar
        elif all([x in data.columns for x in pol_cols]):
            z_re = data["modulus"].values * np.cos(np.pi * data["phase"].values / 180)
            z_im = data["modulus"].values * np.sin(np.pi * data["phase"].values / 180)
            z = z_re + 1j * z_im
        else:
            raise ValueError("Data must contain columns (z_re, z_im) or (modulus, phase)")
        
        if "time" in data.columns:
            time = data["time"].values
        else:
            time = None
            
        return cls(freq, z, time=time, timestamp=timestamp)
    
    
    
class ZData(BaseData):
    fields = ["freq", "z"]
    
    def __init__(self, freq: ndarray, z: ndarray, time: Optional[ndarray] = None, timestamp: Optional[datetime] = None):
        self.freq = freq
        self.z = z
        super().__init__(time=time, timestamp=timestamp)
   
    
    def trim_freq(self, f_min: Optional[float] = None, f_max: Optional[float] = None) -> BaseData:
        """Return a new ZTuple object limited to frequencies between f_min and f_max

        :param Optional[float] f_min: Minimum frequency. Defaults to None (no limit)
        :param Optional[float] f_max: Maximum frequency. Defaults to None (no limit)
        returns: ZTuple
        """
        f_min = -np.inf if f_min is None else f_min
        f_max = np.inf if f_max is None else f_max
        mask = (self.freq >= f_min) & (self.freq <= f_max)
        
        time_input = self.time[mask] if self.time is not None else None
            
        return ZData(self.freq[mask], self.z[mask], time=time_input, timestamp=self.timestamp)
    
    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, timestamp: Optional[datetime] = None):
        try:
            freq = data[ZFields.FREQUENCY.value].values
        except KeyError:
            raise ValueError("Data must contain column freq")
        
        # Cartesian
        cart_cols = ["z_re", "z_im"]
        pol_cols = ["modulus", "phase"]
        if all([x in data.columns for x in cart_cols]):
            z = data["z_re"].values + 1j * data["z_im"].values
        # Polar
        elif all([x in data.columns for x in pol_cols]):
            z_re = data["modulus"].values * np.cos(np.pi * data["phase"].values / 180)
            z_im = data["modulus"].values * np.sin(np.pi * data["phase"].values / 180)
            z = z_re + 1j * z_im
        else:
            raise ValueError("Data must contain columns (z_re, z_im) or (modulus, phase)")
        
        if "time" in data.columns:
            time = data["time"].values
        else:
            time = None
            
        return cls(freq, z, time=time, timestamp=timestamp)
            
        


class TimeSeriesData:
    """Class representing chrono/IV time-series data."""
    def __init__(self, time: List[float], current: List[float], voltage: List[float]):
        self.time = time
        self.current = current
        self.voltage = voltage
        self.timestamps: Optional[List] = None
        self.file_ids: Optional[List[int]] = None
