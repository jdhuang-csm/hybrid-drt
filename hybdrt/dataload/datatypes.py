import numpy as np
from numpy import ndarray
from datetime import datetime
from typing import List, Optional
import pandas as pd
from enum import StrEnum, auto

from ..utils.eis import complex_from_polar


# Define generic immittance fields enum, with members ZFields and YFields
# This class defines field names and cartesian/polar representations
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
    

class YFields(ImmittanceFields):
    FREQUENCY = "freq"
    REAL = "y_re"
    IMAG = "y_im"
    MODULUS = "y_mod"
    PHASE = "y_phase"
    TIME = "time"
    
    
class ChronoFields(StrEnum):
    TIME = "time"
    CURRENT = "i"
    VOLTAGE = "v"
    

class BaseData(object):
    fields: List[str]
    
    def __init__(self, time: Optional[ndarray] = None, timestamp: Optional[datetime] = None, raw_data: Optional[pd.DataFrame] = None):
        self.time = time
        self.timestamp = timestamp
        self.raw_data = raw_data
        
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
        
    
# TODO: make full set of immittance fields available
# TODO: remove fields attribute from immittance data classes; use only field_enum (this may require changes elsewhere, e.g. drt.fit_eis)
class ImmittanceData(BaseData):
    field_enum: ImmittanceFields
    
    def __init__(self, freq: ndarray, x: ndarray, time: Optional[ndarray] = None, timestamp: Optional[datetime] = None,
                 raw_data: Optional[pd.DataFrame] = None):
        self.freq = freq
        self._x = x
        super().__init__(time=time, timestamp=timestamp, raw_data=raw_data)
        
    @property
    def fields(self):
        return [f.value for f in self.field_enum if hasattr(self, f.value)]
        
    @property
    def real(self):
        return self._x.real
    
    @property
    def imag(self):
        return self._x.imag
        
    @property
    def modulus(self):
        return np.abs(self._x)
    
    @property
    def phase(self):
        return np.angle(self._x, deg=True)
    
    def as_generic_dataframe(self):
        """Create a generic immittance dataframe

        :return: DataFrame with fields freq, real, imag, modulus, phase, and time (if available)
        :rtype: DataFrame
        """
        data_dict = {}
        for key in ["freq", "real", "imag", "modulus", "phase", "time"]:
            try:
                data_dict[key] = getattr(self, key)
            except AttributeError:
                pass
            
        return pd.DataFrame.from_dict(data_dict)
        
    def polar(self):
        """Return immittance data in polar form (modulus, phase)

        :return: modulus and phase arrays
        :rtype: Tuple[ndarray, ndarray]
        """
        return self.modulus, self.phase
    
    def cartesian(self):
        """Return immittance data in cartesian form (real, imaginary)

        :return: real and imaginary arrays
        :rtype: Tuple[ndarray, ndarray]
        """
        return self.real, self.imag
    
    def trim_freq(self, f_min: Optional[float] = None, f_max: Optional[float] = None):
        """Return a new ImmittanceData object limited to frequencies between f_min and f_max

        :param Optional[float] f_min: Minimum frequency. Defaults to None (no limit)
        :param Optional[float] f_max: Maximum frequency. Defaults to None (no limit)
        returns: ImmitanceData
        """
        f_min = -np.inf if f_min is None else f_min
        f_max = np.inf if f_max is None else f_max
        mask = (self.freq >= f_min) & (self.freq <= f_max)
        
        time_input = self.time[mask] if self.time is not None else None
            
        return self.__class__(self.freq[mask], self._x[mask], time=time_input, timestamp=self.timestamp)
    
    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, timestamp: Optional[datetime] = None):
        try:
            freq = data[cls.field_enum.FREQUENCY.value].values
        except KeyError:
            raise ValueError("Data must contain column {}".format(cls.field_enum.FREQUENCY.value))
        
        # Cartesian
        cart_cols = cls.field_enum.cartesian_fields()
        pol_cols = cls.field_enum.polar_fields()
        if all([x in data.columns for x in cart_cols]):
            z = data[cls.field_enum.REAL.value].values + 1j * data[cls.field_enum.IMAG.value].values
        # Polar
        elif all([x in data.columns for x in pol_cols]):
            z_re = data[cls.field_enum.MODULUS.value].values * np.cos(np.pi * data[cls.field_enum.PHASE.value].values / 180)
            z_im = data[cls.field_enum.MODULUS.value].values * np.sin(np.pi * data[cls.field_enum.PHASE.value].values / 180)
            z = z_re + 1j * z_im
        else:
            raise ValueError("Data must contain columns (z_re, z_im) or (z_mod, z_phase)")
        
        if cls.field_enum.TIME.value in data.columns:
            time = data[cls.field_enum.TIME.value].values
        else:
            time = None
            
        return cls(freq, z, time=time, timestamp=timestamp, raw_data=data)
    
    def invert(self):
        new_cls = YData if isinstance(self, ZData) else ZData
        return new_cls(self.freq, 1 / self._x, time=self.time, timestamp=self.timestamp)
    
class ZData(ImmittanceData):
    field_enum = ZFields
    # fields = ["freq", "z"]
    
    @property
    def z(self):
        return self._x
    
    @property
    def z_im(self):
        return self.imag
    
    @property
    def z_re(self):
        return self.real
    
    @property
    def z_mod(self):
        return self.modulus
    
    @property
    def z_phase(self):
        return self.phase
    
    
class YData(ImmittanceData):
    field_enum = YFields
    # fields = ["freq", "y"]
    
    @property
    def y(self):
        return self._x
    
    @property
    def y_im(self):
        return self.imag
    
    @property
    def y_re(self):
        return self.real
    
    @property
    def y_mod(self):
        return self.modulus
    
    @property
    def y_phase(self):
        return self.phase
    
# class ZData(BaseData):
#     fields = ["freq", "z"]
    
#     def __init__(self, freq: ndarray, z: ndarray, time: Optional[ndarray] = None, timestamp: Optional[datetime] = None):
#         self.freq = freq
#         self.z = z
#         super().__init__(time=time, timestamp=timestamp)
   
    
#     def trim_freq(self, f_min: Optional[float] = None, f_max: Optional[float] = None) -> BaseData:
#         """Return a new ZTuple object limited to frequencies between f_min and f_max

#         :param Optional[float] f_min: Minimum frequency. Defaults to None (no limit)
#         :param Optional[float] f_max: Maximum frequency. Defaults to None (no limit)
#         returns: ZTuple
#         """
#         f_min = -np.inf if f_min is None else f_min
#         f_max = np.inf if f_max is None else f_max
#         mask = (self.freq >= f_min) & (self.freq <= f_max)
        
#         time_input = self.time[mask] if self.time is not None else None
            
#         return ZData(self.freq[mask], self.z[mask], time=time_input, timestamp=self.timestamp)
    
#     @classmethod
#     def from_dataframe(cls, data: pd.DataFrame, timestamp: Optional[datetime] = None):
#         try:
#             freq = data[ZFields.FREQUENCY.value].values
#         except KeyError:
#             raise ValueError("Data must contain column freq")
        
#         # Cartesian
#         cart_cols = ["z_re", "z_im"]
#         pol_cols = ["modulus", "phase"]
#         if all([x in data.columns for x in cart_cols]):
#             z = data["z_re"].values + 1j * data["z_im"].values
#         # Polar
#         elif all([x in data.columns for x in pol_cols]):
#             z_re = data["modulus"].values * np.cos(np.pi * data["phase"].values / 180)
#             z_im = data["modulus"].values * np.sin(np.pi * data["phase"].values / 180)
#             z = z_re + 1j * z_im
#         else:
#             raise ValueError("Data must contain columns (z_re, z_im) or (modulus, phase)")
        
#         if "time" in data.columns:
#             time = data["time"].values
#         else:
#             time = None
            
#         return cls(freq, z, time=time, timestamp=timestamp)
            
    
# TODO: incorporate optional control mode (galv/pot)
class ChronoData(BaseData):
    field_enum = ChronoFields
    
    @property
    def fields(self):
        return [f.value for f in self.field_enum]
    
    """Class representing chrono/IV time-series data."""
    def __init__(self, time: ndarray, i: ndarray, v: ndarray, timestamp: Optional[datetime] = None, raw_data: Optional[pd.DataFrame] = None):
        self.i = i
        self.v = v
        
        super().__init__(time=time, timestamp=timestamp, raw_data=raw_data)
        
        
    def trim_time(self, t_min: Optional[float] = None, t_max: Optional[float] = None) -> BaseData:
        """Return a new ChronoData object limited to times between t_min and t_max

        :param Optional[float] t_min: Minimum time. Defaults to None (no limit)
        :param Optional[float] t_max: Maximum time. Defaults to None (no limit)
        returns: ChronoData
        """
        t_min = -np.inf if t_min is None else t_min
        t_max = np.inf if t_max is None else t_max
        mask = (self.time >= t_min) & (self.time <= t_max)
            
        return ChronoData(self.time[mask], self.i[mask], self.v[mask], timestamp=self.timestamp)
    
    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, timestamp: Optional[datetime] = None):
        try:
            time = data[cls.field_enum.TIME.value].values
            i = data[cls.field_enum.CURRENT.value].values
            v = data[cls.field_enum.VOLTAGE.value].values
        except KeyError as e:
            raise ValueError(f"Data must contain column {e.args[0]}")
        
        return cls(time, i, v, timestamp=timestamp, raw_data=data)
