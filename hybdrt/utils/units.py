import numpy as np

from typing import Tuple


class UnitPrefix(object):
    scale_map = {
        'G': 1e9,
        'M': 1e6,
        'k': 1e3,
        '': 1,
        'm': 1e-3,
        'mu': 1e-6,
        'n': 1e-9
    }

    reverse_scale_map = {v: k for k, v in scale_map.items()}

    chr_map = {
        'mu': 181
    }
    
    reverse_char_map = {chr(v): k for k, v in chr_map.items()}

    def __init__(self, prefix):
        if prefix not in self.scale_map.keys():
            try:
                # Look up special characters
                prefix = self.reverse_char_map[prefix]
            except KeyError:
                raise ValueError(f"Unrecognized unit prefix: {prefix}")
        self.prefix = prefix

    @classmethod
    def from_value(cls, value, min_factor=None, max_factor=None):
        """
        Select appropriate prefix from value
        """
        if not np.isscalar(value):
            value = np.max(np.abs(value))
            
        # Get scale options from largest to smallest
        scales = list(reversed(sorted(list(cls.reverse_scale_map.keys()))))
        
        # Limit available scales
        if min_factor is not None:
            scales = [s for s in scales if s >= min_factor]
        if max_factor is not None:
            scales = [s for s in scales if s <= max_factor]

        if value == 0 or value is None:
            scale = 1
        else:
            # Set floor on magnitude to ensure that a matching scale is found
            value = max(abs(value), min(scales))
            
            # Get largest scale that is less than value
            scale = next(s for s in scales if value >= s)

        # Get corresponding prefix
        prefix = cls.reverse_scale_map[scale]

        return cls(prefix)


    def set_prefix(self, prefix):
        if prefix not in self.scale_map.keys():
            raise ValueError(f'Invalid prefix {prefix}. Options: {list(self.scale_map.values())}')

        self._prefix = prefix

    def get_prefix(self):
        return self._prefix

    prefix = property(get_prefix, set_prefix)

    @property
    def scale(self):
        return self.scale_map[self.prefix]

    @property
    def char(self):
        if self.chr_map.get(self.prefix, None) is not None:
            return chr(self.chr_map[self.prefix])
        return self.prefix

    def raw_to_scaled(self, raw_value):
        if raw_value is None:
            return None
        return raw_value / self.scale

    def scaled_to_raw(self, scaled_value):
        if scaled_value is None:
            return None
        return scaled_value * self.scale
    

def get_scaled_value(value):
    try:
        return UnitPrefix.from_value(value).raw_to_scaled(value)
    except TypeError:
        return value

def get_prefix_char(value):
    try:
        return UnitPrefix.from_value(value).char
    except TypeError:
        return ""
    
    
def get_scaled_value_and_prefix(value, min_factor: float = None, max_factor: float = None) -> Tuple[float, str]:
    unit = UnitPrefix.from_value(value, min_factor=min_factor, max_factor=max_factor)
    return unit.raw_to_scaled(value), unit.char


# Enumerate all possible prefix characters
ALL_PREFIXES = [get_prefix_char(v) for v in UnitPrefix.scale_map.values()]