import numpy as np
import pandas as pd


def polar_from_complex(data):
    if type(data) == pd.core.frame.DataFrame:
        Zmod = (data['Zreal'].values ** 2 + data['Zimag'].values ** 2) ** 0.5
        Zphz = (180 / np.pi) * np.arctan(data['Zimag'].values / data['Zreal'].values)
    elif type(data) == np.ndarray:
        Zmod = ((data * data.conjugate()) ** 0.5).real
        Zphz = (180 / np.pi) * np.arctan(data.imag / data.real)

    return Zmod, Zphz


def complex_from_polar(data):
    if type(data) == pd.core.frame.DataFrame:
        Zmod = data['Zmod'].values
        Zphz = data['Zphz'].values
    elif type(data) == np.ndarray:
        Zmod = data[:, 0]
        Zphz = data[:, 1]

    Zreal = Zmod * np.cos(np.pi * Zphz / 180)
    Zimag = Zmod * np.sin(np.pi * Zphz / 180)

    return Zreal, Zimag


def construct_eis_df(frequencies, z):
    """
    Construct dataframe from complex impedance array
    :param ndarray frequencies: array of frequencies
    :param ndarray z: array of impedance values
    """
    df = pd.DataFrame(frequencies, columns=['Freq'])
    df['Zreal'] = z.real
    df['Zimag'] = z.imag

    # Get polar data
    zmod, zphz = polar_from_complex(z)
    df['Zmod'] = zmod
    df['Zphz'] = zphz

    return df


def complex_vector_to_concat(z, axis=-1):
    return np.concatenate([z.real, z.imag], axis=axis)


def concat_vector_to_complex(z):
    if len(z) % 2 == 1:
        raise ValueError('z must be of even length')
    else:
        num_complex = int(len(z) / 2)
        return z[:num_complex] + 1j * z[num_complex:]
