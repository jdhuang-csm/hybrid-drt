import pandas as pd
import numpy as np
from hybdrt.dataload import srcconvert
from hybdrt.dataload.core import FileSource


def test_standardize_z_data_gamry():
    # Build a minimal Gamry-style dataframe
    df = pd.DataFrame({
        "Freq": [1.0, 10.0],
        "Zreal": [1.0, 2.0],
        "Zimag": [0.5, -0.5],
        "Zmod": [1.1180, 2.062],
        "Zphz": [30.0, -14.0],
        "Idc": [0.0, 0.1],
        "Vdc": [1.0, 1.1],
    })

    out = srcconvert.standardize_z_data(df.copy(), source=FileSource.GAMRY_DTA)

    # Columns should be renamed to the standard names
    assert "freq" in out.columns
    assert "z_re" in out.columns
    assert "z_im" in out.columns
    assert "z_mod" in out.columns
    assert "z_phase" in out.columns
    assert "i" in out.columns
    assert "v" in out.columns

    # Gamry INVERT_Z_IM is False, so sign should be unchanged
    assert out.loc[0, "z_im"] == 0.5


def test_standardize_z_data_eclab():
    # EC-Lab style headers map and invert imaginary part
    df = pd.DataFrame({
        "freq/Hz": [1.0],
        "Re(Z)/Ohm": [1.0],
        "-Im(Z)/Ohm": [0.2],
        "|Z|/Ohm": [1.02],
        "Phase(Z)/deg": [11.0],
        "I/A": [0.0],
        "Ewe/V": [1.0],
    })

    out = srcconvert.standardize_z_data(df.copy(), source=FileSource.ECLAB_TXT)

    assert "freq" in out.columns
    assert "z_re" in out.columns
    assert "z_im" in out.columns
    assert "z_mod" in out.columns
    assert "z_phase" in out.columns
    assert "i" in out.columns
    assert "v" in out.columns
    # ECLAB sets INVERT_Z_IM = True, so z_im should be multiplied by -1
    assert out.loc[0, "z_im"] == -0.2
