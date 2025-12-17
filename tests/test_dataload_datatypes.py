import numpy as np
import pandas as pd
from hybdrt.dataload.datatypes import ZData, ChronoData
from datetime import datetime


def test_zdata_from_dataframe_cartesian_and_properties():
    df = pd.DataFrame({
        "freq": [1.0, 10.0, 100.0],
        "z_re": [1.0, 2.0, 3.0],
        "z_im": [0.0, -1.0, 1.0],
    })

    zdata = ZData.from_dataframe(df)

    # Check freq and complex values
    assert np.allclose(zdata.freq, np.array([1.0, 10.0, 100.0]))
    assert np.allclose(zdata.z.real, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(zdata.z.imag, np.array([0.0, -1.0, 1.0]))

    # modulus and phase
    expected_mod = np.abs(zdata.z)
    assert np.allclose(zdata.modulus, expected_mod)

    expected_phase = np.angle(zdata.z, deg=True)
    assert np.allclose(zdata.phase, expected_phase)


def test_zdata_trim_freq():
    df = pd.DataFrame({
        "freq": [1.0, 10.0, 100.0],
        "z_re": [1.0, 2.0, 3.0],
        "z_im": [0.0, -1.0, 1.0],
    })
    zdata = ZData.from_dataframe(df)

    trimmed = zdata.trim_freq(f_min=5.0, f_max=50.0)
    assert np.allclose(trimmed.freq, np.array([10.0]))
    assert np.allclose(trimmed.z.real, np.array([2.0]))


def test_chronodata_from_dataframe_and_trim_time_and_timestamps():
    df = pd.DataFrame({
        "time": [0.0, 1.0, 2.0, 3.0],
        "i": [0.1, 0.2, 0.3, 0.4],
        "v": [1.0, 1.1, 1.2, 1.3],
    })

    ts = datetime(2020, 1, 1, 12, 0, 0)
    chrono = ChronoData.from_dataframe(df, timestamp=ts)

    assert np.allclose(chrono.time, np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(chrono.i, np.array([0.1, 0.2, 0.3, 0.4]))

    trimmed = chrono.trim_time(t_min=1.5, t_max=3.0)
    assert np.allclose(trimmed.time, np.array([2.0, 3.0]))
