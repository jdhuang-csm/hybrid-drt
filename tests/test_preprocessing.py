import numpy as np
import pandas as pd
from pathlib import Path

from hybdrt.preprocessing import downsample_data

def test_downsample_data_decimate():
    script_dir = Path(__file__).parent
    # Raw data
    raw_data = pd.read_csv(script_dir / "ChronoTestData.csv", index_col=None)
    # Downsampled data
    downsampled_data = pd.read_csv(script_dir / "ChronoDownsampledData.csv", index_col=None)
    
    # Downsample raw noisy data
    sample_times, sample_i, sample_v, sample_index = downsample_data(
        raw_data["times"].values, raw_data["i_signal"].values, raw_data["v_noisy"].values,     
        step_model="ideal",
        method="decimate", decimation_interval=10, decimation_factor=2, 
        antialiased=True
        )
    
    # Verify downsampled data matches expected values
    assert np.allclose(sample_times, downsampled_data["times"].values)
    assert np.allclose(sample_i, downsampled_data["i_signal"].values)
    assert np.allclose(sample_v, downsampled_data["v_filtered"].values)
    # Verify that sample indices correspond to downsampled data
    assert np.allclose(raw_data["v_signal"].values[sample_index], downsampled_data["v_signal"].values)