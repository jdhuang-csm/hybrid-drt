# Initial distribution of capacitive times (DCT) implementation
# ----------------------------------------------------------------------

import numpy as np
from numpy import ndarray

from .drt1d import DRT


class DCT(DRT):
    def _prep_for_fit(self,
                      # Chrono data
                      times, i_signal, v_signal,
                      # EIS data
                      frequencies, z,
                      # Chrono options
                      step_times, step_sizes, downsample, downsample_kw, offset_steps, smooth_inf_response,
                      # Scaling
                      scale_data, rp_scale,
                      penalty_type, derivative_weights):
        
        data, mat = super()._prep_for_fit(
            times, i_signal, v_signal,
            frequencies, z,
            step_times, step_sizes, downsample, downsample_kw, offset_steps, smooth_inf_response,
            scale_data, rp_scale,
            penalty_type, derivative_weights
        )
        
        
        (rm_drt, induc_rv, inf_rv, cap_rv, rm_dop, zm_drt, induc_zv, cap_zv, zm_dop, penalty_matrices) = mat
        
        for m in [rm_drt, rm_dop, zm_drt, zm_dop]:
            invert_mat(m, inplace=True)
        
        for name in ["response", "impedance", "rm_dop", "zm_dop"]:
            if self.fit_matrices.get(name) is not None:
                invert_mat(self.fit_matrices[name], True)
        
        
        return data, mat
    
    def _prep_impedance_prediction_matrix(self, frequencies):
        mat = super()._prep_impedance_prediction_matrix(frequencies)
        mat = tuple([invert_mat(m) for m in mat])
        return mat
    
    def _prep_chrono_prediction_matrix(self, times, input_signal, step_times, step_sizes,
                                       op_mode, offset_steps, smooth_inf_response):
        mat = super()._prep_chrono_prediction_matrix(times, input_signal, step_times, step_sizes,
                                       op_mode, offset_steps, smooth_inf_response)
        rm, induc_rv, inf_rv, cap_rv, rm_dop = mat
        
        for m in [rm, rm_dop]:
            invert_mat(m, inplace=True)
                
        return mat
    
    
def should_invert(m: ndarray):
    return np.max(m.real) > 0

def invert_mat(m: ndarray, inplace=False):
    # TODO: reconsider this logic for chrono data
    if m is None:
        return m
    
    if should_invert(m):
        if inplace:
            m *= -1
        else:
            m = m * -1
    
    return m
        

def preprocess(frequencies: ndarray, z: ndarray, drt: DRT, keep_errors=True, **kw):
    drt.fit_eis(frequencies, z, **kw)
    z_clean = drt.predict_z(frequencies, include_inductance=False, include_cap=False)
    
    if keep_errors:
        # Add the residuals back to the cleaned data to maintain the 
        # signal-to-noise ratio (avoid false confidence)
        z_pred = drt.predict_z(frequencies)
        z_err = z - z_pred
        z_clean += z_err
        
    return z_clean
