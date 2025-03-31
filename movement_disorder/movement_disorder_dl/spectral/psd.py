# 2025-01-17 Anthony Lee

from typing import Iterable, Tuple
import numpy as np
from scipy.signal import welch, get_window
import numpy.typing as npt
import warnings

__PREFERRED_FUNCTION__ = "psd_welch_contaldi_2023()"

def psd_welch_contaldi_2023(data:npt.NDArray, fs:float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Welch method PSD calculation based on (Contaldi, 2023)

    "Welch’s method (1sec Hamming window, 60% overlap, 250 points)"

    Contaldi, E., Leogrande, G., Fornaro, R., Comi, C., & Magistrelli, L. 
    (2023). Menstrual‐Related Fluctuations in a Juvenile‐Onset Parkinson’s 
    Disease Patient Treated with STN‐DBS: Correlation with Local Field 
    Potentials. Movement Disorders Clinical Practice, 11(1), 101. 
    https://doi.org/10.1002/mdc3.13931
    """
    nperseg = fs
    window_length_sec = 1  # 1-second
    overlap_ratio = 0.6  # 60%

    hamming_window_ndarray = get_window(
        window = "hamming",
        Nx = window_length_sec*fs,
        fftbins = False
    )

    freq, psd = welch(
        x = data, 
        fs = fs, 
        window = hamming_window_ndarray,  # 1 sec Hamming window
        nperseg = nperseg, 
        noverlap = int(nperseg * overlap_ratio),
        detrend=False, # https://github.com/scipy/scipy/issues/8045 - Also the original paper doesn't seem to have detrending
        return_onesided=True,
        scaling="density", 
        axis=-1,
        average="mean"
    )

    return (freq, psd)