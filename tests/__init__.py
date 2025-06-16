"""Test package for ECG analyzer."""

from ecg_analyzer.io import read_ecg_from_edf
from ecg_analyzer.methods import method1_autocorrelation, method2_peak_detection
from ecg_analyzer.visualization import plot_results

__version__ = "1.0.0"
__all__ = [
    "read_ecg_from_edf",
    "method1_autocorrelation", 
    "method2_peak_detection",
    "plot_results"
]
