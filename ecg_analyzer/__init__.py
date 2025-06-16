# ecg_analyzer/__init__.py
"""ECG Heart Rate Analyzer Package."""

from .io import read_ecg_from_edf
from .methods import method1_autocorrelation, method2_peak_detection
from .visualization import plot_results

__version__ = "1.0.0"
__all__ = [
    "read_ecg_from_edf",
    "method1_autocorrelation", 
    "method2_peak_detection",
    "plot_results"
]

# tests/__init__.py
"""Test package for ECG analyzer."""