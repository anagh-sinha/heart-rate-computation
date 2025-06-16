"""Main entry point for ECG heart rate analysis."""

import sys
from ecg_analyzer.io import read_ecg_from_edf
from ecg_analyzer.methods import method1_autocorrelation, method2_peak_detection
from ecg_analyzer.visualization import plot_results


def main():
    """
    Main function to process command line arguments.
    Perform complete ECG analysis using both methods and save the visualization.
    
    Args:
        filename: Path to EDF file
        
    Returns:
        Print: Results from both methods
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <edf_file> [additional_files...]")
        sys.exit(1)

    filename = sys.argv[1]
    signal, fs = read_ecg_from_edf(filename)
    
    if signal is None or fs is None: 
        print("Failed to read ECG data from file")
        sys.exit(1)
    
    autocorr_result = method1_autocorrelation(signal, fs)
    peak_result = method2_peak_detection(signal, fs)
    if autocorr_result[0] and peak_result[0]:
        print(f"Difference between Heart rates: {abs(autocorr_result[0] - peak_result[0]) :.1f} bpm")
    
    plot_results(signal, fs, autocorr_result, peak_result)

if __name__ == "__main__":
    main()
