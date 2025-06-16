"""EDF file reading module for ECG data extraction."""

import pyedflib


def read_ecg_from_edf(filename):
    """
    Read ECG data from an EDF file.
    
    Args:
        filename (str): Path to the EDF file
        
    Returns:
        tuple: (signal, sampling_frequency) or (None, None) if reading fails
    """
    try:
        with pyedflib.EdfReader(filename) as f:
            for i in range(f.signals_in_file):
                label = f.getLabel(i).upper()
                if 'ECG' in label or 'EKG' in label:
                    signal = f.readSignal(i)
                    fs = f.getSampleFrequency(i)
                print(f"Found ECG channel: {f.getLabel(i)}, {fs} Hz, {len(signal)/fs:.1f} seconds")
                return signal, fs
        print("No ECG channel found in file") 
            
    except Exception as e:
        print(f"Error reading EDF file: {e}")
        return None, None