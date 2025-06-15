import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

#Read ECG data from an EDF file and its sampling frequency.
def read_ecg_from_edf(filename):
    with pyedflib.EdfReader(filename) as f:
            signal = f.readSignal(0)
            fs = f.getSampleFrequency(0)
            print(f"Found channel: {f.getLabel(0)}, {fs} Hz, {len(signal)/fs:.1f} seconds")
            return signal, fs

def method1_autocorrelation(signal, fs):
    """    
    ECG signals repeat with each heartbeat. Autocorrelation finds this repetition period by comparing the signal with shifted versions of itself. 
    """
    # Using a 10-second window for analysis
    window_size = int(10 * fs)
    if len(signal) > window_size:
        signal = signal[:window_size]
    
    # Remove mean and normalize
    signal = signal - np.mean(signal)
    signal = signal / np.std(signal)
    
    # Calculate autocorrelation by comparing the signal with time-shifted versions of itself
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Only keeps the forward shifts.
    autocorr = autocorr / autocorr[0]  # Normalizes it so the maximum value is 1 at zero shift (lag 0).
    
    # Search between Reasonable Heartbeat Window 0.3 seconds (200 bpm) and 2 seconds (30 bpm)
    min_lag = int(0.3 * fs)
    max_lag = int(2.0 * fs)
    
    # Find peaks in the autocorrelation by looking for strong peaks (height > 0.3) to avoid false detections from noise.
    peaks, _ = find_peaks(autocorr[min_lag:max_lag], height = 0.3) 

    # The first peak gives us the period.
    period_samples = peaks[0] + min_lag
    period_seconds = period_samples / fs
    heart_rate = 60 / period_seconds
    return heart_rate, autocorr

def method2_peak_detection(signal, fs):
    """
    First enhance R peaks through filtering and then find them using threshold-based peak detection as in Pan-Tompkins algorithm.
    """
    # Bandpass filter (5-15 Hz) to enhance QRS complexes
    b_band, a_band = butter(1, [5/(fs/2), 15/(fs/2)], 'band')
    filtered = filtfilt(b_band, a_band, signal)
    
    # Square the signal to make all values positive and enhance large deflections
    squared = filtered ** 2
    
    # Moving average to smooth (80ms window)
    window_size = int(0.08 * fs)
    ma_filter = np.ones(window_size) / window_size
    detection_signal = np.convolve(squared, ma_filter, mode='same')
    
    # Minimum distance between peaks: 0.3 seconds (200 bpm max)
    min_distance = int(0.3 * fs)
    
    # Optimised dynamic threshold: use 10% of the maximum value to avoid false detections from noise.
    threshold = 0.1 * np.max(detection_signal)
    peaks, _ = find_peaks(detection_signal, height=threshold, distance=min_distance)
    
    # Calculate heart rate from peak intervals
    intervals_samples = np.diff(peaks)
    intervals_seconds = intervals_samples / fs
    heart_rates = 60 / intervals_seconds
    mean_hr = np.mean(heart_rates)
    return mean_hr, peaks, detection_signal 

def main():
    import sys
    filename = sys.argv[1]
    signal, fs = read_ecg_from_edf(filename)
    autocorr_result = method1_autocorrelation(signal, fs)
    peak_result = method2_peak_detection(signal, fs)
    hr_auto = autocorr_result[0]
    hr_peak = peak_result[0]
    
    if hr_auto:
        print(f"Autocorrelation HR: {hr_auto:.1f} bpm")
    else:
        print(f"Autocorrelation HR: Failed to detect")
        
    if hr_peak:
        print(f"Peak Detection HR: {hr_peak:.1f} bpm")
    else:
        print(f"Peak Detection HR: Failed to detect")
    
    if hr_auto and hr_peak:
        print(f"Difference: {abs(hr_auto - hr_peak) :.1f} bpm")

if __name__ == "__main__":
    main()