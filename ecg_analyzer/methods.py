"""Heart rate computation methods for ECG analysis."""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def method1_autocorrelation(signal, fs):
    """
    Compute heart rate using autocorrelation.
    
    The ECG signal is periodic with each heartbeat. Autocorrelation finds
    this period by measuring self-similarity at different time lags.
    
    Args:
        signal: ECG signal array
        fs: Sampling frequency in Hz
        
    Returns:
        tuple: (heart_rate, autocorrelation_function)
    """
    if len(signal) < fs:  # Edge Case: Signal too short
        print("Signal too short for autocorrelation")
        return None, np.array([])
    
    # Using a 10-second window for analysis
    window_size = int(10 * fs)
    if len(signal) > window_size:
        signal = signal[:window_size]
    
    # Remove mean and normalize
    signal = signal - np.mean(signal)
    if np.std(signal) < 1e-10: # Edge Case: Flat signal
        print("Flat signal detected")
        return None, np.array([])
    signal = signal / np.std(signal)
    
    # Calculate autocorrelation by comparing the signal with time-shifted versions of itself
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Only keeps the "forward shifts" (how similar the signal is to itself when we delay it).
    autocorr = autocorr / autocorr[0]  # Normalizes autocorrelation so the maximum value is 1 at zero shift (lag 0).
    
    # Find the first peak after lag 0 by searching between Reasonable Heartbeat Window 0.3 seconds (200 bpm) and 2 seconds (30 bpm)
    min_lag = int(0.3 * fs)
    max_lag = min(int(2.0 * fs), len(autocorr) - 1)  # Ensure we don't exceed array bounds
    if min_lag >= max_lag: # Edge Case: Invalid lag range
        print("Invalid lag range")
        return None, autocorr
    
    # Find peaks in the autocorrelation by looking for strong peaks (height > 0.3) to avoid false detections from noise.
    peaks, _ = find_peaks(autocorr[min_lag:max_lag], height = 0.3) 
    
    if len(peaks) > 0:
        # The first peak gives us the period
        period_samples = peaks[0] + min_lag
        period_seconds = period_samples / fs
        heart_rate = 60 / period_seconds
        
        print(f"Autocorrelation: Found period of {period_seconds:.3f}s = {heart_rate:.1f} bpm")
        return heart_rate, autocorr
    else:
        print("Autocorrelation: No clear periodicity found")
        return None, autocorr


def method2_peak_detection(signal, fs):
    """
    R-peak is the highest point in each QRS complex.
    We first enhance these peaks through filtering and then find them using threshold-based peak detection as in Pan-Tompkins algorithm.
    
    Args:
        signal: ECG signal array
        fs: Sampling frequency in Hz
        
    Returns:
        tuple: (heart_rate, peak_indices, detection_signal)
    """
    # Handle edge case of empty signal
    if len(signal) == 0:
        print("Empty signal")
        return None, np.array([]), np.array([])
    
    try:
        # Bandpass filter (5-15 Hz) to enhance QRS complexes
        b_band, a_band = butter(1, [5/(fs/2), 15/(fs/2)], 'band')
        filtered = filtfilt(b_band, a_band, signal)
        
        # Square the signal to make all values positive and enhance large deflections
        squared = filtered ** 2
        
        # Moving average to smooth (80ms window)
        window_size = max(1, int(0.08 * fs))  # Ensure at least 1 sample
        ma_filter = np.ones(window_size) / window_size
        detection_signal = np.convolve(squared, ma_filter, mode='same')
        
        # Edge Case: Flat signal
        if np.max(detection_signal) < 1e-10:
            print("Detection signal too flat")
            return None, np.array([]), detection_signal
        
        # Minimum distance between peaks: 0.3 seconds (200 bpm max)
        min_distance = int(0.3 * fs)
        
        # Optimised dynamic threshold: use 10% of the maximum value to avoid false detections from noise.
        threshold = 0.1 * np.max(detection_signal)
        peaks, _ = find_peaks(detection_signal, height=threshold, distance=min_distance)
        
        # Calculate heart rate from peak intervals
        if len(peaks) > 1:
            intervals_samples = np.diff(peaks)
            intervals_seconds = intervals_samples / fs
            heart_rates = 60 / intervals_seconds
            mean_hr = np.mean(heart_rates)
            print(f"Peak Detection: Found {len(peaks)} peaks, mean HR = {mean_hr:.1f} bpm")
            
            return mean_hr, peaks, detection_signal
        else:
            print(f"Peak Detection: Only found {len(peaks)} peaks, need at least 2") 
            return None, peaks, detection_signal
    except Exception as e:
        print(f"Peak detection error: {str(e)}")
        return None, np.array([]), signal