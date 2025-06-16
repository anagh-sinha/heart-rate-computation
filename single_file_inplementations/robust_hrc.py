import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

def read_ecg_from_edf(filename): #Read ECG data and sampling frequency.
    with pyedflib.EdfReader(filename) as f:
        for i in range(f.signals_in_file):
            label = f.getLabel(i).upper()
            if 'ECG' in label or 'EKG' in label:
                signal = f.readSignal(i)
                fs = f.getSampleFrequency(i)
                print(f"Found ECG channel: {f.getLabel(i)}, {fs} Hz, {len(signal)/fs:.1f} seconds")
                return signal, fs
        print("No ECG channel found in file") 
        return None, None


def method1_autocorrelation(signal, fs):
    """    
    ECG signals repeat with each heartbeat. Autocorrelation finds this repetition period by comparing the signal with shifted versions of itself. 
    The first peak in the autocorrelation (after lag 0) corresponds to one heartbeat period.
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


def visualize_results(signal, fs, autocorr_result, peak_result):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Original signal with detected peaks
    ax1 = axes[0]
    time = np.arange(len(signal)) / fs
    ax1.plot(time, signal, 'b-', label='ECG Signal')
    ax1.set_title('ECG Signal')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, min(10, time[-1])])  # Show first 10 seconds
    
    # Plot 2: Autocorrelation function
    ax2 = axes[1]
    hr_auto, autocorr = autocorr_result #hr_auto: heartrate from autocorrelation
    
    if len(autocorr) > 0:
        lags = np.arange(len(autocorr)) / fs #lags: time axis for autocorrelation
        ax2.plot(lags, autocorr, 'g-', label='Normalised Autocorrelation')
        if hr_auto:
            period = 60 / hr_auto # Mark the detected period with vertical dashed line.
            ax2.axvline(x=period, color='r', linestyle='--', label=f'Detected Heartbeat Period = {period:.3f}s ({hr_auto:.1f} bpm)') 
    
    ax2.set_title('Normalised Autocorrelation based Heart-Rate Computation')
    ax2.set_xlabel('Lag (seconds)')
    ax2.set_ylabel('Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 3])  # Show first 3 seconds of lag
    
    # Plot 3: Peak detection signal
    ax3 = axes[2]
    mean_hr, peaks, detection_signal = peak_result
    
    if len(detection_signal) > 0:  
        time_det = np.arange(len(detection_signal)) / fs
        ax3.plot(time_det, detection_signal, 'r-', label='Detection Signal (Filtered & Squared)')
        
        if len(peaks) > 0:
            if mean_hr: 
                ax3.plot(time_det[peaks], detection_signal[peaks], 'go', markersize=8, label=f'Detected Peaks ({mean_hr:.1f} bpm)')
            else:
                ax3.plot(time_det[peaks], detection_signal[peaks], 'go', markersize=8, label='Detected Peaks')

    ax3.set_title('Pan-Tompkins (R-Peak Detection) based Heart-Rate Computation')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, min(10, time_det[-1] if len(detection_signal) > 0 else 10)])
    
    plt.tight_layout(pad=3.0) #adjusts subplot parameters to prevent label or title overlap.
    plt.savefig('heart_rate_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    import sys
    filename = sys.argv[1]
    signal, fs = read_ecg_from_edf(filename)
    
    if signal is None or fs is None: 
        print("Failed to read ECG data from file")
        sys.exit(1)
    
    autocorr_result = method1_autocorrelation(signal, fs)
    peak_result = method2_peak_detection(signal, fs)
    if autocorr_result[0] and peak_result[0]:
        print(f"Difference between Heart rates: {abs(autocorr_result[0] - peak_result[0]) :.1f} bpm")
    
    visualize_results(signal, fs, autocorr_result, peak_result)

if __name__ == "__main__":
    main()