"""Visualization module for ECG analysis results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_results(signal, fs, autocorr_result, peak_result, filename='results.png'):
    """
    Create a comprehensive visualization of both analysis methods.
    
    Args:
        signal: Original ECG signal
        fs: Sampling frequency
        autocorr_result: (heart_rate, autocorrelation) from method 1
        peak_result: (heart_rate, peaks, detection_signal) from method 2
        filename: Output filename for the plot
    """
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
    print(f"Visualization saved to heart_rate_analysis.png")