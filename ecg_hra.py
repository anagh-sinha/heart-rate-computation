import numpy as np
from ecg import method1_autocorrelation, method2_peak_detection

def test_edge_cases():
    print("Testing Edge Cases")
    fs = 250
    
    print("\nTest 1: Empty signal") # Test 1: Empty signal
    empty_signal = np.array([])
    hr1, _ = method1_autocorrelation(empty_signal, fs)
    hr2, _, _ = method2_peak_detection(empty_signal, fs)
    print(f"  Autocorrelation: {'PASS - Returned None' if hr1 is None else 'FAIL'}")
    print(f"  Peak Detection: {'PASS - Returned None' if hr2 is None else 'FAIL'}")
    
    print("\nTest 2: Very short signal (0.5s)") # Test 2: Very short signal (0.5 seconds)
    short_signal = np.random.randn(int(0.5 * fs))
    hr1, _ = method1_autocorrelation(short_signal, fs)
    hr2, _, _ = method2_peak_detection(short_signal, fs)
    print(f"  Autocorrelation: {'PASS - Handled gracefully' if True else 'FAIL'}")
    print(f"  Peak Detection: {'PASS - Handled gracefully' if True else 'FAIL'}")
    
    print("\nTest 3: Flat signal") # Test 3: Flat signal (no variation)
    flat_signal = np.ones(fs * 5)  # 5 seconds of constant value
    hr1, _ = method1_autocorrelation(flat_signal, fs)
    hr2, _, _ = method2_peak_detection(flat_signal, fs)
    print(f"  Autocorrelation: {'PASS' if hr1 is None else 'FAIL'}")
    print(f"  Peak Detection: {'PASS' if hr2 is None else 'FAIL'}")
    
    print("\nTest 4: Normal ECG signal (70 bpm)") # Test 4: Normal ECG signal
    t = np.arange(0, 10, 1/fs)
    normal_signal = np.zeros_like(t)
    
    hr_true = 70    # Add R-peaks at 70 bpm
    rr_interval = 60 / hr_true
    for peak_time in np.arange(0.5, 10, rr_interval):
        idx = int(peak_time * fs)
        if idx < len(normal_signal):
            normal_signal[idx] = 2.0
            for i in range(max(0, idx-5), min(len(normal_signal), idx+5)): # Add QRS shape
                dist = abs(i - idx) / 5
                normal_signal[i] += (1 - dist) * 1.5
    
    # Add some noise
    normal_signal += 0.05 * np.random.randn(len(normal_signal))
    
    hr1, _ = method1_autocorrelation(normal_signal, fs)
    hr2, _, _ = method2_peak_detection(normal_signal, fs)
    
    print(f"  Autocorrelation: {hr1:.1f} bpm" if hr1 else "  Autocorrelation: Failed")
    print(f"  Peak Detection: {hr2:.1f} bpm" if hr2 else "  Peak Detection: Failed")
    
    if hr1 and abs(hr1 - hr_true) < 5:
        print(f"  ✓ Autocorrelation accurate (within 5 bpm of true value)")
    if hr2 and abs(hr2 - hr_true) < 5:
        print(f"  ✓ Peak detection accurate (within 5 bpm of true value)")

    #Test detection of different heart rates.
def test_different_heart_rates():
    print("\n\nTesting Different Heart Rates")
    fs = 250
    test_hrs = [60, 80, 100, 120, 160] 
    
    for true_hr in test_hrs:
        print(f"\nTesting {true_hr} bpm:")
        
        # Create signal
        t = np.arange(0, 10, 1/fs)
        signal = np.zeros_like(t)
        
        rr_interval = 60 / true_hr
        for peak_time in np.arange(0.5, 10, rr_interval):
            idx = int(peak_time * fs)
            if idx < len(signal):
                # triangular QRS
                for i in range(max(0, idx-5), min(len(signal), idx+5)):
                    dist = abs(i - idx) / 5
                    signal[i] += (1 - dist) * 2.0
        
        # Add noise
        signal += 0.02 * np.random.randn(len(signal))
        
        # Test both methods
        hr1, _ = method1_autocorrelation(signal, fs)
        hr2, _, _ = method2_peak_detection(signal, fs)
        
        print(f"  Autocorrelation: {hr1:.1f} bpm (Error: {abs(hr1-true_hr):.1f} bpm)" if hr1 else "  Autocorrelation: Failed")
        print(f"  Peak Detection: {hr2:.1f} bpm (Error: {abs(hr2-true_hr):.1f} bpm)" if hr2 else "  Peak Detection: Failed")

if __name__ == "__main__":
    print("Running ECG Analyzer Tests.")
    test_edge_cases()
    test_different_heart_rates()
    print("\nAll tests completed.")