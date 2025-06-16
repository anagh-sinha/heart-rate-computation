# ECG Heart Rate Analyzer

A clean, modular Python implementation for computing heart rate from ECG signals using two different methods.

## Overview

This project implements two classical signal processing methods for heart rate detection:

1. **Autocorrelation Method**: Exploits the periodic nature of ECG signals
2. **Peak Detection Method**: Based on the Pan-Tompkins algorithm for QRS detection

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ecg_heart_rate_analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Analyze a single EDF file
python main.py data/ecg1.edf

# Analyze multiple files
python main.py data/*.edf
```

### Output

The program will:
1. Display heart rate estimates from both methods
2. Save visualization plots
3. Show a summary of all processed files

### Example Output

```
Found ECG channel: ECG, 250.0 Hz, 300.0 seconds
Autocorrelation: Found period of 0.788s = 76.1 bpm
Peak Detection: Found 383 peaks, mean HR = 76.8 bpm
Difference between Heart rates: 0.7 bpm
Visualization saved to heart_rate_analysis.png

```

## Methods 

### Method 1: Autocorrelation

The autocorrelation method works by:
1. Computing the correlation of the signal with time-shifted versions of itself
2. Finding the first peak after lag 0, which corresponds to one heartbeat period
3. Converting the period to heart rate in beats per minute

**Advantages:**
- Robust to noise
- Simple implementation

**Limitations:**
- Requires relatively stable heart rate

### Method 2: Peak Detection (Pan-Tompkins Inspired)

This method enhances and detects QRS complexes:
1. Bandpass filter (5-15 Hz) to isolate QRS frequency content
2. Square the signal to enhance large deflections
3. Apply moving average for smoothing
4. Detect peaks using adaptive thresholding
5. Calculate heart rate from inter-peak intervals

**Advantages:**
- Can handle rhythm variations
- Well-validated approach

**Limitations:**
- Requires parameter tuning

## Project Structure

```
heart-rate-computation/
├── ecg_analyzer/                       # Core analysis module
│   ├── io.py                           # EDF file reading
│   ├── methods.py                      # HR computation methods
│   └── visualization.py                # Results plotting
├── tests/                              # Unit tests
│   └── test_methods.py                 # Method validation
├── main.py                             # Entry point
├── requirements.txt                    # Dependencies
└── single_file_implementations/        # Single file implementations
    ├── robust_hrc.py                   # Robust Heart Rate Computation
    ├── simple_clean_hrc.py             # Simple Clean Heart Rate Computation             
```

## Running Tests

```bash
python tests/ecg_hra.py

```
