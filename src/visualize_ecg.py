"""
Visualize ECG Data
"""

import wfdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from utils import load_config

def visualize_ecg():
    """Visualize ECG signal and annotations"""
    
    config = load_config()
    data_dir = config['data']['data_dir']
    
    # Load record 100
    record_id = '100'
    record_path = Path(data_dir) / record_id
    
    print(f"Loading record {record_id}...")
    
    # Load ECG signal
    record = wfdb.rdrecord(str(record_path))
    
    # Load annotations
    annotation = wfdb.rdann(str(record_path), 'atr')
    
    print(f"✅ Record loaded!")
    print(f"   Duration: {record.sig_len / record.fs / 60:.1f} minutes")
    print(f"   Sampling rate: {record.fs} Hz")
    print(f"   Number of heartbeats: {len(annotation.sample)}")
    
    # Get first 10 seconds of data
    duration = 10  # seconds
    samples = duration * record.fs
    
    signal = record.p_signal[:samples, 0]  # First channel
    time = np.arange(len(signal)) / record.fs
    
    # Get annotations in this window
    ann_samples = annotation.sample[annotation.sample < samples]
    ann_symbols = [annotation.symbol[i] for i, s in enumerate(annotation.sample) if s < samples]
    ann_time = ann_samples / record.fs
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time, signal, 'b-', linewidth=0.5, label='ECG Signal')
    
    # Plot R-peaks
    plt.plot(ann_time, signal[ann_samples], 'ro', markersize=8, label='R-peaks')
    
    # Annotate beat types
    for t, sample, symbol in zip(ann_time, ann_samples, ann_symbols):
        plt.text(t, signal[sample] + 0.2, symbol, 
                fontsize=8, ha='center', color='red')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude (mV)', fontsize=12)
    plt.title(f'ECG Record {record_id} - First {duration} seconds', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    output_path = Path('data') / 'ecg_visualization.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Print beat type statistics
    print("\nBeat type distribution:")
    unique, counts = np.unique(annotation.symbol, return_counts=True)
    for symbol, count in zip(unique, counts):
        print(f"  {symbol}: {count} beats")

if __name__ == "__main__":
    visualize_ecg()