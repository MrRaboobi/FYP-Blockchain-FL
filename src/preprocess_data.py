"""
Preprocess ECG Data
Segment into heartbeats, normalize, and save
"""

import wfdb
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent))
from utils import load_config

# Annotation mapping to classes
ANNOTATION_MAP = {
    'N': 0,  # Normal
    'L': 0,  # Left bundle branch block
    'R': 0,  # Right bundle branch block
    'A': 1,  # Atrial premature
    'a': 1,  # Aberrated atrial premature
    'J': 1,  # Nodal premature
    'S': 1,  # Supraventricular premature
    'V': 2,  # Ventricular premature (PVC)
    'F': 3,  # Fusion
    '/': 4,  # Paced
    'f': 4,  # Fusion of paced and normal
}

CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

def normalize_signal(signal):
    """Z-score normalization"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std

def extract_heartbeats(record_id, data_dir, window_size=360):
    """Extract individual heartbeats from record"""
    
    record_path = Path(data_dir) / str(record_id)
    
    # Load record and annotations
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), 'atr')
    
    signal = record.p_signal[:, 0]  # First channel
    
    segments = []
    labels = []
    
    half_window = window_size // 2
    
    for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        # Skip if annotation not in our mapping
        if symbol not in ANNOTATION_MAP:
            continue
        
        # Check if window is within bounds
        if sample - half_window < 0 or sample + half_window >= len(signal):
            continue
        
        # Extract segment centered on R-peak
        segment = signal[sample - half_window : sample + half_window]
        
        # Normalize
        segment = normalize_signal(segment)
        
        # Map to class
        label = ANNOTATION_MAP[symbol]
        
        segments.append(segment)
        labels.append(label)
    
    return np.array(segments), np.array(labels)

def preprocess_all_records():
    """Preprocess all downloaded records"""
    
    print("=" * 60)
    print("Preprocessing ECG Data")
    print("=" * 60)
    
    config = load_config()
    data_dir = config['data']['data_dir']
    output_dir = config['data']['processed_dir']
    window_size = config['data']['window_size']
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all records
    dat_files = list(Path(data_dir).glob("*.dat"))
    record_ids = [f.stem for f in dat_files]
    
    print(f"\nFound {len(record_ids)} records")
    print(f"Window size: {window_size} samples\n")
    
    all_segments = []
    all_labels = []
    
    for record_id in tqdm(record_ids, desc="Processing records"):
        try:
            segments, labels = extract_heartbeats(record_id, data_dir, window_size)
            all_segments.append(segments)
            all_labels.append(labels)
        except Exception as e:
            print(f"\nError processing record {record_id}: {e}")
            continue
    
    # Combine all data
    X = np.vstack(all_segments)
    y = np.concatenate(all_labels)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    
    # Print class distribution
    print("\nClass Distribution:")
    for cls in range(5):
        count = np.sum(y == cls)
        pct = count / len(y) * 100
        print(f"  {CLASS_NAMES[cls]:20s}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\nTotal segments: {len(y)}")
    print(f"Segment shape: {X.shape}")
    
    # Save processed data
    output_file = Path(output_dir) / 'processed_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    
    print(f"\n✅ Saved to: {output_file}")
    
    return X, y

if __name__ == "__main__":
    preprocess_all_records()