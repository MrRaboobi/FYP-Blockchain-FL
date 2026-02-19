"""
Download MIT-BIH Arrhythmia Dataset
"""

import os
import wfdb
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent))
from utils import load_config

def download_mitbih():
    """Download MIT-BIH dataset"""
    
    print("=" * 60)
    print("Downloading MIT-BIH Arrhythmia Database")
    print("=" * 60)
    
    # Load config
    
    config = load_config()
    data_dir = config['data']['data_dir']
    
    # Create directory
    os.makedirs(data_dir, exist_ok=True)
    
    # MIT-BIH record IDs — 15 records for 10-client experiment
    test_records = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115]
    
    print(f"\nDownloading {len(test_records)} test records...")
    print(f"Target directory: {os.path.abspath(data_dir)}\n")
    
    success_count = 0
    
    for record_id in tqdm(test_records, desc="Downloading"):
        try:
            # Download and immediately write to our directory
            record = wfdb.rdrecord(str(record_id), pn_dir='mitdb')
            annotation = wfdb.rdann(str(record_id), 'atr', pn_dir='mitdb')
            
            # Explicitly write the record to our directory
            output_path = os.path.join(data_dir, str(record_id))
            wfdb.wrsamp(
                record_name=str(record_id),
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=record.p_signal,
                fmt=record.fmt,
                write_dir=data_dir
            )
            
            # Write annotation
            wfdb.wrann(
                record_name=str(record_id),
                extension='atr',
                sample=annotation.sample,
                symbol=annotation.symbol,
                write_dir=data_dir
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"\n  Error with record {record_id}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Download complete! {success_count}/{len(test_records)} records")
    print("=" * 60)
    
    # Verify files
    print("\nVerifying downloaded files...")
    dat_files = list(Path(data_dir).glob("*.dat"))
    atr_files = list(Path(data_dir).glob("*.atr"))
    hea_files = list(Path(data_dir).glob("*.hea"))
    
    print(f"✅ Found {len(dat_files)} .dat files (ECG signals)")
    print(f"✅ Found {len(hea_files)} .hea files (headers)")
    print(f"✅ Found {len(atr_files)} .atr files (annotations)")
    
    if len(dat_files) >= len(test_records):
        print("\n🎉 All files downloaded successfully!")
        return True
    else:
        print(f"\n⚠️  Expected {len(test_records)} files, found {len(dat_files)}")
        return False

if __name__ == "__main__":
    download_mitbih()