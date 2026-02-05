"""
Partition data into non-IID client distributions
"""

import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent))
from utils import load_config

def create_non_iid_partitions(X, y, config):
    """Create label-skewed partitions for 3 clients"""
    
    print("=" * 60)
    print("Creating Non-IID Client Partitions")
    print("=" * 60)
    
    num_clients = config['data']['num_clients']
    distributions = config['data']['client_distributions']
    
    # Separate data by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    # Shuffle each class
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    
    # Map classes to distribution groups
    # 0: Normal, 1: Supraventricular, 2: Ventricular, 3: Fusion, 4: Unknown
    class_groups = {
        'normal': [0],
        'ventricular': [2],
        'rare': [1, 3, 4]  # Supraventricular, Fusion, Unknown
    }
    
    client_data = {}
    
    for client_id in range(1, num_clients + 1):
        client_key = f'client_{client_id}'
        dist = distributions[client_key]
        
        print(f"\n{client_key.upper()}:")
        print(f"  Target distribution:")
        print(f"    Normal: {dist['normal']*100:.1f}%")
        print(f"    Ventricular: {dist['ventricular']*100:.1f}%")
        print(f"    Rare: {dist['rare']*100:.1f}%")
        
        client_indices = []
        
        # Total samples for this client
        total_samples = int(len(y) * dist['data_fraction'])
        
        # Calculate target counts
        target_counts = {
            'normal': int(total_samples * dist['normal']),
            'ventricular': int(total_samples * dist['ventricular']),
            'rare': int(total_samples * dist['rare'])
        }
        
        # Sample from each group
        for group_name, class_list in class_groups.items():
            target = target_counts[group_name]
            per_class = target // len(class_list)
            
            for cls in class_list:
                available = len(class_indices[cls])
                take = min(per_class, available)
                
                indices = class_indices[cls][:take]
                client_indices.extend(indices)
                class_indices[cls] = class_indices[cls][take:]
        
        # Convert to array and shuffle
        client_indices = np.array(client_indices)
        np.random.shuffle(client_indices)
        
        # Extract data
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        # Calculate actual distribution
        print(f"  Actual samples: {len(y_client)}")
        for cls in range(5):
            count = np.sum(y_client == cls)
            pct = count / len(y_client) * 100 if len(y_client) > 0 else 0
            print(f"    Class {cls}: {count} ({pct:.1f}%)")
        
        client_data[client_id] = {
            'X': X_client,
            'y': y_client
        }
    
    return client_data

def split_train_val_test(client_data, config):
    """Split each client's data"""
    
    print("\n" + "=" * 60)
    print("Creating Train/Val/Test Splits")
    print("=" * 60)
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    for client_id, data in client_data.items():
        X = data['X']
        y = data['y']
        
        # Train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), random_state=42
        )
        
        # Val vs test
        relative_test = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test, random_state=42
        )
        
        client_data[client_id] = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
        
        print(f"\nClient {client_id}:")
        print(f"  Train: {len(y_train)}")
        print(f"  Val: {len(y_val)}")
        print(f"  Test: {len(y_test)}")
    
    return client_data

def save_partitions(client_data, config):
    """Save partitioned data"""
    
    output_dir = Path(config['data']['partitioned_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for client_id, data in client_data.items():
        client_dir = output_dir / f'client_{client_id}'
        client_dir.mkdir(exist_ok=True)
        
        with open(client_dir / 'data.pkl', 'wb') as f:
            pickle.dump(data, f)
    
    print(f"\n✅ Partitioned data saved to: {output_dir}")

def main():
    """Main partitioning pipeline"""
    
    # Load config
    config = load_config()
    
    # Load preprocessed data
    processed_file = Path(config['data']['processed_dir']) / 'processed_data.pkl'
    
    print("\nLoading preprocessed data...")
    with open(processed_file, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    
    print(f"Total samples: {len(y)}\n")
    
    # Create partitions
    client_data = create_non_iid_partitions(X, y, config)
    
    # Split train/val/test
    client_data = split_train_val_test(client_data, config)
    
    # Save
    save_partitions(client_data, config)
    
    print("\n" + "=" * 60)
    print("✅ Partitioning Complete!")
    print("=" * 60)

if __name__ == "__main__":
    np.random.seed(42)
    main()