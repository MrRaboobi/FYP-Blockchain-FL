"""
Train Centralized Baseline Model
(All data pooled together)
"""

import torch
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from utils import load_config
from model import create_model
from train_utils import create_data_loaders, train_model, evaluate, print_metrics

def load_all_client_data(partitioned_dir):
    """Load and pool data from all clients"""
    
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    # Load all 3 clients
    for client_id in range(1, 4):
        client_dir = Path(partitioned_dir) / f'client_{client_id}'
        
        with open(client_dir / 'data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        X_train_list.append(data['X_train'])
        y_train_list.append(data['y_train'])
        X_val_list.append(data['X_val'])
        y_val_list.append(data['y_val'])
        X_test_list.append(data['X_test'])
        y_test_list.append(data['y_test'])
    
    # Concatenate all clients
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    
    # Shuffle training data
    indices = np.arange(len(y_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    """Train centralized baseline"""
    
    print("=" * 60)
    print("CENTRALIZED BASELINE TRAINING")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Set device
    device = torch.device('cpu')  # Using CPU for now
    print(f"\nDevice: {device}")
    
    # Load pooled data
    print("\nLoading data from all clients...")
    partitioned_dir = config['data']['partitioned_dir']
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_client_data(partitioned_dir)
    
    print(f"Train samples: {len(y_train)}")
    print(f"Val samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    batch_size = config['training']['batch_size']
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, batch_size
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    model, history, best_val_acc = train_model(
        model, train_loader, val_loader, config, device, verbose=True
    )
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    from train_utils import ECGDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    
    test_dataset = ECGDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\nTest Set Results:")
    print_metrics(test_metrics, prefix="  ")
    
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 60)
    
    # Save model
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / 'centralized_model.pth')
    print(f"\n✅ Model saved to: experiments/centralized_model.pth")
    
    # Save results
    results = {
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'best_val_accuracy': best_val_acc,
        'history': history
    }
    
    with open(output_dir / 'centralized_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✅ Results saved to: experiments/centralized_results.pkl")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()

