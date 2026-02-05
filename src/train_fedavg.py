"""
Federated Averaging (FedAvg) Training
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from copy import deepcopy
import sys
sys.path.append(str(Path(__file__).parent))

from utils import load_config
from model import create_model
from train_utils import (
    load_client_data, create_data_loaders, 
    train_epoch, evaluate, print_metrics, ECGDataset
)
from torch.utils.data import DataLoader

def train_client(model, train_loader, val_loader, epochs, learning_rate, device):
    """Train a client locally for several epochs"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
    
    # Evaluate after local training
    val_metrics = evaluate(model, val_loader, criterion, device)
    
    return model, val_metrics

def aggregate_models(global_model, client_models, client_sizes):
    """
    FedAvg: Weighted average of client models
    
    Args:
        global_model: Global model to update
        client_models: List of client models
        client_sizes: List of client dataset sizes (for weighting)
    """
    
    # Get total samples
    total_samples = sum(client_sizes)
    
    # Get global model state dict
    global_dict = global_model.state_dict()
    
    # Initialize aggregated weights (copy first client as starting point)
    aggregated_dict = {}
    
    for key in global_dict.keys():
        # Skip non-trainable parameters (like num_batches_tracked in BatchNorm)
        if 'num_batches_tracked' in key:
            aggregated_dict[key] = global_dict[key]
            continue
        
        # Initialize with zeros of the correct type
        aggregated_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
    
    # Weighted sum of client models
    for client_model, client_size in zip(client_models, client_sizes):
        weight = client_size / total_samples
        client_dict = client_model.state_dict()
        
        for key in global_dict.keys():
            # Skip non-trainable parameters
            if 'num_batches_tracked' in key:
                continue
            
            aggregated_dict[key] += client_dict[key].float() * weight
    
    # Convert back to original dtype and load
    for key in global_dict.keys():
        if 'num_batches_tracked' not in key:
            aggregated_dict[key] = aggregated_dict[key].to(global_dict[key].dtype)
    
    # Update global model
    global_model.load_state_dict(aggregated_dict)
    
    return global_model

def fedavg_train():
    """Main FedAvg training loop"""
    
    print("=" * 60)
    print("FEDERATED AVERAGING (FedAvg) TRAINING")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Settings
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    learning_rate = config['model']['learning_rate']
    batch_size = config['training']['batch_size']
    device = torch.device('cpu')
    
    print(f"\nFederated Learning Settings:")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Local epochs per round: {local_epochs}")
    print(f"  Number of clients: 3")
    print(f"  Device: {device}")
    
    # Load client data
    print("\nLoading client data...")
    partitioned_dir = config['data']['partitioned_dir']
    
    client_loaders = []
    client_val_loaders = []
    client_test_data = []
    client_sizes = []
    
    for client_id in range(1, 4):
        data = load_client_data(client_id, partitioned_dir)
        
        train_loader, val_loader = create_data_loaders(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            batch_size
        )
        
        client_loaders.append(train_loader)
        client_val_loaders.append(val_loader)
        client_test_data.append((data['X_test'], data['y_test']))
        client_sizes.append(len(data['y_train']))
        
        print(f"  Client {client_id}: {len(data['y_train'])} train samples")
    
    # Create global model
    print("\nCreating global model...")
    global_model = create_model(config)
    global_model = global_model.to(device)
    print(f"Parameters: {global_model.get_num_parameters():,}")
    
    # Training history
    history = {
        'round': [],
        'client_1_acc': [],
        'client_2_acc': [],
        'client_3_acc': [],
        'avg_acc': []
    }
    
    # Federated training loop
    print("\n" + "=" * 60)
    print("FEDERATED TRAINING")
    print("=" * 60)
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        client_models = []
        client_accs = []
        
        # Train each client
        for client_id in range(3):
            # Copy global model to client
            client_model = deepcopy(global_model)
            
            # Train locally
            client_model, val_metrics = train_client(
                client_model,
                client_loaders[client_id],
                client_val_loaders[client_id],
                local_epochs,
                learning_rate,
                device
            )
            
            client_models.append(client_model)
            client_accs.append(val_metrics['accuracy'])
            
            print(f"  Client {client_id + 1} Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Aggregate models (FedAvg)
        global_model = aggregate_models(global_model, client_models, client_sizes)
        
        avg_acc = np.mean(client_accs)
        print(f"  Average Val Acc: {avg_acc:.4f}")
        
        # Record history
        history['round'].append(round_num + 1)
        history['client_1_acc'].append(client_accs[0])
        history['client_2_acc'].append(client_accs[1])
        history['client_3_acc'].append(client_accs[2])
        history['avg_acc'].append(avg_acc)
    
    # Final evaluation on each client's test set
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (Per Client)")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    client_test_results = []
    
    for client_id in range(3):
        X_test, y_test = client_test_data[client_id]
        
        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate global model on this client's test set
        test_metrics = evaluate(global_model, test_loader, criterion, device)
        client_test_results.append(test_metrics)
        
        print(f"\nClient {client_id + 1} Test Results:")
        print_metrics(test_metrics, prefix="  ")
    
    # Overall average
    avg_test_acc = np.mean([r['accuracy'] for r in client_test_results])
    avg_test_f1 = np.mean([r['f1'] for r in client_test_results])
    
    print("\n" + "=" * 60)
    print(f"✅ FEDAVG TRAINING COMPLETE!")
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    print(f"Average Test F1: {avg_test_f1:.4f}")
    print("=" * 60)
    
    # Save model
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    torch.save(global_model.state_dict(), output_dir / 'fedavg_model.pth')
    print(f"\n✅ Model saved to: experiments/fedavg_model.pth")
    
    # Save results
    results = {
        'client_test_results': client_test_results,
        'avg_test_accuracy': avg_test_acc,
        'avg_test_f1': avg_test_f1,
        'history': history
    }
    
    with open(output_dir / 'fedavg_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✅ Results saved to: experiments/fedavg_results.pkl")
    
    return results

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    fedavg_train()