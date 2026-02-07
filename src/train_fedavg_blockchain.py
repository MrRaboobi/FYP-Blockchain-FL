"""
FedAvg with Blockchain Provenance Logging
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from copy import deepcopy
import time
import sys
sys.path.append(str(Path(__file__).parent))

from utils import load_config
from model import create_model
from train_utils import (
    load_client_data, create_data_loaders, 
    train_epoch, evaluate, print_metrics, ECGDataset
)
from torch.utils.data import DataLoader
from blockchain import BlockchainManager

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
    """FedAvg: Weighted average of client models"""
    
    total_samples = sum(client_sizes)
    global_dict = global_model.state_dict()
    aggregated_dict = {}
    
    for key in global_dict.keys():
        if 'num_batches_tracked' in key:
            aggregated_dict[key] = global_dict[key]
            continue
        aggregated_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
    
    for client_model, client_size in zip(client_models, client_sizes):
        weight = client_size / total_samples
        client_dict = client_model.state_dict()
        
        for key in global_dict.keys():
            if 'num_batches_tracked' not in key:
                aggregated_dict[key] += client_dict[key].float() * weight
    
    for key in global_dict.keys():
        if 'num_batches_tracked' not in key:
            aggregated_dict[key] = aggregated_dict[key].to(global_dict[key].dtype)
    
    global_model.load_state_dict(aggregated_dict)
    
    return global_model

def fedavg_blockchain_train():
    """FedAvg training WITH blockchain logging"""
    
    print("=" * 60)
    print("FEDAVG WITH BLOCKCHAIN PROVENANCE")
    print("=" * 60)
    
    # Load config
    config = load_config()
    
    # Settings
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    learning_rate = config['model']['learning_rate']
    batch_size = config['training']['batch_size']
    device = torch.device('cpu')
    
    print(f"\nSettings:")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Clients: 3")
    
    # Initialize blockchain manager
    print("\n" + "=" * 60)
    print("Initializing Blockchain")
    print("=" * 60)
    
    try:
        blockchain = BlockchainManager()
    except Exception as e:
        print(f"❌ Blockchain connection failed: {e}")
        print("Make sure Ganache is running!")
        return
    
    # Load client data
    print("\n" + "=" * 60)
    print("Loading Client Data")
    print("=" * 60)
    
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
        
        print(f"Client {client_id}: {len(data['y_train'])} samples")
    
    # Create global model
    print("\n" + "=" * 60)
    print("Creating Global Model")
    print("=" * 60)
    
    global_model = create_model(config)
    global_model = global_model.to(device)
    print(f"Parameters: {global_model.get_num_parameters():,}")
    
    # Training history
    history = {
        'round': [],
        'client_1_acc': [],
        'client_2_acc': [],
        'client_3_acc': [],
        'avg_acc': [],
        'blockchain_time': []
    }
    
    # Federated training loop
    print("\n" + "=" * 60)
    print("FEDERATED TRAINING WITH BLOCKCHAIN LOGGING")
    print("=" * 60)
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        round_start = time.time()
        
        client_models = []
        client_accs = []
        
        # Train each client
        for client_id in range(3):
            client_model = deepcopy(global_model)
            
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
        
        # Aggregate models
        global_model = aggregate_models(global_model, client_models, client_sizes)
        
        avg_acc = np.mean(client_accs)
        
        # BLOCKCHAIN LOGGING
        print(f"  📝 Logging to blockchain...")
        blockchain_start = time.time()
        
        for client_id in range(3):
            blockchain.log_update(
                round_num + 1,
                client_id + 1,
                client_models[client_id].state_dict(),
                client_sizes[client_id],
                client_accs[client_id]
            )
        
        # Mark round complete
        blockchain.complete_round(round_num + 1)
        
        blockchain_time = time.time() - blockchain_start
        round_time = time.time() - round_start
        
        print(f"  ✅ Blockchain logged (took {blockchain_time:.2f}s)")
        print(f"  Average Val Acc: {avg_acc:.4f}")
        print(f"  Total round time: {round_time:.2f}s")
        
        # Record history
        history['round'].append(round_num + 1)
        history['client_1_acc'].append(client_accs[0])
        history['client_2_acc'].append(client_accs[1])
        history['client_3_acc'].append(client_accs[2])
        history['avg_acc'].append(avg_acc)
        history['blockchain_time'].append(blockchain_time)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    client_test_results = []
    
    for client_id in range(3):
        X_test, y_test = client_test_data[client_id]
        
        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_metrics = evaluate(global_model, test_loader, criterion, device)
        client_test_results.append(test_metrics)
        
        print(f"\nClient {client_id + 1} Test Results:")
        print_metrics(test_metrics, prefix="  ")
    
    avg_test_acc = np.mean([r['accuracy'] for r in client_test_results])
    avg_test_f1 = np.mean([r['f1'] for r in client_test_results])
    
    # Blockchain overhead analysis
    total_blockchain_time = sum(history['blockchain_time'])
    avg_blockchain_time = np.mean(history['blockchain_time'])
    
    print("\n" + "=" * 60)
    print("BLOCKCHAIN OVERHEAD ANALYSIS")
    print("=" * 60)
    print(f"Total blockchain time: {total_blockchain_time:.2f}s")
    print(f"Average per round: {avg_blockchain_time:.2f}s")
    print(f"Total updates logged: {blockchain.get_total_updates()}")
    
    # Print audit trail
    blockchain.print_audit_trail()
    
    print("\n" + "=" * 60)
    print("✅ FEDAVG + BLOCKCHAIN COMPLETE!")
    print("=" * 60)
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    print(f"Average Test F1: {avg_test_f1:.4f}")
    print(f"Blockchain overhead: {avg_blockchain_time:.2f}s per round")
    print("=" * 60)
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    torch.save(global_model.state_dict(), output_dir / 'fedavg_blockchain_model.pth')
    
    results = {
        'client_test_results': client_test_results,
        'avg_test_accuracy': avg_test_acc,
        'avg_test_f1': avg_test_f1,
        'history': history,
        'total_blockchain_time': total_blockchain_time,
        'avg_blockchain_overhead': avg_blockchain_time
    }
    
    with open(output_dir / 'fedavg_blockchain_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Model saved to: experiments/fedavg_blockchain_model.pth")
    print(f"✅ Results saved to: experiments/fedavg_blockchain_results.pkl")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    fedavg_blockchain_train()

    