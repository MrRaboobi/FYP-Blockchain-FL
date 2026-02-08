"""
PHASE 4: FedAvg + Blockchain + Synthetic Data Generation
Complete Integration
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
from synthetic_data import SyntheticDataGenerator, detect_imbalance, calculate_target_samples

def train_client(model, train_loader, val_loader, epochs, learning_rate, device):
    """Train client locally"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
    
    val_metrics = evaluate(model, val_loader, criterion, device)
    return model, val_metrics

def aggregate_models(global_model, client_models, client_sizes):
    """FedAvg aggregation"""
    
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

def fedavg_blockchain_synthetic_train():
    """
    Complete Phase 4: FedAvg + Blockchain + Synthetic Data
    """
    
    print("=" * 70)
    print(" " * 10 + "PHASE 4: BLOCKCHAIN-GOVERNED SYNTHETIC DATA")
    print("=" * 70)
    
    config = load_config()
    
    # Settings
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    learning_rate = config['model']['learning_rate']
    batch_size = config['training']['batch_size']
    device = torch.device('cpu')
    
    # Synthetic data settings
    imbalance_threshold = 0.05  # Classes below 5% need augmentation
    target_ratio = 0.10  # Target: 10% of dataset for rare classes
    
    print(f"\nSettings:")
    print(f"  Federated rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Clients: 3")
    print(f"  Imbalance threshold: {imbalance_threshold:.1%}")
    print(f"  Target rare-class ratio: {target_ratio:.1%}")
    
    # Initialize blockchain
    print("\n" + "=" * 70)
    print("INITIALIZING BLOCKCHAIN")
    print("=" * 70)
    
    try:
        blockchain = BlockchainManager()
    except Exception as e:
        print(f"❌ Blockchain connection failed: {e}")
        print("Make sure Ganache is running!")
        return
    
    # Set synthetic quotas for each client (max 500 samples each)
    print("\nSetting synthetic data quotas...")
    for client_id in range(1, 4):
        blockchain.set_synthetic_quota(client_id, 500)
        quota = blockchain.get_quota(client_id)
        print(f"  Client {client_id}: {quota} samples allowed")
    
    # Load client data
    print("\n" + "=" * 70)
    print("LOADING CLIENT DATA")
    print("=" * 70)
    
    partitioned_dir = config['data']['partitioned_dir']
    
    client_data_dict = {}
    client_loaders = []
    client_val_loaders = []
    client_test_data = []
    client_sizes = []
    
    for client_id in range(1, 4):
        data = load_client_data(client_id, partitioned_dir)
        
        client_data_dict[client_id] = data
        
        train_loader, val_loader = create_data_loaders(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            batch_size
        )
        
        client_loaders.append(train_loader)
        client_val_loaders.append(val_loader)
        client_test_data.append((data['X_test'], data['y_test']))
        client_sizes.append(len(data['y_train']))
        
        print(f"\nClient {client_id}: {len(data['y_train'])} samples")
        
        # Detect imbalance
        imbalance = detect_imbalance(data['y_train'], threshold=imbalance_threshold)
        print(f"  Class distribution:")
        for class_label, info in imbalance.items():
            marker = "⚠" if info['needs_augmentation'] else "✓"
            print(f"    {marker} Class {class_label}: {info['count']} ({info['ratio']:.2%})")
    
    # SYNTHETIC DATA GENERATION PHASE
    print("\n" + "=" * 70)
    print("SYNTHETIC DATA GENERATION (BLOCKCHAIN-GOVERNED)")
    print("=" * 70)
    
    synthetic_generator = SyntheticDataGenerator()
    synthetic_requests_log = []
    
    for client_id in range(1, 4):
        print(f"\n--- Client {client_id} ---")
        
        data = client_data_dict[client_id]
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Detect imbalance
        imbalance = detect_imbalance(y_train, threshold=imbalance_threshold)
        
        needs_augmentation = False
        
        for class_label, info in imbalance.items():
            if info['needs_augmentation']:
                needs_augmentation = True
                
                n_to_generate = calculate_target_samples(
                    info['count'],
                    len(y_train),
                    target_ratio
                )
                
                if n_to_generate > 0:
                    print(f"\n  Class {class_label} needs augmentation:")
                    print(f"    Current: {info['count']} ({info['ratio']:.2%})")
                    print(f"    Requesting: {n_to_generate} synthetic samples")
                    
                    # REQUEST VIA BLOCKCHAIN
                    print(f"    📝 Submitting blockchain request...")
                    request_id = blockchain.request_synthetic(
                        client_id,
                        class_label,
                        n_to_generate
                    )
                    
                    print(f"    ✅ Request #{request_id} logged on blockchain")
                    
                    # AUTO-APPROVE (in production, this would be manual/automated governance)
                    print(f"    🔓 Approving request...")
                    blockchain.approve_synthetic(request_id)
                    
                    quota_remaining = blockchain.get_quota(client_id)
                    print(f"    ✅ Approved! Quota remaining: {quota_remaining}")
                    
                    # GENERATE SYNTHETIC DATA
                    print(f"    🎨 Generating synthetic samples...")
                    try:
                        X_synthetic = synthetic_generator.generate(
                            X_train, y_train,
                            target_class=class_label,
                            n_samples=n_to_generate
                        )
                        
                        y_synthetic = np.full(n_to_generate, class_label)
                        
                        # Augment training data
                        X_train = np.vstack([X_train, X_synthetic])
                        y_train = np.hstack([y_train, y_synthetic])
                        
                        print(f"    ✅ Generated {n_to_generate} samples")
                        
                        # Mark as generated on blockchain
                        blockchain.mark_synthetic_generated(request_id)
                        print(f"    ✅ Logged generation on blockchain")
                        
                        synthetic_requests_log.append({
                            'client_id': client_id,
                            'class_label': class_label,
                            'quantity': n_to_generate,
                            'request_id': request_id
                        })
                        
                    except Exception as e:
                        print(f"    ❌ Generation failed: {e}")
        
        if not needs_augmentation:
            print(f"  ✓ No augmentation needed (all classes balanced)")
        
        # Update client data with augmented dataset
        client_data_dict[client_id]['X_train'] = X_train
        client_data_dict[client_id]['y_train'] = y_train
        client_sizes[client_id - 1] = len(y_train)
        
        # Recreate data loader with augmented data
        train_loader, val_loader = create_data_loaders(
            X_train, y_train,
            data['X_val'], data['y_val'],
            batch_size
        )
        
        client_loaders[client_id - 1] = train_loader
        
        print(f"\n  Final dataset size: {len(y_train)} samples")
    
    # Print synthetic audit trail
    blockchain.print_synthetic_audit()
    
    # Create global model
    print("\n" + "=" * 70)
    print("CREATING GLOBAL MODEL")
    print("=" * 70)
    
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
    print("\n" + "=" * 70)
    print("FEDERATED TRAINING (WITH BLOCKCHAIN LOGGING)")
    print("=" * 70)
    
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
        
        # Aggregate
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
        
        blockchain.complete_round(round_num + 1)
        
        blockchain_time = time.time() - blockchain_start
        round_time = time.time() - round_start
        
        print(f"  ✅ Blockchain logged ({blockchain_time:.2f}s)")
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
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    
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
    
    # Blockchain overhead
    total_blockchain_time = sum(history['blockchain_time'])
    avg_blockchain_time = np.mean(history['blockchain_time'])
    
    print("\n" + "=" * 70)
    print("BLOCKCHAIN OVERHEAD")
    print("=" * 70)
    print(f"Total blockchain time: {total_blockchain_time:.2f}s")
    print(f"Average per round: {avg_blockchain_time:.2f}s")
    
    # Print audit trails
    blockchain.print_audit_trail()
    blockchain.print_synthetic_audit()
    
    print("\n" + "=" * 70)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 70)
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    print(f"Average Test F1: {avg_test_f1:.4f}")
    print(f"Synthetic requests: {len(synthetic_requests_log)}")
    print(f"Blockchain overhead: {avg_blockchain_time:.2f}s per round")
    print("=" * 70)
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    torch.save(global_model.state_dict(), output_dir / 'phase4_model.pth')
    
    results = {
        'client_test_results': client_test_results,
        'avg_test_accuracy': avg_test_acc,
        'avg_test_f1': avg_test_f1,
        'history': history,
        'synthetic_requests': synthetic_requests_log,
        'total_blockchain_time': total_blockchain_time,
        'avg_blockchain_overhead': avg_blockchain_time
    }
    
    with open(output_dir / 'phase4_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Model saved to: experiments/phase4_model.pth")
    print(f"✅ Results saved to: experiments/phase4_results.pkl")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("  ✅ Synthetic data generated and logged on blockchain")
    print("  ✅ All model updates tracked with provenance")
    print("  ✅ Complete audit trail available")
    print("  ✅ Fair, transparent, and auditable federated learning!")
    print("=" * 70)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    fedavg_blockchain_synthetic_train()

