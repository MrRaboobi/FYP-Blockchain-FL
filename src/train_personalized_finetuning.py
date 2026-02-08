"""
Personalized Federated Learning: Fine-Tuning Approach
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

def train_client_fedavg(model, train_loader, val_loader, epochs, learning_rate, device):
    """Standard federated training (for global model)"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
    
    val_metrics = evaluate(model, val_loader, criterion, device)
    return model, val_metrics

def fine_tune_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """Fine-tune the global model on local data"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate * 0.1)  # Lower LR for fine-tuning
    
    print("    Fine-tuning for", epochs, "epochs...")
    
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

def personalized_finetuning_train():
    """FedAvg + Local Fine-Tuning for Personalization"""
    
    print("=" * 60)
    print("PERSONALIZED FL: FINE-TUNING APPROACH")
    print("=" * 60)
    
    config = load_config()
    
    # Settings
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    finetune_epochs = 3  # Additional epochs for personalization
    learning_rate = config['model']['learning_rate']
    batch_size = config['training']['batch_size']
    device = torch.device('cpu')
    
    print(f"\nSettings:")
    print(f"  Federated rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Fine-tune epochs: {finetune_epochs}")
    print(f"  Clients: 3")
    
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
        'global_client_1': [],
        'global_client_2': [],
        'global_client_3': [],
        'personalized_client_1': [],
        'personalized_client_2': [],
        'personalized_client_3': [],
    }
    
    # Federated training loop
    print("\n" + "=" * 60)
    print("FEDERATED TRAINING + PERSONALIZATION")
    print("=" * 60)
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        client_models = []
        
        # Standard FedAvg training
        for client_id in range(3):
            client_model = deepcopy(global_model)
            
            client_model, _ = train_client_fedavg(
                client_model,
                client_loaders[client_id],
                client_val_loaders[client_id],
                local_epochs,
                learning_rate,
                device
            )
            
            client_models.append(client_model)
        
        # Aggregate to update global model
        global_model = aggregate_models(global_model, client_models, client_sizes)
        
        # Evaluate global model on each client
        print("  Global model performance:")
        global_accs = []
        
        criterion = nn.CrossEntropyLoss()
        
        for client_id in range(3):
            val_metrics = evaluate(global_model, client_val_loaders[client_id], criterion, device)
            global_accs.append(val_metrics['accuracy'])
            print(f"    Client {client_id + 1}: {val_metrics['accuracy']:.4f}")
        
        # Personalization: Fine-tune for each client
        print("  Personalized models (fine-tuned):")
        personalized_accs = []
        
        for client_id in range(3):
            # Copy global model
            personalized_model = deepcopy(global_model)
            
            # Fine-tune on local data
            personalized_model, val_metrics = fine_tune_model(
                personalized_model,
                client_loaders[client_id],
                client_val_loaders[client_id],
                finetune_epochs,
                learning_rate,
                device
            )
            
            personalized_accs.append(val_metrics['accuracy'])
            print(f"    Client {client_id + 1}: {val_metrics['accuracy']:.4f} (↑{val_metrics['accuracy'] - global_accs[client_id]:+.4f})")
        
        # Record history
        history['round'].append(round_num + 1)
        history['global_client_1'].append(global_accs[0])
        history['global_client_2'].append(global_accs[1])
        history['global_client_3'].append(global_accs[2])
        history['personalized_client_1'].append(personalized_accs[0])
        history['personalized_client_2'].append(personalized_accs[1])
        history['personalized_client_3'].append(personalized_accs[2])
    
    # Final evaluation on test sets
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    
    global_test_results = []
    personalized_test_results = []
    
    for client_id in range(3):
        X_test, y_test = client_test_data[client_id]
        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Global model performance
        global_metrics = evaluate(global_model, test_loader, criterion, device)
        global_test_results.append(global_metrics)
        
        # Personalized model (fine-tune on client's train data)
        personalized_model = deepcopy(global_model)
        personalized_model, _ = fine_tune_model(
            personalized_model,
            client_loaders[client_id],
            client_val_loaders[client_id],
            finetune_epochs,
            learning_rate,
            device
        )
        
        personalized_metrics = evaluate(personalized_model, test_loader, criterion, device)
        personalized_test_results.append(personalized_metrics)
        
        print(f"\nClient {client_id + 1}:")
        print(f"  Global Model:")
        print_metrics(global_metrics, prefix="    ")
        print(f"  Personalized Model:")
        print_metrics(personalized_metrics, prefix="    ")
        print(f"  Improvement: {personalized_metrics['accuracy'] - global_metrics['accuracy']:+.4f}")
    
    # Summary
    avg_global_acc = np.mean([r['accuracy'] for r in global_test_results])
    avg_personalized_acc = np.mean([r['accuracy'] for r in personalized_test_results])
    
    print("\n" + "=" * 60)
    print("✅ PERSONALIZED FL (FINE-TUNING) COMPLETE!")
    print("=" * 60)
    print(f"Average Global Model Accuracy: {avg_global_acc:.4f}")
    print(f"Average Personalized Accuracy: {avg_personalized_acc:.4f}")
    print(f"Personalization Gain: {avg_personalized_acc - avg_global_acc:+.4f}")
    print("=" * 60)
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'global_test_results': global_test_results,
        'personalized_test_results': personalized_test_results,
        'avg_global_accuracy': avg_global_acc,
        'avg_personalized_accuracy': avg_personalized_acc,
        'personalization_gain': avg_personalized_acc - avg_global_acc,
        'history': history
    }
    
    with open(output_dir / 'personalized_finetuning_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Results saved to: experiments/personalized_finetuning_results.pkl")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    personalized_finetuning_train()

