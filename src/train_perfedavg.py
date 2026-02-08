"""
Personalized Federated Learning: PerFedAvg (Meta-Learning)
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
    evaluate, print_metrics, ECGDataset
)
from torch.utils.data import DataLoader

def perfedavg_step(model, train_loader, alpha, beta, device):
    """
    PerFedAvg meta-learning step
    
    Args:
        model: Current model
        train_loader: Training data
        alpha: Inner learning rate (for virtual update)
        beta: Outer learning rate (for meta update)
        device: CPU/GPU
    
    Returns:
        Updated model
    """
    
    criterion = nn.CrossEntropyLoss()
    
    # Get one batch for inner and outer updates
    data_iter = iter(train_loader)
    
    try:
        # Inner batch (for virtual update)
        X_inner, y_inner = next(data_iter)
        # Outer batch (for meta gradient)
        X_outer, y_outer = next(data_iter)
    except StopIteration:
        # If not enough batches, use same batch twice
        data_iter = iter(train_loader)
        X_inner, y_inner = next(data_iter)
        X_outer, y_outer = X_inner, y_inner
    
    X_inner = X_inner.to(device)
    y_inner = y_inner.to(device)
    X_outer = X_outer.to(device)
    y_outer = y_outer.to(device)
    
    # INNER LOOP: Compute gradient and virtual update
    model.train()
    
    # Forward pass on inner batch
    outputs_inner = model(X_inner)
    loss_inner = criterion(outputs_inner, y_inner)
    
    # Compute gradients
    model.zero_grad()
    loss_inner.backward()
    
    # Virtual update: θ' = θ - α * ∇L(θ)
    with torch.no_grad():
        virtual_model = deepcopy(model)
        for param, virtual_param in zip(model.parameters(), virtual_model.parameters()):
            if param.grad is not None:
                virtual_param.data = param.data - alpha * param.grad.data
    
    # OUTER LOOP: Compute meta gradient using virtual model
    virtual_model.train()
    
    # Forward pass on outer batch with virtual model
    outputs_outer = virtual_model(X_outer)
    loss_outer = criterion(outputs_outer, y_outer)
    
    # Compute gradients w.r.t. original model parameters
    virtual_model.zero_grad()
    loss_outer.backward()
    
    # Meta update: θ = θ - β * ∇L(θ')
    with torch.no_grad():
        for param, virtual_param in zip(model.parameters(), virtual_model.parameters()):
            if virtual_param.grad is not None:
                param.data = param.data - beta * virtual_param.grad.data
    
    return model

def train_client_perfedavg(model, train_loader, val_loader, epochs, alpha, beta, device):
    """Train client using PerFedAvg"""
    
    for epoch in range(epochs):
        perfedavg_step(model, train_loader, alpha, beta, device)
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
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

def perfedavg_train():
    """PerFedAvg meta-learning training"""
    
    print("=" * 60)
    print("PERSONALIZED FL: PerFedAvg (Meta-Learning)")
    print("=" * 60)
    
    config = load_config()
    
    # Settings
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    alpha = 0.01  # Inner learning rate
    beta = 0.001  # Outer (meta) learning rate
    batch_size = config['training']['batch_size']
    device = torch.device('cpu')
    
    print(f"\nSettings:")
    print(f"  Federated rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Inner LR (α): {alpha}")
    print(f"  Outer LR (β): {beta}")
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
        'client_1_acc': [],
        'client_2_acc': [],
        'client_3_acc': [],
        'avg_acc': []
    }
    
    # PerFedAvg training loop
    print("\n" + "=" * 60)
    print("PERFEDAVG META-LEARNING TRAINING")
    print("=" * 60)
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
        
        client_models = []
        client_accs = []
        
        # Train each client with PerFedAvg
        for client_id in range(3):
            client_model = deepcopy(global_model)
            
            client_model, val_metrics = train_client_perfedavg(
                client_model,
                client_loaders[client_id],
                client_val_loaders[client_id],
                local_epochs,
                alpha,
                beta,
                device
            )
            
            client_models.append(client_model)
            client_accs.append(val_metrics['accuracy'])
            
            print(f"  Client {client_id + 1} Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Aggregate
        global_model = aggregate_models(global_model, client_models, client_sizes)
        
        avg_acc = np.mean(client_accs)
        print(f"  Average Val Acc: {avg_acc:.4f}")
        
        # Record history
        history['round'].append(round_num + 1)
        history['client_1_acc'].append(client_accs[0])
        history['client_2_acc'].append(client_accs[1])
        history['client_3_acc'].append(client_accs[2])
        history['avg_acc'].append(avg_acc)
    
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
        
        # Create personalized model via meta-adaptation
        personalized_model = deepcopy(global_model)
        
        # Quick adaptation (few PerFedAvg steps)
        for _ in range(3):
            personalized_model = perfedavg_step(
                personalized_model,
                client_loaders[client_id],
                alpha,
                beta,
                device
            )
        
        test_metrics = evaluate(personalized_model, test_loader, criterion, device)
        client_test_results.append(test_metrics)
        
        print(f"\nClient {client_id + 1} Test Results:")
        print_metrics(test_metrics, prefix="  ")
    
    avg_test_acc = np.mean([r['accuracy'] for r in client_test_results])
    avg_test_f1 = np.mean([r['f1'] for r in client_test_results])
    
    print("\n" + "=" * 60)
    print("✅ PERFEDAVG TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")
    print(f"Average Test F1: {avg_test_f1:.4f}")
    print("=" * 60)
    
    # Save results
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    torch.save(global_model.state_dict(), output_dir / 'perfedavg_model.pth')
    
    results = {
        'client_test_results': client_test_results,
        'avg_test_accuracy': avg_test_acc,
        'avg_test_f1': avg_test_f1,
        'history': history
    }
    
    with open(output_dir / 'perfedavg_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ Model saved to: experiments/perfedavg_model.pth")
    print(f"✅ Results saved to: experiments/perfedavg_results.pkl")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    perfedavg_train()

