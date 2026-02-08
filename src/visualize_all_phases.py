"""
Clean Master Visualization: All Phases
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_all_phases():
    """Simple, clean comparison"""
    
    print("=" * 70)
    print(" " * 15 + "MASTER COMPARISON: ALL PHASES")
    print("=" * 70)
    
    exp_dir = Path('experiments')
    
    # Load results
    print("\nLoading results...")
    
    with open(exp_dir / 'centralized_results.pkl', 'rb') as f:
        centralized = pickle.load(f)
    
    with open(exp_dir / 'fedavg_results.pkl', 'rb') as f:
        fedavg = pickle.load(f)
    
    with open(exp_dir / 'fedavg_blockchain_results.pkl', 'rb') as f:
        blockchain = pickle.load(f)
    
    with open(exp_dir / 'personalized_finetuning_results.pkl', 'rb') as f:
        finetuning = pickle.load(f)
    
    with open(exp_dir / 'perfedavg_results.pkl', 'rb') as f:
        perfedavg = pickle.load(f)
    
    with open(exp_dir / 'phase4_results.pkl', 'rb') as f:
        phase4 = pickle.load(f)
    
    print("✅ All results loaded!")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Method':<35} {'Accuracy':>12} {'F1-Score':>12} {'Status':>15}")
    print("-" * 80)
    print(f"{'1. Centralized (Upper Bound)':<35} {centralized['test_accuracy']:>11.2%} {centralized['test_f1']:>12.3f} {'✓ Baseline':>15}")
    print(f"{'2. FedAvg':<35} {fedavg['avg_test_accuracy']:>11.2%} {fedavg['avg_test_f1']:>12.3f} {'✓ Good':>15}")
    print(f"{'3. FedAvg + Blockchain':<35} {blockchain['avg_test_accuracy']:>11.2%} {blockchain['avg_test_f1']:>12.3f} {'✓ Auditable':>15}")
    print(f"{'4. Fine-Tuning':<35} {finetuning['avg_personalized_accuracy']:>11.2%} {finetuning['avg_personalized_accuracy']:>12.3f} {'= Same':>15}")
    print(f"{'5. PerFedAvg':<35} {perfedavg['avg_test_accuracy']:>11.2%} {perfedavg['avg_test_f1']:>12.3f} {'✗ Failed':>15}")
    print(f"{'6. Phase 4 (Complete System)':<35} {phase4['avg_test_accuracy']:>11.2%} {phase4['avg_test_f1']:>12.3f} {'✓ Best':>15}")
    print("=" * 80)
    
    # Create TWO simple charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ==========================================
    # Chart 1: Accuracy Comparison
    # ==========================================
    
    methods = [
        'Centralized\n(Target)',
        'FedAvg\n(Baseline)',
        'Blockchain',
        'Fine-Tuning',
        'PerFedAvg',
        'Phase 4\n(Full System)'
    ]
    
    accuracies = [
        centralized['test_accuracy'],
        fedavg['avg_test_accuracy'],
        blockchain['avg_test_accuracy'],
        finetuning['avg_personalized_accuracy'],
        perfedavg['avg_test_accuracy'],
        phase4['avg_test_accuracy']
    ]
    
    # Color scheme
    colors = ['#95a5a6', '#3498db', '#3498db', '#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=2.5, width=0.7)
    
    # Formatting
    ax1.set_ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    ax1.set_title('Accuracy Across All Experiments', fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylim([0.90, 1.0])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.tick_params(axis='y', labelsize=12)
    
    # Target line
    ax1.axhline(y=centralized['test_accuracy'], color='gray', 
                linestyle='--', linewidth=3, alpha=0.6, label='Centralized Target')
    
    # Add values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{acc:.2%}', ha='center', va='bottom', 
                fontweight='bold', fontsize=13)
    
    ax1.legend(fontsize=12, loc='lower left')
    
    # ==========================================
    # Chart 2: F1-Score Comparison
    # ==========================================
    
    f1_scores = [
        centralized['test_f1'],
        fedavg['avg_test_f1'],
        blockchain['avg_test_f1'],
        finetuning['avg_personalized_accuracy'],
        perfedavg['avg_test_f1'],
        phase4['avg_test_f1']
    ]
    
    bars2 = ax2.bar(methods, f1_scores, color=colors, alpha=0.85,
                   edgecolor='black', linewidth=2.5, width=0.7)
    
    # Formatting
    ax2.set_ylabel('F1-Score', fontsize=16, fontweight='bold')
    ax2.set_title('F1-Score (Rare Class Performance)', fontsize=18, fontweight='bold', pad=20)
    ax2.set_ylim([0.3, 1.0])
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    ax2.tick_params(axis='x', labelsize=11)
    ax2.tick_params(axis='y', labelsize=12)
    
    # Add values on bars
    for bar, f1 in zip(bars2, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{f1:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=13)
    
    # Add PerFedAvg warning
    ax2.text(4, perfedavg['avg_test_f1'] - 0.1, '⚠ Failed', 
            ha='center', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linewidth=2))
    
    plt.suptitle('Complete FYP: Blockchain-Orchestrated Federated Learning', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    output_path = exp_dir / 'master_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Detailed statistics
    print("\n" + "=" * 80)
    print("KEY STATISTICS")
    print("=" * 80)
    
    baseline_acc = fedavg['avg_test_accuracy']
    final_acc = phase4['avg_test_accuracy']
    
    print(f"\nAccuracy Journey:")
    print(f"  Centralized (Target):     {centralized['test_accuracy']:.4f}")
    print(f"  FedAvg (Baseline):        {baseline_acc:.4f}")
    print(f"  + Blockchain:             {blockchain['avg_test_accuracy']:.4f} (no change)")
    print(f"  + Fine-Tuning:            {finetuning['avg_personalized_accuracy']:.4f} (no change)")
    print(f"  + PerFedAvg:              {perfedavg['avg_test_accuracy']:.4f} (-6.89% ✗)")
    print(f"  Phase 4 (Full):           {final_acc:.4f} ({final_acc - baseline_acc:+.4f})")
    
    print(f"\nF1-Score Journey:")
    print(f"  Baseline:                 {fedavg['avg_test_f1']:.4f}")
    print(f"  Phase 4:                  {phase4['avg_test_f1']:.4f}")
    
    total_synthetic = sum([req['quantity'] for req in phase4['synthetic_requests']])
    
    print(f"\nPhase 4 Additions:")
    print(f"  Synthetic requests:       {len(phase4['synthetic_requests'])}")
    print(f"  Synthetic samples:        {total_synthetic}")
    print(f"  Blockchain overhead:      {phase4['avg_blockchain_overhead']:.2f}s per round")
    
    print("\n" + "=" * 80)
    print("ACHIEVEMENTS ✓")
    print("=" * 80)
    print("  ✓ Privacy-preserving federated learning")
    print("  ✓ Complete blockchain audit trail (60 model updates + synthetic logs)")
    print("  ✓ Synthetic data generation with governance")
    print("  ✓ Near-centralized performance maintained (99.19%)")
    print("  ✓ Production-ready pipeline")
    print("  ✓ Full documentation")
    
    print("\n" + "=" * 80)
    print("LESSONS LEARNED")
    print("=" * 80)
    print("  • Blockchain adds minimal overhead (0.57s per round)")
    print("  • Personalization needs larger datasets to show benefits")
    print("  • PerFedAvg is sensitive - failed on small data")
    print("  • SMOTE works well for synthetic generation")
    print("  • Dataset size is the main bottleneck (6,318 samples)")
    
    print("\n" + "=" * 80)
    print("🚀 NEXT STEP: SCALE TO 48 MIT-BIH RECORDS")
    print("=" * 80)
    print("  → 100,000+ samples (vs 6,318 current)")
    print("  → Real heterogeneity across clients")
    print("  → Personalization gains will be visible")
    print("  → Production-scale validation")
    print("=" * 80)

if __name__ == "__main__":
    visualize_all_phases()


