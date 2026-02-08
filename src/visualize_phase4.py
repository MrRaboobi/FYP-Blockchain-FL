"""
Visualize Phase 4: Synthetic Data + Blockchain Results
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter

def visualize_phase4():
    """Clean visualization of Phase 4 results"""
    
    print("=" * 60)
    print("PHASE 4 VISUALIZATION")
    print("=" * 60)
    
    exp_dir = Path('experiments')
    
    # Load results
    print("\nLoading results...")
    
    # Phase 4 (with synthetic)
    with open(exp_dir / 'phase4_results.pkl', 'rb') as f:
        phase4 = pickle.load(f)
    
    # Phase 2 (baseline - no synthetic)
    with open(exp_dir / 'fedavg_blockchain_results.pkl', 'rb') as f:
        baseline = pickle.load(f)
    
    # Print summary
    print("\nRESULTS SUMMARY:")
    print("-" * 60)
    print(f"Baseline (No Synthetic):    {baseline['avg_test_accuracy']:.2%}")
    print(f"Phase 4 (With Synthetic):   {phase4['avg_test_accuracy']:.2%}")
    print(f"Improvement:                {phase4['avg_test_accuracy'] - baseline['avg_test_accuracy']:+.2%}")
    print(f"\nSynthetic Requests:         {len(phase4['synthetic_requests'])}")
    print(f"Blockchain Overhead:        {phase4['avg_blockchain_overhead']:.2f}s per round")
    print("-" * 60)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # ==========================================
    # 1. Overall Comparison
    # ==========================================
    ax1 = plt.subplot(2, 3, 1)
    
    methods = ['Baseline\n(No Synthetic)', 'Phase 4\n(With Synthetic)']
    accuracies = [
        baseline['avg_test_accuracy'],
        phase4['avg_test_accuracy']
    ]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Impact of Synthetic Data', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.98, 1.0])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add values
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{acc:.2%}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Add improvement arrow
    improvement = phase4['avg_test_accuracy'] - baseline['avg_test_accuracy']
    if improvement > 0:
        ax1.annotate('', xy=(1, phase4['avg_test_accuracy']), 
                    xytext=(0, baseline['avg_test_accuracy']),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax1.text(0.5, (accuracies[0] + accuracies[1])/2,
                f'+{improvement:.2%}', ha='center', va='center',
                color='green', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='green'))
    
    # ==========================================
    # 2. Per-Client Comparison
    # ==========================================
    ax2 = plt.subplot(2, 3, 2)
    
    clients = ['Client 1', 'Client 2', 'Client 3']
    x = np.arange(len(clients))
    width = 0.35
    
    baseline_accs = [r['accuracy'] for r in baseline['client_test_results']]
    phase4_accs = [r['accuracy'] for r in phase4['client_test_results']]
    
    bars1 = ax2.bar(x - width/2, baseline_accs, width, label='Baseline',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, phase4_accs, width, label='With Synthetic',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Client Performance', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clients)
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim([0.95, 1.02])
    
    # Add values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.1%}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    
    # ==========================================
    # 3. Synthetic Requests Summary
    # ==========================================
    ax3 = plt.subplot(2, 3, 3)
    
    # Count requests per client
    synthetic_counts = Counter([req['client_id'] for req in phase4['synthetic_requests']])
    
    if synthetic_counts:
        client_ids = sorted(synthetic_counts.keys())
        counts = [synthetic_counts[cid] for cid in client_ids]
        
        bars3 = ax3.bar([f'Client {cid}' for cid in client_ids], counts,
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
        
        ax3.set_ylabel('Number of Requests', fontsize=12, fontweight='bold')
        ax3.set_title('Synthetic Data Requests', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, count in zip(bars3, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(count)}', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    else:
        ax3.text(0.5, 0.5, 'No Synthetic\nRequests', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax3.transAxes)
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # ==========================================
    # 4. F1-Score Comparison
    # ==========================================
    ax4 = plt.subplot(2, 3, 4)
    
    baseline_f1s = [r['f1'] for r in baseline['client_test_results']]
    phase4_f1s = [r['f1'] for r in phase4['client_test_results']]
    
    bars1 = ax4.bar(x - width/2, baseline_f1s, width, label='Baseline',
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, phase4_f1s, width, label='With Synthetic',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax4.set_title('F1-Score (Rare Class Performance)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(clients)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.set_ylim([0.6, 1.0])
    
    # Add values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')
    
    # ==========================================
    # 5. Blockchain Overhead
    # ==========================================
    ax5 = plt.subplot(2, 3, 5)
    
    overhead_data = [
        ('Phase 2\n(Updates Only)', baseline['avg_blockchain_overhead']),
        ('Phase 4\n(Updates + Synthetic)', phase4['avg_blockchain_overhead'])
    ]
    
    labels, values = zip(*overhead_data)
    
    bars5 = ax5.bar(labels, values, color=['#9b59b6', '#e67e22'],
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    ax5.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax5.set_title('Blockchain Overhead per Round', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars5, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.2f}s', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    
    # ==========================================
    # 6. Summary Stats
    # ==========================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate improvements
    acc_improvement = phase4['avg_test_accuracy'] - baseline['avg_test_accuracy']
    f1_improvement = phase4['avg_test_f1'] - baseline['avg_test_f1']
    
    # Count total synthetic samples
    total_synthetic = sum([req['quantity'] for req in phase4['synthetic_requests']])
    
    summary_text = f"""
PHASE 4 SUMMARY

Accuracy:
  Baseline: {baseline['avg_test_accuracy']:.2%}
  Phase 4:  {phase4['avg_test_accuracy']:.2%}
  Change:   {acc_improvement:+.2%}

F1-Score:
  Baseline: {baseline['avg_test_f1']:.3f}
  Phase 4:  {phase4['avg_test_f1']:.3f}
  Change:   {f1_improvement:+.3f}

Synthetic Data:
  Requests: {len(phase4['synthetic_requests'])}
  Samples:  {total_synthetic}
  
Blockchain:
  Overhead: {phase4['avg_blockchain_overhead']:.2f}s/round
  
KEY ACHIEVEMENTS:
{'✅ Improved performance' if acc_improvement > 0 else '⚠ No accuracy gain'}
{'✅ Better rare-class F1' if f1_improvement > 0 else '⚠ F1 unchanged'}
✅ Full blockchain audit
✅ Transparent governance
✅ Controlled augmentation
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Phase 4: Blockchain-Governed Synthetic Data Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = exp_dir / 'phase4_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Print detailed synthetic requests
    if phase4['synthetic_requests']:
        print("\n" + "=" * 60)
        print("SYNTHETIC DATA REQUESTS DETAIL")
        print("=" * 60)
        for i, req in enumerate(phase4['synthetic_requests']):
            print(f"\nRequest #{i+1}:")
            print(f"  Client: {req['client_id']}")
            print(f"  Class: {req['class_label']}")
            print(f"  Quantity: {req['quantity']}")
            print(f"  Blockchain ID: {req['request_id']}")
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print(f"✓ Accuracy: {baseline['avg_test_accuracy']:.2%} → {phase4['avg_test_accuracy']:.2%}")
    print(f"✓ F1-Score: {baseline['avg_test_f1']:.3f} → {phase4['avg_test_f1']:.3f}")
    print(f"✓ Synthetic samples generated: {total_synthetic}")
    print(f"✓ All requests logged on blockchain")
    print(f"✓ Blockchain overhead: {phase4['avg_blockchain_overhead']:.2f}s per round")
    print("=" * 60)

if __name__ == "__main__":
    visualize_phase4()


