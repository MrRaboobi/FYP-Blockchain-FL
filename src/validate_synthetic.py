"""
Validate Synthetic Data Quality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

def statistical_similarity(X_real, X_synthetic):
    """
    Compare statistical properties of real vs synthetic data
    
    Returns:
        dict of similarity metrics
    """
    
    print("\n" + "=" * 60)
    print("STATISTICAL SIMILARITY ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # Mean comparison
    mean_real = np.mean(X_real, axis=0)
    mean_synthetic = np.mean(X_synthetic, axis=0)
    mean_diff = np.mean(np.abs(mean_real - mean_synthetic))
    
    print(f"\n1. Mean Difference:")
    print(f"   Average: {mean_diff:.6f}")
    print(f"   ✓ Good if < 0.1")
    results['mean_diff'] = mean_diff
    
    # Standard deviation comparison
    std_real = np.std(X_real, axis=0)
    std_synthetic = np.std(X_synthetic, axis=0)
    std_diff = np.mean(np.abs(std_real - std_synthetic))
    
    print(f"\n2. Std Deviation Difference:")
    print(f"   Average: {std_diff:.6f}")
    print(f"   ✓ Good if < 0.15")
    results['std_diff'] = std_diff
    
    # Kolmogorov-Smirnov test (distribution similarity)
    ks_stats = []
    ks_pvalues = []
    
    for i in range(min(10, X_real.shape[1])):  # Test first 10 features
        ks_stat, p_value = stats.ks_2samp(X_real[:, i], X_synthetic[:, i])
        ks_stats.append(ks_stat)
        ks_pvalues.append(p_value)
    
    avg_ks_stat = np.mean(ks_stats)
    avg_p_value = np.mean(ks_pvalues)
    
    print(f"\n3. Kolmogorov-Smirnov Test (Distribution Similarity):")
    print(f"   KS Statistic: {avg_ks_stat:.4f}")
    print(f"   P-value: {avg_p_value:.4f}")
    print(f"   ✓ Good if p-value > 0.05 (distributions similar)")
    results['ks_stat'] = avg_ks_stat
    results['ks_pvalue'] = avg_p_value
    
    # Correlation preservation
    corr_real = np.corrcoef(X_real.T)
    corr_synthetic = np.corrcoef(X_synthetic.T)
    corr_diff = np.mean(np.abs(corr_real - corr_synthetic))
    
    print(f"\n4. Correlation Preservation:")
    print(f"   Difference: {corr_diff:.4f}")
    print(f"   ✓ Good if < 0.2")
    results['corr_diff'] = corr_diff
    
    return results

def discriminative_test(X_real, X_synthetic):
    """
    Train classifier to distinguish real from synthetic
    
    If accuracy is ~50%, synthetic is indistinguishable from real!
    """
    
    print("\n" + "=" * 60)
    print("DISCRIMINATIVE TEST")
    print("=" * 60)
    print("\nCan a classifier tell real from synthetic?")
    
    # Create labels (0 = real, 1 = synthetic)
    y_real = np.zeros(len(X_real))
    y_synthetic = np.ones(len(X_synthetic))
    
    # Combine
    X_combined = np.vstack([X_real, X_synthetic])
    y_combined = np.hstack([y_real, y_synthetic])
    
    # Shuffle
    indices = np.arange(len(y_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    # Split train/test
    split = int(0.7 * len(y_combined))
    X_train, X_test = X_combined[:split], X_combined[split:]
    y_train, y_test = y_combined[:split], y_combined[split:]
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nClassifier Accuracy: {accuracy:.2%}")
    print(f"\nInterpretation:")
    
    if accuracy < 0.55:
        print(f"   ✅ EXCELLENT! Synthetic data is indistinguishable from real")
        quality = "Excellent"
    elif accuracy < 0.65:
        print(f"   ✓ GOOD! Synthetic data is very similar to real")
        quality = "Good"
    elif accuracy < 0.75:
        print(f"   ⚠ FAIR. Synthetic data has noticeable differences")
        quality = "Fair"
    else:
        print(f"   ❌ POOR. Synthetic data is easily distinguishable")
        quality = "Poor"
    
    print(f"\n   Target: ~50% (random guessing means indistinguishable)")
    
    return {
        'accuracy': accuracy,
        'quality': quality
    }

def visual_comparison(X_real, X_synthetic, save_path='experiments/synthetic_validation.png'):
    """
    Visualize real vs synthetic data
    """
    
    print("\n" + "=" * 60)
    print("VISUAL COMPARISON")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. PCA Projection
    ax = axes[0, 0]
    
    # Combine for PCA fitting
    X_combined = np.vstack([X_real, X_synthetic])
    pca = PCA(n_components=2)
    pca.fit(X_combined)
    
    # Transform
    X_real_pca = pca.transform(X_real)
    X_synthetic_pca = pca.transform(X_synthetic)
    
    ax.scatter(X_real_pca[:, 0], X_real_pca[:, 1], 
              alpha=0.5, label='Real', s=30, color='blue')
    ax.scatter(X_synthetic_pca[:, 0], X_synthetic_pca[:, 1], 
              alpha=0.5, label='Synthetic', s=30, color='red')
    ax.set_title('PCA Projection', fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Feature Distribution (First feature)
    ax = axes[0, 1]
    
    ax.hist(X_real[:, 0], bins=30, alpha=0.6, label='Real', color='blue', density=True)
    ax.hist(X_synthetic[:, 0], bins=30, alpha=0.6, label='Synthetic', color='red', density=True)
    ax.set_title('Feature Distribution (Feature 0)', fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Mean Comparison
    ax = axes[0, 2]
    
    mean_real = np.mean(X_real, axis=0)
    mean_synthetic = np.mean(X_synthetic, axis=0)
    
    features = np.arange(min(20, len(mean_real)))
    ax.plot(features, mean_real[:len(features)], 'o-', label='Real', color='blue', linewidth=2)
    ax.plot(features, mean_synthetic[:len(features)], 's-', label='Synthetic', color='red', linewidth=2)
    ax.set_title('Mean Comparison (First 20 Features)', fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Mean Value')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Std Comparison
    ax = axes[1, 0]
    
    std_real = np.std(X_real, axis=0)
    std_synthetic = np.std(X_synthetic, axis=0)
    
    ax.plot(features, std_real[:len(features)], 'o-', label='Real', color='blue', linewidth=2)
    ax.plot(features, std_synthetic[:len(features)], 's-', label='Synthetic', color='red', linewidth=2)
    ax.set_title('Std Deviation Comparison', fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Std Deviation')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Sample Comparison (if time series)
    ax = axes[1, 1]
    
    # Plot a few random samples
    for i in range(min(3, len(X_real))):
        ax.plot(X_real[i], alpha=0.5, color='blue', linewidth=1)
    
    for i in range(min(3, len(X_synthetic))):
        ax.plot(X_synthetic[i], alpha=0.5, color='red', linewidth=1, linestyle='--')
    
    ax.set_title('Sample Waveforms', fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend(['Real', 'Synthetic'])
    ax.grid(alpha=0.3)
    
    # 6. Box Plot Comparison
    ax = axes[1, 2]
    
    # Select first 5 features for box plot
    data_real = [X_real[:, i] for i in range(min(5, X_real.shape[1]))]
    data_synthetic = [X_synthetic[:, i] for i in range(min(5, X_synthetic.shape[1]))]
    
    positions_real = np.arange(len(data_real)) * 2
    positions_synthetic = positions_real + 0.8
    
    bp1 = ax.boxplot(data_real, positions=positions_real, widths=0.6, 
                     patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5))
    bp2 = ax.boxplot(data_synthetic, positions=positions_synthetic, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor='red', alpha=0.5))
    
    ax.set_title('Distribution Box Plots', fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Value Range')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Synthetic'])
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    plt.show()

def comprehensive_validation(X_real, X_synthetic):
    """
    Run all validation tests
    """
    
    print("\n" + "=" * 70)
    print(" " * 15 + "SYNTHETIC DATA QUALITY VALIDATION")
    print("=" * 70)
    
    print(f"\nDataset Info:")
    print(f"  Real samples: {len(X_real)}")
    print(f"  Synthetic samples: {len(X_synthetic)}")
    print(f"  Features: {X_real.shape[1]}")
    
    # Run tests
    stat_results = statistical_similarity(X_real, X_synthetic)
    disc_results = discriminative_test(X_real, X_synthetic)
    visual_comparison(X_real, X_synthetic)
    
    # Overall assessment
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    
    score = 0
    
    if stat_results['mean_diff'] < 0.1:
        score += 1
    if stat_results['std_diff'] < 0.15:
        score += 1
    if stat_results['ks_pvalue'] > 0.05:
        score += 1
    if disc_results['accuracy'] < 0.65:
        score += 2  # More weight on discriminative test
    
    print(f"\nQuality Score: {score}/5")
    
    if score >= 4:
        overall = "✅ EXCELLENT - Synthetic data is high quality!"
    elif score >= 3:
        overall = "✓ GOOD - Synthetic data is usable with minor differences"
    elif score >= 2:
        overall = "⚠ FAIR - Synthetic data has noticeable differences"
    else:
        overall = "❌ POOR - Synthetic data quality needs improvement"
    
    print(f"\n{overall}")
    print("=" * 70)

# Test
if __name__ == "__main__":
    from synthetic_data import SyntheticDataGenerator
    
    # Create test data
    np.random.seed(42)
    
    # Real data (simulate ECG-like time series)
    X_real = np.random.randn(50, 360)  # 50 samples, 360 time steps
    
    # Generate synthetic
    y_dummy = np.zeros(50, dtype=int)
    
    generator = SyntheticDataGenerator()
    X_synthetic = generator.generate(X_real, y_dummy, target_class=0, n_samples=30)
    
    # Validate
    comprehensive_validation(X_real, X_synthetic)

