"""
Synthetic Data Generation using SMOTE
"""

import numpy as np
from collections import Counter

class SyntheticDataGenerator:
    """
    Generate synthetic samples for minority classes using SMOTE
    (Synthetic Minority Over-sampling Technique)
    """
    
    def __init__(self, k_neighbors=5, random_state=42):
        """
        Args:
            k_neighbors: Number of nearest neighbors to use
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate(self, X, y, target_class, n_samples):
        """
        Generate synthetic samples for a specific class
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Labels (n_samples,)
            target_class: Class to generate samples for
            n_samples: Number of synthetic samples to generate
        
        Returns:
            X_synthetic: Generated samples (n_samples, n_features)
        """
        
        # Get samples of target class
        class_indices = np.where(y == target_class)[0]
        
        if len(class_indices) == 0:
            raise ValueError(f"No samples found for class {target_class}")
        
        if len(class_indices) < self.k_neighbors:
            # Not enough samples for k-neighbors, use all available
            k = len(class_indices) - 1
            if k < 1:
                # Only 1 sample, just duplicate with noise
                return self._duplicate_with_noise(X[class_indices], n_samples)
        else:
            k = self.k_neighbors
        
        X_class = X[class_indices]
        
        # Generate synthetic samples using SMOTE
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Randomly select a sample
            idx = np.random.randint(0, len(X_class))
            sample = X_class[idx]
            
            # Find k nearest neighbors
            distances = np.linalg.norm(X_class - sample, axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude itself
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(nearest_indices)
            neighbor = X_class[neighbor_idx]
            
            # Generate synthetic sample along the line segment
            alpha = np.random.random()
            synthetic_sample = sample + alpha * (neighbor - sample)
            
            synthetic_samples.append(synthetic_sample)
        
        return np.array(synthetic_samples)
    
    def _duplicate_with_noise(self, X, n_samples):
        """Fallback: duplicate samples with small noise"""
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            idx = np.random.randint(0, len(X))
            sample = X[idx].copy()
            
            # Add small Gaussian noise
            noise = np.random.normal(0, 0.01, sample.shape)
            synthetic_sample = sample + noise
            
            synthetic_samples.append(synthetic_sample)
        
        return np.array(synthetic_samples)
    
    def balance_dataset(self, X, y, target_ratio=0.5):
        """
        Balance dataset by oversampling minority classes
        
        Args:
            X: Features
            y: Labels
            target_ratio: Target ratio of minority to majority class
        
        Returns:
            X_balanced, y_balanced
        """
        
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        
        target_count = int(majority_count * target_ratio)
        
        X_list = [X]
        y_list = [y]
        
        for class_label, count in class_counts.items():
            if class_label == majority_class:
                continue
            
            if count < target_count:
                n_to_generate = target_count - count
                
                X_synthetic = self.generate(X, y, class_label, n_to_generate)
                y_synthetic = np.full(n_to_generate, class_label)
                
                X_list.append(X_synthetic)
                y_list.append(y_synthetic)
        
        X_balanced = np.vstack(X_list)
        y_balanced = np.hstack(y_list)
        
        # Shuffle
        indices = np.arange(len(y_balanced))
        np.random.shuffle(indices)
        
        return X_balanced[indices], y_balanced[indices]

def detect_imbalance(y, threshold=0.1):
    """
    Detect class imbalance
    
    Args:
        y: Labels
        threshold: Minimum acceptable class ratio
    
    Returns:
        dict of {class_label: (count, ratio, needs_augmentation)}
    """
    
    class_counts = Counter(y)
    total = len(y)
    
    imbalance_info = {}
    
    for class_label, count in class_counts.items():
        ratio = count / total
        needs_augmentation = ratio < threshold
        
        imbalance_info[class_label] = {
            'count': count,
            'ratio': ratio,
            'needs_augmentation': needs_augmentation
        }
    
    return imbalance_info

def calculate_target_samples(current_count, total_samples, target_ratio=0.2):
    """
    Calculate how many synthetic samples needed
    
    Args:
        current_count: Current number of samples for this class
        total_samples: Total samples in dataset
        target_ratio: Target ratio (e.g., 0.2 = 20% of dataset)
    
    Returns:
        Number of synthetic samples to generate
    """
    
    target_count = int(total_samples * target_ratio)
    n_needed = max(0, target_count - current_count)
    
    return n_needed

# Test
if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC DATA GENERATOR TEST")
    print("=" * 60)
    
    # Create dummy imbalanced data
    np.random.seed(42)
    
    # Majority class: 100 samples
    X_majority = np.random.randn(100, 10)
    y_majority = np.zeros(100, dtype=int)
    
    # Minority class: 5 samples
    X_minority = np.random.randn(5, 10)
    y_minority = np.ones(5, dtype=int)
    
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])
    
    print(f"\nOriginal dataset:")
    print(f"  Class 0: {np.sum(y == 0)} samples")
    print(f"  Class 1: {np.sum(y == 1)} samples")
    
    # Detect imbalance
    imbalance = detect_imbalance(y)
    print(f"\nImbalance detection:")
    for class_label, info in imbalance.items():
        print(f"  Class {class_label}: {info['count']} samples ({info['ratio']:.2%}) - Needs augmentation: {info['needs_augmentation']}")
    
    # Generate synthetic samples
    generator = SyntheticDataGenerator()
    
    n_to_generate = calculate_target_samples(5, 105, target_ratio=0.2)
    print(f"\nGenerating {n_to_generate} synthetic samples for class 1...")
    
    X_synthetic = generator.generate(X, y, target_class=1, n_samples=n_to_generate)
    
    print(f"✅ Generated {len(X_synthetic)} synthetic samples")
    print(f"   Shape: {X_synthetic.shape}")
    
    # Balance dataset
    X_balanced, y_balanced = generator.balance_dataset(X, y, target_ratio=0.3)
    
    print(f"\nBalanced dataset:")
    print(f"  Class 0: {np.sum(y_balanced == 0)} samples")
    print(f"  Class 1: {np.sum(y_balanced == 1)} samples")
    print(f"  Total: {len(y_balanced)} samples")
    
    print("\n" + "=" * 60)
    print("✅ SYNTHETIC DATA GENERATOR WORKING!")
    print("=" * 60)




