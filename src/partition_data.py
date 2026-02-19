"""
Partition preprocessed ECG data into non-IID client datasets.
Uses Dirichlet(alpha=0.5) distribution over labels — supports any num_clients.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent))
from utils import load_config


def create_non_iid_partitions(X, y, config):
    """
    Dirichlet-based non-IID partitioning for arbitrary num_clients.

    Each client receives a label distribution drawn from
    Dirichlet(alpha=0.5), giving realistic heterogeneity without
    relying on hardcoded per-client configuration dicts.

    Args:
        X      : np.ndarray of shape (N, window_size)
        y      : np.ndarray of shape (N,) with integer class labels
        config : project config dict

    Returns:
        dict {client_id (int): {'X': ..., 'y': ...}}
    """

    print("=" * 60)
    print("Creating Non-IID Client Partitions (Dirichlet alpha=0.5)")
    print("=" * 60)

    num_clients = config['data']['num_clients']   # e.g. 10
    alpha       = 0.5                             # Dirichlet concentration
    classes     = np.unique(y)
    num_classes = len(classes)

    # Build per-class index lists (shuffled)
    rng = np.random.default_rng(seed=42)
    class_indices = {cls: np.where(y == cls)[0].tolist() for cls in classes}
    for cls in classes:
        rng.shuffle(class_indices[cls])

    # Sample proportion matrix from Dirichlet
    # proportions[k, c] = fraction of class-c data going to client k
    proportions = rng.dirichlet([alpha] * num_clients, size=num_classes)
    # proportions shape: (num_classes, num_clients)

    client_indices = {k: [] for k in range(num_clients)}

    for c_idx, cls in enumerate(classes):
        idxs      = class_indices[cls]
        n_samples = len(idxs)
        # Compute how many samples each client gets from this class
        splits = (proportions[c_idx] * n_samples).astype(int)
        # Fix rounding: add/remove residuals to first client
        splits[0] += n_samples - splits.sum()

        ptr = 0
        for k in range(num_clients):
            end = ptr + splits[k]
            client_indices[k].extend(idxs[ptr:end])
            ptr = end

    # Build client data dicts
    client_data = {}
    for k in range(num_clients):
        idxs = np.array(client_indices[k])
        rng.shuffle(idxs)

        X_c = X[idxs]
        y_c = y[idxs]

        client_id = k + 1
        client_data[client_id] = {'X': X_c, 'y': y_c}

        print(f"\nClient {client_id:2d}:  {len(y_c)} samples")
        for cls in classes:
            cnt = np.sum(y_c == cls)
            pct = cnt / max(len(y_c), 1) * 100
            print(f"    Class {cls}: {cnt:4d} ({pct:5.1f}%)")

    return client_data


def _can_stratify(y_arr):
    """Return True only if every class has at least 2 samples."""
    if len(y_arr) < 2:
        return False
    classes, counts = np.unique(y_arr, return_counts=True)
    return len(classes) > 1 and counts.min() >= 2


def split_train_val_test(client_data, config):
    """Split each client's data into train / val / test subsets."""

    print("\n" + "=" * 60)
    print("Creating Train / Val / Test Splits")
    print("=" * 60)

    train_ratio = config['data']['train_ratio']
    val_ratio   = config['data']['val_ratio']
    test_ratio  = config['data']['test_ratio']

    for client_id, data in client_data.items():
        X, y = data['X'], data['y']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_ratio + test_ratio),
            random_state=42,
            stratify=y if _can_stratify(y) else None
        )

        relative_test = test_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test,
            random_state=42,
            stratify=y_temp if _can_stratify(y_temp) else None
        )

        client_data[client_id] = {
            'X_train': X_train, 'y_train': y_train,
            'X_val':   X_val,   'y_val':   y_val,
            'X_test':  X_test,  'y_test':  y_test,
        }

        print(f"  Client {client_id:2d}: train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    return client_data


def save_partitions(client_data, config):
    """Persist each client partition to disk."""
    output_dir = Path(config['data']['partitioned_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for client_id, data in client_data.items():
        client_dir = output_dir / f'client_{client_id}'
        client_dir.mkdir(exist_ok=True)
        with open(client_dir / 'data.pkl', 'wb') as f:
            pickle.dump(data, f)

    print(f"\n[OK] Partitioned data saved to: {output_dir}")


def main():
    config = load_config()

    processed_file = Path(config['data']['processed_dir']) / 'processed_data.pkl'
    print(f"\nLoading preprocessed data from {processed_file} ...")
    with open(processed_file, 'rb') as f:
        data = pickle.load(f)

    X, y = data['X'], data['y']
    print(f"Total samples: {len(y)}")
    print(f"Num clients  : {config['data']['num_clients']}")

    client_data = create_non_iid_partitions(X, y, config)
    client_data = split_train_val_test(client_data, config)
    save_partitions(client_data, config)

    print("\n" + "=" * 60)
    print("[OK] Partitioning complete — client_1 … client_"
          f"{config['data']['num_clients']}")
    print("=" * 60)


if __name__ == "__main__":
    main()