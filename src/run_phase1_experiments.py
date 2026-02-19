"""
Phase 1.5 — Experiment Automation & Plotting
Runs two FL experiments (Baseline vs Active-Ledger) and generates:
  - robustness_evaluation.pdf  (accuracy per round, both conditions)
  - gas_overhead.pdf           (gas cost comparison bar chart)
"""

import sys
import json
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe on Windows
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Make src importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config
from model import create_model, CNNLSTM
from train_utils import (
    load_client_data,
    create_data_loaders,
    train_epoch,
    evaluate,
    ECGDataset,
)
from torch.utils.data import DataLoader
from blockchain import BlockchainManager, fetch_client_history
from server import calculate_score

# ── Constants ────────────────────────────────────────────────────────────────
NUM_ROUNDS   = 10
NUM_NORMAL   = 4
NUM_MALICIOUS = 1
TOTAL_CLIENTS = NUM_NORMAL + NUM_MALICIOUS   # 5
TOP_K         = 4   # Active-Ledger selects this many per round

OUTPUT_DIR   = Path(__file__).parent.parent  # repo root
GANACHE_URL  = "http://127.0.0.1:8545"


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_client_loaders(config, total_clients, batch_size):
    """
    Load/synthesise data loaders for `total_clients` clients.
    If partitioned data for that client doesn't exist on disk, synthetically
    generate a small random dataset so the experiment can always run.
    """
    partitioned_dir = Path(config["data"]["partitioned_dir"])
    loaders, val_loaders, sizes = [], [], []

    for cid in range(1, total_clients + 1):
        client_dir = partitioned_dir / f"client_{cid}"
        if client_dir.exists():
            data = load_client_data(cid, str(partitioned_dir))
            X_train, y_train = data["X_train"], data["y_train"]
            X_val,   y_val   = data["X_val"],   data["y_val"]
        else:
            # Synthetic fallback — random ECG-shaped data, 5 classes
            rng = np.random.default_rng(seed=cid * 42)
            n_train, n_val = 300, 60
            X_train = rng.standard_normal((n_train, 360)).astype(np.float32)
            y_train = rng.integers(0, 5, n_train)
            X_val   = rng.standard_normal((n_val,   360)).astype(np.float32)
            y_val   = rng.integers(0, 5, n_val)

        tl, vl = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
        loaders.append(tl)
        val_loaders.append(vl)
        sizes.append(len(y_train))

    return loaders, val_loaders, sizes


# ── Core training primitives ─────────────────────────────────────────────────

def _get_weights(model):
    return [p.cpu().detach().numpy() for p in model.parameters()]


def _set_weights(model, weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.tensor(w))


def _fedavg_aggregate(global_model, client_weights_list, sizes):
    """Weighted-average aggregation (FedAvg)."""
    total = sum(sizes)
    global_dict = global_model.state_dict()
    agg = {k: torch.zeros_like(v, dtype=torch.float32)
           for k, v in global_dict.items()
           if "num_batches_tracked" not in k}

    param_keys = [k for k in global_dict if "num_batches_tracked" not in k]

    for weights, size in zip(client_weights_list, sizes):
        factor = size / total
        tmp_model = deepcopy(global_model)
        _set_weights(tmp_model, weights)
        tmp_sd = tmp_model.state_dict()
        for k in param_keys:
            agg[k] += tmp_sd[k].float() * factor

    new_sd = dict(global_dict)
    for k in param_keys:
        new_sd[k] = agg[k].to(global_dict[k].dtype)
    global_model.load_state_dict(new_sd)
    return global_model


def _train_one_client(global_model, train_loader, val_loader,
                      local_epochs, lr, device, is_malicious=False):
    """
    Local training step for one client.
    Returns (weights, num_samples, accuracy).
    Malicious clients return Gaussian noise weights and accuracy=0.10.
    """
    client_model = deepcopy(global_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)

    client_model.train()
    for _ in range(local_epochs):
        train_epoch(client_model, train_loader, criterion, optimizer, device)

    val_metrics = evaluate(client_model, val_loader, criterion, device)
    honest_acc  = float(val_metrics["accuracy"])
    weights     = _get_weights(client_model)
    n_samples   = len(train_loader.dataset)

    if is_malicious:
        noisy = [np.random.normal(0, 1, w.shape).astype(np.float32)
                 for w in weights]
        return noisy, n_samples, 0.10

    return weights, n_samples, honest_acc


def _global_eval(global_model, val_loaders, device):
    """Average accuracy across all clients' val loaders."""
    criterion = nn.CrossEntropyLoss()
    accs = []
    for vl in val_loaders:
        m = evaluate(global_model, vl, criterion, device)
        accs.append(m["accuracy"])
    return float(np.mean(accs))


# ── Experiment A: Baseline Standard FedAvg ───────────────────────────────────

def run_baseline(config, loaders, val_loaders, sizes, device):
    """
    Standard FedAvg: all 5 clients participate every round (including malicious).
    Records global accuracy per round.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A — Standard FedAvg (Baseline)")
    print("=" * 60)

    local_epochs = config["federated"]["local_epochs"]
    lr           = config["model"]["learning_rate"]
    global_model = create_model(config).to(device)
    malicious_ids = set(range(NUM_NORMAL, TOTAL_CLIENTS))  # last client is malicious

    round_accs = []
    for rnd in range(1, NUM_ROUNDS + 1):
        client_weights, client_sizes = [], []

        for cid in range(TOTAL_CLIENTS):
            is_mal = cid in malicious_ids
            w, n, acc = _train_one_client(
                global_model, loaders[cid], val_loaders[cid],
                local_epochs, lr, device, is_malicious=is_mal
            )
            client_weights.append(w)
            client_sizes.append(n)

        # Aggregate ALL clients (including malicious — no defence)
        global_model = _fedavg_aggregate(global_model, client_weights, client_sizes)
        g_acc = _global_eval(global_model, val_loaders, device)
        round_accs.append(g_acc)
        print(f"  Round {rnd:2d}/{NUM_ROUNDS}  global_val_acc={g_acc:.4f}")

    return round_accs


# ── Experiment B: Active-Ledger with PoC Selection ────────────────────────────

def run_active_ledger(config, loaders, val_loaders, sizes, device, blockchain: BlockchainManager):
    """
    Active-Ledger FedAvg:
      - Each client trains locally.
      - After training, the server logs the update on-chain (emits ModelUpdate).
      - Client selection for next round is filtered by PoC score from on-chain
        history; only top-K clients are included in the aggregation.
    Records global accuracy per round.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B — Active-Ledger FedAvg (PoC Selection)")
    print("=" * 60)

    local_epochs  = config["federated"]["local_epochs"]
    lr            = config["model"]["learning_rate"]
    global_model  = create_model(config).to(device)
    malicious_ids = set(range(NUM_NORMAL, TOTAL_CLIENTS))

    # Map each simulated client-id to an Ethereum account (round-robin)
    eth_accounts = blockchain.w3.eth.accounts[:TOTAL_CLIENTS]
    # Pad if Ganache has fewer than TOTAL_CLIENTS accounts
    while len(eth_accounts) < TOTAL_CLIENTS:
        eth_accounts = list(eth_accounts) + [blockchain.deployer]

    round_accs = []

    for rnd in range(1, NUM_ROUNDS + 1):
        all_results  = []   # (client_idx, weights, size, accuracy, eth_addr)

        # — Local training (all clients) —
        for cid in range(TOTAL_CLIENTS):
            is_mal = cid in malicious_ids
            w, n, acc = _train_one_client(
                global_model, loaders[cid], val_loaders[cid],
                local_epochs, lr, device, is_malicious=is_mal
            )
            all_results.append((cid, w, n, acc, eth_accounts[cid]))

        # — On-chain logging (all participated clients) —
        for cid, w, n, acc, addr in all_results:
            try:
                # Use eth account directly as the transaction sender
                acc_int   = int(acc * 10000)
                dummy_hash = bytes([cid % 256] * 32)
                tx = blockchain.contract.functions.logUpdate(
                    rnd, cid + 1, dummy_hash, n, acc_int
                ).transact({"from": addr})
                blockchain.w3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                # Non-fatal: continue experiment if a log tx fails
                print(f"    [warn] blockchain log failed for client {cid+1}: {e}")

        # — PoC-based client selection for aggregation —
        scored = []
        for cid, w, n, acc, addr in all_results:
            history = fetch_client_history(addr, blockchain.contract, blockchain.w3)
            score   = calculate_score(history)
            scored.append((score, cid, w, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:TOP_K]

        sel_weights = [w for _, _, w, _ in selected]
        sel_sizes   = [n for _, _, _, n in selected]

        # — Aggregate only selected (high-PoC) clients —
        global_model = _fedavg_aggregate(global_model, sel_weights, sel_sizes)
        g_acc = _global_eval(global_model, val_loaders, device)
        round_accs.append(g_acc)

        top_scores = [f"{s:.3f}" for s, *_ in selected]
        print(f"  Round {rnd:2d}/{NUM_ROUNDS}  selected={[c+1 for _,c,_,_ in selected]}"
              f"  scores={top_scores}  global_val_acc={g_acc:.4f}")

    return round_accs


# ── Gas Cost Estimation ────────────────────────────────────────────────────────

def estimate_gas_costs(blockchain: BlockchainManager):
    """
    Estimate on-chain gas cost of two approaches using eth_estimateGas:
      1. logUpdate() — which now emits the ModelUpdate event (current approach)
      2. A hypothetical function that pushes a 64-element uint256 array to storage
         (array-push pattern — simulated via repeated SSTORE estimation)

    Returns:
        (gas_event: int, gas_array: int)
    """
    deployer = blockchain.deployer
    contract  = blockchain.contract

    # Cost of current logUpdate (emits event)
    gas_event = contract.functions.logUpdate(
        1, 1, b"\x00" * 32, 1000, 9000
    ).estimate_gas({"from": deployer})

    # Hypothetical array-push: We estimate SSTORE overhead for a 64-element
    # uint256 array relative to a single-value write.
    # An SSTORE to a fresh slot costs 20000 gas (Berlin/London EIP-2929).
    # A 64-element push therefore costs approximately 64 * SSTORE_COLD.
    SSTORE_COLD = 20_000
    CALL_OVERHEAD = 21_000
    ARRAY_WRITE_ELEMENTS = 64
    gas_array = CALL_OVERHEAD + ARRAY_WRITE_ELEMENTS * SSTORE_COLD

    return gas_event, gas_array


# ── Plot helpers ──────────────────────────────────────────────────────────────

FONT_SIZE  = 11
COLOR_BASE = "#2C7BB6"
COLOR_POC  = "#D7191C"
COLOR_BAR1 = "#4D9DE0"
COLOR_BAR2 = "#E15554"


def plot_robustness(baseline_accs, active_accs, output_path):
    rounds = list(range(1, len(baseline_accs) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, baseline_accs, marker="o", linewidth=2,
            color=COLOR_BASE, label="Standard FedAvg")
    ax.plot(rounds, active_accs,  marker="s", linewidth=2,
            color=COLOR_POC,  label="Active-Ledger (PoC)")

    ax.set_xlabel("Communication Round", fontsize=FONT_SIZE)
    ax.set_ylabel("Global Validation Accuracy", fontsize=FONT_SIZE)
    ax.set_title("Robustness Evaluation Under Byzantine Attack\n"
                 f"({NUM_NORMAL} normal + {NUM_MALICIOUS} malicious client)",
                 fontsize=FONT_SIZE + 1)
    ax.set_xticks(rounds)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_gas_overhead(gas_event, gas_array, output_path):
    labels = ["logUpdate()\n(Event-Emit)", "Hypothetical\n(Array-Push ×64)"]
    values = [gas_event, gas_array]
    colors = [COLOR_BAR1, COLOR_BAR2]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=FONT_SIZE)

    ax.set_ylabel("Estimated Gas Units", fontsize=FONT_SIZE)
    ax.set_title("On-Chain Gas Cost Comparison\n"
                 "Event-emit vs. Array-storage pattern",
                 fontsize=FONT_SIZE + 1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    config = load_config()
    device = torch.device("cpu")

    batch_size   = config["training"]["batch_size"]
    total_clients = TOTAL_CLIENTS

    print("=" * 60)
    print("PHASE 1.5 — EXPERIMENT AUTOMATION")
    print("=" * 60)
    print(f"  Clients  : {NUM_NORMAL} normal + {NUM_MALICIOUS} malicious")
    print(f"  Rounds   : {NUM_ROUNDS}")
    print(f"  Top-K    : {TOP_K}  (Active-Ledger selection)")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[..] Loading client data ...")
    loaders, val_loaders, sizes = _load_client_loaders(config, total_clients, batch_size)
    print(f"[OK] Loaded {total_clients} client datasets: {sizes}")

    # ── Blockchain init ───────────────────────────────────────────────────────
    print("\n[..] Connecting to blockchain ...")
    try:
        blockchain = BlockchainManager(GANACHE_URL)
    except Exception as e:
        print(f"[FAIL] Blockchain init: {e}")
        raise

    # ── Experiment A ──────────────────────────────────────────────────────────
    t0 = time.time()
    baseline_accs = run_baseline(config, loaders, val_loaders, sizes, device)
    print(f"\n[A] Baseline done in {time.time()-t0:.1f}s")

    # ── Experiment B ──────────────────────────────────────────────────────────
    t1 = time.time()
    active_accs = run_active_ledger(config, loaders, val_loaders, sizes, device, blockchain)
    print(f"\n[B] Active-Ledger done in {time.time()-t1:.1f}s")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[..] Generating plots ...")
    plot_robustness(
        baseline_accs, active_accs,
        OUTPUT_DIR / "robustness_evaluation.pdf"
    )

    gas_event, gas_array = estimate_gas_costs(blockchain)
    print(f"[OK] Gas — logUpdate (event): {gas_event:,}  |  hypothetical array-push: {gas_array:,}")
    plot_gas_overhead(
        gas_event, gas_array,
        OUTPUT_DIR / "gas_overhead.pdf"
    )

    print("\n" + "=" * 60)
    print("PHASE 1.5 COMPLETE")
    print(f"  robustness_evaluation.pdf -> {OUTPUT_DIR}")
    print(f"  gas_overhead.pdf          -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
