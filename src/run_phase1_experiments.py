"""
Phase 1.5 v2 — Experiment Automation & Plotting (10-client, 15 MIT-BIH records)
Generates:
  robustness_evaluation.pdf  — dual subplot: accuracy + round latency
  gas_overhead.pdf           — gas cost bar chart
"""

import sys
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config
from model import create_model, CNNLSTM
from train_utils import load_client_data, create_data_loaders, train_epoch, evaluate
from blockchain import BlockchainManager, fetch_client_history
from server import calculate_score

# ── Experiment constants ──────────────────────────────────────────────────────
NUM_ROUNDS    = 15
NUM_NORMAL    = 8
NUM_MALICIOUS = 2
TOTAL_CLIENTS = 10     # NUM_NORMAL + NUM_MALICIOUS
TOP_K         = 7      # Active-Ledger selects this many per round

OUTPUT_DIR  = Path(__file__).parent.parent   # repo root
GANACHE_URL = "http://127.0.0.1:8545"

assert TOTAL_CLIENTS == NUM_NORMAL + NUM_MALICIOUS, "client counts must sum"


# ── Data loading — NO synthetic fallback ─────────────────────────────────────

def _load_client_loaders(config, total_clients, batch_size):
    """
    Load data for `total_clients` clients from disk.
    Raises FileNotFoundError if any client directory is missing (enforces
    data integrity — synthetic fallback is intentionally removed).
    """
    partitioned_dir = Path(config["data"]["partitioned_dir"])
    loaders, val_loaders, sizes = [], [], []

    for cid in range(1, total_clients + 1):
        client_dir = partitioned_dir / f"client_{cid}"
        if not client_dir.exists():
            raise FileNotFoundError(
                f"[FATAL] Partition not found: {client_dir}\n"
                "Run: python src/download_data.py && "
                "python src/preprocess_data.py && "
                "python src/partition_data.py"
            )
        data = load_client_data(cid, str(partitioned_dir))
        tl, vl = create_data_loaders(
            data["X_train"], data["y_train"],
            data["X_val"],   data["y_val"],
            batch_size
        )
        loaders.append(tl)
        val_loaders.append(vl)
        sizes.append(len(data["y_train"]))

    return loaders, val_loaders, sizes


# ── Core training primitives ─────────────────────────────────────────────────

def _get_weights(model):
    return [p.cpu().detach().numpy() for p in model.parameters()]


def _set_weights(model, weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.tensor(w))


def _fedavg_aggregate(global_model, client_weights_list, sizes):
    total     = sum(sizes)
    global_sd = global_model.state_dict()
    param_keys = [k for k in global_sd if "num_batches_tracked" not in k]
    agg = {k: torch.zeros_like(v, dtype=torch.float32)
           for k, v in global_sd.items() if k in param_keys}

    for weights, size in zip(client_weights_list, sizes):
        factor   = size / total
        tmp      = deepcopy(global_model)
        _set_weights(tmp, weights)
        tmp_sd   = tmp.state_dict()
        for k in param_keys:
            agg[k] += tmp_sd[k].float() * factor

    new_sd = dict(global_sd)
    for k in param_keys:
        new_sd[k] = agg[k].to(global_sd[k].dtype)
    global_model.load_state_dict(new_sd)
    return global_model


def _train_one_client(global_model, train_loader, val_loader,
                      local_epochs, lr, device, is_malicious=False):
    client_model = deepcopy(global_model).to(device)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(client_model.parameters(), lr=lr)

    client_model.train()
    for _ in range(local_epochs):
        train_epoch(client_model, train_loader, criterion, optimizer, device)

    val_metrics = evaluate(client_model, val_loader, criterion, device)
    honest_acc  = float(val_metrics["accuracy"])
    weights     = _get_weights(client_model)
    n_samples   = len(train_loader.dataset)

    if is_malicious:
        noisy = [np.random.normal(0, 1, w.shape).astype(np.float32) for w in weights]
        return noisy, n_samples, 0.10

    return weights, n_samples, honest_acc


def _global_eval(global_model, val_loaders, device):
    criterion = nn.CrossEntropyLoss()
    accs = [evaluate(global_model, vl, criterion, device)["accuracy"]
            for vl in val_loaders]
    return float(np.mean(accs))


# ── Experiment A: Baseline FedAvg ────────────────────────────────────────────

def run_baseline(config, loaders, val_loaders, sizes, device):
    print("\n" + "=" * 60)
    print("EXPERIMENT A — Standard FedAvg (Baseline)")
    print(f"  {NUM_NORMAL} normal + {NUM_MALICIOUS} malicious, all included each round")
    print("=" * 60)

    local_epochs  = config["federated"]["local_epochs"]
    lr            = config["model"]["learning_rate"]
    global_model  = create_model(config).to(device)
    malicious_ids = set(range(NUM_NORMAL, TOTAL_CLIENTS))

    round_accs, round_latencies = [], []

    for rnd in range(1, NUM_ROUNDS + 1):
        t_start = time.time()
        cw, cs  = [], []
        for cid in range(TOTAL_CLIENTS):
            w, n, _ = _train_one_client(
                global_model, loaders[cid], val_loaders[cid],
                local_epochs, lr, device, is_malicious=(cid in malicious_ids)
            )
            cw.append(w); cs.append(n)

        global_model = _fedavg_aggregate(global_model, cw, cs)
        g_acc  = _global_eval(global_model, val_loaders, device)
        t_rnd  = time.time() - t_start

        round_accs.append(g_acc)
        round_latencies.append(t_rnd)
        print(f"  Round {rnd:2d}/{NUM_ROUNDS}  acc={g_acc:.4f}  latency={t_rnd:.1f}s")

    return round_accs, round_latencies


# ── Experiment B: Active-Ledger FedAvg ───────────────────────────────────────

def run_active_ledger(config, loaders, val_loaders, sizes, device,
                      blockchain: BlockchainManager):
    print("\n" + "=" * 60)
    print("EXPERIMENT B — Active-Ledger FedAvg (PoC Selection)")
    print(f"  top-K={TOP_K} clients selected per round via on-chain PoC score")
    print("=" * 60)

    local_epochs  = config["federated"]["local_epochs"]
    lr            = config["model"]["learning_rate"]
    global_model  = create_model(config).to(device)
    malicious_ids = set(range(NUM_NORMAL, TOTAL_CLIENTS))

    # Ethereum accounts — use as many as Ganache provides
    eth_accounts = list(blockchain.w3.eth.accounts)
    while len(eth_accounts) < TOTAL_CLIENTS:
        eth_accounts.append(blockchain.deployer)

    round_accs, round_latencies = [], []

    for rnd in range(1, NUM_ROUNDS + 1):
        t_start = time.time()

        # Local training
        results = []
        for cid in range(TOTAL_CLIENTS):
            w, n, acc = _train_one_client(
                global_model, loaders[cid], val_loaders[cid],
                local_epochs, lr, device, is_malicious=(cid in malicious_ids)
            )
            results.append((cid, w, n, acc, eth_accounts[cid]))

        # On-chain logging
        for cid, w, n, acc, addr in results:
            try:
                acc_int    = int(acc * 10000)
                dummy_hash = bytes([cid % 256] * 32)
                tx = blockchain.contract.functions.logUpdate(
                    rnd, cid + 1, dummy_hash, n, acc_int
                ).transact({"from": addr})
                blockchain.w3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                print(f"    [warn] blockchain log failed cid={cid+1}: {e}")

        # PoC selection
        scored = []
        for cid, w, n, acc, addr in results:
            history = fetch_client_history(addr, blockchain.contract, blockchain.w3)
            score   = calculate_score(history)
            scored.append((score, cid, w, n))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected   = scored[:TOP_K]
        sel_weights = [w for _, _, w, _ in selected]
        sel_sizes   = [n for _, _, _, n in selected]

        global_model = _fedavg_aggregate(global_model, sel_weights, sel_sizes)
        g_acc  = _global_eval(global_model, val_loaders, device)
        t_rnd  = time.time() - t_start

        round_accs.append(g_acc)
        round_latencies.append(t_rnd)
        top_ids    = [c + 1 for _, c, _, _ in selected]
        top_scores = [f"{s:.3f}" for s, *_ in selected]
        print(f"  Round {rnd:2d}/{NUM_ROUNDS}  sel={top_ids}  scores={top_scores}"
              f"  acc={g_acc:.4f}  latency={t_rnd:.1f}s")

    return round_accs, round_latencies


# ── Gas cost estimation ───────────────────────────────────────────────────────

def estimate_gas_costs(blockchain: BlockchainManager):
    deployer  = blockchain.deployer
    contract  = blockchain.contract

    gas_event = contract.functions.logUpdate(
        1, 1, b"\x00" * 32, 1000, 9000
    ).estimate_gas({"from": deployer})

    # Hypothetical array-push: 64 × cold SSTORE (EIP-2929) + call overhead
    SSTORE_COLD = 20_000
    CALL_OVERHEAD = 21_000
    gas_array = CALL_OVERHEAD + 64 * SSTORE_COLD

    return gas_event, gas_array


# ── Plotting ──────────────────────────────────────────────────────────────────

C_BASE = "#2C7BB6"
C_POC  = "#D7191C"
C_BAR1 = "#4D9DE0"
C_BAR2 = "#E15554"
FS     = 10


def plot_robustness(baseline_accs, baseline_lat, active_accs, active_lat, output_path):
    """
    Dual-subplot PDF:
      Top    : Global Validation Accuracy vs Round
      Bottom : Round Latency (seconds) vs Round
    """
    rounds = list(range(1, len(baseline_accs) + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # — Top subplot: accuracy —
    ax1.plot(rounds, baseline_accs, marker="o", lw=2, color=C_BASE,
             label="Standard FedAvg")
    ax1.plot(rounds, active_accs,   marker="s", lw=2, color=C_POC,
             label="Active-Ledger (PoC)")
    ax1.set_ylabel("Global Validation Accuracy", fontsize=FS)
    ax1.set_title(
        f"Robustness Evaluation Under Byzantine Attack\n"
        f"({NUM_NORMAL} normal + {NUM_MALICIOUS} Byzantine clients, "
        f"{NUM_ROUNDS} rounds)",
        fontsize=FS + 1
    )
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax1.legend(fontsize=FS)
    ax1.grid(True, linestyle="--", alpha=0.45)

    # — Bottom subplot: latency —
    ax2.plot(rounds, baseline_lat, marker="o", lw=2, color=C_BASE,
             label="Standard FedAvg")
    ax2.plot(rounds, active_lat,   marker="s", lw=2, color=C_POC,
             label="Active-Ledger (PoC)")
    ax2.set_xlabel("Communication Round", fontsize=FS)
    ax2.set_ylabel("Round Latency (seconds)", fontsize=FS)
    ax2.set_title("Average Round Latency vs Communication Round", fontsize=FS + 1)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax2.legend(fontsize=FS)
    ax2.grid(True, linestyle="--", alpha=0.45)

    plt.xticks(rounds)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_gas_overhead(gas_event, gas_array, output_path):
    labels = ["logUpdate()\n(Event-Emit)", "Hypothetical\n(Array-Push ×64)"]
    values = [gas_event, gas_array]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=[C_BAR1, C_BAR2], width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=FS)
    ax.set_ylabel("Estimated Gas Units", fontsize=FS)
    ax.set_title("On-Chain Gas Cost Comparison\n"
                 "Event-emit vs. Array-storage pattern", fontsize=FS + 1)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
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

    config     = load_config()
    device     = torch.device("cpu")
    batch_size = config["training"]["batch_size"]

    print("=" * 60)
    print("PHASE 1.5 v2 — EXPERIMENT AUTOMATION (10-client)")
    print("=" * 60)
    print(f"  Clients  : {NUM_NORMAL} normal + {NUM_MALICIOUS} malicious = {TOTAL_CLIENTS}")
    print(f"  Rounds   : {NUM_ROUNDS}")
    print(f"  Top-K    : {TOP_K}")

    # Load real partitioned data (no fallback)
    print("\n[..] Loading client data (no synthetic fallback) ...")
    loaders, val_loaders, sizes = _load_client_loaders(config, TOTAL_CLIENTS, batch_size)
    print(f"[OK] Loaded {TOTAL_CLIENTS} clients: {sizes}")

    # Blockchain
    print("\n[..] Connecting to blockchain ...")
    blockchain = BlockchainManager(GANACHE_URL)

    # Experiments
    t0 = time.time()
    baseline_accs, baseline_lat = run_baseline(config, loaders, val_loaders, sizes, device)
    print(f"\n[A] Baseline done in {time.time()-t0:.1f}s")

    t1 = time.time()
    active_accs, active_lat = run_active_ledger(
        config, loaders, val_loaders, sizes, device, blockchain
    )
    print(f"\n[B] Active-Ledger done in {time.time()-t1:.1f}s")

    # Plots
    print("\n[..] Generating plots ...")
    plot_robustness(
        baseline_accs, baseline_lat,
        active_accs,   active_lat,
        OUTPUT_DIR / "robustness_evaluation.pdf"
    )

    gas_event, gas_array = estimate_gas_costs(blockchain)
    print(f"[OK] Gas — logUpdate (event): {gas_event:,} | array-push: {gas_array:,}")
    plot_gas_overhead(gas_event, gas_array, OUTPUT_DIR / "gas_overhead.pdf")

    print("\n" + "=" * 60)
    print("PHASE 1.5 v2 COMPLETE")
    print(f"  robustness_evaluation.pdf  -> {OUTPUT_DIR}")
    print(f"  gas_overhead.pdf           -> {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
