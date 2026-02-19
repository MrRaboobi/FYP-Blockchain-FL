"""
Flower Server Strategy — Proof-of-Contribution (PoC) Active-Ledger Orchestration.
Phase 1.4 — Active Orchestration (v2: corrected PoC bounds, EMA weighting)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Dict, List, Optional, Tuple
import numpy as np

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitIns,
    Parameters,
)

from blockchain import fetch_client_history


# ---------------------------------------------------------------------------
# Proof-of-Contribution helpers
# ---------------------------------------------------------------------------

def _ema(values: List[float], alpha: float = 0.7) -> float:
    """
    Exponential Moving Average (most-recent-first EMA).
    Entries are assumed to be ordered oldest → newest.
    Returns the EMA value (same range as the input values).
    """
    if not values:
        return 0.0
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
    return ema


def calculate_score(history: List[Dict]) -> float:
    """
    Proof-of-Contribution (PoC) reputation score derived from on-chain history.

    The on-chain `accuracy` field is stored as an integer (accuracy × 10000).
    This function normalises it by dividing by 10000.0 before computing the
    score.

    Score formula:
        ema_acc   = EMA(normalised accuracy values, alpha=0.7)
        max_round = highest round index seen in history
        participation = len(history) / max_round   (← always ≤ 1)
        raw_score = ema_acc × participation

    The final score is clamped strictly to (0, 1) via:
        score = min(max(raw_score, 1e-6), 1.0 - 1e-6)

    Args:
        history: list returned by `fetch_client_history`; each element is
                 {'round': int, 'accuracy': float/int, 'timestamp': int}.
                 NOTE: the accuracy value from fetch_client_history is already
                 divided by 10000 (see blockchain.py).  This function handles
                 both: if the value is > 1 it treats it as raw integer form.

    Returns:
        float strictly in (0, 1).
    """
    if not history:
        return 0.5   # baseline for unseen clients

    # Normalise: fetch_client_history already divides by 10000,
    # but guard against raw integer leakage from direct contract reads.
    acc_values = []
    for entry in sorted(history, key=lambda e: e['round']):
        raw = entry['accuracy']
        normalised = raw / 10000.0 if raw > 1.0 else float(raw)
        # Clamp individual accuracy to [0, 1]
        normalised = min(max(normalised, 0.0), 1.0)
        acc_values.append(normalised)

    ema_acc = _ema(acc_values, alpha=0.7)

    max_round = max(entry['round'] for entry in history)
    if max_round <= 0:
        participation = 1.0
    else:
        participation = len(history) / max_round          # ∈ (0, 1]

    raw_score = ema_acc * participation

    # Strict (0, 1) bound
    score = min(max(raw_score, 1e-6), 1.0 - 1e-6)
    return float(score)


# ---------------------------------------------------------------------------
# Custom strategy
# ---------------------------------------------------------------------------

class PoCFedAvg(FedAvg):
    """
    FedAvg variant that selects clients according to their PoC reputation
    score queried from the on-chain `ModelUpdate` event log.

    Args:
        contract      : deployed Web3 contract instance (FLLogger)
        web3_instance : live Web3 connection
        eth_accounts  : list of Ethereum account strings (one per client proxy)
        top_k_fraction: fraction of available clients to select (default 0.8)
        **kwargs      : passed through to FedAvg
    """

    def __init__(
        self,
        contract,
        web3_instance,
        eth_accounts: List[str],
        top_k_fraction: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contract = contract
        self.web3_instance = web3_instance
        self.eth_accounts = eth_accounts
        self.top_k_fraction = top_k_fraction

    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Override configure_fit to rank available clients by PoC score and
        select only the top-K fraction.
        """
        config  = {"local_epochs": 1, "server_round": server_round}
        fit_ins = FitIns(parameters, config)

        sample_size = max(1, int(client_manager.num_available()))
        clients     = client_manager.sample(num_clients=sample_size)

        scored: List[Tuple[float, ClientProxy]] = []
        for proxy in clients:
            try:
                idx = int(proxy.cid) % len(self.eth_accounts)
            except (ValueError, TypeError):
                idx = 0
            addr    = self.eth_accounts[idx]
            history = fetch_client_history(addr, self.contract, self.web3_instance)
            score   = calculate_score(history)
            scored.append((score, proxy))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k    = max(1, round(len(scored) * self.top_k_fraction))
        selected = [proxy for _, proxy in scored[:top_k]]

        return [(proxy, fit_ins) for proxy in selected]
