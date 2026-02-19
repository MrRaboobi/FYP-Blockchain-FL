"""
Flower Server Strategy — Proof-of-Contribution (PoC) Active-Ledger Orchestration.
Phase 1.4 — Active Orchestration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)

from blockchain import fetch_client_history


# ---------------------------------------------------------------------------
# Proof-of-Contribution helpers
# ---------------------------------------------------------------------------

def calculate_score(history: List[Dict]) -> float:
    """
    Compute a Proof-of-Contribution (PoC) reputation score from a client's
    on-chain history.

    Args:
        history: list returned by `fetch_client_history`; each element is
                 {'round': int, 'accuracy': float, 'timestamp': int}.

    Returns:
        float: PoC score in [0, 1].
               Baseline 0.5 when no history exists.
               Otherwise  avg(accuracy) * (len(history) / current_round),
               where current_round is len(history) used as a proxy so that
               the participation ratio is always <= 1.
    """
    if not history:
        return 0.5

    accuracies = [entry["accuracy"] for entry in history]
    avg_acc = float(np.mean(accuracies))

    # participation ratio: rounds participated / total rounds seen
    current_round = max(entry["round"] for entry in history)
    if current_round == 0:
        return avg_acc

    participation = len(history) / current_round
    poc_score = avg_acc * participation
    return float(poc_score)


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
        config = {"local_epochs": 1, "server_round": server_round}
        fit_ins = FitIns(parameters, config)

        # Sample all currently available clients
        sample_size = max(1, int(client_manager.num_available()))
        clients = client_manager.sample(num_clients=sample_size)

        # Score each client by their on-chain history
        scored: List[Tuple[float, ClientProxy]] = []
        for proxy in clients:
            # Map proxy to an Ethereum address by index (round-robin fallback)
            try:
                idx = int(proxy.cid) % len(self.eth_accounts)
            except (ValueError, TypeError):
                idx = 0
            addr = self.eth_accounts[idx]

            history = fetch_client_history(addr, self.contract, self.web3_instance)
            score = calculate_score(history)
            scored.append((score, proxy))

        # Sort descending by score, take top K
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = max(1, round(len(scored) * self.top_k_fraction))
        selected = [proxy for _, proxy in scored[:top_k]]

        return [(proxy, fit_ins) for proxy in selected]
