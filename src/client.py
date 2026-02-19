"""
Flower Client for FL with optional malicious behaviour simulation.
Phase 1.4 — Malicious Client Simulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from model import create_model, CNNLSTM
from train_utils import train_epoch, evaluate


class FLClient(fl.client.NumPyClient):
    """
    Flower NumPy client wrapping the CNN-LSTM model.

    Args:
        client_id  : integer identifier (1-based)
        model      : instantiated CNNLSTM model
        train_loader: PyTorch DataLoader for training set
        val_loader : PyTorch DataLoader for validation set
        config     : project config dict
        is_malicious: when True the client returns Gaussian noise instead of
                      real weight updates (Byzantine attack simulation)
    """

    def __init__(
        self,
        client_id: int,
        model: CNNLSTM,
        train_loader,
        val_loader,
        config: Dict,
        is_malicious: bool = False,
    ):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_malicious = is_malicious
        self.device = torch.device("cpu")
        self.model.to(self.device)

    # ------------------------------------------------------------------
    def get_parameters(self, config: Dict) -> NDArrays:
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    # ------------------------------------------------------------------
    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.parameters(), parameters)
        for p, new_p in params_dict:
            with torch.no_grad():
                p.copy_(torch.tensor(new_p))

    # ------------------------------------------------------------------
    def fit(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local training step."""
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", self.config["federated"]["local_epochs"]))
        lr = self.config["model"]["learning_rate"]

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(local_epochs):
            train_epoch(self.model, self.train_loader, criterion, optimizer, self.device)

        # Evaluate to get honest accuracy before (potentially) poisoning
        val_metrics = evaluate(self.model, self.val_loader, criterion, self.device)
        honest_accuracy = float(val_metrics["accuracy"])

        updated_weights = self.get_parameters({})
        num_samples = len(self.train_loader.dataset)

        if self.is_malicious:
            # Byzantine attack: replace weights with scaled Gaussian noise
            poisoned_weights = [
                np.random.normal(0, 1, w.shape).astype(np.float32)
                for w in updated_weights
            ]
            return poisoned_weights, num_samples, {"accuracy": 0.10, "is_malicious": 1}

        return updated_weights, num_samples, {"accuracy": honest_accuracy, "is_malicious": 0}

    # ------------------------------------------------------------------
    def evaluate(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate current global model on local validation set."""
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(self.model, self.val_loader, criterion, self.device)

        return (
            float(metrics["loss"]),
            len(self.val_loader.dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
            },
        )
