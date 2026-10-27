import copy
import random
from typing import Dict, List, Tuple, Set, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from fl.client import FLClient
from utils.checkpoint import save_checkpoint, load_checkpoint


class FedAvgServer:
    def __init__(
        self,
        global_model: nn.Module,
        clients: Dict[int, FLClient],
        harmful_ids: Set[int],
        test_dataset: Dataset,
        clients_per_round: int,
        num_rounds: int,
        device: str,
        seed: int,
        checkpoint_dir: str,
        resume: bool = False,
        batch_size: int = 256,
    ):
        self.global_model = global_model
        self.clients = clients
        self.harmful_ids = harmful_ids
        self.test_dataset = test_dataset
        self.clients_per_round = clients_per_round
        self.num_rounds = num_rounds
        self.device = device
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        self.batch_size = batch_size
        self.round_history: List[Tuple[int, Dict]] = []

    @staticmethod
    def fedavg(updates: List[Tuple[Dict, int]]) -> Dict:
        total = sum(n for _, n in updates)
        aggregated = {k: torch.zeros_like(v.float()) for k, v in updates[0][0].items()}
        for state_dict, n in updates:
            w = n / total
            for k in aggregated:
                aggregated[k] += state_dict[k].float() * w
        return aggregated

    def train(self) -> nn.Module:
        rng = random.Random(self.seed)
        all_ids = list(self.clients.keys())
        start_round = 0
        if self.resume:
            start_round = load_checkpoint(self.checkpoint_dir, self.global_model)

        for rnd in range(start_round, self.num_rounds):
            selected = rng.sample(all_ids, min(self.clients_per_round, len(all_ids)))
            updates = [self.clients[cid].local_train(self.global_model) for cid in selected]
            new_weights = self.fedavg(updates)
            self.global_model.load_state_dict({k: v.to(self.device) for k, v in new_weights.items()})
            self.round_history.append((rnd, {k: v.cpu() for k, v in new_weights.items()}))
            if len(self.round_history) > 20:
                self.round_history.pop(0)
            if (rnd + 1) % 10 == 0 or (rnd + 1) == self.num_rounds:
                save_checkpoint(self.checkpoint_dir, rnd, self.global_model)

        return self.global_model
