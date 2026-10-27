import copy
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

from data.cifar10 import PoisonedSubset


class FLClient:
    def __init__(
        self,
        client_id: int,
        base_dataset: Dataset,
        indices: List[int],
        device: str,
        local_epochs: int,
        local_lr: float,
        local_momentum: float,
        local_weight_decay: float,
        batch_size: int,
        attack_fn: Optional[Callable] = None,
    ):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.local_weight_decay = local_weight_decay
        self.batch_size = batch_size
        self.indices = indices

        if attack_fn is not None:
            self.local_ds: Dataset = PoisonedSubset(base_dataset, indices, attack_fn)
        else:
            self.local_ds = Subset(base_dataset, indices)

    def local_train(self, global_model: nn.Module) -> Tuple[Dict, int]:
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        loader = DataLoader(self.local_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.local_lr,
            momentum=self.local_momentum,
            weight_decay=self.local_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
        return {k: v.cpu() for k, v in model.state_dict().items()}, len(self.local_ds)

    def gradient_signature(self, global_model: nn.Module, n_batches: int = 5) -> torch.Tensor:
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        loader = DataLoader(self.local_ds, batch_size=min(64, len(self.local_ds)), shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        model.zero_grad()
        batches_done = 0
        try:
            for x, y in loader:
                if batches_done >= n_batches:
                    break
                x, y = x.to(self.device), y.to(self.device)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    continue
                (loss / n_batches).backward()
                batches_done += 1
            if batches_done == 0:
                raise RuntimeError
            sig = torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters() if p.grad is not None])
            return torch.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return torch.zeros(sum(p.numel() for p in model.parameters()))
