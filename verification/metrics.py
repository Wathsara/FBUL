from typing import Optional, Callable

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import roc_auc_score


def compute_mta(model: nn.Module, test_dataset: Dataset, device: str, batch_size: int = 256) -> float:
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def compute_asr(model: nn.Module, test_dataset: Dataset, attack_fn: Callable, device: str, batch_size: int = 256) -> float:
    from data.cifar10 import PoisonedSubset
    all_indices = list(range(len(test_dataset)))
    poisoned_ds = PoisonedSubset(test_dataset, all_indices, attack_fn)
    loader = DataLoader(poisoned_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    triggered, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            triggered += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return triggered / max(total, 1)


def compute_mia_auc(
    model: nn.Module,
    member_dataset: Dataset,
    non_member_dataset: Dataset,
    device: str,
    batch_size: int = 256,
    n_shadow: int = 5,
) -> float:
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()

    def get_losses(ds: Dataset) -> np.ndarray:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                losses.extend(criterion(model(x), y).cpu().numpy())
        return np.array(losses)

    member_losses = get_losses(member_dataset)
    non_member_losses = get_losses(non_member_dataset)
    scores = np.concatenate([-member_losses, -non_member_losses])
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.5


def compute_grad_sim(
    model: nn.Module,
    theta_pre: dict,
    forget_dataset: Dataset,
    device: str,
    batch_size: int = 64,
) -> float:
    model_copy = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    pre_vec = torch.cat([theta_pre[k].float().flatten() for k in theta_pre]).to(device)
    delta = model_copy - pre_vec
    model.eval()
    loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    model.zero_grad()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        criterion(model(x), y).backward()
        break
    forget_grad = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
    forget_grad = torch.nan_to_num(forget_grad)
    cos = nn.functional.cosine_similarity(delta.unsqueeze(0), forget_grad.unsqueeze(0)).item()
    model.zero_grad()
    return float(cos)
