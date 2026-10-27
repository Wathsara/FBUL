import random
from typing import Callable, List, Tuple

import torch
from torch.utils.data import Dataset


def label_flip_fn(source_label: int, target_label: int) -> Callable:
    def attack(x: torch.Tensor, y: int):
        if y == source_label or source_label == -1:
            return x, target_label
        return x, y
    return attack


def backdoor_fn(trigger_size: int, target_label: int) -> Callable:
    def attack(x: torch.Tensor, y: int):
        x = x.clone()
        x[:, -trigger_size:, -trigger_size:] = 1.0
        return x, target_label
    return attack


def mixed_fn(trigger_size: int, target_label: int, source_label: int, cluster_id: int) -> Callable:
    if cluster_id % 2 == 0:
        return backdoor_fn(trigger_size, target_label)
    return label_flip_fn(source_label, target_label)


def build_attack_functions(
    harmful_ids: List[int],
    attack_type: str,
    trigger_size: int,
    target_label: int,
    source_label: int,
) -> dict:
    attack_map = {}
    for i, cid in enumerate(harmful_ids):
        if attack_type == "label_flip":
            attack_map[cid] = label_flip_fn(source_label, target_label)
        elif attack_type == "backdoor":
            attack_map[cid] = backdoor_fn(trigger_size, target_label)
        elif attack_type == "mixed":
            attack_map[cid] = mixed_fn(trigger_size, target_label, source_label, i)
    return attack_map
