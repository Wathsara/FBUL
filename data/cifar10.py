import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


def load_cifar10(data_dir: str) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    train_tf, test_tf = build_transforms()
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return train_ds, test_ds


def iid_partition(dataset: Dataset, num_clients: int, seed: int = 42) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    shard_size = len(indices) // num_clients
    return {i: indices[i * shard_size:(i + 1) * shard_size] for i in range(num_clients)}


def noniid_partition(dataset: Dataset, num_clients: int, alpha: float, seed: int = 42) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(class_idx)).astype(int)
        proportions[-1] = len(class_idx) - proportions[:-1].sum()
        start = 0
        for i, count in enumerate(proportions):
            client_indices[i].extend(class_idx[start:start + count].tolist())
            start += count
    return client_indices


def partition_data(dataset: Dataset, num_clients: int, distribution: str, alpha: float, seed: int) -> Dict[int, List[int]]:
    if distribution == "iid":
        return iid_partition(dataset, num_clients, seed)
    return noniid_partition(dataset, num_clients, alpha, seed)


class PoisonedSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int], attack_fn):
        self.base = Subset(base_dataset, indices)
        self.attack_fn = attack_fn

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.attack_fn(x, y)
