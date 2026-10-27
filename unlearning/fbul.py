import copy
import time
from typing import Dict, List, Set, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from sklearn.preprocessing import normalize

from fl.client import FLClient
from unlearning.clustering import SubclusterDetector


class GradientSignatureCollector:
    def __init__(self, global_model: nn.Module, clients: Dict[int, FLClient], device: str):
        self.global_model = global_model
        self.clients = clients
        self.device = device

    def collect(self, client_ids: List[int]) -> np.ndarray:
        signatures = [self.clients[cid].gradient_signature(self.global_model).numpy() for cid in client_ids]
        return np.array(signatures)


class HarmfulClientDetector:
    def __init__(self, global_model: nn.Module, clients: Dict[int, FLClient], device: str):
        self.collector = GradientSignatureCollector(global_model, clients, device)
        self.clients = clients

    def detect(self, known_harmful: Optional[Set[int]] = None) -> Set[int]:
        if known_harmful is not None:
            return set(known_harmful)
        all_ids = list(self.clients.keys())
        signatures = self.collector.collect(all_ids)
        normed = normalize(signatures, norm="l2")
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(normed)
        c0_size = np.sum(labels == 0)
        c1_size = np.sum(labels == 1)
        bad_label = 0 if c0_size <= c1_size else 1
        return {all_ids[i] for i in np.where(labels == bad_label)[0]}


class BulkAscentEngine:
    def __init__(self, device: str, ascent_lr: float, ascent_steps: int, batch_size: int = 64):
        self.device = device
        self.ascent_lr = ascent_lr
        self.ascent_steps = ascent_steps
        self.batch_size = batch_size

    def ascend(self, model: nn.Module, dataset: Dataset) -> nn.Module:
        if dataset is None or len(dataset) == 0:
            return model
        model.train()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.ascent_lr, momentum=0.9)
        loader_iter = iter(loader)
        for _ in range(self.ascent_steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            (-criterion(model(x), y)).backward()
            optimizer.step()
        return model


class DeltaAggregator:
    @staticmethod
    def aggregate(theta_pre: Dict, ascent_states: List[Dict], weights: List[float]) -> Dict:
        delta = {k: torch.zeros_like(v.float()) for k, v in theta_pre.items()}
        for state, w in zip(ascent_states, weights):
            for k in theta_pre:
                delta[k] += w * (state[k].float() - theta_pre[k].float())
        return delta


class StabilityConstrainedUpdater:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def apply(self, theta_pre: Dict, delta: Dict) -> Dict:
        new_state = {}
        for k in theta_pre:
            theta_k = theta_pre[k].float()
            delta_k = delta[k].float()
            budget = self.epsilon * (theta_k.norm().item() + 1e-8)
            d_norm = delta_k.norm().item()
            scale = (budget / (d_norm + 1e-8)) if d_norm > budget else 1.0
            new_state[k] = theta_k - scale * delta_k
        return new_state


class CorrectionEngine:
    def __init__(self, device: str, lr: float, max_rounds: int, convergence_tol: float, batch_size: int):
        self.device = device
        self.lr = lr
        self.max_rounds = max_rounds
        self.convergence_tol = convergence_tol
        self.batch_size = batch_size

    def correct(self, model: nn.Module, proxy_clients: List[FLClient], base_dataset: Dataset) -> Tuple[nn.Module, int]:
        benign_data = ConcatDataset([Subset(base_dataset, c.indices) for c in proxy_clients])
        loader = DataLoader(benign_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        model.train()
        prev_loss = float("inf")
        rounds_done = 0
        for rnd in range(self.max_rounds):
            epoch_loss, n = 0.0, 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            avg = epoch_loss / max(n, 1)
            rounds_done = rnd + 1
            if abs(prev_loss - avg) < self.convergence_tol:
                break
            prev_loss = avg
        return model, rounds_done


class FBULUnlearner:
    def __init__(
        self,
        global_model: nn.Module,
        clients: Dict[int, FLClient],
        base_dataset: Dataset,
        device: str,
        ascent_lr: float,
        ascent_steps: int,
        stability_eps: float,
        correction_lr: float,
        correction_rounds: int,
        correction_tol: float,
        proxy_fraction: float,
        max_subclusters: int,
        sil_threshold: float,
        batch_size: int,
    ):
        self.global_model = global_model
        self.clients = clients
        self.base_dataset = base_dataset
        self.device = device
        self.proxy_fraction = proxy_fraction

        self.detector = HarmfulClientDetector(global_model, clients, device)
        self.signature_collector = GradientSignatureCollector(global_model, clients, device)
        self.subclustering = SubclusterDetector(max_subclusters, sil_threshold)
        self.ascent_engine = BulkAscentEngine(device, ascent_lr, ascent_steps, batch_size)
        self.aggregator = DeltaAggregator()
        self.updater = StabilityConstrainedUpdater(stability_eps)
        self.correction_engine = CorrectionEngine(device, correction_lr, correction_rounds, correction_tol, batch_size)

    def _build_cluster_dataset(self, cluster_ids: List[int]) -> Dataset:
        return ConcatDataset([Subset(self.base_dataset, self.clients[cid].indices) for cid in cluster_ids])

    def _select_proxy_clients(self, benign_ids: List[int]) -> List[FLClient]:
        import random
        n = max(1, int(len(benign_ids) * self.proxy_fraction))
        selected = random.sample(benign_ids, min(n, len(benign_ids)))
        return [self.clients[cid] for cid in selected]

    def unlearn(self, known_harmful: Optional[Set[int]] = None) -> Tuple[nn.Module, Dict]:
        t0 = time.time()

        detected_harmful = self.detector.detect(known_harmful)
        harmful_ids = list(detected_harmful)

        signatures = self.signature_collector.collect(harmful_ids)
        self.subclustering.fit(signatures)
        cluster_index_groups = self.subclustering.cluster_indices()

        theta_pre = {k: v.cpu() for k, v in self.global_model.state_dict().items()}

        ascent_states, cluster_weights = [], []
        for idx_group in cluster_index_groups:
            cluster_ids = [harmful_ids[i] for i in idx_group]
            cluster_ds = self._build_cluster_dataset(cluster_ids)
            ascent_model = copy.deepcopy(self.global_model).to(self.device)
            self.ascent_engine.ascend(ascent_model, cluster_ds)
            ascent_states.append({k: v.cpu() for k, v in ascent_model.state_dict().items()})
            cluster_weights.append(len(cluster_ids) / len(harmful_ids))

        delta = self.aggregator.aggregate(theta_pre, ascent_states, cluster_weights)
        new_state = self.updater.apply(theta_pre, delta)
        self.global_model.load_state_dict({k: v.to(self.device) for k, v in new_state.items()})

        benign_ids = [cid for cid in self.clients if cid not in detected_harmful]
        proxy_clients = self._select_proxy_clients(benign_ids)
        self.global_model, correction_rounds = self.correction_engine.correct(
            self.global_model, proxy_clients, self.base_dataset
        )

        meta = {
            "detected_harmful": detected_harmful,
            "n_clusters": self.subclustering.n_clusters_,
            "silhouette_score": self.subclustering.silhouette_score_,
            "correction_rounds": correction_rounds,
            "unlearn_time_s": time.time() - t0,
        }
        return self.global_model, meta
