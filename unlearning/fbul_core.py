"""
FBUL — Federated Bulk Unlearning (Cluster-Aware)

Algorithm:
  1. Detection : cluster all-client gradient signatures → harmful set H.
  2. Sub-cluster H: partition harmful clients by their own gradient-signature
     similarity using adaptive k (silhouette-score threshold = 0.15).
     Falls back to k=1 (homogeneous) when clustering adds no signal.
  3. Per-cluster Gradient Ascent: one ascent model per sub-cluster,
     trained only on that cluster's data.  Stops early once loss ≥
     ascent_loss_threshold (≈ random-guess level) to avoid over-erasing.
  4. Weighted Delta Aggregation: δ_total = Σ (|C_i|/|H|) · δ_i
  5. Per-layer Stability-Constrained Subtraction:
       θ_new_k = θ_pre_k − clip(δ_k,  ε·‖θ_pre_k‖)
     Clipping is relative to each layer's own magnitude so that small
     layers (biases) receive proportionate updates.
  6. Correction: short fine-tuning on a randomly-sampled subset of
     verified benign clients.  Stops early when per-epoch loss converges.
"""

import copy
import random as _random
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from sklearn.cluster    import KMeans
from sklearn.metrics    import silhouette_score
from sklearn.preprocessing import normalize


# ── 1. Detection ─────────────────────────────────────────────────────────────

def detect_harmful_clients(clients: dict, global_model, cfg: dict,
                            known_harmful: set = None) -> set:
    """
    Collects multi-batch gradient signatures from all clients and
    2-means clusters them. Minority cluster = harmful.
    Oracle bypass via known_harmful.
    """
    if known_harmful is not None:
        return set(known_harmful)

    cids = list(clients.keys())
    sigs = [clients[cid].get_gradient_signature(global_model).numpy()
            for cid in cids]
    sigs_norm = normalize(np.array(sigs), norm="l2")

    km     = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = km.fit_predict(sigs_norm)

    c0, c1  = np.where(labels == 0)[0], np.where(labels == 1)[0]
    bad_lbl = 0 if len(c0) <= len(c1) else 1
    return {cids[i] for i in np.where(labels == bad_lbl)[0]}


# ── 2. Adaptive sub-clustering of harmful clients ─────────────────────────────

def _cluster_harmful_clients(clients: dict, harmful_ids: set,
                              global_model, cfg: dict) -> tuple[list[set], float]:
    """
    Partitions harmful_ids into sub-clusters based on gradient-signature
    similarity.  Threshold lowered to 0.15 so mixed-attack heterogeneity
    (e.g. label-flip + backdoor clients together) reliably triggers k > 1.

    Falls back to k=1 when:
      - fewer than 4 harmful clients
      - no k reaches the silhouette threshold
      - all signatures are identical
    """
    ids = list(harmful_ids)
    n   = len(ids)

    threshold = cfg.get("cluster_sil_threshold", 0.15)
    max_k     = cfg.get("cluster_max_k", 8)

    if n < 4:
        return [set(ids)], 0.0

    sigs = [clients[cid].get_gradient_signature(global_model).numpy()
            for cid in ids]
    sigs_arr = np.array(sigs)
    sigs_arr = np.nan_to_num(sigs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(sigs_arr):
        return [set(ids)], 0.0
    sigs_norm = normalize(sigs_arr, norm="l2")

    best_k, best_score = 1, -1.0
    for k in range(2, min(max_k + 1, n)):
        km     = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(sigs_norm)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(sigs_norm, labels)
        if score > best_score:
            best_score, best_k = score, k
        if score >= threshold:
            best_k = k
            break

    if best_k == 1 or best_score < threshold:
        return [set(ids)], 0.0

    km     = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    labels = km.fit_predict(sigs_norm)
    clusters = [set() for _ in range(best_k)]
    for i, lbl in enumerate(labels):
        clusters[lbl].add(ids[i])
    non_empty = [c for c in clusters if c]
    return non_empty, float(best_score)


# ── 3. Gradient Ascent (per-cluster, convergence-aware) ──────────────────────

def _build_cluster_dataset(clients: dict, cluster_ids: set, train_dataset):
    """Concatenates local datasets for one cluster of harmful clients."""
    subsets = [Subset(train_dataset, clients[cid].indices) for cid in cluster_ids]
    return ConcatDataset(subsets) if subsets else None


class _TriggeredDataset(Dataset):
    """Wraps a dataset and applies a pixel-patch trigger + target label to every sample."""
    def __init__(self, base_ds, trigger_size: int, target_label: int):
        self.base_ds      = base_ds
        self.trigger_size = trigger_size
        self.target_label = target_label

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        x, _ = self.base_ds[i]
        x = x.clone()
        _, h, w = x.shape
        ts = self.trigger_size
        x[:, h-ts:h, w-ts:w] = 1.0
        return x, self.target_label


def _gradient_ascent(model, dataset, device: str, lr: float, steps: int,
                     cfg: dict = None):
    """
    Maximises cross-entropy on dataset (gradient ascent).

    Improvements over original:
      - Uses cfg["local_batch_size"] instead of hardcoded 64.
      - Stops early when loss ≥ ascent_loss_threshold, meaning the model
        is already as confused as random guessing on the harmful data.
    """
    if dataset is None or len(dataset) == 0:
        return model
    model.train()
    batch_size     = cfg.get("local_batch_size", 64) if cfg else 64
    loss_threshold = cfg.get("ascent_loss_threshold", None) if cfg else None

    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion   = nn.CrossEntropyLoss()
    opt         = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader_iter = iter(loader)

    for _ in range(steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        (-loss).backward()   # negate → ascent
        opt.step()

        # Early stopping: model is already confused enough on harmful data
        if loss_threshold is not None and loss.item() >= loss_threshold:
            break

    return model


# ── 4. Weighted delta aggregation ────────────────────────────────────────────

def _aggregate_cluster_deltas(theta_pre: dict,
                               cluster_ascent_states: list,
                               cluster_weights: list) -> dict:
    """
    δ_total = Σ w_i · (θ_ascent_i − θ_pre),  w_i = |C_i| / |H|
    """
    delta_total = {k: torch.zeros_like(v.float()) for k, v in theta_pre.items()}
    for state, w in zip(cluster_ascent_states, cluster_weights):
        for k in theta_pre:
            delta_total[k] += w * (state[k].float() - theta_pre[k].float())
    return delta_total


# ── 5. Per-layer stability-constrained update ─────────────────────────────────

def _apply_stability_constrained_update(theta_pre: dict,
                                         delta: dict,
                                         epsilon: float,
                                         lam: float = 1.0) -> dict:
    """
    Per-layer clipping: each layer's delta is bounded relative to that
    layer's own parameter magnitude rather than a single global L2 norm.

        budget_k = ε · (‖θ_pre_k‖ + 1e-8)
        θ_new_k  = θ_pre_k − lam · clip(δ_k, budget_k)

    This prevents large layers (e.g. first conv in ResNet) from consuming
    the entire ε budget, leaving bias and batch-norm layers untouched.
    """
    new_weights = {}
    for k in theta_pre:
        theta_k  = theta_pre[k].float()
        delta_k  = delta[k].float()
        budget   = epsilon * (theta_k.norm().item() + 1e-8)
        dnorm    = delta_k.norm().item()
        scale    = (budget / (dnorm + 1e-8)) if dnorm > budget else 1.0
        new_weights[k] = theta_k - lam * scale * delta_k
    return new_weights


# ── 6. Correction fine-tuning (random proxy, adaptive stopping) ──────────────

def _correction_round(model, benign_clients: list, train_dataset,
                       device: str, cfg: dict):
    """
    Fine-tunes on verified benign proxy clients.

    Improvements over original:
      - benign_clients is already a randomly-sampled subset (done in run_fbul).
      - Stops early when per-epoch loss has converged (|Δloss| < 1e-4),
        avoiding unnecessary rounds when accuracy has already recovered.
    """
    benign_data = ConcatDataset(
        [Subset(train_dataset, c.indices) for c in benign_clients]
    )
    loader    = DataLoader(benign_data, batch_size=cfg["local_batch_size"],
                           shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    opt       = torch.optim.SGD(model.parameters(),
                                lr=cfg["correction_lr"], momentum=0.9)
    model.train()
    prev_loss = float("inf")

    for _ in range(cfg["correction_rounds"]):
        epoch_loss, n_batches = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1
        avg_loss = epoch_loss / max(n_batches, 1)

        if abs(prev_loss - avg_loss) < 1e-4:   # converged — stop early
            break
        prev_loss = avg_loss

    return model


# ── Main FBUL entry point ─────────────────────────────────────────────────────

def run_fbul(global_model, clients: dict, harmful_ids: set,
             train_dataset, device: str, cfg: dict,
             known_harmful: set = None, attack_type: str = None) -> tuple:
    """
    Cluster-aware bulk unlearning.
    Returns (unlearned_model, meta).
    """
    t0 = time.time()

    # ── Step 1: Detect harmful set ────────────────────────────────────────────
    detected = detect_harmful_clients(clients, global_model, cfg, known_harmful)

    # ── Step 2: Sub-cluster the harmful set ───────────────────────────────────
    # Allow caller to inject pre-formed clusters (e.g. RandomGroup baseline).
    if "_forced_clusters" in cfg:
        clusters, decision_sil = cfg["_forced_clusters"], 0.0
    else:
        clusters, decision_sil = _cluster_harmful_clients(
            clients, detected, global_model, cfg
        )
    n_clusters    = len(clusters)
    cluster_sizes = [len(c) for c in clusters]
    total_harmful = sum(cluster_sizes)

    # ── Step 3: Per-cluster gradient ascent ───────────────────────────────────
    theta_pre    = {k: v.cpu() for k, v in global_model.state_dict().items()}
    is_backdoor  = attack_type in ("backdoor", "mixed")

    cluster_ascent_states = []
    cluster_weights       = []

    for cluster_ids in clusters:
        cluster_ds   = _build_cluster_dataset(clients, cluster_ids, train_dataset)
        ascent_model = copy.deepcopy(global_model).to(device)

        # Phase 1: standard gradient ascent on harmful client data
        _gradient_ascent(ascent_model, cluster_ds, device,
                         lr=cfg["grad_ascent_lr"],
                         steps=cfg["grad_ascent_steps"],
                         cfg=cfg)

        # Phase 2 (backdoor only): additional ascent on trigger-injected samples
        if is_backdoor and cluster_ds is not None and len(cluster_ds) > 0:
            triggered_ds = _TriggeredDataset(
                cluster_ds,
                trigger_size  = cfg.get("backdoor_trigger_size", 3),
                target_label  = cfg.get("backdoor_target_label", 0),
            )
            _gradient_ascent(ascent_model, triggered_ds, device,
                             lr=cfg["grad_ascent_lr"],
                             steps=cfg.get("backdoor_ascent_steps",
                                           cfg["grad_ascent_steps"] * 3),
                             cfg=cfg)

        cluster_ascent_states.append(
            {k: v.cpu() for k, v in ascent_model.state_dict().items()}
        )
        cluster_weights.append(len(cluster_ids) / max(total_harmful, 1))

    # ── Step 4: Weighted delta aggregation ────────────────────────────────────
    delta_total = _aggregate_cluster_deltas(
        theta_pre, cluster_ascent_states, cluster_weights
    )

    # ── Step 5: Per-layer stability-constrained subtraction ───────────────────
    new_weights = _apply_stability_constrained_update(
        theta_pre, delta_total, epsilon=cfg["stability_eps"]
    )
    global_model.load_state_dict({k: v.to(device) for k, v in new_weights.items()})

    # ── Step 6: Correction on randomly-sampled benign proxy clients ───────────
    benign_clients = [c for cid, c in clients.items() if cid not in detected]
    n_proxy        = max(1, int(len(benign_clients) * cfg["proxy_benign_fraction"]))
    proxy_clients  = _random.sample(benign_clients, min(n_proxy, len(benign_clients)))
    _correction_round(global_model, proxy_clients, train_dataset, device, cfg)

    meta = {
        "detected_harmful":    detected,
        "true_harmful":        harmful_ids,
        "n_clusters":          n_clusters,
        "cluster_sizes":       cluster_sizes,
        "silhouette_score":    decision_sil,
        "detection_precision": len(detected & harmful_ids) / max(len(detected), 1),
        "detection_recall":    len(detected & harmful_ids) / max(len(harmful_ids), 1),
        "unlearn_time_s":      time.time() - t0,
    }
    return global_model, meta
