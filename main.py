import argparse
import os
import random
import sys
from typing import Dict, Set

import numpy as np
import torch

from config import Config, FederatedConfig, AttackConfig, UnlearningConfig
from data.cifar10 import load_cifar10, partition_data
from models.resnet import build_resnet18
from fl.client import FLClient
from fl.server import FedAvgServer
from attacks.injector import build_attack_functions
from unlearning.fbul import FBULUnlearner
from verification.metrics import compute_mta, compute_asr, compute_mia_auc, compute_grad_sim
from utils.logger import CSVLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FBUL: Federated Bulk Unlearning — CIFAR-10")
    p.add_argument("--data-dir",           type=str,   default="./data")
    p.add_argument("--output-dir",         type=str,   default="./results")
    p.add_argument("--checkpoint-dir",     type=str,   default="./checkpoints")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--device",             type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",             action="store_true")
    p.add_argument("--num-clients",        type=int,   default=100)
    p.add_argument("--num-rounds",         type=int,   default=100)
    p.add_argument("--clients-per-round",  type=int,   default=10)
    p.add_argument("--local-epochs",       type=int,   default=5)
    p.add_argument("--local-lr",           type=float, default=0.01)
    p.add_argument("--local-momentum",     type=float, default=0.9)
    p.add_argument("--local-weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size",         type=int,   default=64)
    p.add_argument("--distribution",       type=str,   default="iid", choices=["iid", "noniid"])
    p.add_argument("--alpha",              type=float, default=0.5)
    p.add_argument("--k-harmful",          type=int,   default=30)
    p.add_argument("--attack-type",        type=str,   default="label_flip", choices=["label_flip", "backdoor", "mixed"])
    p.add_argument("--poison-fraction",    type=float, default=0.20)
    p.add_argument("--trigger-size",       type=int,   default=4)
    p.add_argument("--target-label",       type=int,   default=0)
    p.add_argument("--source-label",       type=int,   default=-1)
    p.add_argument("--ascent-lr",          type=float, default=0.01)
    p.add_argument("--ascent-steps",       type=int,   default=100)
    p.add_argument("--stability-eps",      type=float, default=0.10)
    p.add_argument("--correction-lr",      type=float, default=0.001)
    p.add_argument("--correction-rounds",  type=int,   default=20)
    p.add_argument("--correction-tol",     type=float, default=1e-4)
    p.add_argument("--proxy-fraction",     type=float, default=0.20)
    p.add_argument("--max-subclusters",    type=int,   default=8)
    p.add_argument("--sil-threshold",      type=float, default=0.15)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        federated=FederatedConfig(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            clients_per_round=args.clients_per_round,
            local_epochs=args.local_epochs,
            local_lr=args.local_lr,
            local_momentum=args.local_momentum,
            local_weight_decay=args.local_weight_decay,
            batch_size=args.batch_size,
            distribution=args.distribution,
            alpha=args.alpha,
        ),
        attack=AttackConfig(
            k_harmful=args.k_harmful,
            attack_type=args.attack_type,
            poison_fraction=args.poison_fraction,
            trigger_size=args.trigger_size,
            target_label=args.target_label,
            source_label=args.source_label,
        ),
        unlearning=UnlearningConfig(
            ascent_lr=args.ascent_lr,
            ascent_steps=args.ascent_steps,
            stability_eps=args.stability_eps,
            correction_lr=args.correction_lr,
            correction_rounds=args.correction_rounds,
            correction_convergence_tol=args.correction_tol,
            proxy_fraction=args.proxy_fraction,
            max_subclusters=args.max_subclusters,
            sil_threshold=args.sil_threshold,
        ),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_clients(
    train_dataset,
    partition: Dict[int, list],
    harmful_ids: Set[int],
    attack_fns: Dict[int, callable],
    cfg: Config,
) -> Dict[int, FLClient]:
    clients = {}
    fl = cfg.federated
    for cid, indices in partition.items():
        clients[cid] = FLClient(
            client_id=cid,
            base_dataset=train_dataset,
            indices=indices,
            device=cfg.device,
            local_epochs=fl.local_epochs,
            local_lr=fl.local_lr,
            local_momentum=fl.local_momentum,
            local_weight_decay=fl.local_weight_decay,
            batch_size=fl.batch_size,
            attack_fn=attack_fns.get(cid),
        )
    return clients


def run_evaluation(model, test_dataset, forget_dataset, theta_pre, attack_fn, cfg: Config) -> Dict:
    from torch.utils.data import Subset
    import random as rnd
    non_member_indices = rnd.sample(range(len(test_dataset)), min(len(forget_dataset), len(test_dataset)))
    non_member_ds = Subset(test_dataset, non_member_indices)
    return {
        "mta":      compute_mta(model, test_dataset, cfg.device),
        "asr":      compute_asr(model, test_dataset, attack_fn, cfg.device),
        "mia_auc":  compute_mia_auc(model, forget_dataset, non_member_ds, cfg.device),
        "grad_sim": compute_grad_sim(model, theta_pre, forget_dataset, cfg.device),
    }


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    print(f"[FBUL] Device: {cfg.device} | Distribution: {cfg.federated.distribution} | k={cfg.attack.k_harmful} | Attack: {cfg.attack.attack_type}")

    train_dataset, test_dataset = load_cifar10(cfg.data_dir)
    partition = partition_data(
        train_dataset,
        cfg.federated.num_clients,
        cfg.federated.distribution,
        cfg.federated.alpha,
        cfg.seed,
    )

    rng_harm = random.Random(cfg.seed + 1)
    harmful_ids: Set[int] = set(rng_harm.sample(list(partition.keys()), cfg.attack.k_harmful))

    attack_fns = build_attack_functions(
        list(harmful_ids),
        cfg.attack.attack_type,
        cfg.attack.trigger_size,
        cfg.attack.target_label,
        cfg.attack.source_label,
    )

    clients = build_clients(train_dataset, partition, harmful_ids, attack_fns, cfg)

    global_model = build_resnet18().to(cfg.device)
    server = FedAvgServer(
        global_model=global_model,
        clients=clients,
        harmful_ids=harmful_ids,
        test_dataset=test_dataset,
        clients_per_round=cfg.federated.clients_per_round,
        num_rounds=cfg.federated.num_rounds,
        device=cfg.device,
        seed=cfg.seed,
        checkpoint_dir=cfg.checkpoint_dir,
        resume=cfg.resume,
        batch_size=cfg.federated.batch_size,
    )

    print("[FBUL] Starting FL training...")
    trained_model = server.train()

    mta_before = compute_mta(trained_model, test_dataset, cfg.device)
    print(f"[FBUL] Pre-unlearning MTA: {mta_before:.4f}")

    theta_pre_snapshot = {k: v.cpu().clone() for k, v in trained_model.state_dict().items()}

    unlearner = FBULUnlearner(
        global_model=trained_model,
        clients=clients,
        base_dataset=train_dataset,
        device=cfg.device,
        ascent_lr=cfg.unlearning.ascent_lr,
        ascent_steps=cfg.unlearning.ascent_steps,
        stability_eps=cfg.unlearning.stability_eps,
        correction_lr=cfg.unlearning.correction_lr,
        correction_rounds=cfg.unlearning.correction_rounds,
        correction_tol=cfg.unlearning.correction_convergence_tol,
        proxy_fraction=cfg.unlearning.proxy_fraction,
        max_subclusters=cfg.unlearning.max_subclusters,
        sil_threshold=cfg.unlearning.sil_threshold,
        batch_size=cfg.federated.batch_size,
    )

    print("[FBUL] Running FBUL unlearning...")
    unlearned_model, unlearn_meta = unlearner.unlearn(known_harmful=harmful_ids)

    print(
        f"[FBUL] Unlearning complete | Clusters: {unlearn_meta['n_clusters']} | "
        f"Sil: {unlearn_meta['silhouette_score']:.3f} | Time: {unlearn_meta['unlearn_time_s']:.1f}s"
    )

    from torch.utils.data import Subset as TSubset
    forget_indices = [idx for cid in harmful_ids for idx in partition[cid]]
    forget_dataset = TSubset(train_dataset, forget_indices)

    attack_fn_eval = list(attack_fns.values())[0] if attack_fns else (lambda x, y: (x, y))
    metrics = run_evaluation(unlearned_model, test_dataset, forget_dataset, theta_pre_snapshot, attack_fn_eval, cfg)

    print(
        f"[FBUL] Post-unlearning MTA: {metrics['mta']:.4f} | ASR: {metrics['asr']:.4f} | "
        f"MIA-AUC: {metrics['mia_auc']:.4f} | GradSim: {metrics['grad_sim']:.4f}"
    )

    logger = CSVLogger(cfg.output_dir, "fbul_results.csv", [
        "seed", "distribution", "alpha", "k_harmful", "attack_type",
        "num_rounds", "local_epochs", "local_lr", "batch_size",
        "ascent_lr", "ascent_steps", "stability_eps", "correction_lr",
        "mta_before", "mta_after", "asr_after", "mia_auc", "grad_sim",
        "n_clusters", "silhouette_score", "correction_rounds", "unlearn_time_s",
    ])
    logger.log({
        "seed":              cfg.seed,
        "distribution":      cfg.federated.distribution,
        "alpha":             cfg.federated.alpha,
        "k_harmful":         cfg.attack.k_harmful,
        "attack_type":       cfg.attack.attack_type,
        "num_rounds":        cfg.federated.num_rounds,
        "local_epochs":      cfg.federated.local_epochs,
        "local_lr":          cfg.federated.local_lr,
        "batch_size":        cfg.federated.batch_size,
        "ascent_lr":         cfg.unlearning.ascent_lr,
        "ascent_steps":      cfg.unlearning.ascent_steps,
        "stability_eps":     cfg.unlearning.stability_eps,
        "correction_lr":     cfg.unlearning.correction_lr,
        "mta_before":        round(mta_before, 6),
        "mta_after":         round(metrics["mta"], 6),
        "asr_after":         round(metrics["asr"], 6),
        "mia_auc":           round(metrics["mia_auc"], 6),
        "grad_sim":          round(metrics["grad_sim"], 6),
        "n_clusters":        unlearn_meta["n_clusters"],
        "silhouette_score":  round(unlearn_meta["silhouette_score"], 6),
        "correction_rounds": unlearn_meta["correction_rounds"],
        "unlearn_time_s":    round(unlearn_meta["unlearn_time_s"], 3),
    })
    logger.close()
    print(f"[FBUL] Results saved to {os.path.join(cfg.output_dir, 'fbul_results.csv')}")


if __name__ == "__main__":
    main()
