# FBUL: Federated Bulk Unlearning

Official implementation of **Federated Bulk Unlearning (FBUL)**, an integrated framework for coordinated knowledge removal and recovery in federated learning systems.

> **Paper:** *Federated Bulk Unlearning: Coordinated Knowledge Removal and Recovery in Federated Data Systems*  
> IEEE Transactions on Knowledge and Data Engineering (TKDE), under review.

---

## Overview

FBUL removes the combined influence of multiple corrupted data sources from a federated model without full retraining. The framework consists of four integrated stages:

1. **Update Signature Analysis** — characterises each corrupted source's contribution using stored gradient signatures.
2. **Adaptive Source Grouping** — partitions the removal set into sub-clusters using silhouette-score-guided KMeans (falls back to a single group when clustering adds no signal).
3. **Coordinated Erasure** — performs per-cluster gradient ascent and combines the resulting deltas with per-layer stability constraints so that no single layer is over-perturbed.
4. **Clean Proxy Correction** — fine-tunes the post-erasure model on a randomly-sampled subset of verified benign sources, with early stopping on loss convergence.

![FBUL Architecture](figures/FBUL-Architecture.png)

---

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
scikit-learn>=1.2
scipy>=1.10
```

Install:

```bash
pip install -r requirements.txt
```

---

## Quick Start

**Train a federated model and run FBUL unlearning on CIFAR-10:**

```bash
# IID, k=30 harmful clients, label-flip attack
python main.py --dataset cifar10 --k-harmful 30 --attack-type label_flip --seed 42

# Non-IID (Dirichlet α=0.1), backdoor attack
python main.py --dataset cifar10 --k-harmful 30 --attack-type backdoor \
               --distribution noniid --alpha 0.1 --seed 42

# Smoke test (fast, small config for verification)
python main.py --smoke
```

Results are written to `./results/` as CSV files.

---

## Repository Structure

```
fbul-release/
├── main.py                    # Entry point
├── config.py                  # All hyperparameters
├── requirements.txt
│
├── fl/
│   ├── client.py              # FLClient: local training + gradient signature
│   └── server.py              # FedAvgServer: global aggregation + checkpointing
│
├── attacks/
│   └── injector.py            # Label-flip and backdoor attack builders
│
├── unlearning/
│   ├── fbul.py                # FBUL unlearner (OOP interface)
│   ├── fbul_core.py           # FBUL core algorithm (functional, documented)
│   └── clustering.py          # Adaptive sub-clustering (silhouette KMeans)
│
├── models/
│   └── resnet.py              # ResNet-18 for CIFAR-10
│
├── data/
│   └── cifar10.py             # Dataset loading and Dirichlet partitioning
│
├── verification/
│   └── metrics.py             # MTA, ASR, MIA-AUC, GradSim, relearn rounds
│
└── utils/
    ├── checkpoint.py          # Checkpoint save/load
    └── logger.py              # CSV result logger
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--k-harmful` | 30 | Number of sources to remove |
| `--num-clients` | 100 | Total FL participants |
| `--num-rounds` | 100 | Global training rounds |
| `--ascent-lr` | 0.01 | Learning rate for gradient ascent erasure |
| `--ascent-steps` | 100 | Gradient ascent steps per sub-cluster |
| `--stability-eps` | 0.10 | Per-layer stability clipping coefficient |
| `--correction-rounds` | 20 | Maximum correction rounds (early-stopped) |
| `--sil-threshold` | 0.15 | Silhouette score threshold for sub-clustering |
| `--proxy-fraction` | 0.20 | Fraction of benign sources used for correction |

---

## Baselines

The following baselines are included for comparison:

| Method | Reference |
|--------|-----------|
| Sequential | Per-source gradient ascent (k operations) |
| FedEraser | Wu et al., IEEE TIFS 2022 |
| RAPID | Cao et al., IEEE S&P 2023 (FedRecover) |

---

## Reproducing Paper Results

The full experiment suite (15 experiments, 4 datasets, 5 seeds each) was run on NVIDIA Tesla V100 GPUs. To reproduce a specific result:

```bash
# Main IID comparison, k=30, CIFAR-10
python main.py --dataset cifar10 --k-harmful 30 --attack-type label_flip \
               --distribution iid --seed 42

# Non-IID hard (α=0.1)
python main.py --dataset cifar10 --k-harmful 30 --attack-type label_flip \
               --distribution noniid --alpha 0.1 --seed 42

# Backdoor (trigger injection)
python main.py --dataset cifar10 --k-harmful 30 --attack-type backdoor \
               --distribution iid --seed 42
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{fbul2025,
  title   = {Federated Bulk Unlearning: Coordinated Knowledge Removal and Recovery in Federated Data Systems},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.
