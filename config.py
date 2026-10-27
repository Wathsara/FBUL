from dataclasses import dataclass, field


@dataclass
class FederatedConfig:
    num_clients: int = 100
    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9
    local_weight_decay: float = 1e-4
    batch_size: int = 64
    distribution: str = "iid"
    alpha: float = 0.5


@dataclass
class AttackConfig:
    k_harmful: int = 30
    attack_type: str = "label_flip"
    poison_fraction: float = 0.20
    trigger_size: int = 4
    target_label: int = 0
    source_label: int = -1


@dataclass
class UnlearningConfig:
    ascent_lr: float = 0.01
    ascent_steps: int = 100
    stability_eps: float = 0.10
    correction_lr: float = 0.001
    correction_rounds: int = 20
    correction_convergence_tol: float = 1e-4
    proxy_fraction: float = 0.20
    max_subclusters: int = 8
    sil_threshold: float = 0.15


@dataclass
class Config:
    data_dir: str = "./data"
    output_dir: str = "./results"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
    device: str = "cuda"
    resume: bool = False
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
