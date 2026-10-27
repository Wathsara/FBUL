import os
import torch
import torch.nn as nn
from typing import Optional


def save_checkpoint(checkpoint_dir: str, round_num: int, model: nn.Module) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"round_{round_num:04d}.pt")
    torch.save({"round": round_num, "model": model.state_dict()}, path)
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.endswith(".pt") and f != os.path.basename(path):
            try:
                os.remove(os.path.join(checkpoint_dir, f))
            except OSError:
                pass


def load_checkpoint(checkpoint_dir: str, model: nn.Module) -> int:
    if not os.path.isdir(checkpoint_dir):
        return 0
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
    if not files:
        return 0
    state = torch.load(os.path.join(checkpoint_dir, files[-1]), map_location="cpu")
    model.load_state_dict(state["model"])
    return state["round"] + 1
