"""Configuration loading, reproducibility controls, and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file and require a top-level mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration {path} must be a YAML mapping.")
    return config


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable local experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs(base: str | Path = "outputs") -> dict[str, Path]:
    """Create local output folders without adding generated files to Git."""
    root = Path(base)
    folders = {"root": root, "figures": root / "figures", "results": root / "results", "models": root / "models"}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders
