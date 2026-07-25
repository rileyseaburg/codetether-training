"""Durable runtime and metric evidence for a training run."""

import json
import platform

from pathlib import Path

import torch

from .constants import BASE_MODEL, BASE_REVISION, MAX_LENGTH, SEED


def write(
    path: Path,
    baseline: dict[str, float],
    final: dict[str, float],
    train_metrics: dict[str, float],
) -> None:
    """Persist hardware identity, configuration, and observed metrics."""
    device = torch.cuda.get_device_properties(0)
    value = {
        'base_model': BASE_MODEL,
        'base_revision': BASE_REVISION,
        'seed': SEED,
        'max_length': MAX_LENGTH,
        'hardware': {
            'hostname': platform.node(),
            'gpu': device.name,
            'gpu_memory_bytes': device.total_memory,
            'cuda': torch.version.cuda,
            'torch': torch.__version__,
        },
        'baseline': baseline,
        'train': train_metrics,
        'final': final,
        'eval_loss_improvement': baseline['eval_loss'] - final['eval_loss'],
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
