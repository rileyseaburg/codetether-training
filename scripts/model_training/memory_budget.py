"""Estimate QLoRA memory so a run fails fast instead of mid-epoch.

A mixture-of-experts model keeps every expert resident under QLoRA, so
total parameters decide feasibility even when only a few are active.
"""

BYTES_PER_4BIT_PARAM = 0.55
"""NF4 weight plus double-quantization and per-block scale overhead."""

ADAPTER_OVERHEAD_GB = 1.5
ACTIVATION_GB_PER_1K = 0.45
"""Approximate activation cost per 1k tokens with gradient checkpointing."""


def estimate(parameters_billion: float, sequence: int) -> dict[str, float]:
    """Return estimated gigabytes for weights, activations, and total."""
    weights = parameters_billion * BYTES_PER_4BIT_PARAM
    activations = (sequence / 1000.0) * ACTIVATION_GB_PER_1K
    total = weights + activations + ADAPTER_OVERHEAD_GB
    return {
        'weights_gb': round(weights, 1),
        'activations_gb': round(activations, 1),
        'total_gb': round(total, 1),
    }


def fits(parameters_billion: float, sequence: int, available_gb: float) -> bool:
    """Return whether the configuration fits with a safety margin."""
    total = estimate(parameters_billion, sequence)['total_gb']
    return total <= available_gb * 0.9
