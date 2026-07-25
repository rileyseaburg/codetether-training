"""Choose a trainable base model from available VRAM.

Qwen3-Coder-30B-A3B activates ~3B of 30B parameters per token, but QLoRA
still keeps every expert resident, so memory decides feasibility.
"""

from .memory_budget import estimate


LARGE_GB = 70.0
MEDIUM_GB = 40.0
SMALL_GB = 22.0
LARGE_MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
SMALL_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
LARGE_PARAMS = 30.5
SMALL_PARAMS = 4.0


def plan(gigabytes: float) -> dict[str, object]:
    """Return the recommended model and sequence length for a device."""
    if gigabytes >= LARGE_GB:
        return _entry(LARGE_MODEL, LARGE_PARAMS, 8192, 'h100-class')
    if gigabytes >= MEDIUM_GB:
        return _entry(LARGE_MODEL, LARGE_PARAMS, 4096, 'a100-40g')
    if gigabytes >= SMALL_GB:
        return _entry(LARGE_MODEL, LARGE_PARAMS, 4096, 'l4-24g')
    return _entry(SMALL_MODEL, SMALL_PARAMS, 2048, 't4-class')


def _entry(
    model: str, params: float, length: int, tier: str
) -> dict[str, object]:
    return {
        'recommended_model': model,
        'max_length': length,
        'tier': tier,
        'estimated': estimate(params, length),
    }
