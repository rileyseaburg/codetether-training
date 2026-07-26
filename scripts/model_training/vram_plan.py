"""Choose a trainable base model from available VRAM.

Qwen3-Coder-30B-A3B activates ~3B of 30B parameters per token, but QLoRA
still keeps every expert resident, so memory decides feasibility.

An A100 40 GB cannot hold the 30B mixture-of-experts model: a real run
reached 39.47 GiB and hit CUDA OOM during loading. The 30B tier therefore
requires an 80 GB device, and 40 GB trains the dense 4B model instead.
"""

from .memory_budget import estimate


LARGE_GB = 70.0
MEDIUM_GB = 20.0
LARGE_MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
SMALL_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
LARGE_PARAMS = 30.5
SMALL_PARAMS = 4.0


def plan(gigabytes: float) -> dict[str, object]:
    """Return the recommended model and sequence length for a device."""
    if gigabytes >= LARGE_GB:
        return _entry(LARGE_MODEL, LARGE_PARAMS, 4096, 'h100-80g', True)
    if gigabytes >= MEDIUM_GB:
        return _entry(SMALL_MODEL, SMALL_PARAMS, 8192, 'a100-40g-or-l4', False)
    return _entry(SMALL_MODEL, SMALL_PARAMS, 2048, 't4-class', False)


def _entry(
    model: str, params: float, length: int, tier: str, moe: bool
) -> dict[str, object]:
    return {
        'recommended_model': model,
        'max_length': length,
        'tier': tier,
        'mixture_of_experts': moe,
        'estimated': estimate(params, length, moe),
    }
