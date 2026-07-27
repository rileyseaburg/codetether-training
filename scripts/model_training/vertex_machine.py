"""Select a Vertex machine and accelerator for a base model.

Verified quota in us-central1 for project spotlessbinco:
`restricted_image_training_nvidia_a100_80gb_gpus` and
`custom_model_training_preemptible_nvidia_a100_gpus`, both limit 8.
"""

A100_80GB = ('a2-ultragpu-1g', 'NVIDIA_A100_80GB')
A100_40GB = ('a2-highgpu-1g', 'NVIDIA_TESLA_A100')
LARGE_MARKER = '30B'


def select(model: str) -> tuple[str, str]:
    """Return the machine type and accelerator for a model.

    The 30B mixture-of-experts model needs 45.4 GB and hit CUDA OOM on a
    40 GB device, so it requires the 80 GB tier. The dense 4B model needs
    7.4 GB and runs on either.
    """
    return A100_80GB if LARGE_MARKER in model else A100_40GB
