"""Build the worker pool specification for a Vertex custom training job."""

BOOT_DISK_GB = 500
"""The 30B base model is 61 GB, plus checkpoints and a merged export."""


def build(
    image: str,
    machine: str,
    accelerator: str,
    environment: dict[str, str],
    count: int = 1,
) -> dict[str, object]:
    """Return a single worker pool running the training entrypoint."""
    return {
        'workerPoolSpecs': [
            {
                'machineSpec': {
                    'machineType': machine,
                    'acceleratorType': accelerator,
                    'acceleratorCount': count,
                },
                'replicaCount': 1,
                'diskSpec': {
                    'bootDiskType': 'pd-ssd',
                    'bootDiskSizeGb': BOOT_DISK_GB,
                },
                'containerSpec': {
                    'imageUri': image,
                    'command': [
                        'bash',
                        '/workspace/ct/scripts/model_training/vertex_entrypoint.sh',
                    ],
                    'env': [
                        {'name': key, 'value': value}
                        for key, value in sorted(environment.items())
                    ],
                },
            }
        ]
    }
