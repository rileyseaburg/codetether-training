"""Prepare a new or existing adapter for quantized training."""

from pathlib import Path

from peft import PeftModel, prepare_model_for_kbit_training

from .lora_config import build


def configure(
    model: object, adapter: Path | None
) -> tuple[object, object | None]:
    """Prepare k-bit layers and optionally attach a trainable adapter."""
    prepared = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    if adapter is None:
        return prepared, build()
    attached = PeftModel.from_pretrained(
        prepared,
        adapter,
        is_trainable=True,
    )
    return attached, None
