"""Hugging Face dataset loading for rendered training files."""

from pathlib import Path

from datasets import Dataset, load_dataset


def splits(train: Path, validation: Path) -> tuple[Dataset, Dataset]:
    """Load immutable JSONL paths as independent datasets."""
    train_data = load_dataset(
        'json',
        data_files=str(train),
        split='train',
    )
    validation_data = load_dataset(
        'json',
        data_files=str(validation),
        split='train',
    )
    return train_data, validation_data
