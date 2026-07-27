"""Hugging Face dataset loading for rendered training files."""

from pathlib import Path

from datasets import Dataset, load_dataset

from .constants import MAX_EVAL_SAMPLES, SEED


SUPERVISION_COLUMNS = ('prompt', 'completion', 'length')
"""TRL infers prompt-completion mode from the column names alone.

Any extra column, such as provenance metadata, breaks collation, so it is
dropped after loading while the manifests retain full provenance.

`length` is kept because `group_by_length` reads that column to batch
similar sizes together. Without it the trainer silently passes `None` and
performs no grouping, leaving a measured 69 percent padding waste in place.
"""


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
    return _project(train_data), _project(_subset(validation_data))


def _subset(data: Dataset) -> Dataset:
    """Bound the validation split so evaluation stays affordable."""
    if len(data) <= MAX_EVAL_SAMPLES:
        return data
    shuffled = data.shuffle(seed=SEED)
    return shuffled.select(range(MAX_EVAL_SAMPLES))


def _project(data: Dataset) -> Dataset:
    """Keep only the columns TRL expects for supervised training."""
    if 'prompt' not in data.column_names:
        return data
    extra = [c for c in data.column_names if c not in SUPERVISION_COLUMNS]
    return data.remove_columns(extra) if extra else data
