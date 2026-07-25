"""Pin a tokenizer and render GPU-bounded SFT datasets."""

import argparse
import json

from pathlib import Path

from transformers import AutoTokenizer

from .constants import BASE_MODEL, BASE_REVISION
from .render_file import build


def main() -> None:
    """Render train and validation text using the pinned base template."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=8192)
    values = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION
    )
    evidence = {
        'base_model': BASE_MODEL,
        'base_revision': BASE_REVISION,
        'tokenizer_class': type(tokenizer).__name__,
        'vocab_size': len(tokenizer),
        'max_tokens': values.max_tokens,
        'train': build(
            values.directory / 'train.jsonl',
            values.directory / 'train-1024.jsonl',
            tokenizer,
            values.max_tokens,
        ),
        'validation': build(
            values.directory / 'validation.jsonl',
            values.directory / 'validation-1024.jsonl',
            tokenizer,
            values.max_tokens,
        ),
    }
    path = values.directory / 'render-manifest.json'
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'manifest': str(path), **evidence}, sort_keys=True))


if __name__ == '__main__':
    main()
