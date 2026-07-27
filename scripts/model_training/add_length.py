"""Add a token-length column to already rendered pairs.

`group_by_length` batches similar sizes together, but it reads a `length`
column and silently performs no grouping when that column is absent. The
v4 pairs were rendered before the column existed, and measurement showed
69 percent of every 8,192 token window was padding as a result.
"""

import argparse
import json

from pathlib import Path

from transformers import AutoTokenizer

from .model_target import resolve_target


def main() -> None:
    """Rewrite a pairs file with a precomputed length column."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    values = parser.parse_args()
    model, revision = resolve_target()
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, use_fast=True
    )
    written = 0
    with values.pairs.open() as source, values.output.open('w') as target:
        for line in source:
            record = json.loads(line)
            if 'length' not in record:
                record['length'] = _count(tokenizer, record)
            target.write(json.dumps(record, sort_keys=True) + '\n')
            written += 1
    print(json.dumps({'path': str(values.output), 'records': written}))


def _count(tokenizer: object, record: dict[str, str]) -> int:
    """Return the combined prompt and completion token count."""
    text = record['prompt'] + record['completion']
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


if __name__ == '__main__':
    main()
