"""Evaluate a trained adapter on real hardware.

Runs on the deployment-class GPU rather than the training host, so the
measurement reflects what the served model will actually do. Loads the base
model in 4-bit, attaches the adapter, and scores the behaviour probes that
decide promotion.
"""

import argparse
import json

from pathlib import Path

from model_training.adapter_eval import load_for_eval
from model_training.behaviour_probe import score


def main() -> None:
    """Print behaviour metrics for one checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--base', default=None)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', action='store_true')
    values = parser.parse_args()
    adapter = None if values.baseline else values.adapter
    model, tokenizer = load_for_eval(adapter, values.base)
    report = score(model, tokenizer)
    report['adapter'] = 'none' if values.baseline else str(values.adapter)
    text = json.dumps(report, indent=2, sort_keys=True)
    if values.output:
        values.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
