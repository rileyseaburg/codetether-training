# codetether-training

Reproducible QLoRA fine-tuning pipeline for CodeTether agent models.

Trains a coding-agent model on real agent transcripts: user requests,
assistant replies, and native tool calls harvested from local session
stores and Codex rollouts.

## Why this pipeline exists in this shape

An earlier revision (v2) **regressed against its own base model**. It
improved validation loss from 3.4744 to 1.1585 while its code-fix pass rate
fell from 0.75 to 0.00, and half of all short prompts returned nothing.

`REGRESSION_REPORT.md` documents the measured failure. Four causes:

| Defect | v2 | current |
|---|---|---|
| Loss target | every token, incl. user/system | assistant completions only |
| Tool schemas in prompts | never rendered | 20,107 train pairs |
| Short instructions | none | 240 primers |
| Corpus | one table, 8 KB cap | 1,719 stores + Codex rollouts |

The lesson is encoded as a rule: **validation loss is not the promotion
gate.** A candidate must beat the untuned base on behavioural benchmarks.

## Dataset

`DATASET_REPORT.md` has full provenance. Summary:

| Measure | Value |
|---|---:|
| Conversations ingested | 27,217 |
| Messages | 1,485,449 |
| Assistant turns | 708,950 |
| Native tool calls | 722,858 |
| Rendered train pairs | 40,716 |
| Rendered validation pairs | 2,498 |
| Training tokens | 113,051,545 |

Splits are assigned per whole conversation, so no conversation spans train
and validation. Exact prompt+completion overlap is zero.

## Quick start on a cloud GPU

See `COLAB_RUNBOOK.md`. The short version:

```bash
export VAULT_ADDR=https://vault.example.com
export VAULT_TOKEN=...          # never committed
export CODETETHER_HF_REPO=owner/dataset-repo

CODETETHER_BUNDLE=$PWD bash scripts/model_training/colab_bootstrap.sh
```

The bootstrap installs GPU dependencies, fetches the dataset from
HuggingFace, installs the CodeTether CLI, probes Vault for provider
credentials, prints a device plan, then trains detached.

## Device planning

`vram_plan.py` picks the model and sequence length from available VRAM, and
`memory_budget.py` estimates cost so an undersized device fails fast.

| VRAM | Model | Sequence | Estimated |
|---:|---|---:|---:|
| 80 GB | Qwen3-Coder-30B-A3B | 8192 | 22.0 GB |
| 40 GB | Qwen3-Coder-30B-A3B | 4096 | 20.1 GB |
| 24 GB | Qwen3-Coder-30B-A3B | 4096 | 20.1 GB |
| under 22 GB | Qwen3-4B-Instruct-2507 | 2048 | 4.6 GB |

Mixture-of-experts models keep every expert resident under QLoRA, so total
parameters decide feasibility even when few are active per token.

## Promotion gate

Run both benchmarks against the candidate and the untuned base:

```bash
python3 -m model_training.bench_local --base-url http://127.0.0.1:8099 \
    --output bench.json
python3 -m model_training.bench_empty --base-url http://127.0.0.1:8099 \
    --output empty.json
```

Promote only if the candidate raises the code-fix pass rate, keeps the
empty-response rate near zero, and emits valid tool calls.

## Credentials

Secrets are read at runtime from HashiCorp Vault and are never written to
this repository, generated notebooks, or logs. `vault_probe.py` reports
which providers are reachable without returning any value.

`HF_TOKEN` is honoured for local runs when Vault is unavailable.

## Layout

```text
scripts/model_training/
  corpus_index.py     locate transcript stores across the host
  ingest_all.py       build one deduplicated conversation corpus
  corpus_split.py     leakage-free train and validation split
  render_v3.py        masked, tool-aware, short-primed rendering
  train.py            QLoRA supervised fine-tuning
  merge.py            merge adapter into base weights
  gpu_probe.py        device capacity and model plan
  bench_local.py      tool-call and code-fix benchmark
  bench_empty.py      empty-response rate benchmark
```

## Conventions

Python is formatted and linted with `ruff`. Every module stays within a
50-line budget and holds one responsibility. Tests run with:

```bash
PYTHONPATH=scripts python3 -m unittest discover \
    -s scripts/model_training/tests
```

## License

MIT
