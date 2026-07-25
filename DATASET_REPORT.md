# v4 training corpus

Evidence level: **static/local**. All counts came from the ingest, split, and
render manifests in this directory plus direct file inspection.

## Why v4 exists

The v2 fine-tune regressed against its own base model: 0.00 code-fix pass
rate versus 0.75, and a 50% empty-response rate. Three defects caused it,
and a fourth capped the corpus at a fraction of what was available.

| Defect | v2 | v4 |
|---|---|---|
| Loss target | every token, incl. user/system | assistant completions only |
| Tool schemas in prompts | never rendered | 20,107 pairs |
| Short instructions | none | 240 primers |
| Corpus source | one Iceberg table, 8 KB cap | 1,719 stores + Codex rollouts |

## Source discovery

Session data is not centralized. Every project keeps its own
`.codetether-agent/sessions` store, so surveying only the global directory
undercounted the corpus by orders of magnitude.

| Measure | Value |
|---|---:|
| Transcript stores indexed | 1,719 |
| Transcript files | 45,206 |
| CodeTether session bytes | 6.97 GB |
| Codex rollout bytes | 3.33 GB |

## Ingested corpus

| Measure | Value |
|---|---:|
| Conversations | 27,217 |
| Messages | 1,485,449 |
| Assistant turns | 708,950 |
| Native tool calls | 722,858 |
| Duplicate conversations removed | 5,393 |
| Low-signal conversations skipped | 51,656 |

Sources: 26,574 CodeTether sessions, 643 Codex rollouts.

## Split and rendering

Whole conversations are assigned to one split by digest bucket, so no
conversation spans train and validation.

| Measure | Train | Validation |
|---|---:|---:|
| Conversations | 25,546 | 1,671 |
| Rendered pairs | 40,716 | 2,498 |
| Pairs with tool schema | 20,107 | 1,254 |
| Mean tokens | 2,785 | 2,770 |
| Total tokens | 113,051,545 | 6,939,285 |

Exact prompt+completion overlap after pruning: **0** (114 duplicates removed).

## Verified invariants

- 40,716 of 40,716 prompts end at `<|im_start|>assistant`.
- 40,716 of 40,716 completions end with `<|im_end|>`.
- Context is windowed to 12 recent turns; conversations contribute at most
  24 turns, which keeps rendering linear. Some conversations exceed 1,600
  messages, and unbounded prefix rendering was ~200x slower.

## Promotion gate

A v4 adapter may not become a default route until it beats the untuned base
on `scripts/model_training/bench_local.py` and `bench_empty.py`. Validation
loss alone is not evidence; v2 improved loss while regressing capability.