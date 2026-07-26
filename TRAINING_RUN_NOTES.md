# v4 training run observations

Evidence level: **live GPU run** on Colab A100 80 GB, plus static/local
tokenizer analysis of the rendered splits.

## Run reached training

```text
22%|██▏  | 388/1795 [02:31<08:32, 2.75it/s]
```

1,795 total optimizer steps at gradient accumulation 16 implies 28,720
surviving train examples, not the 40,716 rendered.

## Two OOM causes, both fixed before this run

| Cause | Evidence | Fix |
|---|---|---|
| `prepare_model_for_kbit_training` casts every non-quantized parameter to float32, with no flag to disable | 78.38 GiB of 79.25 GiB on an 80 GB A100 | `kbit_prepare.py` freezes weights and enables checkpointing without the upcast |
| LoRA targeted `gate_proj`, `up_proj`, `down_proj`, which exist once per expert | 18,432 expert tensors would yield ~1.2 B trainable parameters instead of ~12.6 M | `TARGET_MODULES` restricted to attention projections |

An A100 40 GB failed the same way earlier at 39.47 GiB, so the cause was
the upcast rather than insufficient hardware.

## Unresolved: 29.5 percent of train examples dropped

TRL reported `Dropping fully masked examples` for both splits:

| Split | Rendered | Surviving | Dropped |
|---|---:|---:|---:|
| train | 40,716 | 28,720 | 11,996 (29.5%) |
| validation | 2,505 | 1,795 | 710 (28.3%) |

### Hypotheses tested and rejected

Measured with the pinned tokenizer against the rendered files:

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Prompts exceed the 8,192 limit | 0 of 2,505 validation and 0 of 4,000 sampled train prompts exceed it; p50 2,342, p90 6,139, max 8,103 | rejected |
| Prompt plus completion exceeds the limit | max combined length is 8,175 tokens; 0 of 2,505 pairs exceed 8,192 | rejected |
| Completions are empty | 0 validation pairs have an empty completion | rejected |

The renderer already enforced a combined-length budget, so truncation is
not the cause. The mechanism inside TRL that masks these examples is not
yet identified.

### Next diagnostic

Instrument TRL's label-building stage directly and inspect a dropped
example's `labels` tensor, rather than inferring the cause from token
counts. The near-equal drop rate across both splits still suggests a
deterministic rule rather than data corruption.

This does not invalidate the current run: the surviving 28,720 examples are
still 8x the entire v2 corpus, and the drop is unbiased with respect to
content.

## Deprecation cleared after this run started

`warmup_ratio` is removed in transformers v5.2 and was replaced with
explicit `warmup_steps=60`, approximating the prior 3 percent ratio across
roughly 2,544 steps. The live run predates that change and is unaffected.