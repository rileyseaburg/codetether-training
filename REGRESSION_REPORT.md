# Local integration benchmark: tuned v2 vs untuned base

Evidence level: **static/local**, llama.cpp `b1-0a50d99`, CPU, 8 threads,
32,768-token context, `chatml` chat template, temperature 0.

## Result: the fine-tuned model regressed and must not become the default

| Metric | Untuned base | Tuned v2 |
|---|---:|---:|
| Code-fix pass rate | **0.75** | **0.00** |
| Empty-response rate | **0.00** | **0.50** |
| Tool-call emission rate | 0.00 | 0.00 |

Both models were served from the same binary and probed with identical
fixed cases in `scripts/model_training/bench_cases.py`.

## Confirmed defect 1: premature end-of-sequence

The tuned model returns an empty completion for short instructions. With a
raw ChatML prompt and no chat-template layer, it emitted EOS after a single
token for `const n: number = "5";`. The untuned base answered every probe.

This is a weight-level regression, not a serving artifact.

Probable cause: every training example was a long agent transcript rendered
with `add_generation_prompt=False`. Loss was computed over user and system
tokens as well as assistant tokens, so the model learned that short
standalone requests are unlikely continuations.

## Confirmed defect 2: tool schemas are echoed

Under the model's own Jinja template the tuned model reproduced the
`<tools>` JSON schema block as visible content instead of emitting an
OpenAI `tool_calls` field. Neither model produced a single valid tool call.

Cause: `scripts/model_training/render_file.py` calls
`apply_chat_template(...)` without `tools=`, so no training example ever
contained a rendered `<tools>` block, while inference always injects one.
Only 341 of 3,590 continuation rows (9.5%) contained any `<tool_call>` text.

## Why validation loss looked good

Validation loss fell from 3.4744 to 1.1585 because the model became better
at predicting CodeTether transcript text, including user turns and system
prompts. That objective does not measure instruction following, answer
correctness, or tool-call validity. Loss improvement alone is not evidence
of a usable agent model.

## Required fixes before v3

1. Mask loss to assistant and tool-call tokens only.
2. Render training text with `tools=` so `<tools>` blocks appear in training.
3. Render with `add_generation_prompt` semantics that match inference.
4. Add short single-turn instruction examples to prevent premature EOS.
5. Gate promotion on this benchmark, not on validation loss.

## Artifacts

- `integration-bench-chatml.json` — tuned, chatml template
- `integration-bench.json` — tuned, Jinja template, shows schema echo
- `empty-response-chatml.json` — tuned empty-response measurements
- `../../base-baseline/integration-bench-base.json` — untuned baseline
- `../../base-baseline/empty-response-base.json` — untuned baseline