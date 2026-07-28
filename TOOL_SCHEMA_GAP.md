# Tool-schema supervision gap

Evidence level: **live hardware** for the baseline measurement, **static/local**
for the corpus analysis.

## Measured baseline on the deployment GPU

Untuned `Qwen/Qwen3.5-9B`, 4-bit NF4, on the RTX 2080 SUPER that will serve
the model:

```json
{"adapter": "none", "empty_rate": 0.0, "probes": 8, "tool_call_rate": 0.875}
```

The base model already emits a correctly formatted tool call for 7 of 8
probe prompts. It fits in 7520 MiB of 8192 MiB.

## Why that invalidates the current gate

The promotion gate required `tool_call_rate >= 0.10` against a recorded base
rate of 0.00. That 0.00 came from an earlier model and a probe that supplied
no tool schema. With the schema supplied, the untuned base scores 0.875, so
the threshold is met before any training occurs and cannot distinguish a
trained adapter from an untrained one.

## The supervision gap

Sampling 6,000 rendered pairs:

| Measurement | Count |
|---|---:|
| Prompts containing a tool call | 4,989 |
| Completions containing `<tool_call>` | 14 |
| Completions containing `<function=` | 0 |

Tool calls appear almost entirely in **prompts**, which are masked out of the
loss. Only the assistant's prose is supervised. The corpus therefore teaches
the model to talk about tool results, not to emit tool calls.

Tool names present in prompts, showing the schema the agent actually uses:

```text
bash 7205, read 3357, exec_command 1507, grep 1432, agent 1061,
edit 823, glob 791, list 617, batch 422
```

`example_function_name` appears 3,061 times because the chat template's
instruction block is counted; it is not a real tool.

## Consequence

Learning our tool schemas requires supervising turns where the assistant
**emits** a call. Those turns exist in the source corpus: ingestion recorded
722,858 tool calls across 27,217 conversations. Rendering dropped them from
completions, so the fix belongs in the renderer, not the trainer.

## What the gate must measure instead

Because the base model already formats calls correctly, the useful signals
are schema-specific rather than syntactic:

- calls naming a tool that exists in our schema, versus invented names
- required parameters present and correctly named
- the correct tool chosen for the request

A syntactic `tool_call_rate` cannot separate a trained model from the base.
