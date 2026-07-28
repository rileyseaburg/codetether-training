# Checkpoint-300 evaluation on the deployment GPU

Evidence level: **live hardware**. Both runs executed on the RTX 2080 SUPER
in Proxmox VM 141, `Qwen/Qwen3.5-9B` in 4-bit NF4, greedy decoding, eight
probe prompts with the agent tool schema supplied.

## Result

| Model | `tool_call_rate` | `empty_rate` |
|---|---:|---:|
| Untuned base | **0.875** | 0.0 |
| `checkpoint-300` adapter | **0.125** | 0.0 |

Training made tool calling **7× worse**, from 7 of 8 probes down to 1 of 8.

## Why

`checkpoint-300` was trained on the v4 pairs. Sampling 6,000 of those pairs:

| Measurement | Count |
|---|---:|
| Prompts containing a tool call | 4,989 |
| Completions containing a real `<function=` call | **0** |

Loss applied only to assistant prose. The model was therefore trained, on
40,716 examples, to respond to tool-rich contexts with text instead of
tool calls. It learned exactly what the data taught, and that suppressed a
capability the base model already had.

Descending loss confirmed this was real learning, not divergence:

```text
eval_loss  1.5960 → 1.5103 → 1.4682
```

Loss measured fluency on prose completions. It could not observe that the
completions omitted every tool call.

## Fit on the target card

The 9B model loaded and generated within budget:

```text
7520 MiB of 8192 MiB   (4-bit NF4, 8,192-token context)
```

Weight loading initially ran at 93 s/shard because the VM had 7 GB of RAM and
2 GB of exhausted swap. Adding 32 GB of swap reduced this to about 2 s/shard,
a 40× improvement, and full load completed in 17 minutes.

## Corrections applied

The v5 render supervises emitted calls. Sampling 4,000 v5 pairs:

| Measurement | v4 | v5 |
|---|---:|---:|
| Completions with a real tool call | 0.0% | **84.0%** |

Supervised tool names now match the agent's own schema:

```text
bash 1477, exec_command 1182, read 400, write_stdin 232,
grep 204, apply_patch 150, browserctl 121, write 90
```

The gate also needed replacing. A syntactic `tool_call_rate` cannot separate
a trained model from a base model that already scores 0.875, so scoring now
reports `known_tool_rate`, `invented_tool_rate`, and `valid_params_rate`
against the real schema.
