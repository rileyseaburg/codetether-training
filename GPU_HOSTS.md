# GPU host options for the v4 run

Colab recycles runtimes long before a multi-hour epoch finishes, so the
full run belongs on a rented persistent box.

Prices below are per GPU-hour as published by CloudMart on 22 June 2026.
Verify current rates before renting; GPU pricing moves quickly.

## A100 80 GB — the value tier

An A100 80 GB delivers roughly 60–70 percent of H100 training throughput at
about half the price, and the measured plan needs 45.4 GB of 80 GB.

| Provider | On-demand /hr | Est. run cost |
|---|---:|---:|
| Paperspace | $1.15 | $14–17 |
| RunPod (SXM) | $1.64 | $20–25 |
| Lambda Labs | $1.99 | $24–30 |
| Crusoe | $2.42 | $29–36 |
| CoreWeave | $2.70 | $32–41 |

## H100 80 GB — roughly 1.5x faster

| Provider | On-demand /hr | Spot /hr |
|---|---:|---:|
| RunPod | $2.69 | $1.99 |
| Lambda Labs | $2.99 | — |
| Crusoe | $3.90 | $1.60 |
| CoreWeave | $6.16 | — |
| Google Cloud | $6.98 | $2.09 |

An H100 finishes in roughly 8–10 hours instead of 12–15, so RunPod spot at
$1.99 costs about the same as a Paperspace A100 while returning results
sooner.

## Recommendation

**Paperspace A100 80 GB** for the lowest total cost, or **RunPod** when
per-second billing and pre-built PyTorch templates matter more than the
hourly rate.

Avoid spot or interruptible instances unless checkpoint mirroring is
verified first: checkpoints land every 500 steps, and a reclaim between
checkpoints loses that interval.

## Requirements

| Resource | Minimum | Reason |
|---|---:|---|
| VRAM | 80 GB | 45.4 GB measured; 40 GB hit CUDA OOM |
| Disk | 200 GB | 61 GB base weights, checkpoints, merged export |
| RAM | 80 GB | merging dequantizes on CPU, needs 67 GB |

## Launch

```bash
export VAULT_ADDR=https://vault.example.com
export VAULT_TOKEN=...
export CODETETHER_HF_REPO=owner/codetether-agent-traces-v4
export CODETETHER_WORKSPACE=/workspace

git clone --depth 1 \
  https://github.com/rileyseaburg/codetether-training /workspace/ct
bash /workspace/ct/scripts/model_training/rent_bootstrap.sh
```

The launcher clones the repo, installs dependencies, fetches the dataset
from HuggingFace, probes the GPU, guards disk, then trains detached so an
SSH disconnect cannot end the run.

## Monitor

```bash
tail -f /workspace/codetether-state/logs/train.log
PYTHONPATH=/workspace/ct/scripts python3 -m model_training.run_status
```

## Expected timing

Measured on an A100 80 GB at 8,192 tokens: 27.32 s/it across 2,545 steps.

| Item | Value |
|---|---:|
| Training steps | 2,545 |
| Observed rate | 27.32 s/it |
| Raw training | 19.3 h |
| After eval and padding fixes | 12–15 h |

Evaluation previously consumed 3.2 hours of that total; capping the
validation subset at 400 examples and widening the interval to 500 steps
reduces it to about 0.25 hours. Sequences are also grouped by length, since
a mean prompt of 2,712 tokens padded to 8,192 wasted about two thirds of
each batch.
