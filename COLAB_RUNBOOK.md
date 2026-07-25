# Colab v4 training runbook

Pulls data from HuggingFace, installs CodeTether, and reads credentials from
public Vault at runtime. No secret is stored in this repository or notebook.

## Runtime selection

H100 requires Pro+ or paid compute units. **Choose A100 GPU.** A measured
budget shows the 30B QLoRA needs about 20.1 GB at 4096 tokens, so a 40 GB
A100 has roughly 2x headroom. `gpu_probe` adapts automatically.

| Colab runtime | VRAM | Trains | Sequence | Estimated | Verdict |
|---|---:|---|---:|---:|---|
| H100 | 80 GB | Qwen3-Coder-30B-A3B | 8192 | 22.0 GB | Best, often locked |
| A100 | 40 GB | Qwen3-Coder-30B-A3B | 4096 | 20.1 GB | **Use this** |
| L4 | 24 GB | Qwen3-Coder-30B-A3B | 4096 | 20.1 GB | Works, slower |
| T4 | 16 GB | Qwen3-4B-Instruct-2507 | 2048 | 4.6 GB | Proving run only |

T4 lacks bfloat16, so the trainer selects float16 automatically.

## Credentials

The HuggingFace token is stored in Vault and verified working:

```text
secret/codetether/providers/huggingface   field: token
field must contain a token with write access
```

Store one with:

```bash
vault kv put secret/codetether/providers/huggingface token=hf_xxx
```

## Published dataset

Already uploaded and round-trip verified (download SHA-256 matches source):

```text
owner/codetether-agent-traces-v4  (private dataset)
  train-pairs.jsonl        426,293,333 bytes   40,716 pairs
  validation-pairs.jsonl    26,282,817 bytes    2,498 pairs
  render-v3-manifest.json        1,196 bytes
```

## 1. Configure the Colab session

```python
import os
os.environ['VAULT_ADDR'] = 'https://vault.example.com'
os.environ['VAULT_TOKEN'] = ''  # paste at runtime; never commit
os.environ['CODETETHER_HF_REPO'] = 'owner/codetether-agent-traces-v4'
```

## 2. Fetch the scripts

```bash
!git clone --depth 1 https://github.com/owner/codetether-training /content/ct
```

## 3. Start training

```bash
!cd /content/ct && CODETETHER_BUNDLE=/content/ct \
  bash scripts/model_training/colab_bootstrap.sh
```

The bootstrap installs GPU dependencies, downloads the splits from
HuggingFace, installs the CodeTether CLI, probes Vault, writes
`logs/gpu-probe.json`, then trains detached with assistant-only loss
masking. Detaching matters: the v1 run stalled when a PTY reader vanished.

## 4. Watch progress

```bash
!tail -20 /content/ct/logs/train.log
```

## 5. Survive disconnects

Checkpoints land every 250 steps in `output/`. Mirror them to Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!mkdir -p /content/drive/MyDrive/codetether-v4 \
  && rsync -a /content/ct/output/ /content/drive/MyDrive/codetether-v4/
```

Resume after a disconnect:

```bash
!cd /content/ct && CODETETHER_BUNDLE=/content/ct \
  CODETETHER_RESUME=/content/ct/output/checkpoint-LATEST \
  bash scripts/model_training/colab_bootstrap.sh
```

## 6. Publish the adapter

```bash
!cd /content/ct && PYTHONPATH=scripts python3 -m model_training.hf_publish \
  --repo owner/codetether-qwen3-coder-v4 \
  --files output/final-adapter/adapter_model.safetensors \
          output/final-adapter/adapter_config.json \
  --output output/hf-adapter.json
```

## Promotion gate

A v4 adapter must beat the untuned base on `bench_local.py` and
`bench_empty.py` before becoming a default route. The v2 model improved
validation loss while regressing capability, so loss is not the gate.