#!/usr/bin/env bash
# Bootstrap the v4 QLoRA run on a Colab GPU runtime.
#
# Pulls the dataset from HuggingFace, installs CodeTether, loads provider
# credentials from public Vault, then trains with assistant-only loss
# masking. Requires VAULT_ADDR and VAULT_TOKEN in the environment; no
# credential is ever written into this script or the notebook.
set -euo pipefail

repo=${CODETETHER_HF_REPO:?CODETETHER_HF_REPO is required}
bundle=${CODETETHER_BUNDLE:-/content/ct}
output=${CODETETHER_OUTPUT:-$bundle/output}
log=${CODETETHER_LOG:-$bundle/logs/train.log}
resume=${CODETETHER_RESUME:-}

: "${VAULT_ADDR:?VAULT_ADDR is required}"
: "${VAULT_TOKEN:?VAULT_TOKEN is required}"

mkdir -p "$bundle/data" "$output" "$(dirname "$log")"

python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet huggingface_hub
python3 -m pip install --quiet -r \
    "$bundle/scripts/model_training/requirements-gpu.txt"

export PYTHONPATH="$bundle/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python3 -m model_training.hf_fetch \
    --repo "$repo" \
    --output "$bundle/data"

python3 -m model_training.codetether_setup \
    --output "$bundle/codetether-setup.json"

python3 -m model_training.gpu_probe | tee "$bundle/logs/gpu-probe.json"

args=(
    --train "$bundle/data/train-pairs.jsonl"
    --validation "$bundle/data/validation-pairs.jsonl"
    --output "$output"
    --epochs "${CODETETHER_EPOCHS:-1}"
    --masked
)
if [[ -n "$resume" ]]; then
    args+=(--resume "$resume")
fi

setsid nohup python3 -m model_training.train "${args[@]}" \
    </dev/null >>"$log" 2>&1 &

echo "{\"pid\": $!, \"log\": \"$log\", \"output\": \"$output\"}"
