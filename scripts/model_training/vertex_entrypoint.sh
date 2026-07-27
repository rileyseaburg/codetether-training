#!/usr/bin/env bash
# Entrypoint for the Vertex AI custom training container.
#
# Vertex runs this in the foreground and owns the process lifecycle, so
# training must NOT detach here: exiting early would cancel the job.
# Checkpoints stream to Cloud Storage so a preemption is resumable.
set -euo pipefail

workspace=/workspace
bundle=$workspace/ct
state=$workspace/codetether-state
bucket=${CODETETHER_GCS_BUCKET:?CODETETHER_GCS_BUCKET is required}
origin=${CODETETHER_GIT_REPO:-https://github.com/rileyseaburg/codetether-training}

if [[ ! -d "$bundle/.git" ]]; then
    git clone --depth 1 "$origin" "$bundle"
fi

export PYTHONPATH="$bundle/scripts"
export CODETETHER_STATE="$state"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$state/data" "$state/output" "$state/logs"

pip install --quiet -r "$bundle/scripts/model_training/requirements-gpu.txt"

HF_TOKEN=$(python3 -m model_training.hf_export_token)
export HF_TOKEN

python3 -m model_training.hf_fetch \
    --repo "$CODETETHER_HF_REPO" --output "$state/data"
python3 -m model_training.gpu_probe | tee "$state/logs/gpu-probe.json"
python3 -m model_training.mask_audit \
    --pairs "$state/data/train-pairs.jsonl" --sample 200 \
    --output "$state/logs/mask-audit.json"

bash "$bundle/scripts/model_training/gcs_mirror.sh" "$state" "$bucket" &

exec python3 -m model_training.train \
    --train "$state/data/train-pairs.jsonl" \
    --validation "$state/data/validation-pairs.jsonl" \
    --output "$state/output" \
    --epochs "${CODETETHER_EPOCHS:-1}" \
    --masked