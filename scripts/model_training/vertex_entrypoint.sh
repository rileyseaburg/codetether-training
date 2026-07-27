#!/usr/bin/env bash
# Entrypoint for the Vertex AI custom training container.
#
# Vertex runs this in the foreground and owns the process lifecycle, so
# training must NOT detach here: exiting early would cancel the job.
# Checkpoints and logs stream to Cloud Storage because this project's
# service account cannot read Cloud Logging.
set -euo pipefail

workspace=/workspace
bundle=$workspace/ct
state=$workspace/codetether-state
bucket=${CODETETHER_GCS_BUCKET:?CODETETHER_GCS_BUCKET is required}

# The container command already cloned the repository before invoking this
# script, so cloning again would fail on a non-empty target directory.
test -d "$bundle/scripts/model_training"

export PYTHONPATH="$bundle/scripts"
export CODETETHER_STATE="$state"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$state/data" "$state/output" "$state/logs"

# Start mirroring before anything can fail, so early errors are recoverable.
bash "$bundle/scripts/model_training/gcs_mirror.sh" "$state" "$bucket" &

# Every stage appends here; the file is mirrored to Cloud Storage each minute.
log=$state/logs/train.log
exec > >(tee -a "$log") 2>&1

echo "=== stage: dependencies ==="
pip install -r "$bundle/scripts/model_training/requirements-gpu.txt" 2>&1 |
    tail -20
python3 -m model_training.version_report

echo "=== stage: dataset ==="
HF_TOKEN=$(python3 -m model_training.hf_export_token)
export HF_TOKEN
python3 -m model_training.hf_fetch \
    --repo "$CODETETHER_HF_REPO" --output "$state/data"

echo "=== stage: preflight ==="
python3 -m model_training.gpu_probe | tee "$state/logs/gpu-probe.json"
python3 -m model_training.mask_audit \
    --pairs "$state/data/train-pairs.jsonl" --sample 200 \
    --output "$state/logs/mask-audit.json"

echo "=== stage: training ==="
set +e
python3 -u -m model_training.train \
    --train "$state/data/train-pairs.jsonl" \
    --validation "$state/data/validation-pairs.jsonl" \
    --output "$state/output" \
    --epochs "${CODETETHER_EPOCHS:-1}" \
    --masked
status=$?

echo "=== training exited with status $status ==="
gsutil -q -m cp "$state/logs/"* \
    "gs://$bucket/model-training/qwen3-coder-v4/logs/" 2>/dev/null || true
gsutil -q -m rsync -r "$state/output" \
    "gs://$bucket/model-training/qwen3-coder-v4/output" 2>/dev/null || true
exit "$status"
