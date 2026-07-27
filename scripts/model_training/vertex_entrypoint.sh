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

# The container command already cloned the repository before invoking this
# script, so cloning again would fail on a non-empty target directory.
test -d "$bundle/scripts/model_training"

export PYTHONPATH="$bundle/scripts"
export CODETETHER_STATE="$state"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "$state/data" "$state/output" "$state/logs"

# Record dependency resolution: a conflict here is invisible otherwise
# because Cloud Logging is not readable by this service account.
pip install -r "$bundle/scripts/model_training/requirements-gpu.txt" \
    2>&1 | tail -30 | tee "$state/logs/pip.log"
python3 -c 'import torch, transformers, trl, peft, bitsandbytes as bnb; \
print({"torch": torch.__version__, "transformers": transformers.__version__, \
"trl": trl.__version__, "peft": peft.__version__, "bnb": bnb.__version__})' \
    2>&1 | tee "$state/logs/versions.log"

HF_TOKEN=$(python3 -m model_training.hf_export_token)
export HF_TOKEN

python3 -m model_training.hf_fetch \
    --repo "$CODETETHER_HF_REPO" --output "$state/data"
python3 -m model_training.gpu_probe | tee "$state/logs/gpu-probe.json"
python3 -m model_training.mask_audit \
    --pairs "$state/data/train-pairs.jsonl" --sample 200 \
    --output "$state/logs/mask-audit.json"

bash "$bundle/scripts/model_training/gcs_mirror.sh" "$state" "$bucket" &

# Tee training output to a file as well as stdout. Cloud Logging access is
# denied to this project's service account, so the mirrored log file is the
# only usable diagnostic when a run fails.
set +e
python3 -u -m model_training.train \
    --train "$state/data/train-pairs.jsonl" \
    --validation "$state/data/validation-pairs.jsonl" \
    --output "$state/output" \
    --epochs "${CODETETHER_EPOCHS:-1}" \
    --masked 2>&1 | tee -a "$state/logs/train.log"
status=${PIPESTATUS[0]}

gsutil -q cp "$state/logs/train.log" \
    "gs://$bucket/model-training/qwen3-coder-v4/logs/" 2>/dev/null || true
exit "$status"