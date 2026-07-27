#!/usr/bin/env bash
# Submit the v4 QLoRA run as a Vertex AI custom training job.
#
# The project's service account holds roles/aiplatform.admin but lacks
# Compute Engine permissions, so Vertex custom training is the available
# path. Verified quota in us-central1:
#
#   restricted_image_training_nvidia_a100_80gb_gpus     limit 8
#   custom_model_training_preemptible_nvidia_a100_gpus  limit 8
#
# Vertex jobs are server-side: they survive local disconnects entirely.
set -euo pipefail

project=${CODETETHER_GCP_PROJECT:-$(gcloud config get-value project)}
region=${CODETETHER_GCP_REGION:-us-central1}
bucket=${CODETETHER_GCS_BUCKET:?CODETETHER_GCS_BUCKET is required}
repo=${CODETETHER_HF_REPO:?CODETETHER_HF_REPO is required}
image=${CODETETHER_TRAIN_IMAGE:?CODETETHER_TRAIN_IMAGE is required}
machine=${CODETETHER_MACHINE:-a2-ultragpu-1g}
accelerator=${CODETETHER_ACCELERATOR:-NVIDIA_A100_80GB}
name=${CODETETHER_JOB_NAME:-codetether-qwen3-v4-$(date +%Y%m%d-%H%M%S)}

: "${VAULT_ADDR:?VAULT_ADDR is required}"
: "${VAULT_TOKEN:?VAULT_TOKEN is required}"

config=$(mktemp)
trap 'rm -f "$config"' EXIT

python3 -m model_training.vertex_config \
    --image "$image" \
    --machine "$machine" \
    --accelerator "$accelerator" \
    --bucket "$bucket" \
    --hf-repo "$repo" \
    --output "$config"

gcloud ai custom-jobs create \
    --region="$region" \
    --project="$project" \
    --display-name="$name" \
    --config="$config"

echo "{\"job\": \"$name\", \"region\": \"$region\"}"