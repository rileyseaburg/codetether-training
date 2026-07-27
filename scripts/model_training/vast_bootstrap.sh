#!/usr/bin/env bash
# Provision a Vast.ai instance and run the v4 QLoRA training to completion.
#
# Colab recycles runtimes long before a 19 hour epoch finishes, so long runs
# belong on a persistent box. State lives outside the clone and training is
# detached, so an SSH disconnect cannot end the run.
set -euo pipefail

bundle=${CODETETHER_BUNDLE:-/workspace/ct}
state=${CODETETHER_STATE:-/workspace/codetether-state}
repo=${CODETETHER_HF_REPO:?CODETETHER_HF_REPO is required}

: "${VAULT_ADDR:?VAULT_ADDR is required}"
: "${VAULT_TOKEN:?VAULT_TOKEN is required}"

export CODETETHER_BUNDLE="$bundle"
export CODETETHER_STATE="$state"
export CODETETHER_HF_REPO="$repo"
export CODETETHER_MIRROR=${CODETETHER_MIRROR:-$state/mirror}

if [[ ! -d "$bundle/.git" ]]; then
    git clone --depth 1 \
        https://github.com/rileyseaburg/codetether-training "$bundle"
fi

command -v rsync >/dev/null 2>&1 || apt-get install -y -qq rsync

bash "$bundle/scripts/model_training/colab_bootstrap.sh"

cat <<INFO
Training is detached. Useful commands:

  tail -f $state/logs/train.log
  PYTHONPATH=$bundle/scripts python3 -m model_training.run_status

When the final adapter appears, package it:

  CODETETHER_BUNDLE=$bundle CODETETHER_STATE=$state \\
    bash $bundle/scripts/model_training/colab_package.sh
INFO