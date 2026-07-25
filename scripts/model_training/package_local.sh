#!/usr/bin/env bash
set -euo pipefail

adapter=${1:?adapter directory is required}
output=${2:?output directory is required}
venv=${3:?merge virtualenv is required}
llama=${4:?llama.cpp directory is required}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

mkdir -p "$output"
export PYTHONPATH="$root/scripts"
"$venv/bin/python" -m model_training.merge \
    --adapter "$adapter" \
    --output "$output/merged"

"$venv/bin/python" "$llama/convert_hf_to_gguf.py" \
    "$output/merged" \
    --outfile "$output/codetether-qwen25-coder-15b-f16.gguf" \
    --outtype f16 \
    --model-name CodeTether-Qwen2.5-Coder-1.5B-v1

"$llama/build/bin/llama-quantize" \
    "$output/codetether-qwen25-coder-15b-f16.gguf" \
    "$output/codetether-qwen25-coder-15b-q4_k_m.gguf" \
    Q4_K_M

sha256sum \
    "$output/codetether-qwen25-coder-15b-f16.gguf" \
    "$output/codetether-qwen25-coder-15b-q4_k_m.gguf"