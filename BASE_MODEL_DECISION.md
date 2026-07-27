# Base model selection

Evidence level: **static/local**. Sizes come from the HuggingFace API
(`safetensors.total` and blob sizes); the VRAM figures are arithmetic on
those counts, not measured inference.

## The deciding constraint

The trained model is served on an **NVIDIA RTX 2080 Super, 8 GB VRAM**.
Deployment, not training capacity, therefore bounds model size. A model that
fine-tunes comfortably on a rented A100 is worthless if it cannot be served.

Estimated at Q4_K_M with an 8,192 token KV cache:

| candidate | params | weights | total | fits 8 GB |
|---|---:|---:|---:|---|
| Qwen3.6-27B | 27.8B | 15.3 GB | 16.3 GB | no |
| Qwen3-Coder-30B-A3B | 30.5B | 16.8 GB | 17.8 GB | no |
| **Qwen3.5-9B** | **9.7B** | **5.3 GB** | **7.0 GB** | **yes** |
| Qwen3.5-4B | 4.7B | 2.6 GB | 4.3 GB | yes |

`Qwen3.5-9B` is the largest candidate that serves on the target card with a
useful context window, so it is the base model.

## Rejected candidates

**Qwen3.6-27B and Qwen3.6-35B-A3B.** Newer and stronger, but 15.3 GB and
19.8 GB at Q4_K_M respectively. Neither serves on 8 GB.

**microsoft/Fara1.5-27B and Fara1.5-9B.** Built on the same `qwen3_5`
architecture, but the model card describes a multimodal browser computer-use
agent that observes screenshots and emits browser actions
(`pipeline_tag: image-text-to-text`). Its post-training targets a different
task than a text coding corpus.

**moonshotai/Kimi-K3.** 2.78 trillion parameters, 1.56 TB across 96 shards,
already FP8-quantized. Holding the weights alone needs about 19.5 devices of
80 GB, against a quota of 8. Unusable directly; viable only as an API
teacher for distillation.

## Toolchain consequence

`Qwen3.5-9B` declares `model_type: qwen3_5`, which is **absent** from
transformers 4.57.1, 4.58.0, 4.60.0, and 5.0.0, and present only from
5.14.1. Verified directly against `configuration_auto.py` in each release.

transformers 5.x pairs with `peft` 0.19.1, which imports `DTensor` from
`torch.distributed.tensor` and therefore needs torch 2.5 or newer. Vertex
AI's newest prebuilt training image ships torch 2.4, so a custom image is
required: `Dockerfile.trainer` builds on torch 2.6 and fails the build if
the architecture is not importable.

## Turing note

The target card is Turing (sm_75) and has no bfloat16 support, so served
inference runs in float16. Training on Ampere or newer still uses bfloat16;
the quantized GGUF export is dtype-independent.