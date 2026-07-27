"""Immutable base-model and training hyperparameters.

Qwen3-Coder-30B-A3B is a mixture-of-experts model that activates about 3B
of 30B parameters per token, so it fits constrained GPUs far better than a
dense model of comparable quality while supporting agentic tool use.
"""

BASE_MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
BASE_REVISION = 'b2cff646eb4bb1d68355c01b18ae02e7cf42d120'
SMALL_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
SMALL_REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
MAX_LENGTH = 8192
WARMUP_STEPS = 60
"""Explicit warmup steps.

`warmup_ratio` is deprecated in transformers and removed in v5.2. With
40,716 examples at an effective batch of 16, one epoch is roughly 2,500
optimizer steps, so 60 steps approximates the previous 3 percent ratio.
"""
ACCUMULATION_STEPS = 16
EVAL_STEPS = 500
"""Evaluation interval in optimizer steps.

A full validation pass over 2,505 examples took 19 minutes on an A100
80 GB. At a 250 step interval that is 3.2 hours of a 19 hour run spent
evaluating, so the interval is widened and the subset bounded below.
"""
MAX_EVAL_SAMPLES = 400
"""Cap validation examples per evaluation pass.

A 400 example subset estimates loss closely enough to select checkpoints
while cutting each pass from about 19 minutes to about 3.
"""
SEED = 42
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
]
"""Attention projections only.

Qwen3-Coder-30B-A3B is a mixture of experts whose `gate_proj`, `up_proj`,
and `down_proj` modules exist once per expert: 18,432 tensors in total.
Targeting them multiplies adapter parameters by the expert count and
exhausts an 80 GB device. Attention layers are shared across experts.
"""
