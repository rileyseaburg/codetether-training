"""Generate from fixed prompts and measure agent-relevant behaviour.

Probes must supply a tool schema. Qwen's chat template only emits the
tool-call instruction block when `tools` is passed, so scoring bare prompts
reported `tool_call_rate: 0.0` at every checkpoint regardless of what the
model had learned. The rate was unobservable rather than zero.
"""

import torch

from .behaviour_prompts import PROMPTS
from .probe_tools import TOOL_SCHEMA


MAX_NEW_TOKENS = 128

CALL_MARKERS = ('<tool_call>', '<function=')
"""Qwen3.5 nests `<function=name>` inside `<tool_call>`."""


def score(model: object, tokenizer: object) -> dict[str, object]:
    """Return empty-response and tool-call rates for a small probe set."""
    empty = 0
    tool_calls = 0
    for prompt in PROMPTS:
        text = _generate(model, tokenizer, prompt)
        if not text.strip():
            empty += 1
        if any(marker in text for marker in CALL_MARKERS):
            tool_calls += 1
    total = len(PROMPTS)
    return {
        'probes': total,
        'empty_rate': round(empty / total, 4),
        'tool_call_rate': round(tool_calls / total, 4),
    }


def _generate(model: object, tokenizer: object, prompt: str) -> str:
    """Return the model's continuation for one chat prompt."""
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tools=TOOL_SCHEMA,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][inputs['input_ids'].shape[1] :]
    return str(tokenizer.decode(generated, skip_special_tokens=False))
