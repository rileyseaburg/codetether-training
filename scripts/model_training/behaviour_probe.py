"""Generate from fixed prompts and measure agent-relevant behaviour."""

import torch

from .behaviour_prompts import PROMPTS


MAX_NEW_TOKENS = 128


def score(model: object, tokenizer: object) -> dict[str, object]:
    """Return empty-response and tool-call rates for a small probe set."""
    empty = 0
    tool_calls = 0
    for prompt in PROMPTS:
        text = _generate(model, tokenizer, prompt)
        if not text.strip():
            empty += 1
        if '<tool_call>' in text:
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
