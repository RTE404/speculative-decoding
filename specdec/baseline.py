"""Normal token-by-token generation from the target model alone."""

import torch

from specdec.cache import CachedModel
from specdec.sampling import sample, to_probs


def generate(target: CachedModel, prompt: list[int], *, max_new_tokens: int, temperature: float = 0.0,
             eos_ids: tuple[int, ...] = (), generator: torch.Generator | None = None) -> list[int]:
    tokens = list(prompt)
    out: list[int] = []
    while len(out) < max_new_tokens:
        probs = to_probs(target.logits(tokens, keep=1)[0], temperature)
        token = sample(probs, generator).item()
        out.append(token)
        tokens.append(token)
        if token in eos_ids:
            break
    return out
