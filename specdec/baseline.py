"""Normal token-by-token generation from the target model alone. No KV cache yet."""

import torch

from specdec.models import real_logits
from specdec.sampling import sample, to_probs


@torch.inference_mode()
def generate(model, prompt_ids: torch.Tensor, *, max_new_tokens: int, temperature: float = 0.0,
             eos_ids: tuple[int, ...] = (), vocab_size: int,
             generator: torch.Generator | None = None) -> list[int]:
    out: list[int] = []
    seq = prompt_ids
    while len(out) < max_new_tokens:
        logits = real_logits(model(input_ids=seq, logits_to_keep=1).logits[0, -1], vocab_size)
        token = sample(to_probs(logits, temperature), generator).item()
        out.append(token)
        if token in eos_ids:
            break
        seq = torch.cat([seq, seq.new_tensor([[token]])], dim=1)
    return out
