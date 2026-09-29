"""The speculative decoding loop: draft, verify, accept or reject. No KV cache yet."""

from dataclasses import dataclass, field

import torch

from specdec.models import real_logits
from specdec.sampling import to_probs, verify


@dataclass
class Round:
    proposed: int         # draft tokens proposed (k)
    accepted: int         # draft tokens accepted (n)
    sum_min: list[float]  # sum over the vocabulary of min(p, q), at each evaluated position

    @property
    def evaluated(self) -> int:
        """Draft positions the target actually judged: the accepted ones plus the first rejection."""
        return len(self.sum_min)


@dataclass
class Output:
    tokens: list[int]
    rounds: list[Round] = field(default_factory=list)


@torch.inference_mode()
def generate(target, draft, prompt_ids: torch.Tensor, *, max_new_tokens: int, gamma: int,
             temperature: float = 0.0, eos_ids: tuple[int, ...] = (), vocab_size: int,
             generator: torch.Generator | None = None) -> Output:
    result = Output(tokens=[])
    seq = prompt_ids
    while len(result.tokens) < max_new_tokens:
        # Never draft past the length limit: the round's own target token needs one slot.
        room = max_new_tokens - len(result.tokens) - 1
        draft_tokens, q = draft.propose(seq, min(gamma, room), temperature, generator)
        k = len(draft_tokens)

        # One target pass over the draft. Output row t is the distribution for token t+1,
        # so the last k+1 rows judge the k draft tokens and give the bonus token.
        ids = torch.cat([seq, draft_tokens.view(1, -1)], dim=1)
        logits = real_logits(target(input_ids=ids, logits_to_keep=k + 1).logits[0], vocab_size)
        p = to_probs(logits, temperature)

        n, next_token = verify(p.unsqueeze(0), q.unsqueeze(0), draft_tokens.unsqueeze(0), generator)
        n = n.item()
        evaluated = n + 1 if n < k else k
        sum_min = torch.minimum(p[:evaluated], q[:evaluated]).sum(-1).tolist()
        result.rounds.append(Round(proposed=k, accepted=n, sum_min=sum_min))

        # Stop at EOS or the length limit, even in the middle of an accepted block.
        for token in draft_tokens[:n].tolist() + [next_token.item()]:
            result.tokens.append(token)
            if token in eos_ids or len(result.tokens) >= max_new_tokens:
                return result
        seq = torch.cat([prompt_ids, prompt_ids.new_tensor([result.tokens])], dim=1)
    return result
