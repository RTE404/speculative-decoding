"""The speculative decoding loop: draft, verify, accept or reject."""

from dataclasses import dataclass, field

import torch

from specdec.cache import CachedModel
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


def generate(target: CachedModel, draft, prompt: list[int], *, max_new_tokens: int, gamma: int,
             temperature: float = 0.0, eos_ids: tuple[int, ...] = (),
             generator: torch.Generator | None = None) -> Output:
    result = Output(tokens=[])
    tokens = list(prompt)
    while len(result.tokens) < max_new_tokens:
        # Never draft past the length limit: the round's own target token needs one slot.
        room = max_new_tokens - len(result.tokens) - 1
        draft_tokens, q = draft.propose(tokens, min(gamma, room), temperature, generator)
        k = len(draft_tokens)

        # One target pass over the draft. Row i judges draft token i; row k gives the bonus token.
        p = to_probs(target.logits(tokens + draft_tokens, keep=k + 1), temperature)
        drafted = torch.tensor([draft_tokens], dtype=torch.long, device=p.device)
        n, next_token = verify(p.unsqueeze(0), q.unsqueeze(0), drafted, generator)
        n = n.item()
        evaluated = n + 1 if n < k else k
        sum_min = torch.minimum(p[:evaluated], q[:evaluated]).sum(-1).tolist()
        result.rounds.append(Round(proposed=k, accepted=n, sum_min=sum_min))

        # Stop at EOS or the length limit, even in the middle of an accepted block.
        for token in draft_tokens[:n] + [next_token.item()]:
            result.tokens.append(token)
            tokens.append(token)
            if token in eos_ids or len(result.tokens) >= max_new_tokens:
                return result
    return result
