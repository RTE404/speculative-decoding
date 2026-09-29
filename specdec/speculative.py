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
    # Per round: (k, n, sum of min(p, q) at every draft position). The sums stay on the GPU
    # until the end, so collecting them adds no synchronisation to the timed loop.
    pending: list[tuple[int, int, torch.Tensor]] = []

    def finish() -> Output:
        for k, n, sums in pending:
            evaluated = n + 1 if n < k else k
            result.rounds.append(Round(proposed=k, accepted=n, sum_min=sums[:evaluated].tolist()))
        return result

    while len(result.tokens) < max_new_tokens:
        # Never draft past the length limit: the round's own target token needs one slot.
        room = max_new_tokens - len(result.tokens) - 1
        draft_tokens, q = draft.propose(tokens, min(gamma, room), temperature, generator)
        k = len(draft_tokens)

        # One target pass over the draft. Row i judges draft token i; row k gives the bonus token.
        p = to_probs(target.logits(tokens + draft_tokens, keep=k + 1), temperature)
        q = q.to(p.device)  # the draft may live on another GPU
        drafted = torch.tensor([draft_tokens], dtype=torch.long, device=p.device)
        n, next_token = verify(p.unsqueeze(0), q.unsqueeze(0), drafted, generator)
        n, next_token = torch.cat([n, next_token]).tolist()  # the round's one synchronisation
        pending.append((k, n, torch.minimum(p[:k], q).sum(-1)))

        # Stop at EOS or the length limit, even in the middle of an accepted block.
        for token in draft_tokens[:n] + [next_token]:
            result.tokens.append(token)
            tokens.append(token)
            if token in eos_ids or len(result.tokens) >= max_new_tokens:
                return finish()
    return finish()
