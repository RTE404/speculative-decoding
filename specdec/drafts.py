"""Drafts propose up to gamma tokens and the distributions q they were drawn from.

The speculative loop does not know which draft it is using. Both drafts return
(tokens, q) with len(tokens) = k <= gamma and q of shape [k, V]; k may be 0.
"""

import torch

from specdec.cache import CachedModel
from specdec.sampling import sample, to_probs


class NeuralDraft:
    """A small language model that samples gamma tokens one at a time."""

    def __init__(self, lm: CachedModel):
        self.lm = lm
        self.vocab_size = lm.vocab_size

    def propose(self, tokens: list[int], gamma: int, temperature: float,
                generator: torch.Generator | None = None) -> tuple[list[int], torch.Tensor]:
        context = list(tokens)
        proposed, dists = [], []
        for _ in range(gamma):
            q = to_probs(self.lm.logits(context, keep=1)[0], temperature)
            token = sample(q, generator).item()
            proposed.append(token)
            dists.append(q)
            context.append(token)
        if not proposed:
            return [], torch.empty(0, self.vocab_size, device=self.lm.device)
        return proposed, torch.stack(dists)


class PromptLookupDraft:
    """Copies tokens from earlier in the sequence (prompt lookup decoding). Costs no model call.

    Takes the last n tokens, finds the most recent earlier place they appeared, and proposes
    the tokens that followed. Tries n = max_ngram down to 1. Its q is one-hot on each proposal.
    """

    def __init__(self, vocab_size: int, device: torch.device | str = "cpu", max_ngram: int = 3):
        self.vocab_size = vocab_size
        self.device = device
        self.max_ngram = max_ngram

    def find(self, tokens: list[int], gamma: int) -> list[int]:
        for n in range(self.max_ngram, 0, -1):
            if len(tokens) <= n:
                continue
            pattern = tokens[-n:]
            # Start positions whose match ends before the final token, most recent first.
            for start in range(len(tokens) - n - 1, -1, -1):
                if tokens[start:start + n] == pattern:
                    return tokens[start + n:start + n + gamma]
        return []

    def propose(self, tokens: list[int], gamma: int, temperature: float,
                generator: torch.Generator | None = None) -> tuple[list[int], torch.Tensor]:
        proposed = self.find(tokens, gamma) if gamma > 0 else []
        # Build q directly on the GPU: a one-hot row is 0.6 MB, too much to copy over every round.
        ids = torch.tensor(proposed, dtype=torch.long, device=self.device)
        return proposed, torch.nn.functional.one_hot(ids, self.vocab_size).float()
