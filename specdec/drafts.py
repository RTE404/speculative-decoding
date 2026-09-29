"""Drafts propose up to gamma tokens and the distributions q they were drawn from.

The speculative loop does not know which draft it is using. Both drafts return
(tokens [k], q [k, V]) with k <= gamma; k may be 0.
"""

import torch

from specdec.models import real_logits
from specdec.sampling import sample, to_probs


class NeuralDraft:
    """A small language model that samples gamma tokens one at a time. No KV cache yet."""

    def __init__(self, model, vocab_size: int):
        self.model = model
        self.vocab_size = vocab_size

    @torch.inference_mode()
    def propose(self, seq: torch.Tensor, gamma: int, temperature: float,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, dists = [], []
        for _ in range(gamma):
            ids = torch.cat([seq, torch.stack(tokens).view(1, -1)], dim=1) if tokens else seq
            logits = real_logits(self.model(input_ids=ids, logits_to_keep=1).logits[0, -1], self.vocab_size)
            q = to_probs(logits, temperature)
            tokens.append(sample(q, generator))
            dists.append(q)
        if not tokens:
            return seq.new_empty(0), torch.empty(0, self.vocab_size, device=seq.device)
        return torch.stack(tokens), torch.stack(dists)


class PromptLookupDraft:
    """Copies tokens from earlier in the sequence (prompt lookup decoding). Costs no model call.

    Takes the last n tokens, finds the most recent earlier place they appeared, and proposes
    the tokens that followed. Tries n = max_ngram down to 1. Its q is one-hot on each proposal.
    """

    def __init__(self, vocab_size: int, max_ngram: int = 3):
        self.vocab_size = vocab_size
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

    def propose(self, seq: torch.Tensor, gamma: int, temperature: float,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        found = self.find(seq[0].tolist(), gamma) if gamma > 0 else []
        tokens = torch.tensor(found, dtype=torch.long, device=seq.device)
        q = torch.nn.functional.one_hot(tokens, self.vocab_size).float()
        return tokens, q
