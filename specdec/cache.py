"""A model plus its KV cache, kept in step with the token sequence.

The wrapper remembers which tokens its cache holds. On each call it keeps the longest
cached prefix that matches the new sequence, crops the rest, and runs only the new tokens.
That makes rollback automatic: rejected draft tokens are cropped on the next call, and a
token the cache never saw (the draft's last guess when every guess is accepted) is simply
fed in again. A cached entry depends only on the tokens before it, so any matching prefix
is valid.
"""

import torch

from specdec.models import real_logits


class CachedModel:
    def __init__(self, model, vocab_size: int, use_cache: bool = True):
        self.model = model
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.device = model.device
        self.cache = None
        self.cached_ids: list[int] = []

    def reset(self) -> None:
        self.cache = None
        self.cached_ids = []

    @torch.inference_mode()
    def logits(self, tokens: list[int], keep: int) -> torch.Tensor:
        """Float32 logits for the last `keep` positions of `tokens`: row i predicts the token
        after position len(tokens) - keep + i. Shape [keep, vocab_size]."""
        if not self.use_cache:
            ids = torch.tensor([tokens], device=self.device)
            return real_logits(self.model(input_ids=ids, logits_to_keep=keep).logits[0], self.vocab_size)

        # Reuse the longest matching prefix, but always feed at least `keep` tokens.
        limit = min(len(self.cached_ids), len(tokens) - keep)
        common = 0
        while common < limit and self.cached_ids[common] == tokens[common]:
            common += 1
        if common == 0:
            self.reset()
        elif common < len(self.cached_ids):
            # transformers >= 5.15: a negative argument removes that many tokens.
            # Never call crop(0) or crop(positive): their meaning changed between versions.
            self.cache.crop(-(len(self.cached_ids) - common))

        ids = torch.tensor([tokens[common:]], device=self.device)
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True, logits_to_keep=keep)
        self.cache = out.past_key_values
        self.cached_ids = list(tokens)
        if self.cache.get_seq_length() != len(self.cached_ids):
            raise AssertionError(f"cache holds {self.cache.get_seq_length()} tokens, expected {len(self.cached_ids)}")
        return real_logits(out.logits[0], self.vocab_size)
