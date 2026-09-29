"""The speculative sampling acceptance rule (Leviathan et al., Algorithm 1), in float32.

Greedy decoding is handled as the special case where p and q are one-hot at their argmax:
the same rule then accepts a draft token exactly when it equals the target's argmax, and
otherwise emits the target's argmax. So one code path covers greedy and sampling.
"""

import torch

# Below this, the leftover distribution max(0, p - q) is treated as empty. Its total is
# 1 - beta, so it only vanishes when the draft and target distributions are almost equal.
RESIDUAL_EPS = 1e-9


def to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Turn logits into a float32 distribution over the last dimension. Temperature 0 means greedy."""
    logits = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits / temperature, dim=-1)


def sample(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one token from each distribution in the last dimension."""
    flat = probs.reshape(-1, probs.shape[-1])
    return torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])


def verify(p: torch.Tensor, q: torch.Tensor, draft_tokens: torch.Tensor,
           generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Decide how many draft tokens to keep, and sample the token that follows them.

    p:            [B, k+1, V] target distributions. Row i is the target's distribution for the
                  token at draft position i; row k is for the bonus token after all k drafts.
    q:            [B, k, V] draft distributions the draft tokens were sampled from.
    draft_tokens: [B, k] the proposed tokens. k may be 0 (a copy draft that found no match).

    Returns (n, next_token): n [B] draft tokens accepted, and next_token [B], sampled from
    norm(max(0, p_n - q_n)) after a rejection, or from p_k when all k were accepted.
    """
    batch, k, _ = q.shape
    idx = draft_tokens.unsqueeze(-1)
    p_x = p[:, :k].gather(-1, idx).squeeze(-1)
    q_x = q.gather(-1, idx).squeeze(-1)

    # Accept token i with probability min(1, p/q). Comparing r*q < p avoids dividing by q.
    r = torch.rand(batch, k, generator=generator, device=p.device)
    accepted = r * q_x < p_x
    n = accepted.int().cumprod(dim=-1).sum(dim=-1)

    # Distribution for the next token. Padding q with a zero row makes the all-accepted
    # case fall out naturally: max(0, p_k - 0) = p_k.
    rows = torch.arange(batch, device=p.device)
    q_padded = torch.cat([q, q.new_zeros(batch, 1, q.shape[-1])], dim=1)
    p_n, q_n = p[rows, n], q_padded[rows, n]
    residual = (p_n - q_n).clamp_min(0)
    total = residual.sum(dim=-1, keepdim=True)
    degenerate = (total <= RESIDUAL_EPS) | ~torch.isfinite(total)
    dist = torch.where(degenerate, p_n, residual / total.clamp_min(RESIDUAL_EPS))
    return n, sample(dist, generator)
