"""Experiment 1: the acceptance rule on synthetic distributions.

Each test runs the rule on a batch of one million independent rounds and checks that the
emitted tokens follow the target distribution p exactly (chi-square test).
"""

import numpy as np
import pytest
import torch

from specdec.sampling import sample, to_probs, verify
from tests.stats import P_VALUE_FLOOR, chi_square_p

N = 1_000_000
V = 12


def random_dist(generator: torch.Generator, sharpness: float = 1.0) -> torch.Tensor:
    return torch.softmax(sharpness * torch.randn(V, generator=generator), dim=-1)


def run_rounds(p_rows: torch.Tensor, q_rows: torch.Tensor, seed: int):
    """Run N rounds with the same per-position p and q. Draft tokens are sampled from q."""
    g = torch.Generator().manual_seed(seed)
    k = q_rows.shape[0]
    p = p_rows.expand(N, k + 1, V)
    q = q_rows.expand(N, k, V)
    draft_tokens = sample(q, g)
    n, next_token = verify(p, q, draft_tokens, g)
    first = torch.where(n > 0, draft_tokens[:, 0], next_token)
    return draft_tokens, n, next_token, first


def counts(tokens: torch.Tensor) -> np.ndarray:
    return torch.bincount(tokens, minlength=V).numpy()


def case(name: str):
    g = torch.Generator().manual_seed(1)
    p0, p1, p2 = random_dist(g), random_dist(g), random_dist(g)
    if name == "random":
        q0 = random_dist(g)
    elif name == "one_hot":  # the copy draft
        q0 = torch.nn.functional.one_hot(torch.tensor(int(p0.argmin())), V).float()
    elif name == "nearly_equal":  # leftover max(0, p - q) is almost empty
        q0 = p0 * (1 + 1e-7 * torch.randn(V, generator=g)).abs()
        q0 = q0 / q0.sum()
    elif name == "sharp_draft":  # q much more confident than p
        q0 = random_dist(g, sharpness=6.0)
    elif name == "disjoint":  # q puts no mass where p does: nothing is ever accepted
        p0 = torch.zeros(V).index_fill_(0, torch.arange(V // 2), 1.0)
        p0 = p0 / p0.sum()
        q0 = torch.zeros(V).index_fill_(0, torch.arange(V // 2, V), 1.0)
        q0 = q0 / q0.sum()
    else:
        raise ValueError(name)
    q1 = random_dist(g)
    return torch.stack([p0, p1, p2]), torch.stack([q0, q1])


@pytest.mark.parametrize("name", ["random", "one_hot", "nearly_equal", "sharp_draft", "disjoint"])
def test_first_token_follows_p(name):
    p_rows, q_rows = case(name)
    _, _, _, first = run_rounds(p_rows, q_rows, seed=2)
    assert chi_square_p(counts(first), p_rows[0].numpy()) > P_VALUE_FLOOR


@pytest.mark.parametrize("name", ["random", "one_hot", "sharp_draft"])
def test_acceptance_rate_matches_sum_min(name):
    """The chance of accepting the first draft token is sum(min(p, q)) (Theorem 3.5)."""
    p_rows, q_rows = case(name)
    _, n, _, _ = run_rounds(p_rows, q_rows, seed=3)
    expected = torch.minimum(p_rows[0], q_rows[0]).sum().item()
    observed = (n > 0).float().mean().item()
    assert abs(observed - expected) < 4 * np.sqrt(expected * (1 - expected) / N) + 1e-6


def test_second_position_and_bonus_token():
    """Token 2 after an acceptance follows p1; the bonus token after accepting all follows p2."""
    p_rows, q_rows = case("random")
    draft_tokens, n, next_token, _ = run_rounds(p_rows, q_rows, seed=4)
    second = torch.where(n > 1, draft_tokens[:, 1], next_token)[n >= 1]
    assert chi_square_p(counts(second), p_rows[1].numpy()) > P_VALUE_FLOOR
    bonus = next_token[n == 2]
    assert len(bonus) > 10_000
    assert chi_square_p(counts(bonus), p_rows[2].numpy()) > P_VALUE_FLOOR


def test_no_draft_tokens():
    """With k = 0 (copy draft found no match) the rule simply samples from p."""
    p_rows, _ = case("random")
    g = torch.Generator().manual_seed(5)
    p = p_rows[:1].expand(N, 1, V)
    n, next_token = verify(p, torch.empty(N, 0, V), torch.empty(N, 0, dtype=torch.long), g)
    assert (n == 0).all()
    assert chi_square_p(counts(next_token), p_rows[0].numpy()) > P_VALUE_FLOOR


def test_greedy_is_argmax():
    """With temperature 0, a draft token is kept exactly when it is the target's argmax."""
    g = torch.Generator().manual_seed(6)
    target_logits = torch.randn(1000, 3, V, generator=g)
    draft_logits = torch.randn(1000, 2, V, generator=g)
    p, q = to_probs(target_logits, 0), to_probs(draft_logits, 0)
    draft_tokens = draft_logits.argmax(-1)
    n, next_token = verify(p, q, draft_tokens, g)
    target_argmax = target_logits.argmax(-1)
    matches = (draft_tokens == target_argmax[:, :2]).int().cumprod(-1).sum(-1)
    assert torch.equal(n, matches)
    assert torch.equal(next_token, target_argmax[torch.arange(1000), n])
