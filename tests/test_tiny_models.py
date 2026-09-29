"""Experiment 2: the full speculative loop on tiny random models, checked exactly.

The vocabulary is small enough to list every possible 3-token continuation, so the exact
target distribution over continuations is known and the sampled one can be tested against it.
"""

import itertools
from collections import Counter

import numpy as np
import pytest
import torch

from specdec import baseline, speculative
from specdec.drafts import NeuralDraft, PromptLookupDraft
from tests.stats import P_VALUE_FLOOR, chi_square_p
from tests.tiny import tiny_model

V = 6
N_TOKENS = 3
PROMPT = torch.tensor([[1, 2, 3, 1, 2, 4, 1, 2]])  # repeats, so the copy draft finds matches


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Tiny models run several times faster on one CPU thread than on many."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def models():
    return tiny_model(seed=10, vocab_size=V), tiny_model(seed=11, vocab_size=V)


@torch.inference_mode()
def exact_distribution(model, prompt: torch.Tensor, n_tokens: int, temperature: float = 1.0) -> np.ndarray:
    """P(t1, ..., tn | prompt) for every continuation, by running the model on every prefix."""
    joint = np.ones([V] * n_tokens)
    for depth in range(n_tokens):
        prefixes = list(itertools.product(range(V), repeat=depth))
        ids = torch.cat([prompt.expand(len(prefixes), -1),
                         torch.tensor(prefixes, dtype=torch.long).view(len(prefixes), depth)], dim=1)
        probs = torch.softmax(model(input_ids=ids).logits[:, -1].float() / temperature, dim=-1).numpy()
        shape = [V] * depth + [V] + [1] * (n_tokens - depth - 1)
        joint = joint * probs.reshape(shape)
    return joint


def sampled_counts(outputs: list[list[int]]) -> np.ndarray:
    counts = np.zeros([V] * N_TOKENS)
    for tokens, count in Counter(tuple(t) for t in outputs).items():
        counts[tokens] = count
    return counts


def make_draft(kind: str, draft_model):
    return NeuralDraft(draft_model, V) if kind == "neural" else PromptLookupDraft(V)


@pytest.mark.parametrize("kind,gamma", [("neural", 2), ("neural", 4), ("copy", 3)])
def test_distribution_matches_target(models, kind, gamma):
    target, draft_model = models
    draft = make_draft(kind, draft_model)
    g = torch.Generator().manual_seed(20)
    outputs, rounds = [], []
    for _ in range(4000):
        out = speculative.generate(target, draft, PROMPT, max_new_tokens=N_TOKENS, gamma=gamma,
                                   temperature=1.0, vocab_size=V, generator=g)
        outputs.append(out.tokens)
        rounds += out.rounds

    assert all(len(t) == N_TOKENS for t in outputs)
    exact = exact_distribution(target, PROMPT, N_TOKENS)
    assert chi_square_p(sampled_counts(outputs), exact) > P_VALUE_FLOOR

    # Coverage: the test is only meaningful if both branches of the rule actually ran.
    with_draft = [r for r in rounds if r.proposed > 0]
    rejected = sum(r.accepted < r.proposed for r in with_draft) / len(with_draft)
    all_accepted = sum(r.accepted == r.proposed for r in with_draft)
    assert rejected >= 0.3, f"only {rejected:.0%} of rounds hit the rejection branch"
    assert all_accepted >= 50, f"only {all_accepted} rounds accepted every draft token (bonus-token path)"


def test_baseline_matches_target(models):
    """The reference sampler itself follows the exact distribution."""
    target, _ = models
    g = torch.Generator().manual_seed(21)
    outputs = [baseline.generate(target, PROMPT, max_new_tokens=N_TOKENS, temperature=1.0, vocab_size=V,
                                 generator=g) for _ in range(4000)]
    assert chi_square_p(sampled_counts(outputs), exact_distribution(target, PROMPT, N_TOKENS)) > P_VALUE_FLOOR


@pytest.mark.parametrize("kind", ["neural", "copy"])
@pytest.mark.parametrize("gamma", [1, 3, 5])
def test_greedy_exact_match(models, kind, gamma):
    target, draft_model = models
    draft = make_draft(kind, draft_model)
    g = torch.Generator().manual_seed(22)
    for trial in range(5):
        prompt = torch.randint(0, V, (1, 6), generator=g)
        expected = baseline.generate(target, prompt, max_new_tokens=40, vocab_size=V)
        got = speculative.generate(target, draft, prompt, max_new_tokens=40, gamma=gamma, vocab_size=V).tokens
        assert got == expected, f"trial {trial}"


@pytest.mark.parametrize("kind", ["neural", "copy"])
def test_stops_at_eos_and_length(models, kind):
    """No token after EOS, never more than max_new_tokens, and the same stop point as the baseline."""
    target, draft_model = models
    draft = make_draft(kind, draft_model)
    g = torch.Generator().manual_seed(23)
    stopped_early = 0
    for eos in range(V):
        for trial in range(10):
            prompt = torch.randint(0, V, (1, 6), generator=g)
            expected = baseline.generate(target, prompt, max_new_tokens=25, eos_ids=(eos,), vocab_size=V)
            got = speculative.generate(target, draft, prompt, max_new_tokens=25, gamma=4, eos_ids=(eos,),
                                       vocab_size=V).tokens
            assert got == expected
            assert len(got) <= 25
            assert eos not in got[:-1]
            stopped_early += len(got) < 25
    assert stopped_early > 0, "EOS never came up, so the test checked nothing"


def test_alpha_estimators_agree(models):
    """Counted acceptance (accepted / evaluated) and mean sum(min(p, q)) estimate the same alpha."""
    target, draft_model = models
    draft = NeuralDraft(draft_model, V)
    g = torch.Generator().manual_seed(24)
    rounds = []
    for _ in range(300):
        rounds += speculative.generate(target, draft, PROMPT, max_new_tokens=20, gamma=4, temperature=1.0,
                                       vocab_size=V, generator=g).rounds
    accepted = sum(r.accepted for r in rounds)
    evaluated = sum(r.evaluated for r in rounds)
    counted = accepted / evaluated
    from_dists = float(np.mean([s for r in rounds for s in r.sum_min]))
    assert abs(counted - from_dists) < 4 * np.sqrt(counted * (1 - counted) / evaluated)
