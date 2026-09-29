"""Experiment 2: the full speculative loop on tiny random models, checked exactly.

The vocabulary is small enough to list every possible 3-token continuation, so the exact
target distribution over continuations is known and the sampled one can be tested against it.
Most tests run both with and without the KV cache.
"""

import itertools
from collections import Counter

import numpy as np
import pytest
import torch

from specdec import baseline, speculative
from specdec.cache import CachedModel
from specdec.drafts import NeuralDraft, PromptLookupDraft
from tests.stats import P_VALUE_FLOOR, chi_square_p
from tests.tiny import tiny_model

V = 6
N_TOKENS = 3
PROMPT = [1, 2, 3, 1, 2, 4, 1, 2]  # repeats, so the copy draft finds matches


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


def setup(models, kind: str, use_cache: bool):
    target_model, draft_model = models
    target = CachedModel(target_model, V, use_cache=use_cache)
    draft = NeuralDraft(CachedModel(draft_model, V, use_cache=use_cache)) if kind == "neural" else PromptLookupDraft(V)
    return target, draft


@torch.inference_mode()
def exact_distribution(model, prompt: list[int], n_tokens: int) -> np.ndarray:
    """P(t1, ..., tn | prompt) for every continuation, by running the model on every prefix."""
    joint = np.ones([V] * n_tokens)
    for depth in range(n_tokens):
        prefixes = list(itertools.product(range(V), repeat=depth))
        ids = torch.tensor([prompt + list(prefix) for prefix in prefixes])
        probs = torch.softmax(model(input_ids=ids).logits[:, -1].float(), dim=-1).numpy()
        shape = [V] * depth + [V] + [1] * (n_tokens - depth - 1)
        joint = joint * probs.reshape(shape)
    return joint


def sampled_counts(outputs: list[list[int]]) -> np.ndarray:
    counts = np.zeros([V] * N_TOKENS)
    for tokens, count in Counter(tuple(t) for t in outputs).items():
        counts[tokens] = count
    return counts


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("kind,gamma", [("neural", 2), ("neural", 4), ("copy", 3)])
def test_distribution_matches_target(models, kind, gamma, use_cache):
    target, draft = setup(models, kind, use_cache)
    g = torch.Generator().manual_seed(20)
    outputs, rounds = [], []
    for _ in range(4000):
        out = speculative.generate(target, draft, PROMPT, max_new_tokens=N_TOKENS, gamma=gamma,
                                   temperature=1.0, generator=g)
        outputs.append(out.tokens)
        rounds += out.rounds

    assert all(len(t) == N_TOKENS for t in outputs)
    exact = exact_distribution(models[0], PROMPT, N_TOKENS)
    assert chi_square_p(sampled_counts(outputs), exact) > P_VALUE_FLOOR

    # Coverage: the test is only meaningful if both branches of the rule actually ran.
    with_draft = [r for r in rounds if r.proposed > 0]
    rejected = sum(r.accepted < r.proposed for r in with_draft) / len(with_draft)
    all_accepted = sum(r.accepted == r.proposed for r in with_draft)
    assert rejected >= 0.3, f"only {rejected:.0%} of rounds hit the rejection branch"
    assert all_accepted >= 50, f"only {all_accepted} rounds accepted every draft token (bonus-token path)"


@pytest.mark.parametrize("use_cache", [False, True])
def test_baseline_matches_target(models, use_cache):
    """The reference sampler itself follows the exact distribution."""
    target = CachedModel(models[0], V, use_cache=use_cache)
    g = torch.Generator().manual_seed(21)
    outputs = [baseline.generate(target, PROMPT, max_new_tokens=N_TOKENS, temperature=1.0, generator=g)
               for _ in range(4000)]
    assert chi_square_p(sampled_counts(outputs), exact_distribution(models[0], PROMPT, N_TOKENS)) > P_VALUE_FLOOR


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("kind", ["neural", "copy"])
@pytest.mark.parametrize("gamma", [1, 3, 5])
def test_greedy_exact_match(models, kind, gamma, use_cache):
    """Speculative greedy output equals plain greedy output from the target without a cache."""
    target, draft = setup(models, kind, use_cache)
    reference = CachedModel(models[0], V, use_cache=False)
    g = torch.Generator().manual_seed(22)
    for trial in range(5):
        prompt = torch.randint(0, V, (6,), generator=g).tolist()
        expected = baseline.generate(reference, prompt, max_new_tokens=40)
        got = speculative.generate(target, draft, prompt, max_new_tokens=40, gamma=gamma).tokens
        assert got == expected, f"trial {trial}"


@pytest.mark.parametrize("kind", ["neural", "copy"])
def test_stops_at_eos_and_length(models, kind):
    """No token after EOS, never more than max_new_tokens, and the same stop point as the baseline."""
    target, draft = setup(models, kind, use_cache=True)
    reference = CachedModel(models[0], V, use_cache=False)
    g = torch.Generator().manual_seed(23)
    stopped_early = 0
    for eos in range(V):
        for _ in range(10):
            prompt = torch.randint(0, V, (6,), generator=g).tolist()
            expected = baseline.generate(reference, prompt, max_new_tokens=25, eos_ids=(eos,))
            got = speculative.generate(target, draft, prompt, max_new_tokens=25, gamma=4, eos_ids=(eos,)).tokens
            assert got == expected
            assert len(got) <= 25
            assert eos not in got[:-1]
            stopped_early += len(got) < 25
    assert stopped_early > 0, "EOS never came up, so the test checked nothing"


def test_alpha_estimators_agree(models):
    """Counted acceptance (accepted / evaluated) and mean sum(min(p, q)) estimate the same alpha."""
    target, draft = setup(models, "neural", use_cache=True)
    g = torch.Generator().manual_seed(24)
    rounds = []
    for _ in range(300):
        rounds += speculative.generate(target, draft, PROMPT, max_new_tokens=20, gamma=4, temperature=1.0,
                                       generator=g).rounds
    accepted = sum(r.accepted for r in rounds)
    evaluated = sum(r.evaluated for r in rounds)
    counted = accepted / evaluated
    from_dists = float(np.mean([s for r in rounds for s in r.sum_min]))
    assert abs(counted - from_dists) < 4 * np.sqrt(counted * (1 - counted) / evaluated)


# --- KV cache ---------------------------------------------------------------

def test_cached_logits_match_fresh_forward(models):
    """After any mix of appends and rollbacks, cached logits equal a fresh forward pass."""
    model = models[0]
    cached = CachedModel(model, V)
    fresh = CachedModel(model, V, use_cache=False)
    g = torch.Generator().manual_seed(30)
    tokens = torch.randint(0, V, (5,), generator=g).tolist()
    for step in range(300):
        action = torch.randint(0, 4, (1,), generator=g).item()
        if action == 0 and len(tokens) > 3:  # roll back up to 4 tokens
            tokens = tokens[:len(tokens) - torch.randint(1, 5, (1,), generator=g).item()]
        elif action == 1 and len(tokens) > 3:  # rewrite the tail, like a rejected draft
            cut = torch.randint(1, 4, (1,), generator=g).item()
            tokens = tokens[:-cut] + torch.randint(0, V, (cut,), generator=g).tolist()
        else:  # append a block, like a verify pass
            tokens += torch.randint(0, V, (torch.randint(1, 6, (1,), generator=g).item(),), generator=g).tolist()
        tokens = tokens[-100:] if len(tokens) > 100 else tokens
        keep = torch.randint(1, min(len(tokens), 6) + 1, (1,), generator=g).item()
        got, expected = cached.logits(tokens, keep), fresh.logits(tokens, keep)
        assert torch.allclose(got, expected, atol=1e-5), f"step {step}: max diff {(got - expected).abs().max()}"
        assert cached.cache.get_seq_length() == len(tokens)


@pytest.mark.parametrize("gamma", [2, 4])
def test_cache_does_not_change_draft_proposals(models, gamma):
    """The draft's proposals, and so every round's acceptance count, are identical with and
    without the cache. A draft cache one token behind would change them (it lowers alpha
    without biasing the output, so the distribution test cannot see it)."""
    g = torch.Generator().manual_seed(31)
    for _ in range(10):
        prompt = torch.randint(0, V, (6,), generator=g).tolist()
        runs = []
        for use_cache in (False, True):
            target, draft = setup(models, "neural", use_cache)
            runs.append(speculative.generate(target, draft, prompt, max_new_tokens=40, gamma=gamma))
        assert runs[0].tokens == runs[1].tokens
        assert [(r.proposed, r.accepted) for r in runs[0].rounds] == [(r.proposed, r.accepted) for r in runs[1].rounds]
        assert any(r.accepted == r.proposed for r in runs[1].rounds), "no all-accepted round, so the owed-token case was not tested"
