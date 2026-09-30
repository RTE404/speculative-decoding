"""Experiment 6: predicted speedup from the paper's theory against measured speedup.

Times are in units of one target step (one target pass over one new token), so the
baseline costs 1 per token. Three predictions, each adding one measured ingredient,
show where any gap between theory and measurement comes from:

A. Paper formula (Theorem 3.8): mean alpha, fixed gamma, cost ratio c, and a verification
   pass assumed to cost the same as one target step.
B. Cost model: the same equal-acceptance assumption, but with the measured verification
   cost for k+1 tokens and the measured number of draft tokens proposed in each round
   (the copy draft proposes fewer than gamma, or none, when it finds no match).
C. Round by round: the cost model of B, but with the tokens each round actually produced
   instead of the equal-acceptance expectation.

A to B is the effect of the paper's cost simplifications, B to C the effect of the
equal-acceptance assumption, and C to measured is everything the cost model leaves out
(reading the prompt, Python and synchronisation overheads).
"""

import numpy as np


def expected_tokens(alpha: float, k: int) -> float:
    """Tokens produced by a round that proposes k draft tokens (Equation 1)."""
    if alpha >= 1:
        return k + 1
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def paper_speedup(alpha: float, gamma: int, c: float) -> float:
    """Theorem 3.8."""
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def verify_cost(verify_ratio: dict, tokens: int) -> float:
    """Measured cost of one target pass over `tokens` tokens, interpolated between measured sizes."""
    points = sorted((int(size), ratio) for size, ratio in verify_ratio.items())  # JSON keys are strings
    return float(np.interp(tokens, [size for size, _ in points], [ratio for _, ratio in points]))


def round_cost(draft: str, k: int, costs: dict) -> float:
    """Cost of one round in target steps: drafting plus one verification pass over k+1 tokens."""
    drafting = k * costs["c_neural"] if draft == "neural" else costs["c_copy"]
    return drafting + verify_cost(costs["verify_ratio"], k + 1)


def predictions(records: list[dict], draft: str, gamma: int, costs: dict) -> dict:
    """A, B and C for one configuration on one set of runs (see module docstring)."""
    rounds = [r for rec in records for r in rec["rounds"]]
    with_draft = [r for r in rounds if r[0] > 0]
    accepted = sum(r[1] for r in with_draft)
    evaluated = sum(len(r[2]) for r in with_draft)
    alpha = accepted / evaluated if evaluated else 0.0
    c = costs["c_neural"] if draft == "neural" else costs["c_copy"]

    total_cost = sum(round_cost(draft, r[0], costs) for r in rounds)
    expected = sum(expected_tokens(alpha, r[0]) for r in rounds)
    produced = sum(rec["new_tokens"] for rec in records)
    return {
        "alpha": alpha,
        "A_paper": paper_speedup(alpha, gamma, c),
        "B_cost_model": expected / total_cost,
        "C_round_by_round": produced / total_cost,
    }
