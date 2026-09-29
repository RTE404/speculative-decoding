"""Experiment 4: the inputs to the paper's speedup formula.

- c for the neural draft: one draft step divided by one target step.
- c for the copy draft: one n-gram lookup divided by one target step.
- Verification cost: one target pass over b tokens divided by one target step.

Every measurement goes through CachedModel, the same code path the generation loop uses,
including the cache crop that rolls back the previous measurement's tokens.
"""

import statistics
import time

from specdec.cache import CachedModel
from specdec.drafts import PromptLookupDraft
from specdec.timing import timed

WARMUP = 5
REPEATS = 20
BLOCK_SIZES = (1, 2, 3, 4, 5, 7, 9, 16)  # gamma + 1 for every gamma in the sweep, plus 16


def step_time(lm: CachedModel, context: list[int], block: int) -> float:
    """Median seconds for one pass over `block` new tokens after `context`."""
    times: list[float] = []
    for i in range(WARMUP + REPEATS):
        # A different block each time, so every call crops the previous one: as in the real loop.
        start = (i * block) % max(1, len(context) - block)
        tokens = context + context[start:start + block]
        with timed(lm.device, times if i >= WARMUP else []):
            lm.logits(tokens, keep=block)
    return statistics.median(times)


def lookup_time(copy: PromptLookupDraft, context: list[int], gamma: int = 8) -> float:
    times = []
    for i in range(WARMUP + REPEATS):
        start = time.perf_counter()
        copy.find(context, gamma)
        if i >= WARMUP:
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def measure(target: CachedModel, draft: CachedModel, copy: PromptLookupDraft, context: list[int]) -> dict:
    """`context` should be a realistic sequence: a prompt plus some generated tokens."""
    target_step = step_time(target, context, 1)
    draft_step = step_time(draft, context, 1)
    verify = {b: step_time(target, context, b) for b in BLOCK_SIZES}
    lookup = lookup_time(copy, context)
    return {
        "context_tokens": len(context),
        "target_step_ms": 1000 * target_step,
        "draft_step_ms": 1000 * draft_step,
        "lookup_ms": 1000 * lookup,
        "c_neural": draft_step / target_step,
        "c_copy": lookup / target_step,
        "verify_ms": {b: 1000 * t for b, t in verify.items()},
        "verify_ratio": {b: t / verify[1] for b, t in verify.items()},
    }
