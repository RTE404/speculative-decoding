"""Experiment 5: tokens per second for the baseline and every draft setting, greedy.

Configurations are interleaved: for each prompt every configuration runs once, in an order
rotated by one position per prompt, so any drift in GPU speed affects all of them equally.
Each run is appended to runs.jsonl as soon as it finishes.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from specdec import baseline, speculative
from specdec.cache import CachedModel
from specdec.drafts import NeuralDraft, PromptLookupDraft
from specdec.models import EOS_IDS
from specdec.timing import gpu_state, timed

NEURAL_GAMMAS = (1, 2, 3, 4, 6)
COPY_GAMMAS = (2, 4, 8)


@dataclass(frozen=True)
class Config:
    draft: str | None  # None for the baseline, else "neural" or "copy"
    gamma: int = 0

    @property
    def name(self) -> str:
        return "baseline" if self.draft is None else f"{self.draft}-g{self.gamma}"


def all_configs() -> list[Config]:
    return ([Config(None)] + [Config("neural", g) for g in NEURAL_GAMMAS]
            + [Config("copy", g) for g in COPY_GAMMAS])


class Runner:
    def __init__(self, target: CachedModel, neural: NeuralDraft, copy: PromptLookupDraft):
        self.target = target
        self.drafts = {"neural": neural, "copy": copy}

    def run(self, config: Config, prompt_ids: list[int], max_new_tokens: int) -> dict:
        # Start from empty caches, so every run pays for reading the prompt. Otherwise the cache
        # would reuse the prompt from the previous configuration's run on the same prompt.
        self.target.reset()
        self.drafts["neural"].lm.reset()
        times: list[float] = []
        with timed(self.target.device, times):
            if config.draft is None:
                tokens, rounds = baseline.generate(self.target, prompt_ids, max_new_tokens=max_new_tokens,
                                                   eos_ids=EOS_IDS), []
            else:
                out = speculative.generate(self.target, self.drafts[config.draft], prompt_ids,
                                           max_new_tokens=max_new_tokens, gamma=config.gamma, eos_ids=EOS_IDS)
                tokens, rounds = out.tokens, out.rounds
        seconds = times[0]
        return {
            "config": config.name, "draft": config.draft, "gamma": config.gamma,
            "new_tokens": len(tokens), "seconds": seconds, "tokens_per_s": len(tokens) / seconds,
            "tokens": tokens,
            "rounds": [[r.proposed, r.accepted, r.sum_min] for r in rounds],
        }


def warm_up(runner: Runner, prompt_ids: list[int], configs: list[Config]) -> None:
    """Run every configuration once, untimed, so every input length has been seen."""
    for config in configs:
        runner.run(config, prompt_ids, max_new_tokens=24)


def sweep(runner: Runner, prompts: list[dict], configs: list[Config], max_new_tokens: int,
          out_path: Path, repeat: int = 0) -> list[dict]:
    records = []
    with out_path.open("a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts):
            shift = i % len(configs)
            for config in configs[shift:] + configs[:shift]:
                before = gpu_state()
                record = runner.run(config, prompt["ids"], max_new_tokens)
                record.update(prompt_id=prompt["source_id"], task=prompt["task"], repeat=repeat,
                              gpu_before=before, gpu_after=gpu_state())
                f.write(json.dumps(record) + "\n")
                f.flush()
                records.append(record)
                print(f"  {prompt['source_id']:>16}  {config.name:<10} {record['new_tokens']:>4} tokens "
                      f"{record['tokens_per_s']:6.1f} tok/s", flush=True)
    return records
