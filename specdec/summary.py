"""Summarise experiment 5 runs: speedup and acceptance by configuration and task."""

import statistics
from collections import defaultdict

TASKS = ("code", "maths", "chat")


def acceptance(records: list[dict]) -> dict:
    """Alpha two ways over the same evaluated positions (plan, section 6), plus related rates."""
    rounds = [r for rec in records for r in rec["rounds"]]
    with_draft = [r for r in rounds if r[0] > 0]  # copy-draft rounds with no match are excluded
    accepted = sum(r[1] for r in with_draft)
    evaluated = sum(len(r[2]) for r in with_draft)
    proposed = sum(r[0] for r in with_draft)
    sums = [s for r in with_draft for s in r[2]]
    tokens = sum(rec["new_tokens"] for rec in records)
    return {
        "rounds": len(rounds),
        "match_rate": len(with_draft) / len(rounds) if rounds else None,
        "alpha_counted": accepted / evaluated if evaluated else None,
        "alpha_sum_min": statistics.fmean(sums) if sums else None,
        "draft_efficiency": accepted / proposed if proposed else None,
        "tokens_per_round": tokens / len(rounds) if rounds else None,
    }


def summarise(records: list[dict]) -> dict:
    main = [r for r in records if r["repeat"] == 0]
    baseline_tokens = {r["prompt_id"]: r["tokens"] for r in main if r["draft"] is None}
    groups = defaultdict(list)
    for r in main:
        groups[(r["config"], r["task"])].append(r)
        groups[(r["config"], "all")].append(r)

    def rate(recs):
        return sum(r["new_tokens"] for r in recs) / sum(r["seconds"] for r in recs)

    table = {}
    for (config, task), recs in sorted(groups.items()):
        base = groups[("baseline", task)]
        base_by_prompt = {r["prompt_id"]: r for r in base}
        per_prompt = [r["tokens_per_s"] / base_by_prompt[r["prompt_id"]]["tokens_per_s"] for r in recs]
        entry = {
            "prompts": len(recs),
            "tokens_per_s": rate(recs),
            "speedup": rate(recs) / rate(base),
            "median_prompt_speedup": statistics.median(per_prompt),
            "differs_from_baseline": sum(r["tokens"] != baseline_tokens[r["prompt_id"]] for r in recs),
        }
        if recs[0]["draft"] is not None:
            entry.update(acceptance(recs))
        table.setdefault(config, {})[task] = entry

    # Session-to-session spread from the repeated subset: speedup per repeat, per config.
    repeats = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["repeat"] > 0:
            repeats[r["repeat"]][r["config"]].append(r)
    spread = {}
    for config in table:
        values = [rate(by_config[config]) / rate(by_config["baseline"])
                  for by_config in repeats.values() if by_config.get(config) and by_config.get("baseline")]
        if len(values) > 1:
            spread[config] = {"speedups": values, "stdev": statistics.stdev(values)}
    return {"by_config": table, "repeat_spread": spread}


def print_table(summary: dict) -> None:
    def num(value, width: int, fmt: str) -> str:
        return f"{'-':>{width}}" if value is None else f"{value:>{width}{fmt}}"

    columns = (*TASKS, "all")
    header = f"{'config':<11}" + "".join(f"{task:>17}" for task in columns) + f"{'alpha':>8}{'tok/round':>11}"
    print(header)
    print("-" * len(header))
    for config, per_task in summary["by_config"].items():
        cells = "".join(
            f"{num(per_task[t]['speedup'], 8, '.2f')}x {num(per_task[t]['tokens_per_s'], 6, '.1f')}/s"
            if t in per_task else f"{'':>17}" for t in columns)
        overall = per_task["all"]
        print(f"{config:<11}{cells}{num(overall.get('alpha_counted'), 8, '.2f')}"
              f"{num(overall.get('tokens_per_round'), 11, '.2f')}")
