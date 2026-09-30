"""One command for the whole run (plan, section 8).

    python -m specdec.run           # full run, about 1 to 1.5 hours on a T4
    python -m specdec.run --quick   # smoke test: 1 prompt per task, 32 tokens, fast tests only

Steps: correctness tests (experiments 1 and 2), exact match on the real models (experiment 3),
cost measurements (experiment 4), the speedup sweep (experiment 5), and a summary table.
Everything goes into a new folder results/<run id>/; earlier runs are never overwritten.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from specdec import baseline, benchmark, costs, exactness, report, summary
from specdec.cache import CachedModel
from specdec.check_env import environment
from specdec.drafts import NeuralDraft, PromptLookupDraft
from specdec.models import (DRAFT_ID, EOS_IDS, REAL_VOCAB, TARGET_ID, encode_prompt, find_kaggle_model,
                            load_pair)

ROOT = Path(__file__).resolve().parent.parent
QUICK_TESTS = "not distribution_matches and not baseline_matches and not alpha_estimators"


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_tests(quick: bool) -> bool:
    command = [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"]
    if quick:
        command += ["-k", QUICK_TESTS]
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode == 0


def select_prompts(tokenizer, device, per_task: int) -> list[dict]:
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
    chosen = []
    for task in summary.TASKS:
        for prompt in [p for p in prompts if p["task"] == task][:per_task]:
            chosen.append({**prompt, "ids": encode_prompt(tokenizer, prompt["prompt"], device)[0].tolist()})
    return chosen


def check_mismatches(target_model, prompts: list[dict], records: list[dict]) -> list[dict]:
    """Teacher-forced check (experiment 3's) of every speculative output that differs from the
    baseline output for the same prompt and pass: is each token the target's top choice or a near-tie?"""
    ids = {p["source_id"]: p["ids"] for p in prompts}
    base = {(r["prompt_id"], r["repeat"]): r["tokens"] for r in records if r["draft"] is None}
    checks = []
    for r in records:
        reference = base[(r["prompt_id"], r["repeat"])]
        if r["draft"] is None or r["tokens"] == reference:
            continue
        prompt = ids[r["prompt_id"]]
        checks.append({
            "prompt_id": r["prompt_id"], "config": r["config"], "repeat": r["repeat"],
            "first_mismatch": exactness.first_mismatch(r["tokens"], reference),
            "speculative": exactness.teacher_forced_check(target_model, prompt, r["tokens"]),
            "baseline": exactness.teacher_forced_check(target_model, prompt, reference),
        })
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--prompts-per-task", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repeats", type=int, help="extra passes over one prompt per task, for the spread")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-exactness", action="store_true")
    parser.add_argument("--target", default=find_kaggle_model("3b-instruct") or TARGET_ID)
    parser.add_argument("--draft", default=find_kaggle_model("0.5b-instruct") or DRAFT_ID)
    args = parser.parse_args()
    per_task = args.prompts_per_task or (1 if args.quick else 10)
    max_new = args.max_new_tokens or (32 if args.quick else 128)
    repeats = args.repeats if args.repeats is not None else (1 if args.quick else 3)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ("-quick" if args.quick else "")
    run_dir = ROOT / "results" / run_id
    run_dir.mkdir(parents=True)
    settings = {"run_id": run_id, "quick": args.quick, "prompts_per_task": per_task,
                "max_new_tokens": max_new, "repeats": repeats, "target": args.target, "draft": args.draft,
                "git_commit": git_commit(), "environment": environment()}
    write_json(run_dir / "settings.json", settings)
    print(json.dumps(settings, indent=2), flush=True)

    print("\n== Experiments 1 and 2: correctness tests ==", flush=True)
    if not args.skip_tests and not run_tests(args.quick):
        sys.exit("correctness tests failed: stopping before any timing")

    if not args.skip_exactness:
        print("\n== Experiment 3: exact match on the real models ==", flush=True)
        report = exactness.run(args.target, args.draft, max_new)
        write_json(run_dir / "exactness.json", report)
        exactness.print_verdict(report)
        if not report["float32_strict_pass"]:
            sys.exit("float32 exact match failed: stopping before any timing")

    pair = load_pair(args.target, args.draft)
    if pair.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(pair.device)
    target = CachedModel(pair.target, REAL_VOCAB)
    neural = NeuralDraft(CachedModel(pair.draft, REAL_VOCAB))
    copy = PromptLookupDraft(REAL_VOCAB, device=pair.device)
    prompts = select_prompts(pair.tokenizer, pair.device, per_task)

    print("\n== Experiment 4: cost measurements ==", flush=True)
    first = prompts[0]["ids"]
    context = first + baseline.generate(target, first, max_new_tokens=64, eos_ids=())
    cost_report = costs.measure(target, neural.lm, copy, context)
    write_json(run_dir / "costs.json", cost_report)
    print(json.dumps(cost_report, indent=2), flush=True)

    print("\n== Experiment 5: speedup sweep ==", flush=True)
    runner = benchmark.Runner(target, neural, copy)
    configs = benchmark.all_configs()
    benchmark.warm_up(runner, prompts[0]["ids"], configs)
    runs_path = run_dir / "runs.jsonl"
    records = benchmark.sweep(runner, prompts, configs, max_new, runs_path)
    subset = [next(p for p in prompts if p["task"] == task) for task in summary.TASKS]
    for repeat in range(1, repeats + 1):
        print(f"-- repeat {repeat} of {repeats} (one prompt per task) --", flush=True)
        records += benchmark.sweep(runner, subset, configs, max_new, runs_path, repeat=repeat)

    print("\n== Checking every output that differs from the baseline ==", flush=True)
    mismatches = check_mismatches(pair.target, prompts, records)
    write_json(run_dir / "mismatches.json", mismatches)
    explained = sum(not m["speculative"]["failures"] and not m["baseline"]["failures"] for m in mismatches)
    print(f"{len(mismatches)} outputs differ from the baseline; {explained} are explained by near-ties", flush=True)

    result = summary.summarise(records)
    if pair.device.type == "cuda":
        result["peak_memory_gb"] = torch.cuda.max_memory_allocated(pair.device) / 1e9
    result["costs"] = cost_report
    write_json(run_dir / "summary.json", result)
    print("\n== Summary: speedup vs baseline (tokens/s) ==")
    summary.print_table(result)
    print(f"\nc (neural) = {cost_report['c_neural']:.3f}, c (copy) = {cost_report['c_copy']:.4f}")
    print(f"saved everything in results/{run_id}/")
    print(f"report written to {report.write(run_dir).relative_to(ROOT)}/README.md")


if __name__ == "__main__":
    main()
