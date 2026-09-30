"""Build report/<run id>/README.md and its charts from a run folder.

    python -m specdec.report results/<run id>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from specdec import theory  # noqa: E402
from specdec.benchmark import COPY_GAMMAS, NEURAL_GAMMAS  # noqa: E402
from specdec.summary import TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Reference palette (dataviz skill), light mode; validated for two series, all pairs.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8985"
SERIES = {"neural": {"color": "#2a78d6", "marker": "o", "label": "0.5B neural draft"},
          "copy": {"color": "#eb6834", "marker": "s", "label": "copy draft (prompt lookup)"}}
GAMMAS = {"neural": NEURAL_GAMMAS, "copy": COPY_GAMMAS}
COLUMNS = (*TASKS, "all")


def load(run_dir: Path) -> dict:
    def read(name):
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    runs = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return {"settings": read("settings.json"), "costs": read("costs.json"), "summary": read("summary.json"),
            "exactness": read("exactness.json"), "mismatches": read("mismatches.json"), "runs": runs}


def theory_rows(data: dict) -> list[dict]:
    rows = []
    measured = data["summary"]["by_config"]
    for draft, gammas in GAMMAS.items():
        for gamma in gammas:
            config = f"{draft}-g{gamma}"
            for task in COLUMNS:
                records = [r for r in data["runs"] if r["config"] == config and r["repeat"] == 0
                           and (task == "all" or r["task"] == task)]
                rows.append({"config": config, "draft": draft, "gamma": gamma, "task": task,
                             **theory.predictions(records, draft, gamma, data["costs"]),
                             "measured": measured[config][task]["speedup"]})
    return rows


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def figure(ncols: int, width: float, height: float):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE, squeeze=False)
    for ax in axes[0]:
        style(ax)
    return fig, axes[0]


def chart_speedup_vs_gamma(data: dict, path: Path) -> None:
    table = data["summary"]["by_config"]
    fig, axes = figure(len(COLUMNS), 12, 3.6)
    for ax, task in zip(axes, COLUMNS):
        ax.axhline(1, color=REFERENCE, linewidth=1.2, linestyle="--")
        for draft, gammas in GAMMAS.items():
            s = SERIES[draft]
            ax.plot(gammas, [table[f"{draft}-g{g}"][task]["speedup"] for g in gammas], color=s["color"],
                    marker=s["marker"], markersize=6, linewidth=2, label=s["label"])
        ax.set_title(task if task != "all" else "all tasks", color=INK, fontsize=11)
        ax.set_xlabel("draft length γ", color=INK_SECONDARY, fontsize=9)
        ax.set_xticks(sorted(set(NEURAL_GAMMAS) | set(COPY_GAMMAS)))
    axes[0].set_ylabel("speedup vs target alone", color=INK_SECONDARY, fontsize=9)
    top = max(1.2, max(table[c][t]["speedup"] for c in table for t in COLUMNS) * 1.08)
    for ax in axes:
        ax.set_ylim(0, top)
    axes[0].text(8, 0.96, "target alone", color=INK_SECONDARY, fontsize=8, ha="right", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_theory(rows: list[dict], path: Path) -> None:
    rows = [r for r in rows if r["task"] != "all"]
    fig, axes = figure(2, 10, 4.6)
    top = max(max(r["measured"], r["A_paper"], r["C_round_by_round"]) for r in rows) * 1.05
    for ax, key, title in ((axes[0], "A_paper", "A. Paper formula (Theorem 3.8)"),
                           (axes[1], "C_round_by_round", "C. Cost model, round by round")):
        ax.plot([0, top], [0, top], color=REFERENCE, linewidth=1.2, linestyle="--", label="prediction = measurement")
        for draft, s in SERIES.items():
            points = [r for r in rows if r["draft"] == draft]
            ax.scatter([r[key] for r in points], [r["measured"] for r in points], color=s["color"],
                       marker=s["marker"], s=40, edgecolors=SURFACE, linewidths=1.5, label=s["label"], zorder=3)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("predicted speedup", color=INK_SECONDARY, fontsize=9)
    axes[0].set_ylabel("measured speedup", color=INK_SECONDARY, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_costs(costs: dict, path: Path) -> None:
    points = sorted((int(b), v) for b, v in costs["verify_ratio"].items())
    fig, axes = figure(1, 6, 3.6)
    ax = axes[0]
    ax.axhline(1, color=REFERENCE, linewidth=1.2, linestyle="--")
    ax.plot([b for b, _ in points], [v for _, v in points], color=SERIES["neural"]["color"], marker="o",
            markersize=6, linewidth=2)
    ax.axhline(costs["c_neural"], color=REFERENCE, linewidth=1.2, linestyle=":")
    ax.text(16, 1.02, "one target step", color=INK_SECONDARY, fontsize=8, ha="right", va="bottom")
    ax.text(16, costs["c_neural"] + 0.02, f"one 0.5B draft step (c = {costs['c_neural']:.2f})",
            color=INK_SECONDARY, fontsize=8, ha="right", va="bottom")
    ax.set_ylim(0, 1.2)
    ax.set_xticks([b for b, _ in points])
    ax.set_xlabel("tokens checked in one target pass", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("time, in target steps", color=INK_SECONDARY, fontsize=9)
    ax.set_title("Cost of checking several tokens in one target pass", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fmt(value, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def markdown(data: dict, rows: list[dict]) -> str:
    s, costs, table = data["settings"], data["costs"], data["summary"]["by_config"]
    env = s["environment"]
    configs = ["baseline"] + [f"{d}-g{g}" for d, gs in GAMMAS.items() for g in gs]
    best = {d: max((f"{d}-g{g}" for g in gs), key=lambda c: table[c]["all"]["speedup"]) for d, gs in GAMMAS.items()}
    runs = data["runs"]
    temps = [r[k]["temperature_c"] for r in runs for k in ("gpu_before", "gpu_after") if r.get(k)]
    clocks = [r[k]["sm_clock_mhz"] for r in runs for k in ("gpu_before", "gpu_after") if r.get(k)]
    out = [f"# Results: run {s['run_id']}", ""]
    out += [f"Qwen2.5-3B-Instruct target, greedy decoding, {s['prompts_per_task']} prompts per task, "
            f"up to {s['max_new_tokens']} new tokens. {', '.join(env['gpus'])}; torch {env['torch']}, "
            f"transformers {env['transformers']}, CUDA {env['cuda']}. Code at commit `{(s['git_commit'] or '?')[:7]}`.",
            ""]
    out += ["## Headline", ""]
    for draft, config in best.items():
        per_task = ", ".join(f"{t} {table[config][t]['speedup']:.2f}×" for t in TASKS)
        out.append(f"- **{SERIES[draft]['label']}**: best setting {config}, **{table[config]['all']['speedup']:.2f}×** "
                   f"overall ({per_task}).")
    out.append(f"- Cost ratio c = {costs['c_neural']:.3f} for the 0.5B draft and {costs['c_copy']:.4f} for the copy "
               f"draft. Checking 2 to 16 tokens in one pass costs {min(costs['verify_ratio'].values()):.2f} to "
               f"{max(v for b, v in costs['verify_ratio'].items() if int(b) > 1):.2f} of one target step.")
    out += ["", "![Speedup vs draft length](speedup_vs_gamma.png)", ""]

    out += ["## Speedup by task (experiment 5)", "",
            "Tokens per second relative to the target alone, total tokens over total time per task.", "",
            "| Setting | " + " | ".join(COLUMNS) + " | tokens/s (all) |", "|---" * (len(COLUMNS) + 2) + "|"]
    for c in configs:
        out.append(f"| {c} | " + " | ".join(f"{table[c][t]['speedup']:.2f}×" for t in COLUMNS)
                   + f" | {table[c]['all']['tokens_per_s']:.1f} |")

    out += ["", "## Acceptance", "",
            "α is accepted draft tokens over evaluated draft tokens (accepted plus the first rejection per round). "
            "Match rate is the share of rounds where the draft proposed anything (the copy draft proposes nothing "
            "when it finds no match; those rounds are excluded from α). In greedy decoding the two α estimators of "
            "the plan are identical by construction, since min(p, q) is 1 or 0.", "",
            "| Setting | α code | α maths | α chat | match rate | tokens per round |", "|---|---|---|---|---|---|"]
    for c in configs[1:]:
        a = table[c]
        out.append(f"| {c} | " + " | ".join(fmt(a[t]["alpha_counted"]) for t in TASKS)
                   + f" | {fmt(a['all']['match_rate'])} | {fmt(a['all']['tokens_per_round'])} |")

    out += ["", "## Theory vs measurement (experiment 6)", "",
            "Predicted speedup in three layers (see `specdec/theory.py`): **A** the paper's formula with the "
            "measured α, γ and c; **B** adds the measured verification cost and the draft tokens actually proposed "
            "each round; **C** also uses the tokens each round actually produced, dropping the equal-acceptance "
            "assumption. The cost model ignores reading the prompt and Python overhead.", "",
            "![Theory vs measurement](theory_vs_measured.png)", "",
            "| Setting | Task | α | A paper | B cost model | C round by round | Measured |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['config']} | {r['task']} | {r['alpha']:.2f} | {r['A_paper']:.2f} | {r['B_cost_model']:.2f} "
                   f"| {r['C_round_by_round']:.2f} | {r['measured']:.2f} |")

    out += ["", "## Costs (experiment 4)", "", "![Verification cost](verify_cost.png)", "",
            f"Target step {costs['target_step_ms']:.1f} ms, 0.5B draft step {costs['draft_step_ms']:.1f} ms, "
            f"n-gram lookup {costs['lookup_ms']:.3f} ms, measured after a {costs['context_tokens']}-token context.", "",
            "| Tokens checked | " + " | ".join(str(b) for b in costs["verify_ratio"]) + " |",
            "|---" * (len(costs["verify_ratio"]) + 1) + "|",
            "| Time (target steps) | " + " | ".join(f"{v:.2f}" for v in costs["verify_ratio"].values()) + " |"]

    out += ["", "## Correctness", ""]
    ex = data["exactness"]
    if ex:
        out.append(f"- Experiment 3, {ex['max_new_tokens']} tokens on one prompt per task: float32 strict exact match "
                   f"**{'pass' if ex['float32_strict_pass'] else 'FAIL'}**, float16 teacher-forced check "
                   f"**{'pass' if ex['float16_teacher_forced_pass'] else 'FAIL'}** (near-tie threshold "
                   f"{ex['near_tie_threshold']}).")
    differs = {c: table[c]["all"]["differs_from_baseline"] for c in configs[1:]}
    out.append(f"- Speculative outputs that differ from the baseline in float16: {sum(differs.values())} of "
               f"{sum(table[c]['all']['prompts'] for c in configs[1:])}.")
    mm = data["mismatches"]
    if mm is not None:
        explained = sum(1 for m in mm if not m["speculative"]["failures"] and not m["baseline"]["failures"])
        out.append(f"- Teacher-forced check of every differing output: {explained} of {len(mm)} are explained by "
                   f"near-ties (every token is the target's top choice or within the threshold of it).")
    out += ["", "## Run conditions", ""]
    if temps:
        out.append(f"- GPU temperature {min(temps):.0f} to {max(temps):.0f} °C, SM clock {min(clocks):.0f} to "
                   f"{max(clocks):.0f} MHz. Configurations were interleaved, so clock changes affect all of them.")
    spread = data["summary"].get("repeat_spread", {})
    if spread:
        worst = max(spread.items(), key=lambda kv: kv[1]["stdev"])
        out.append(f"- Across {s['repeats']} repeated passes over one prompt per task, the largest standard "
                   f"deviation of a speedup was {worst[1]['stdev']:.3f} ({worst[0]}).")
    if data["summary"].get("peak_memory_gb"):
        out.append(f"- Peak GPU memory {data['summary']['peak_memory_gb']:.1f} GB.")
    return "\n".join(out) + "\n"


def write(run_dir: Path, out_dir: Path | None = None) -> Path:
    data = load(run_dir)
    out_dir = out_dir or ROOT / "report" / data["settings"]["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = theory_rows(data)
    (out_dir / "theory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    chart_speedup_vs_gamma(data, out_dir / "speedup_vs_gamma.png")
    chart_theory(rows, out_dir / "theory_vs_measured.png")
    chart_costs(data["costs"], out_dir / "verify_cost.png")
    (out_dir / "README.md").write_text(markdown(data, rows), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print("wrote", write(args.run_dir, args.out))


if __name__ == "__main__":
    main()
