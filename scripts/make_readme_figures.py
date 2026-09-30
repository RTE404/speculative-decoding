"""Build the README's figures, in light and dark versions, from a run's saved results.

    python scripts/make_readme_figures.py results/20260930-035822

Writes docs/figures/<name>-light.png and <name>-dark.png with transparent backgrounds;
the README picks one with <picture> and prefers-color-scheme.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from specdec import theory  # noqa: E402

OUT = ROOT / "docs" / "figures"
TASKS = ("code", "maths", "chat")

# Reference palette (dataviz skill); two categorical slots validated all-pairs in both modes.
THEMES = {
    "light": {"ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "ref": "#8a8985",
              "neural": "#2a78d6", "copy": "#eb6834", "target": "#8a8985"},
    "dark": {"ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#3a3a37", "ref": "#8a8985",
             "neural": "#3987e5", "copy": "#d95926", "target": "#8a8985"},
}
LABELS = {"neural": "0.5B model as the draft", "copy": "copy-from-context draft"}


def load(run_dir: Path) -> dict:
    runs = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return {"summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["by_config"],
            "costs": json.loads((run_dir / "costs.json").read_text(encoding="utf-8")), "runs": runs}


def axes_style(ax, t, grid_axis="y"):
    ax.set_facecolor("none")
    ax.grid(True, axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["ref"])
    ax.tick_params(colors=t["ink2"], labelsize=10)


def save(fig, name: str, theme: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", dpi=160, transparent=True, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def hero(data, theme):
    """Best speedup per task for each draft, as horizontal bars."""
    t, table = THEMES[theme], data["summary"]
    best = {"neural": "neural-g1", "copy": "copy-g8"}
    rows = [*TASKS, "all"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    axes_style(ax, t, grid_axis="x")
    height = 0.36
    for i, task in enumerate(rows):
        y = len(rows) - 1 - i
        for j, draft in enumerate(("copy", "neural")):
            value = table[best[draft]][task]["speedup"]
            yy = y + (0.2 if j == 0 else -0.2)
            ax.barh(yy, value, height=height, color=t[draft], edgecolor="none", zorder=3)
            ax.text(value + 0.05, yy, f"{value:.2f}×", va="center", ha="left", color=t["ink"], fontsize=10,
                    fontweight="bold" if task == "all" else "normal")
    ax.axvline(1, color=t["ref"], linewidth=1.2, linestyle="--", zorder=4)
    ax.text(1.02, len(rows) - 0.45, "3B model alone", color=t["ink2"], fontsize=9, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["all tasks" if r == "all" else r for r in reversed(rows)], color=t["ink"], fontsize=11)
    ax.set_xlim(0, 4.7)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["0", "1×", "2×", "3×", "4×"])
    ax.set_xlabel("speedup (tokens per second, relative to the 3B model alone)", color=t["ink2"], fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=t[d]) for d in ("copy", "neural")]
    ax.legend(handles, [f"{LABELS['copy']} (γ = 8)", f"{LABELS['neural']} (γ = 1)"], loc="lower right",
              frameon=False, fontsize=10, labelcolor=t["ink"])
    save(fig, "hero", theme)


def speedup_vs_gamma(data, theme):
    t, table = THEMES[theme], data["summary"]
    gammas = {"neural": (1, 2, 3, 4, 6), "copy": (2, 4, 8)}
    markers = {"neural": "o", "copy": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        axes_style(ax, t)
        ax.axhline(1, color=t["ref"], linewidth=1.2, linestyle="--")
        for draft, gs in gammas.items():
            ax.plot(gs, [table[f"{draft}-g{g}"][task]["speedup"] for g in gs], color=t[draft],
                    marker=markers[draft], markersize=6, linewidth=2, label=LABELS[draft])
        ax.set_title(task, color=t["ink"], fontsize=12)
        ax.set_xticks([1, 2, 3, 4, 6, 8])
        ax.set_xlabel("γ (tokens drafted per round)", color=t["ink2"], fontsize=9)
        ax.set_ylim(0, 4.4)
    axes[0].set_ylabel("speedup", color=t["ink2"], fontsize=10)
    axes[0].text(8, 0.93, "3B alone", color=t["ink2"], fontsize=8, ha="right", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, labelcolor=t["ink"],
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "speedup-vs-gamma", theme)


def costs(data, theme):
    """Where the time goes: one step of each model, a 16-token check, a lookup."""
    t, c = THEMES[theme], data["costs"]
    rows = [("3B model: 1 token", c["target_step_ms"], "target"),
            ("3B model: check 16 tokens at once", c["verify_ms"]["16"], "target"),
            ("0.5B draft: 1 token", c["draft_step_ms"], "neural"),
            ("copy draft: 1 lookup", c["lookup_ms"], "copy")]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    axes_style(ax, t, grid_axis="x")
    for i, (label, ms, kind) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, ms, height=0.55, color=t[kind], edgecolor="none", zorder=3)
        text = f"{ms:.1f} ms" if ms >= 1 else f"{ms:.3f} ms   (about {round(c['target_step_ms'] / ms, -2):,.0f}× faster than a 3B step)"
        ax.text(ms + 0.8, y, text, va="center", color=t["ink"], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=t["ink"], fontsize=10)
    ax.set_xlim(0, 55)
    ax.set_xlabel("time per step on a Tesla T4 (ms)", color=t["ink2"], fontsize=10)
    save(fig, "costs", theme)


def theory_fig(data, theme):
    t = THEMES[theme]
    points = {"neural": [], "copy": []}
    for draft, gs in (("neural", (1, 2, 3, 4, 6)), ("copy", (2, 4, 8))):
        for g in gs:
            for task in TASKS:
                recs = [r for r in data["runs"] if r["config"] == f"{draft}-g{g}" and r["repeat"] == 0
                        and r["task"] == task]
                p = theory.predictions(recs, draft, g, data["costs"])
                points[draft].append((p["A_paper"], p["C_round_by_round"],
                                      data["summary"][f"{draft}-g{g}"][task]["speedup"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    top = 6
    for ax, idx, title in ((axes[0], 0, "paper's formula, as written"),
                           (axes[1], 1, "same theory + measured costs")):
        axes_style(ax, t, grid_axis="both")
        ax.plot([0, top], [0, top], color=t["ref"], linewidth=1.2, linestyle="--", label="perfect prediction")
        for draft, pts in points.items():
            ax.scatter([p[idx] for p in pts], [p[2] for p in pts], color=t[draft], s=46,
                       marker="o" if draft == "neural" else "s", edgecolors="none", zorder=3, label=LABELS[draft])
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_title(title, color=t["ink"], fontsize=12)
        ax.set_xlabel("predicted speedup", color=t["ink2"], fontsize=10)
    axes[0].set_ylabel("measured speedup", color=t["ink2"], fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, labelcolor=t["ink"],
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "theory", theme)


def token_rows(record: dict, tokenizer, max_rounds: int) -> list[list[tuple[str, str]]]:
    """Split a real generation into target passes: accepted draft tokens, then the target's token."""
    rows, pos, tokens = [], 0, record["tokens"]
    for k, n, _ in record["rounds"][:max_rounds]:
        row = [(tokenizer.decode([tok]), "draft") for tok in tokens[pos:pos + n]]
        if pos + n < len(tokens):
            row.append((tokenizer.decode([tokens[pos + n]]), "target"))
        rows.append(row)
        pos += n + 1
    return rows


def show(text: str) -> str:
    if text.strip() == "":
        return "↵" * text.count("\n") if "\n" in text else "␣" * min(len(text), 4)
    return text.replace("\n", "↵").strip(" ") if text.strip() else text


def trace(data, theme, tokenizer):
    """Real generations, one line per target pass (like the paper's Figure 1)."""
    t = THEMES[theme]
    examples = [("copy-g8", "HumanEval/122", "copy", "copy draft, γ = 8, on a code prompt"),
                ("neural-g4", "spec_bench/413", "neural", "0.5B draft, γ = 4, on a maths prompt")]
    rows_per = 7
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6))
    for ax, (config, prompt_id, draft, title) in zip(axes, examples):
        rec = next(r for r in data["runs"] if r["config"] == config and r["prompt_id"] == prompt_id
                   and r["repeat"] == 0)
        rows = token_rows(rec, tokenizer, rows_per)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, rows_per - 0.3)
        ax.axis("off")
        ax.set_title(title, color=t["ink"], fontsize=11, loc="left")
        for i, row in enumerate(rows):
            y = rows_per - 1 - i
            ax.text(-1, y, f"pass {i + 1}", color=t["ink2"], fontsize=8.5, ha="right", va="center")
            x = 0.5
            for text, kind in row:
                label = show(text)
                width = 0.95 * max(len(label), 1) + 1.4
                if x + width > 99:
                    ax.text(x, y, "…", color=t["ink2"], fontsize=10, va="center")
                    break
                color = t[draft] if kind == "draft" else t["target"]
                ax.add_patch(FancyBboxPatch((x, y - 0.32), width, 0.64, boxstyle="round,pad=0,rounding_size=0.25",
                                            facecolor=color, alpha=0.22 if kind == "draft" else 0.35,
                                            edgecolor=color, linewidth=1.2))
                ax.text(x + width / 2, y, label, color=t["ink"], fontsize=8.5, ha="center", va="center",
                        family="DejaVu Sans Mono")
                x += width + 0.5
    handles = [FancyBboxPatch((0, 0), 1, 1, facecolor=t["copy"], alpha=0.3, edgecolor=t["copy"]),
               FancyBboxPatch((0, 0), 1, 1, facecolor=t["neural"], alpha=0.3, edgecolor=t["neural"]),
               FancyBboxPatch((0, 0), 1, 1, facecolor=t["target"], alpha=0.45, edgecolor=t["target"])]
    fig.legend(handles, ["guessed by the copy draft, accepted", "guessed by the 0.5B draft, accepted",
                         "written by the 3B model"], loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               labelcolor=t["ink"], bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.04, 0.05, 1, 1))
    save(fig, "trace", theme)


def main() -> None:
    from transformers import AutoTokenizer
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "20260930-035822"
    data = load(run_dir)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    for theme in THEMES:
        hero(data, theme)
        speedup_vs_gamma(data, theme)
        costs(data, theme)
        theory_fig(data, theme)
        trace(data, theme, tokenizer)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
