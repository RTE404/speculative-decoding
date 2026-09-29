"""Build prompts.json from pinned upstream sources.

Run once; the output is committed. Rerunning must reproduce the same file.

    python scripts/make_prompts.py
"""

import gzip
import io
import json
import random
import urllib.request
from pathlib import Path

HUMANEVAL_COMMIT = "6d43fb980f9fee3c892a914eda09951f772ad10d"
SPEC_BENCH_COMMIT = "fd2c1cd7d2201ef71db4c5f4e455008f017967bf"
HUMANEVAL_URL = f"https://raw.githubusercontent.com/openai/human-eval/{HUMANEVAL_COMMIT}/data/HumanEval.jsonl.gz"
SPEC_BENCH_URL = f"https://raw.githubusercontent.com/hemingkx/Spec-Bench/{SPEC_BENCH_COMMIT}/data/spec_bench/question.jsonl"

MT_BENCH_CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
PER_TASK = 10
SEED = 0

CODE_INSTRUCTION = "Complete the following Python function. Reply with the full function in a single code block.\n\n```python\n{prompt}```"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def load_humaneval() -> list[dict]:
    raw = gzip.decompress(fetch(HUMANEVAL_URL)).decode("utf-8")
    return [json.loads(line) for line in io.StringIO(raw) if line.strip()]


def load_spec_bench() -> list[dict]:
    raw = fetch(SPEC_BENCH_URL).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def main() -> None:
    rng = random.Random(SEED)
    records = []

    humaneval = load_humaneval()
    for row in sorted(rng.sample(humaneval, PER_TASK), key=lambda r: int(r["task_id"].split("/")[1])):
        records.append({
            "task": "code",
            "source": "HumanEval (openai/human-eval)",
            "source_id": row["task_id"],
            "license": "MIT",
            "prompt": CODE_INSTRUCTION.format(prompt=row["prompt"]),
        })

    spec_bench = load_spec_bench()

    maths = [r for r in spec_bench if r["category"] == "math_reasoning"]
    for row in sorted(rng.sample(maths, PER_TASK), key=lambda r: r["question_id"]):
        records.append({
            "task": "maths",
            "source": "GSM8K via Spec-Bench (hemingkx/Spec-Bench)",
            "source_id": f"spec_bench/{row['question_id']}",
            "license": "MIT (GSM8K); Apache-2.0 (Spec-Bench)",
            "prompt": row["turns"][0],
        })

    # One MT-Bench question per category, then the remainder drawn from the rest.
    chat_pool = [r for r in spec_bench if r["category"] in MT_BENCH_CATEGORIES]
    chosen = [rng.choice([r for r in chat_pool if r["category"] == c]) for c in MT_BENCH_CATEGORIES]
    rest = [r for r in chat_pool if r not in chosen]
    chosen += rng.sample(rest, PER_TASK - len(chosen))
    for row in sorted(chosen, key=lambda r: r["question_id"]):
        records.append({
            "task": "chat",
            "source": f"MT-Bench first turn via Spec-Bench, category {row['category']}",
            "source_id": f"spec_bench/{row['question_id']}",
            "license": "Apache-2.0",
            "prompt": row["turns"][0],
        })

    out = Path(__file__).resolve().parent.parent / "prompts.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} prompts to {out}")


if __name__ == "__main__":
    main()
