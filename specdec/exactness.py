"""Experiment 3: greedy exact match on the real models (plan, section 5).

Part 1, float32: speculative greedy output must equal plain greedy output token for token.
The 3B target (about 11.5 GiB) runs on cuda:0 and the draft on cuda:1.

Part 2, float16: mismatches from near-tied logits are expected (verifying several tokens at
once uses different matrix kernels). Teacher-forced check: run the target once, without a
cache, over the prompt plus the emitted tokens, and require every emitted token to be the
target's top choice, or a near-tie (top-two logit gap below NEAR_TIE) that is logged.

    python -m specdec.exactness
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from specdec import baseline, speculative
from specdec.cache import CachedModel
from specdec.drafts import NeuralDraft, PromptLookupDraft
from specdec.models import (DRAFT_ID, EOS_IDS, REAL_VOCAB, TARGET_ID, encode_prompt, find_kaggle_model,
                            load_pair, real_logits)

ROOT = Path(__file__).resolve().parent.parent
GAMMA = 4
# Fixed before running. Qwen's logits are mostly between 10 and 40, where float16 values
# are 0.0078 to 0.031 apart, so kernel differences of a few steps move logits by up to ~0.1.
NEAR_TIE = 0.1


def one_prompt_per_task() -> list[dict]:
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
    return [next(p for p in prompts if p["task"] == task) for task in ("code", "maths", "chat")]


def run_all(pair, prompts: list[dict], max_new_tokens: int) -> dict:
    """Greedy baseline plus both drafts, for each prompt. Returns tokens per prompt and method."""
    target = CachedModel(pair.target, REAL_VOCAB)
    drafts = {
        "neural": NeuralDraft(CachedModel(pair.draft, REAL_VOCAB)),
        "copy": PromptLookupDraft(REAL_VOCAB, device=pair.device),
    }
    outputs = {}
    for prompt in prompts:
        ids = encode_prompt(pair.tokenizer, prompt["prompt"], pair.device)[0].tolist()
        runs = {"baseline": baseline.generate(target, ids, max_new_tokens=max_new_tokens, eos_ids=EOS_IDS)}
        for name, draft in drafts.items():
            runs[name] = speculative.generate(target, draft, ids, max_new_tokens=max_new_tokens, gamma=GAMMA,
                                              eos_ids=EOS_IDS).tokens
        outputs[prompt["source_id"]] = {"prompt_ids": ids, "runs": runs}
        print(prompt["source_id"], {name: len(tokens) for name, tokens in runs.items()}, flush=True)
    return outputs


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


@torch.inference_mode()
def teacher_forced_check(model, prompt_ids: list[int], emitted: list[int]) -> dict:
    """Is each emitted token the target's argmax given everything before it (one no-cache pass)?"""
    ids = torch.tensor([prompt_ids + emitted], device=model.device)
    logits = real_logits(model(input_ids=ids, logits_to_keep=len(emitted) + 1).logits[0, :-1])
    top2 = logits.topk(2, dim=-1)
    chosen = logits.gather(-1, torch.tensor(emitted, device=model.device).unsqueeze(-1)).squeeze(-1)
    gap_to_top = (top2.values[:, 0] - chosen).tolist()
    not_top = [i for i, gap in enumerate(gap_to_top) if gap > 0]
    near_ties = [{"position": i, "gap": gap_to_top[i]} for i in not_top if gap_to_top[i] < NEAR_TIE]
    failures = [{"position": i, "gap": gap_to_top[i]} for i in not_top if gap_to_top[i] >= NEAR_TIE]
    return {"tokens": len(emitted), "not_top_choice": len(not_top), "near_ties": near_ties, "failures": failures}


def part_float32(args, prompts) -> dict:
    two_gpus = torch.cuda.device_count() >= 2
    pair = load_pair(args.target, args.draft, device=torch.device("cuda:0"), dtype=torch.float32,
                     draft_device=torch.device("cuda:1" if two_gpus else "cuda:0"))
    outputs = run_all(pair, prompts, args.tokens)
    results = {}
    for source_id, out in outputs.items():
        base = out["runs"]["baseline"]
        results[source_id] = {name: {"identical": tokens == base, "first_mismatch": first_mismatch(tokens, base)}
                              for name, tokens in out["runs"].items() if name != "baseline"}
    del pair
    gc.collect()
    torch.cuda.empty_cache()
    return results


def part_float16(args, prompts) -> dict:
    pair = load_pair(args.target, args.draft)
    outputs = run_all(pair, prompts, args.tokens)
    results = {}
    for source_id, out in outputs.items():
        base = out["runs"]["baseline"]
        results[source_id] = {
            name: {"identical_to_baseline": tokens == base, "first_mismatch": first_mismatch(tokens, base),
                   **teacher_forced_check(pair.target, out["prompt_ids"], tokens)}
            for name, tokens in out["runs"].items()
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=find_kaggle_model("3b-instruct") or TARGET_ID)
    parser.add_argument("--draft", default=find_kaggle_model("0.5b-instruct") or DRAFT_ID)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    prompts = one_prompt_per_task()

    print("== float32: strict exact match ==", flush=True)
    fp32 = part_float32(args, prompts)
    print("== float16: teacher-forced check ==", flush=True)
    fp16 = part_float16(args, prompts)

    strict_pass = all(r["identical"] for per_prompt in fp32.values() for r in per_prompt.values())
    fp16_pass = all(not r["failures"] for per_prompt in fp16.values() for r in per_prompt.values())
    report = {"near_tie_threshold": NEAR_TIE, "gamma": GAMMA, "max_new_tokens": args.tokens,
              "float32": fp32, "float16": fp16,
              "float32_strict_pass": strict_pass, "float16_teacher_forced_pass": fp16_pass}
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "exactness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nfloat32 strict exact match: {'PASS' if strict_pass else 'FAIL'}")
    print(f"float16 teacher-forced check: {'PASS' if fp16_pass else 'FAIL'}")
    print("saved results/exactness.json")


if __name__ == "__main__":
    main()
