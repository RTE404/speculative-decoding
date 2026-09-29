"""Step 1 check: load both models on one T4 and answer the plan's open questions.

Prints library versions, confirms the vocabulary facts, checks for finite float16 logits,
tests cache rollback with crop(), and takes a first rough measurement of c and of the
verification cost. Saves everything to results/env_check.json.

    python -m specdec.check_env
    python -m specdec.check_env --target /kaggle/input/... --draft /kaggle/input/...
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers

from specdec.models import (DRAFT_ID, EOS_IDS, TARGET_ID, encode_prompt, find_kaggle_model, load_pair,
                            real_logits)

ROOT = Path(__file__).resolve().parent.parent
WARMUP_STEPS = 5


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def greedy_step_times(model, ids: torch.Tensor, n_tokens: int) -> tuple[list[int], list[float]]:
    """Greedy decode with a KV cache; return the tokens and per-step times in seconds."""
    device = ids.device
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    next_token = real_logits(out.logits[:, -1]).argmax(-1, keepdim=True)
    tokens, times = [next_token.item()], []
    for _ in range(n_tokens - 1):
        sync(device)
        start = time.perf_counter()
        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        next_token = real_logits(out.logits[:, -1]).argmax(-1, keepdim=True)
        sync(device)
        times.append(time.perf_counter() - start)
        tokens.append(next_token.item())
        if tokens[-1] in EOS_IDS:
            break
    return tokens, times


@torch.inference_mode()
def verify_costs(model, ids: torch.Tensor, block_sizes: list[int], repeats: int = 10) -> dict[int, float]:
    """Median time for one target pass over k new tokens, rolling the cache back after each."""
    device = ids.device
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    base_len = cache.get_seq_length()
    costs = {}
    for k in block_sizes:
        block = ids[:, :k]
        times = []
        for i in range(WARMUP_STEPS + repeats):
            sync(device)
            start = time.perf_counter()
            real_logits(model(input_ids=block, past_key_values=cache, use_cache=True).logits)
            sync(device)
            if i >= WARMUP_STEPS:
                times.append(time.perf_counter() - start)
            cache.crop(-k)
            if cache.get_seq_length() != base_len:
                raise AssertionError(f"crop(-{k}) left {cache.get_seq_length()} tokens, expected {base_len}")
        costs[k] = statistics.median(times)
    return costs


def environment() -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        import accelerate
        info["accelerate"] = accelerate.__version__
    except ImportError:
        info["accelerate"] = None
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=find_kaggle_model("3b-instruct") or TARGET_ID)
    parser.add_argument("--draft", default=find_kaggle_model("0.5b-instruct") or DRAFT_ID)
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args()

    report = {"environment": environment(), "target": args.target, "draft": args.draft}
    print(json.dumps(report, indent=2))

    pair = load_pair(args.target, args.draft)
    report["dtype"] = str(pair.dtype)
    report["models"] = {
        name: {
            "vocab_size": m.config.vocab_size,
            "tie_word_embeddings": m.config.tie_word_embeddings,
            "num_hidden_layers": m.config.num_hidden_layers,
            "attn_implementation": m.config._attn_implementation,
            "generation_config": {k: getattr(m.generation_config, k, None)
                                  for k in ("do_sample", "temperature", "top_p", "top_k",
                                            "repetition_penalty", "eos_token_id")},
        }
        for name, m in (("target", pair.target), ("draft", pair.draft))
    }
    report["tokenizer_len"] = len(pair.tokenizer)
    print("vocabulary checks passed")

    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
    prompt = next(p for p in prompts if p["task"] == "maths")
    ids = encode_prompt(pair.tokenizer, prompt["prompt"], pair.device)
    report["prompt_tokens"] = ids.shape[1]

    step_ms = {}
    for name, model in (("target", pair.target), ("draft", pair.draft)):
        tokens, times = greedy_step_times(model, ids, args.tokens)
        steady = times[WARMUP_STEPS:] or times
        step_ms[name] = 1000 * statistics.median(steady)
        report[f"{name}_text"] = pair.tokenizer.decode(tokens)
    report["step_ms"] = step_ms
    report["tokens_per_s"] = {k: 1000 / v for k, v in step_ms.items()}
    report["c_estimate"] = step_ms["draft"] / step_ms["target"]

    costs = verify_costs(pair.target, ids, [1, 2, 4, 8, 16])
    report["verify_cost_ratio"] = {k: v / costs[1] for k, v in costs.items()}
    report["crop_rollback"] = "ok"

    if pair.device.type == "cuda":
        report["peak_memory_gb"] = torch.cuda.max_memory_allocated(pair.device) / 1e9

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "env_check.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("models", "tokenizer_len", "step_ms", "tokens_per_s",
                                              "c_estimate", "verify_cost_ratio", "crop_rollback")},
                     indent=2, default=str))
    print("\ntarget output:", report["target_text"])
    print("\nsaved results/env_check.json")


if __name__ == "__main__":
    main()
