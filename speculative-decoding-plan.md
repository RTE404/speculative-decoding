# Speculative Decoding in a Day: Project Plan

**Paper:** *Fast Inference from Transformers via Speculative Decoding*, Leviathan, Kalman and Matias, ICML 2023 ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192))
**Related paper:** *Accelerating Large Language Model Decoding with Speculative Sampling*, Chen et al., DeepMind, 2023 ([arXiv:2302.01318](https://arxiv.org/abs/2302.01318)). Same idea, found independently. Note that Chen et al. swap the symbols (their q is the target), so code copied from that paper inverts the ratio.
**Owner:** Rohan Tiwari
**Budget:** Zero. Everything runs on a free Kaggle GPU.
**Last reviewed:** 2026-09-29, against the research in `reports/Speculative decoding plan review.md`.

---

## 1. What we are trying to do

Implement speculative decoding from scratch in PyTorch, prove it gives exactly the target model's output, and measure the speedup on a Kaggle T4. Then use the paper's formula to explain the result, and test whether a draft that costs nothing (copying tokens from the context) beats a small neural draft.

**The main constraint:** the full set of results must be reproducible by running one Kaggle notebook, in about an hour, well inside Kaggle's free limits.

The questions this project answers:

> **1. On a free T4, does a 0.5B draft speed up Qwen2.5-3B at all, and does the paper's formula explain the result once we measure c, α and the verification cost?**
>
> **2. Does a zero-cost draft that copies tokens from the context beat the 0.5B neural draft?**

**Expected answer, based on published numbers.** In Hugging Face eager decoding, a small model's step time is mostly software overhead, so the 0.5B draft costs about two-thirds of a 3B step (c ≈ 0.5 to 0.7). The formula then predicts roughly 0.8× to 1.2× for the neural draft, with the best γ at 1 or 2, while the copy draft (c ≈ 0) should give about 1.2× to 1.7× on code and maths and close to 1× on open chat. If that holds, the headline finding is that **on cheap hardware in plain PyTorch, the draft's cost is software overhead, and a draft that costs nothing wins**. This tests Corollary 3.9 (α > c) directly. A small or negative speedup, well explained, is a complete result.

---

## 2. The paper in plain terms

A language model generates text one token at a time, and each token needs a full pass through the model. For a large model, most of that time goes to loading its weights from memory, not to the maths. Checking 5 tokens in one pass therefore costs about the same as generating 1.

Speculative decoding uses this:

1. A small, fast **draft** guesses the next few tokens (say 4).
2. The large **target model** checks all of them in a single pass.
3. Guesses are accepted from the left, one at a time, until the target model disagrees.
4. At the first disagreement, the target model's own choice replaces the rejected token, and the rest are discarded.
5. If all guesses are accepted, the target model adds one extra token for free.

The acceptance rule guarantees the output follows exactly the target model's probabilities, for any draft (Appendix A.1). With greedy decoding the output is identical token for token, in exact arithmetic. The draft does not have to be a neural model: the paper notes (Section 3.6) that copying tokens from the context can work well at almost zero cost.

**Paper results for reference** (Table 2): T5-XXL (11B) target, T5-small (77M) draft, one TPU-v4, batch size 1.

| Task | Temp | γ | α | Speedup |
|---|---|---|---|---|
| English to German translation | 0 | 7 | 0.75 | 3.4× |
| CNN/DM summarisation | 0 | 5 | 0.65 | 3.1× |

The paper's drafts are about 100 times smaller than the target, with c below 0.05. Our neural draft is only about 6 times smaller than our target. Hugging Face's own guidance is that a draft should be at least 10 times smaller. We compare the pattern and the theory, not the numbers.

**Published numbers closest to our setup:**
- Prompt lookup on Spec-Bench (Vicuna-7B, RTX 3090): 1.72× on GSM8K, 1.64× on MT-Bench conversation, 1.64× overall.
- Qwen2.5-0.5B drafting for Qwen2.5-7B (on a CPU): about 62% acceptance, 2.64× on MBPP code.
- A case of assisted generation being slower than the baseline: Llama-3.1-8B with a 1B draft on an A100 ran about 28% slower (transformers issue #36337).

---

## 3. The theory we will test

- **α (alpha), the acceptance rate:** how often the target accepts a draft token. Defined in the paper as the mean of Σ min(p, q) (Corollary 3.6).
- **γ (gamma), the draft length:** how many tokens the draft guesses each round.
- **c, the cost ratio:** time for one draft step divided by time for one target step.

From the paper:

- **Expected tokens per round:** (1 − α^(γ+1)) / (1 − α)
- **Expected speedup:** (1 − α^(γ+1)) / ((1 − α)(γc + 1))
- **Speedup is possible only if α > c** (Corollary 3.9).
- **Zero-cost draft (c ≈ 0):** speedup is (1 − α^(γ+1)) / (1 − α), never more than 1 / (1 − α).

### Expected c

Qwen's official Hugging Face benchmark (A100, batch size 1) gives **30.8 tokens/s for the 3B and only 47.4 tokens/s for the 0.5B**, and the 7B is faster than the 3B. vLLM runs the same models 4 to 6.6 times faster on the same GPU. So Hugging Face eager decoding at this size is limited by per-layer framework and kernel-launch overhead, and step time follows layer count (24 against 36, ratio 0.67) more than parameter count. The implied c is 30.8 / 47.4 ≈ **0.65**. On a T4 we expect **c ≈ 0.5 to 0.7**. This is an estimate: no T4 measurement for this pair was found. Experiment 4 replaces it.

**First measurement (2026-09-29, `specdec.check_env`, one prompt, 48 greedy tokens, median step time):** target 46.3 ms/token (21.6 tokens/s), draft 29.9 ms/token (33.4 tokens/s), so **c ≈ 0.65**, as predicted. Checking 2 to 16 tokens in one target pass cost 0.93 to 0.97 times a single-token pass, so **verification is effectively free** up to 16 tokens. This is a rough first number from one prompt; experiment 4 measures it properly.

### Predicted speedup for the neural draft

| α | γ | c = 0.65 | c = 0.3 |
|---|---|---|---|
| 0.7 | 4 | 0.77× | — |
| 0.8 | 1 | 1.09× | — |
| 0.8 | 4 | 0.93× | 1.53× |
| 0.9 | 2 | 1.18× | — |
| 0.9 | 4 | 1.14× | — |

At high c, the best γ for the neural draft is 1 or 2, and longer drafts only slow things down. The copy draft has c ≈ 0 but lower α, and proposes nothing when it finds no match.

### Expected gaps between theory and measurement
- **Verification cost.** The formula assumes checking γ+1 tokens costs the same as one target step. For a 3B model on a T4 this should be nearly true up to about 9 tokens (a k-token pass has an arithmetic intensity of about k FLOP per byte, far below the T4's balance point of about 200). We measure it directly.
- **Unequal acceptance.** The formula assumes every token has the same acceptance chance. We save per-step values to test this.
- **Variable γ.** The copy draft's proposal length varies, so its prediction uses the measured proposal lengths.
- **Python and synchronisation overheads** that the formula ignores.

---

## 4. Scope

### Kaggle setup
- **Accelerator:** "GPU T4 x2". Since September 2026 this is the only T4 option: single T4 has been removed and the P100 was retired on 2026-09-15. Timing runs load both models on **one** T4 with `device_map={"": 0}`. Never use `device_map="auto"`, which would split the 3B across both GPUs and distort timings. The second T4 is used only for the float32 exact-match test (experiment 3).
- **First cell checks:** assert that CUDA is available, that there are two devices, and that both are named T4. Fail immediately otherwise. Runs pushed with the Kaggle API have been silently given the wrong image or accelerator.
- **Internet:** on, to install pinned packages and download the models. Kaggle forum reports say this, and the GPU itself, need a phone-verified account; the official docs do not say so explicitly.
- **Hugging Face access:** Qwen2.5 is not gated, but anonymous downloads are rate-limited per IP address, and Hugging Face names a missing token as the main cause of rate limiting. Store `HF_TOKEN` in Kaggle Secrets. Keep the model cache in the default location (`/root/.cache/huggingface`), not in `/kaggle/working`, so the weights do not fill the 20 GB saved output. `HF_HUB_ENABLE_HF_TRANSFER` is deprecated; downloads now use hf-xet.
- **Kaggle Models alternative:** `qwen-lm/qwen2.5/transformers/3b-instruct` exists on Kaggle Models. The 0.5B variant is likely there but not confirmed. If attached, find the model folder by searching `/kaggle/input` for `config.json` rather than hard-coding the path.
- **Limits** (from Kaggle's docs, checked 2026-09-29):
  - Up to **12 hours** per GPU session. "Save & Run All" must finish within 12 hours.
  - GPU quota of **30 hours per week**, "or sometimes higher depending on demand and resources". It resets weekly.
  - T4 x2 machine: 2 T4s (16 GB each), **4 CPU cores, 29 GB RAM**.
  - **20 GB** of saved output in `/kaggle/working`. Other disk space is wiped at the end of the session.
  - Idle timeout for interactive sessions: 20 minutes according to the Notebooks docs, 60 minutes according to the GPU tips page. "Save & Run All" does not time out while idle.
  - Free GPUs can have a queue at busy times.
- **Environment:** the current Kaggle image (v170, released 2026-06-29) ships **torch 2.10.0 with CUDA 12.8, transformers 5.x, accelerate 1.13.0 and Python 3.12**. A new image with **Python 3.13 and torch 2.11** has been merged but not yet released, so the environment may change during development. Kaggle updates its image about every two weeks. Once the code works, pin the notebook's environment (Session options, Environment) so later runs use the same image. The T4 is supported by current PyTorch builds.

### Precision and attention
- **Float16** for all timing runs (the T4 does not support bfloat16). Float32 only for experiment 3.
- **Attention:** load both models with `attn_implementation="sdpa"`. FlashAttention-2 does not support the T4, and PyTorch's own flash backend needs a newer GPU, so SDPA will use its memory-efficient or math backend. Do not use `"eager"`: its attention scaling is where Qwen2's documented float16 overflow happens.
- **Float16 risk:** Qwen's Qwen2 documentation advises against float16, citing inf/NaN errors or repeated `!!!!` output. The documented cause is an overflow in the last layer of 7B-sized models. No failures of the 0.5B or 3B in float16 were found, but that is not proof. So:
  - check `torch.isfinite` on the logits in the baseline and the speculative loop, and stop with a clear error if it fails;
  - do all softmax, probability and acceptance maths in float32 (model outputs come back in float16).

### Models

| Role | Model | Size (float16) | Licence |
|---|---|---|---|
| Target | Qwen2.5-3B-Instruct | about 6.2 GB (3.09B parameters) | Qwen Research License (non-commercial: research or evaluation only) |
| Neural draft | Qwen2.5-0.5B-Instruct | about 1.0 GB (0.49B parameters) | Apache-2.0 |
| Copy draft | Prompt lookup, no model | none | ours (MIT) |

Confirmed compatibility (from the model files on Hugging Face):
- Both models have an output size (`vocab_size`) of **151,936**, tied embeddings, and byte-identical tokenizer files and chat template.
- The tokenizer's real vocabulary is **151,665** tokens (151,643 regular plus 22 special). The other 271 output rows are padding. **Slice logits to `[..., :151665]` before any softmax**, in both models. This also makes the 7B extension work, since the 7B has an output size of 152,064.
- `models.py` still asserts these facts at load time and fails loudly if they differ.
- Architecture: the 0.5B has 24 layers (hidden size 896), the 3B has 36 layers (hidden size 2048).

**Qwen's default generation settings must be overridden.** Both models ship a `generation_config.json` with sampling turned on (`do_sample: true`, temperature 0.7, top_p 0.8, top_k 20) and a repetition penalty that **differs between the two models (1.05 for the 3B, 1.1 for the 0.5B)**. Hugging Face's `generate()` applies the repetition penalty even when `do_sample=False`. Our loops build every distribution from raw `model(...).logits` and never use `generate()`. Any use of `generate()` (for example the Hugging Face comparison extension) must pass `do_sample=False` and `repetition_penalty=1.0`.

**Stopping rule:** stop on **either** end token: 151645 (`<|im_end|>`) or 151643 (`<|endoftext|>`).

**Chat template:** when no system message is given, the template adds "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." Both models see the same text, so the comparison stays fair, but it adds about 20 tokens to every prompt. Keep it, and document it.

### Prompts
Ten prompts per task, 30 in total, saved in the repository as `prompts.json` so no dataset download is needed:
- **Code:** 10 HumanEval problem prompts (MIT, copyright OpenAI).
- **Maths:** 10 GSM8K questions, taken from **Spec-Bench's** maths subset (GSM8K is MIT; Spec-Bench is Apache-2.0).
- **Chat:** 10 **MT-Bench first-turn questions**, taken from Spec-Bench's conversation subset (Apache-2.0, via FastChat).

Taking maths and chat from Spec-Bench lets us compare our prompt-lookup results with published numbers for the same categories. Spec-Bench has no code category, so HumanEval comes from its own repository. Choose the prompts once, record their IDs, and never change them.

Each record in `prompts.json` has `task`, `source`, `source_id`, `license` and `prompt` fields. Do not use Alpaca or AlpacaEval (non-commercial), lmsys-chat-1m (forbids redistribution), or Spec-Bench's Natural Questions, retrieval and CNN/DailyMail categories (share-alike or publisher copyright).

Maximum output: **128 new tokens**.

### Out of scope
- Sampling-based timing runs (sampling is tested for correctness only)
- Other targets (1.5B, 7B), batching, compiled models, Medusa or EAGLE
- Training a draft, adaptive γ

These are listed in section 12 as later extensions.

---

## 5. Experiments

| # | Experiment | Time on Kaggle | What it tells us |
|---|---|---|---|
| 1 | **Acceptance rule unit test:** synthetic distributions p and q (including a one-hot q, the copy draft's case, and a case where p ≈ q so the leftover probability is close to zero), one million samples, chi-square test against p | Seconds | The acceptance and resampling rule is exactly right, including its edge cases |
| 2 | **Full loop on tiny random models:** two randomly initialised Qwen2-architecture models with a vocabulary of about 50 and 2 layers. See the details below | Under a minute | The whole loop, including the KV cache, rollback, bonus token and EOS handling, is exact, without needing real models |
| 3 | **Exact match on real models:** 1 prompt per task, both drafts. See the details below | A few minutes | The real setup is exact |
| 4 | **Cost measurements:** c for each draft, and the verification cost for 1, 2, 4, 8 and 16 tokens in one target pass, median of 20 timed steps each | A few minutes | The inputs to the formula, and the first check of the c ≈ 0.5 to 0.7 estimate |
| 5 | **Speedup vs γ:** neural draft with γ = 1, 2, 3, 4, 6, copy draft with γ = 2, 4, 8, plus the baseline, on all 30 prompts, greedy | About 30 to 50 minutes | Best γ for each draft, speedup by task, measured α |
| 6 | **Theory vs measurement:** predicted speedup from measured α, γ, c and verification cost against measured speedup, in the format of the paper's Table 4 | Seconds | Whether the formula explains our results, and which assumption causes any gap |

**Experiment 2 details.**
- **Make p and q genuinely different.** With the default initialisation, tiny random models give nearly uniform distributions, so p ≈ q, α ≈ 1, and the rejection path almost never runs. Raise the initialisation scale (or divide the logits by a small temperature) and give the two models different seeds.
- **Require coverage.** Assert that a minimum share of rounds (for example 30%) hit the rejection branch, and that some rounds hit the all-accepted bonus-token branch.
- **Distribution test.** Compare the sampled distribution of the first 3 output tokens with the exact distribution, computed by listing every sequence (about 50³ cells). Use a chi-square test, merging rare cells so each has an expected count of at least 5.
- **Cache test.** After every round, compare the cached next-token logits with a fresh forward pass without a cache. This catches the draft-cache bug in section 8, which lowers α but does not bias the output, so the distribution test cannot see it.
- **EOS test.** Make EOS reachable inside a draft window and check that generation stops exactly there.
- Greedy exact match in float32.

**Experiment 3 details.**
- **Strict test in float32:** must be 100% identical. The 3B needs about 11.5 GiB in float32, so the target runs on `cuda:0` and the neural draft on `cuda:1`. Move logits to one device before comparing.
- **Float16 test:** mismatches are expected, not a bug. Hugging Face maintainers confirm that verifying several tokens at once uses different matrix kernels, so near-tied logits can flip. Fix the near-tie threshold in advance: **a top-two logit gap below 0.1** (fixed 2026-09-30, before any run). Qwen's logits are mostly between 10 and 40, where float16 values are 0.008 to 0.03 apart, so kernel differences of a few steps can move a logit by up to about 0.1. The check is **teacher-forced**: run the target once without a cache over the full emitted sequence, and require every emitted token to be the target's top choice or a logged near-tie.

---

## 6. What we measure

- **Tokens per second** for the baseline and each draft setting, and the **speedup**
- **Acceptance rate α**, overall and per task, two ways, over the **same positions**: the accepted draft tokens plus the first rejected one in each round. Positions after a rejection are never evaluated and are excluded.
  - **Counted:** accepted draft tokens divided by **evaluated** draft tokens. (Dividing by all proposed tokens is biased: at a true α of 0.8 and γ = 4 it gives about 0.55.)
  - **From the distributions:** mean of Σ min(p, q) at the evaluated positions (Corollary 3.6). For the copy draft this is the target's probability of the proposed token.
  - The two should agree within sampling error. A gap signals a bug.
- **Draft efficiency:** accepted draft tokens divided by (γ × rounds), reported separately from α
- **Per-step acceptance values**, to test the equal-acceptance assumption
- **Average tokens per round**, and for the copy draft, the **match rate** (rounds with no match are reported separately and excluded from α) and proposal lengths
- **c** for each draft and the **verification cost**
- **Peak GPU memory**, **wall-clock time** of each run, and **GPU clock and temperature** (from `nvidia-smi`) during timing

---

## 7. Rules that keep the results honest

- **The baseline uses a KV cache too.** A slow baseline would inflate the speedup.
- **The baseline is our own greedy loop on raw logits**, not `generate()`, so Qwen's default sampling and repetition-penalty settings cannot leak in.
- **Time properly.** Warm up before timing, covering every input length used: the prompt, a single token, and every γ+1. Call `torch.cuda.synchronize()` before reading the clock. Report medians.
- **Interleave configurations** (baseline, draft, baseline, draft) rather than running them in blocks, so any drift in GPU speed affects all of them equally. The T4 is passively cooled and can slow down when hot, so log its clock speed and temperature.
- **Keep GPU-to-CPU syncs to one per round.** Do the accept and reject maths as tensor operations.
- **Never overshoot.** Check for EOS and `max_new_tokens` inside every accepted block and discard anything after them. Overshooting by up to γ tokens would inflate tokens per second.
- **Same settings everywhere:** same prompts, chat template, maximum length, stopping rule and seeds. Report tokens per second, since outputs can end at different lengths.
- **Record the environment** in every results file: GPU name, precision, attention implementation, CUDA version, Kaggle image version, and exact `torch`, `transformers` and `accelerate` versions.
- **Report honestly.** If the neural draft is slower than the baseline, or the copy draft loses, say so and explain it with the measured numbers.

---

## 8. How the code fits together

```
specdec/
  models.py        # loads target and draft, checks output sizes, slices logits, sets sdpa and fp16
  baseline.py      # normal token-by-token greedy generation with a KV cache, on raw logits
  drafts.py        # draft interface: neural draft and copy draft
  speculative.py   # draft, verify, accept or reject, roll back the cache
  sampling.py      # the acceptance rule and the corrected resampling step (float32)
  run.py           # runs every experiment, saves results, builds the report
tests/             # experiments 1 to 3
prompts.json       # the 30 prompts, with source, source_id and license
requirements.txt   # exact pinned versions
kaggle/run.ipynb   # the one notebook that reproduces everything
results/           # one JSON line per prompt per run, never overwritten
report/            # generated tables and charts
DATA_LICENSES.md   # licences for prompts, models and outputs
LICENSE            # MIT, for the code
```

**Reproducing on Kaggle.** `kaggle/run.ipynb` does five things:
1. Checks the hardware (two T4s).
2. Installs `requirements.txt`, **before anything imports `transformers`**. "Save & Run All" cannot restart the kernel, so a package installed after an import will not take effect.
3. Clones the repository.
4. Runs `python -m specdec.run`.
5. Copies `results/` and `report/` to `/kaggle/working/` so they are saved as the notebook's output.

Use "Save Version" with "Save & Run All" so the run happens in the background and the saved version is a permanent, public record of the run. Download the output and commit the results to the repository.

```
python -m specdec.run           # full run, about 1 hour on a T4
python -m specdec.run --quick   # smoke test: 1 prompt per task, 32 tokens, a few minutes
```

The full run ends by writing `report/README.md` with the speedup tables, the theory-vs-measurement table, and the charts.

**Pin versions.** Keep Kaggle's preinstalled `torch` and never reinstall it. Pin `transformers` to an exact **5.x** version (for example 5.17.0) and `accelerate` to 1.13.0. Do not pin a 4.x version: that is a cross-major downgrade that conflicts with the preinstalled `huggingface_hub` and `peft`. Print the versions at start-up and record them in every results file.

**Draft interface.** `speculative.py` does not know which draft it is using. Each draft takes the current tokens and returns up to γ proposed tokens with their distributions q.

**The acceptance rule, in float32.**
- Accept draft token x if `r * q(x) < p(x)`, with r uniform on [0, 1). This avoids dividing by q.
- On rejection, sample from norm(max(0, p − q)). If that leftover sums to almost zero (it equals 1 − β, so it vanishes when p ≈ q) or is not finite, sample from p instead.
- For greedy decoding, compare with the target's argmax directly.

**The copy draft (prompt lookup).** Take the last n tokens of the sequence (prompt plus output so far) and find the most recent earlier place they appeared. If found, propose the γ tokens that followed it. Try n = 3, then 2, then 1. If nothing matches, propose nothing, and the round is a normal target step. Its q is one-hot, and it goes through the same acceptance rule as the neural draft: with a one-hot q, the rule accepts a proposed token with probability p(x) and otherwise samples from p with x removed. One code path for both drafts means one set of tests covers both (experiment 1 includes a one-hot q). About 50 lines.

**Target logits line-up.** In the verify pass, the target's output at position t is the distribution for token t+1. Checking γ draft tokens gives γ+1 useful rows; the last one gives the bonus token when all are accepted.

**KV cache rollback** (transformers 5.x `DynamicCache`):
- The API changed in v5.15: `crop()` now takes a **negative** number of tokens to remove. A positive argument is treated as an absolute length, logs a deprecation warning, and will be removed in v5.18.
- `crop(0)` emptied the whole cache up to v5.13 and does nothing from v5.14.
- So always write: `if n_reject > 0: cache.crop(-n_reject)`. Never call `crop(0)` or `crop(absolute_length)`.
- After every round, assert `cache.get_seq_length()` equals the number of committed tokens. Since v5.4, Qwen2 computes positions and the attention mask from the cache length, and `cache_position` is no longer an argument. A cache that is one token too long does not raise an error: the output just goes subtly wrong.
- Pass only `input_ids` and `past_key_values` (batch size 1, no padding). Do not pass `cache_position`.
- For the prompt pass, use `logits_to_keep=1`. By default the model returns logits for every position.
- Build it without a cache first, get experiment 2 passing, then add the cache.

**The classic rollback bug:** when all γ guesses are accepted, the neural draft never ran on its own last guess, so its cache is one token behind. The next round must feed it both that token and the target's bonus token. The per-round cache test in experiment 2 catches this.

---

## 9. Time budget

Rough estimates, to be replaced with measured times after the first run.

| Part | Estimate |
|---|---|
| Install packages and download models (about 7 GB) | 5 to 10 minutes |
| Experiments 1 to 4 | about 10 minutes |
| Experiment 5 (9 configurations × 30 prompts × up to 128 tokens, about 35,000 tokens; the 3B baseline at an estimated 18 to 30 tokens/s, and neural-draft settings that may run slower than the baseline) | about 30 to 50 minutes |
| Experiment 6 and report | about 1 minute |
| **Full run** | **about 1 to 1.5 hours** |
| `--quick` | about 5 to 10 minutes, including model download |

A full run uses about 1 to 1.5 of the 30 weekly GPU-hours. If it turns out much slower than estimated, reduce the prompts with a `--prompts-per-task` option rather than changing the design.

**Planned full runs: three.**
1. **Trial run:** finds problems that `--quick` misses and gives real timings. Its results are not published.
2. **Final run:** "Save & Run All" with the environment pinned. This is the published record.
3. **Repeat run:** the same notebook version in a fresh session, to report how much speedups vary between sessions.

Timing repeats for medians happen inside each run, so extra full runs are not needed for that. Total: about 3 to 6 GPU-hours.

**Settled by the step 1 check (2026-09-29):**
- Environment: Python 3.12.13, torch 2.10.0+cu128, transformers 5.17.0, accelerate 1.13.0, two Tesla T4s.
- Vocabulary: 151,936 output rows in both models, 151,665 real tokens, identical tokenizers. Qwen's shipped defaults confirmed (sampling on, repetition penalty 1.05 for the 3B and 1.1 for the 0.5B).
- Float16: finite logits for both models under SDPA on the test prompt.
- Cache rollback: `crop(-k)` restored the exact cache length for k = 1 to 16.
- c ≈ 0.65 and a flat verification cost (section 3).
- Target speed 21.6 tokens/s, inside the 18 to 30 tokens/s estimate, so the time budget above stands.
- Download: about 7 GB from Hugging Face in under a minute.
- A second session (2026-09-30) repeated the check: c = 0.657, target 22.1 tokens/s, verification 0.90 to 0.94 of a single step. Within about 2% of the first.

**Final run (2026-09-30, run 20260930-035822, published in `results/` and `report/`):** 39 minutes end to end with "Save & Run All" (tests 5.5 minutes, exact match 3 minutes, sweep and repeats 29 minutes). All 37 tests and both exact-match checks passed; 11 of 240 float16 outputs differed from the baseline and all 11 were near-ties (logit gaps 0.016 to 0.031). Results match the trial run within 1.1% on average and 4.3% at most. Copy draft γ = 8: 1.76× overall; 0.5B draft γ = 1: 1.19×; c = 0.650. Layer C of experiment 6 is within about 3% of every measurement. The run crashed at its last step (building the report) because of a variable name in `run.py`; the report was generated from the saved results, and the bug is fixed.

**Trial run (2026-09-30, run 20260930-024815, not published):** the full pipeline ran end to end in about an hour.
- Best settings: copy draft γ = 8 at 1.69× overall (code 3.91×, maths 1.50×, chat 1.16×); 0.5B draft γ = 1 at 1.20× (code 1.27×, maths 1.23×, chat 1.10×). The 0.5B draft falls below 1× on chat from γ = 3.
- α for the 0.5B draft: code 0.98, maths 0.92, chat 0.72. Copy draft: α 0.88 / 0.50 / 0.32 and match rate 78% / 55% / 43%.
- c = 0.640; verification of 2 to 16 tokens 0.93 to 0.95 of a step.
- Theory: layer C of experiment 6 predicts the 0.5B draft within 2 to 6% on every task and γ. The paper's formula (A) overshoots the copy draft (for example 5.7× predicted against 3.9× on code) because it assumes a proposal every round; using measured proposal lengths (B) closes most of the gap.
- 11 of 240 speculative outputs differed from the baseline in float16. From the final run on, each is re-checked automatically.
- The T4 ran at 75 to 81 °C with the SM clock between 1170 and 1575 MHz, so interleaving matters.
- Disclosure for the write-up: the code prompts ask for "the full function", so the model repeats the docstring, which favours the copy draft.

**Step 3 result (experiment 3, 2026-09-30):** on one code, one maths and one chat prompt, 128 tokens each, with γ = 4:
- Float32 (target on `cuda:0`, draft on `cuda:1`): both drafts **identical** to the target alone.
- Float16: both drafts and the baseline identical, and the teacher-forced check found every token to be the target's top choice, with **no near-ties at all**.

---

## 10. Build order

| Step | Work | Done when |
|---|---|---|
| 1. Set up | Create the repository and the Kaggle notebook with the hardware check. Add `HF_TOKEN` to Kaggle Secrets. Pin `transformers` 5.x. Load both models on one T4 with SDPA. Write `prompts.json` and `DATA_LICENSES.md`. | Both models generate finite logits on Kaggle, and the vocabulary checks pass |
| 2. Rule and tiny models | `sampling.py`, `drafts.py`, `speculative.py` without a cache. Experiments 1 and 2 (in a CPU-only Kaggle session or on a laptop, to save GPU quota). | Both tests pass, with the required rejection-branch coverage |
| 3. Cache | Add KV cache and rollback to the baseline and speculative loop. | Experiment 2 still passes with the cache (including the per-round cache test), and experiment 3 passes on real models |
| 4. Measure and run | Experiments 4 and 5, `run.py`, `--quick` mode. | `--quick` runs end to end from the Kaggle notebook, and c is measured |
| 5. Report | Experiment 6, the generated report, the top-level README. | One full "Save & Run All" run, with results committed to the repository |

---

## 11. Risks and plans

| Risk | Plan |
|---|---|
| Neural draft is slower than the baseline (c too high) | Likely, given published numbers. This is a result: explain it with α > c and the verification cost, and compare with the copy draft |
| Copy draft loses on every task | Still a result. Report its match rate and α, and explain with the formula |
| Qwen's default `generation_config` (sampling on, different repetition penalties) leaks into a "greedy" run | Build every distribution from raw logits. Pass `do_sample=False` and `repetition_penalty=1.0` to any `generate()` call |
| `crop()` behaves differently between transformers versions | Pin transformers. Use only `crop(-n)` with n > 0. Assert the cache length every round |
| Float16 overflow (inf or NaN logits) | SDPA attention, `isfinite` checks, float32 probability maths. If the 3B fails, report it and fall back to running the target in float32 on its own T4 |
| Float16 near-ties break the exact-match test | Strict test in float32. In float16, the teacher-forced check allows only logged near-ties under a threshold fixed in advance |
| Cache rollback bugs | Experiment 2 on tiny models, with the per-round cache test, catches them in seconds. Keep the no-cache version as a reference |
| Tiny test models never reach the rejection path | Larger initialisation scale, different seeds, and a required minimum share of rejection rounds |
| Kaggle image changes during development (the Python 3.13 / torch 2.11 image is pending) | Pin the notebook environment once it works, pin `transformers` and `accelerate`, record all versions in every results file |
| A run pushed with the Kaggle API gets the wrong image or accelerator | Remove any `docker_image` digest from `kernel-metadata.json`, use `machine_shape: "NvidiaTeslaT4"`, and rely on the first-cell hardware check |
| Hugging Face rate-limits the download | `HF_TOKEN` from Kaggle Secrets, or attach the models from Kaggle Models |
| GPU queue at busy times | Start runs early, or push the notebook with the Kaggle API (`kaggle kernels push`) |
| GPU speed varies within or between sessions | Interleave configurations, report medians, log clocks and temperature, record the GPU name |

---

## 12. Deliverables

1. A public GitHub repository: the from-scratch implementation, both drafts, tests, `prompts.json`, pinned requirements, and the notebook.
2. A public Kaggle notebook whose saved version contains the full run and its outputs.
3. Committed results and a generated report.
4. A top-level README with: speedup by task and γ for both drafts, the theory-vs-measurement table, measured c and verification cost, a "Reproduce on Kaggle" section, and a list of differences from the paper.
5. `DATA_LICENSES.md`, listing for each source its name, link, copyright holder, licence and the item IDs used: HumanEval and GSM8K (MIT notices), MT-Bench and Spec-Bench (Apache-2.0), and the models (3B under the Qwen Research License, 0.5B under Apache-2.0, weights not redistributed). It states that the committed 3B outputs fall under the Qwen Research License (research or evaluation only), not under the repository's MIT licence, and that "Built with Qwen" labelling applies if anyone trains a model on them.
6. Optional: a short blog post built around the findings. The title should match what the results actually show.

**Resume line (write it only after the results are in, and make the claim match them). Example shape, assuming the expected outcome:**
> Implemented speculative decoding (ICML 2023) from scratch in PyTorch with KV-cache rollback, verified exact against standard decoding. On a free T4, a zero-cost prompt-lookup draft gave Y× on Qwen2.5-3B, while a 0.5B neural draft was limited to X× by a measured cost ratio of c ≈ Z; the paper's cost model explained both. Fully reproducible from one Kaggle notebook in about an hour.

### Later extensions (not part of this plan), most informative first
1. **Compiled models** (static cache with `torch.compile`). If c is mostly software overhead, compiling should lower it sharply; Hugging Face reports up to 4× faster decoding. This is the direct test of the main finding. Note that `StaticCache` has no `crop()` in transformers 5.17, so rollback must be written by hand.
2. **Comparison with Hugging Face assisted generation and prompt lookup decoding.** Set `num_assistant_tokens=γ`, `num_assistant_tokens_schedule="constant"` and `assistant_confidence_threshold=0.0` on the **draft model's** `generation_config` (the default threshold of 0.4 stops drafting early), and pass `do_sample=False` and `repetition_penalty=1.0`.
3. Sampling at temperature 0.7 and 1.0 for timing runs.
4. Qwen2.5-1.5B target (Apache-2.0), to see how the draft-to-target size ratio changes c and the winner.
5. Qwen2.5-7B target split across both T4s. It has an output size of 152,064 and untied embeddings, and has exactly the shape of Qwen2's documented float16 overflow, so test for inf/NaN first.
