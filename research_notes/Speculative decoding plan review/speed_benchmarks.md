# Speed benchmarks: small Qwen models, T4-class GPUs, speculative / assisted / prompt-lookup decoding (HF transformers)

Research date: 2026-09-29. 15 tool calls. Some pages could not be fetched (Medium T4 benchmark returned 403; HF "universal-assisted-generation" blog URL returned 404, so its GitHub markdown source was used instead). No per-token numbers measured on a T4 for Qwen2.5-3B/0.5B in HF transformers were found. The T4 figures below are **estimates** built from A100 HF numbers and memory bandwidth.

## Q1. Tokens/s for Qwen2.5-3B and Qwen2.5-0.5B (HF transformers, batch 1)

### Takeaway
Official Qwen numbers for HF transformers on an **A100** are only 30.8 tok/s (3B) and 47.4 tok/s (0.5B) at batch 1. The 7B model (40.4 tok/s) is *faster* than the 3B, so HF eager decoding of these small models is limited by per-layer overhead, not by the GPU. A T4 estimate of 20-25 tok/s for the 3B is plausible. However, the 0.5B will probably **not** run 2-3x faster than the 3B in HF eager mode.

### Cited Findings
- Qwen2.5 official speed benchmark, HF transformers BF16, batch 1, generating 2048 tokens. Setup: NVIDIA A100 80GB, CUDA 12.1, PyTorch 2.3.1, Flash Attention 2.5.8, Transformers 4.46.0 (Qwen2.5 docs, 2024). — [Qwen speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)

  | Model | Input length 1 (tok/s) | Input length 6144 (tok/s) |
  |---|---|---|
  | 0.5B | 47.40 | 47.45 |
  | 1.5B | 39.68 | 40.88 |
  | 3B | 30.80 | 32.20 |
  | 7B | 40.38 | 38.76 |

- The same page reports vLLM 0.6.3 on A100: 0.5B = 311.55 tok/s and 3B = 127.61 tok/s. That is 6.6x and 4.1x faster than HF for the same GPU and batch 1, which shows how much of HF's time is framework overhead. — [Qwen speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
- The T4 has no BF16 tensor cores; FP16 is the precision to use on a T4. (Blog dated June 2026, vLLM 0.15.1, Qwen2.5-0.5B.) — [Bear the Tokens blog](https://www.dhruvvakharwala.dev/blogs/bear-the-tokens-qwen-t4-throughput)
- The only T4 Qwen2.5-0.5B throughput figures found are **batched** vLLM figures (concurrency 50): AWQ INT4 about 4,084 tok/s, FP16 with suffix decoding 9,052 tok/s (7,800 on a verified re-run). They do not apply to batch-1 HF. — [Bear the Tokens blog](https://www.dhruvvakharwala.dev/blogs/bear-the-tokens-qwen-t4-throughput)
- A Medium post benchmarks Qwen models on T4, L4 and H100, but it could not be fetched (HTTP 403), so its numbers are not verified. — [Medium, Wltsankalpa](https://medium.com/@wltsankalpa/benchmarking-qwen-models-across-nvidia-gpus-t4-l4-h100-architectures-finding-your-sweet-spot-a59a0adf9043)
- No T4 HF batch-1 tok/s numbers for Llama-3.2-3B were found. — search result only; no source

### Inferences
- **Estimate: bandwidth floor on a T4.** Qwen2.5-3B has about 3.1B params, so about 6.2 GB of FP16 weights. At 320 GB/s the lower bound is about 19 ms/token, a ceiling of about 52 tok/s. At a typical 60-75% of achievable bandwidth that becomes roughly 25-33 ms/token, or about 30-40 tok/s from GPU time alone.
- **Estimate: overhead on Kaggle.** On an A100 the HF 3B takes 32 ms/token, almost all of it CPU/launch overhead (the A100 bandwidth floor is about 3 ms). Kaggle's CPUs are weaker than an A100 host, so the overhead there is probably ≥30 ms/token. Because CUDA runs asynchronously, time per step is roughly the larger of CPU time and GPU time. **Estimated T4 3B HF eager speed: about 18-30 tok/s.** The plan's 20-25 tok/s is reasonable, but it should be measured in the first minutes of the run.
- **Estimate: 0.5B on a T4.** Its weights are about 1 GB, a 3 ms floor, so it is fully overhead-bound: roughly 20-25 ms/token, or 40-50 tok/s, similar to the A100 HF figure.
- **Implication for the cost ratio c.** From the A100 HF numbers, c = t(0.5B)/t(3B) = 30.80/47.40 ≈ **0.65**. This is consistent with per-token latency scaling with layer count rather than parameters (Qwen2.5-0.5B has 24 layers and 3B has 36; 24/36 = 0.67). The layer counts come from the model configs and were not re-fetched in this session. On a T4 the 3B's GPU time is a larger share, so c may be a little lower, **around 0.5-0.7**. The plan's range of 0.3-0.6 is at risk of being optimistic at the low end. The plan should measure c directly (time the draft step and the target step separately).

### Gaps
- No measured Qwen2.5-3B, 0.5B, Llama-3.2-3B or Phi numbers were found for T4 + HF transformers + FP16 + batch 1.

## Q2. Overhead-bound small models; effect of torch.compile and static cache

### Takeaway
The evidence strongly suggests that HF eager decoding of models of 3B and smaller at batch 1 is overhead-bound. HF documents static cache plus torch.compile as giving "up to a 4x" speedup, with smaller gains for larger models.

### Cited Findings
- HF docs: "The static kv-cache ... you can combine it with torch.compile for up to a 4x speed up. Your speed up may vary depending on the model size (larger models have a smaller speed up) and hardware." They recommend `cache_implementation="static"` and `torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)`. The first calls are slow because of compilation, and changing the batch size or max length triggers recompilation. — [HF LLM optimisation docs](https://huggingface.co/docs/transformers/main/en/llm_optims)
- In the Qwen benchmark, the HF 7B (40.4 tok/s) is faster than the HF 3B (30.8 tok/s) on an A100, and vLLM is 4-6.6x faster than HF on the same GPU. — [Qwen speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)

### Inferences
- In eager mode the 0.5B draft costs almost as much per step as the 3B target, which makes a draft model a poor choice there. If torch.compile or CUDA graphs were used for the draft only, its step time could fall several-fold and c could drop toward 0.1-0.2 (**estimate**). However, HF assisted generation with a compiled or static-cache assistant may not be supported; this needs checking against the installed transformers version.
- For a from-scratch loop, compiling needs fixed shapes. Verifying k+1 tokens changes the input length, which triggers recompiles unless k is fixed and padded.

### Gaps
- No measured torch.compile speedups on a T4 for Qwen2.5-0.5B or 3B were found.

## Q3. Speculative / assisted decoding speedups with Qwen2.5 pairs; acceptance rates

### Takeaway
Published results for Qwen2.5-0.5B drafting for Qwen2.5-7B show about 62% acceptance and a 2.64x speedup on code, but that was on a CPU, where the target is heavily memory-bound. No T4 results were found for 0.5B → 3B. Speedups reported by HF on GPUs are typically 1.5-2x, and those use targets of 7B and larger.

### Cited Findings
- ML-SpecQD (arXiv 2503.13565, 2025): Qwen2.5-0.5B draft → Qwen2.5-7B BF16 target, HF transformers assisted decoding, 8 draft tokens, MBPP code (100 prompts). Geomean speedup **2.64x** over greedy, draft acceptance about **62%**. Hardware: **Intel Core Ultra 9 285K CPU**, DDR5-7200 (not a GPU). — [ML-SpecQD](https://arxiv.org/html/2503.13565)
- Search snippet (not verified in the full paper): consultant decoding with a Qwen2.5-0.5B draft and Qwen2.5-7B target reported 3.75x on GSM8K and 3.59x on HumanEval. It is a modified (lossy) verification scheme, so it is not comparable to standard speculative decoding. — [Consultant Decoding, arXiv 2506.02391](https://arxiv.org/pdf/2506.02391)
- HF Universal Assisted Generation blog. Rows that use a Qwen draft: Qwen2-0.5B-Instruct drafting for Phi-3-medium-128k (14B) on tau/scrolls gave **1.91x** on a single A6000. Qwen2-0.5B drafting for Llama-3.1-70B (2x A100) and for Mixtral-8x22B (4x A100) both gave **1.78x**. Other rows: CodeLlama-13B + tiny_starcoder on HumanEval 1.90x; gemma-2-9b + vicuna-68m on CNN/DM 1.76x. — [HF blog source (GitHub)](https://github.com/huggingface/blog/blob/main/universal_assisted_generation.md)
- HF assisted generation blog (2023, RTX 3090): up to 2x speedup for greedy without quantisation (3x with INT8). The assistant should be "at least an order of magnitude smaller" than the main model. Speedups shrink with sampling at high temperature. Assisted generation is limited to batch size 1. — [HF assisted generation blog](https://huggingface.co/blog/assisted-generation)
- HF dynamic speculation (Intel + HF): up to 2.7x depending on task. Dynamic lookahead has been the default mode for assisted decoding since transformers **4.45.0**. — [HF dynamic speculation blog](https://huggingface.co/blog/dynamic_speculation_lookahead) (via search summary)
- Spec-Bench, Vicuna-7B-v1.3, FP16, batch 1, PyTorch 2.5.1, CUDA 12.1. SpS (standard draft-model speculative sampling, 68M draft) gives mean accepted tokens per step of **2.28** and speedup of **1.79x on an RTX 3090** / 1.52x on an A100. Per-task on the 3090: conversation 1.94x, translation 1.37x, summarisation 1.96x, QA 1.86x, math 1.81x, RAG 1.83x. — [Spec-Bench leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)

### Inferences
- Using the standard formula speedup = (1−α^(k+1)) / ((1−α)(c·k+1)), with **c ≈ 0.65** (the eager HF estimate above):
  - α = 0.7, k = 4 gives about 0.77x, a **slowdown**.
  - α = 0.8, k = 4 gives about 0.93x.
  - Even α = 0.9, k = 4 gives about 1.14x.

  With c = 0.3 and α = 0.8, k = 4, the speedup is about 1.53x. **The 0.5B → 3B result on a T4 in eager HF is likely to be about 1x or a slowdown.** The project should expect this and report it as a finding. Accepted-token rates will look healthy, perhaps 60-80% on code and math, since Qwen2.5 0.5B and 3B share training data and tokenizer; this range is an estimate that follows the 62% on MBPP reported for 0.5B → 7B.
- Accepted tokens per step in Spec-Bench, which uses a 68M draft for a 7B target, is only about 2.3. The Qwen 0.5B/3B pair is better aligned, so it will likely be higher, but that cannot make up for a high c.

### Gaps
- No published acceptance rates or speedups were found specifically for Qwen2.5-0.5B → Qwen2.5-1.5B or → Qwen2.5-3B on a GPU.
- No per-task (HumanEval, GSM8K, chat) acceptance rates on a GPU were found for Qwen2.5-0.5B → 7B beyond the CPU MBPP number.

## Q4. Cases where assisted generation is slower than the baseline

### Takeaway
There are documented slowdowns even on large GPUs with an 8B/1B pair. HF itself warns that the assistant must be much smaller than the target and that low-quality drafts erase the gains.

### Cited Findings
- transformers issue #36337 (Feb 2025): A100-SXM4-80GB, PyTorch 2.2.2, transformers 4.49.0, Llama-3.1-8B target with Llama-3.2-1B assistant. Assisted generation took **31.12 ms/token versus 24.36 ms/token** for the baseline, about 28% slower. No maintainer resolution was visible. — [GH issue #36337](https://github.com/huggingface/transformers/issues/36337)
- HF: "If the assistant has poor quality, you get the cost of using the assistant model with little to no benefits." The assistant should be at least 10x smaller. — [HF assisted generation blog](https://huggingface.co/blog/assisted-generation)
- Whisper speculative decoding on a T4 was faster only up to batch size 4; above batch size 4 it was slower than the main model alone. — [HF Whisper spec-dec blog](https://github.com/huggingface/blog/blob/main/whisper-speculative-decoding.md)
- Spec-Bench on an A100: Lookahead decoding on translation gave only 1.00-1.14x; PLD on translation gave 1.06x. — [Spec-Bench leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)
- The transformers issue tracker also has "Speculative decoding is surprisingly slow on Whisper-large-v3" (#32366); it was seen in search results but its contents were not fetched. — [GH issue #32366](https://github.com/huggingface/transformers/issues/32366)

### Inferences
- An 8B/1B pair has a parameter ratio of 8, and 3B/0.5B has a ratio of 6. Both are below HF's "10x smaller" guideline, and in the overhead-bound regime the effective step-time ratio is much worse than the parameter ratio. A slowdown for 0.5B → 3B is a realistic outcome.

### Gaps
- No slowdown report specific to the T4 with a 0.5B → 3B pair was found.

## Q5. Prompt lookup decoding (PLD) speedups by task

### Takeaway
PLD gives about 2.4x on input-grounded tasks (summarisation, context QA, code editing) with a 7B model on an A100. On independent benchmarks it gives about 1.6x overall. Gains are smallest on translation, open-ended QA and roleplay, where the output overlaps little with the input. Math (GSM8K) still reached about 1.6-1.7x in Spec-Bench.

### Cited Findings
- apoorvumang/prompt-lookup-decoding: Mistral-7B-Instruct-v0.1, single A100 40GB, greedy, max n-gram 3, continuation length 10. About **2.4x** average on CNN/DailyMail summarisation (100 examples) and HAGRID context QA. On MT-Bench, turn 0 had smaller gains and turn 1 about 2.4x. Coding showed "very high gain in 2nd turn, because there is lots of code copying". Extraction had the highest first-turn gains, and roleplay was the poorest. "Throughput of PLD was always more than that of greedy (or within margin of error)." — [PLD repo](https://github.com/apoorvumang/prompt-lookup-decoding)
- Spec-Bench, Vicuna-7B-v1.3, FP16, batch 1:
  - RTX 3090: conversation (MT-bench) 1.64x, translation 1.15x, summarisation 2.46x, QA 1.28x, **math (GSM8K) 1.72x**, RAG 1.71x. Mean accepted tokens 1.73; overall 1.64x.
  - A100: 1.60 / 1.06 / 2.66 / 1.19 / 1.62 / 1.86; overall 1.66x.

  — [Spec-Bench leaderboard](https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md)
- HF exposes PLD through `generate(..., prompt_lookup_num_tokens=N)`. The docs describe it as working "especially well for input-grounded tasks - such as summarization". — [HF LLM optimisation docs](https://huggingface.co/docs/transformers/main/en/llm_optims)

### Inferences
- PLD's draft cost is essentially zero (c ≈ 0), so on the overhead-bound T4 setup it is the method most likely to show a real speedup. Rough expectations (**estimates**):
  - HumanEval: modest, about 1.2-1.6x. The model re-emits the function signature and docstring identifiers, but the prompt is short.
  - GSM8K: about 1.3-1.7x (in line with Spec-Bench math), from repeated numbers and phrases in the reasoning.
  - Open chat: about 1.0-1.4x.
- The expected mean accepted tokens per step is about 1.7, based on Spec-Bench.
- With max 128 new tokens and short prompts, there is less context to copy from than in Spec-Bench, so the results may fall at the lower end of these ranges.

### Gaps
- No PLD results specifically for Qwen2.5-3B or for T4 hardware were found.

## Q6. Time to verify k tokens in one forward pass vs 1 token

### Takeaway
No measured T4 curve was found. Theory and HF's explanation (decoding is memory-bandwidth bound) imply that verifying k ≤ 8 tokens costs nearly the same as 1 token on a T4 for a 3B model. This is an inference and should be measured.

### Cited Findings
- HF: "the bottleneck in the forward pass comes from loading the model layer weights into the computation cores of your device, not from performing the computations themselves." A single forward pass returns logits for all positions, which is what allows parallel verification. — [HF assisted generation blog](https://huggingface.co/blog/assisted-generation)

### Inferences
- **Estimate:** T4 FP16 tensor-core peak is about 65 TFLOPS and bandwidth is 320 GB/s, so the ridge point is about 200 FLOP/byte. A decode forward pass with k tokens has an arithmetic intensity of about k FLOP/byte for the weight GEMMs. So k ≤ 8 (or even about 32) stays memory/overhead-bound, and latency should be roughly flat, perhaps +5-20% at k = 8 from attention and non-GEMM ops. In the HF eager overhead-bound regime the kernel count does not change with k, which flattens the curve further. The plan should measure this directly: time a target forward with 1, 2, 4, 8 and 16 tokens.

### Gaps
- No published T4 or consumer-GPU measurement of forward latency versus number of query tokens was found for 0.5-3B models.

## Q7. Published from-scratch reproductions of speculative decoding (Leviathan et al.) on free GPUs

### Takeaway
Popular from-scratch repositories exist, but the ones checked do not publish hardware-specific speedup tables. The strongest free-GPU (T4) speculative decoding evidence is HF's Whisper blog, which found about 2x.

### Cited Findings
- romsto/Speculative-Decoding: a PyTorch implementation of Leviathan et al. 2023, plus "Ngram Assisted Speculative Decoding (NASD)". It notes that "Increasing the value of γ will not always lead to faster generation". The README gives no speedup, acceptance or hardware numbers. — [romsto repo](https://github.com/romsto/Speculative-Decoding)
- HF Whisper speculative decoding on a **T4 16GB** (Whisper large-v2 target, distil-whisper draft), batch 1: English 73 s → 33 s (**2.2x**); multilingual Dutch 117 s → 62 s (1.9x). — [HF Whisper spec-dec blog](https://github.com/huggingface/blog/blob/main/whisper-speculative-decoding.md)

### Gaps
- feifeibear/LLMSpeculativeSampling was not fetched (tool budget). No blog reproducing Leviathan et al. with Qwen on a Kaggle or Colab T4 with reported speedups was found.
