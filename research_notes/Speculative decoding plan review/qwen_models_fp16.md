# Qwen2.5-Instruct model family: compatibility facts and float16 risks on NVIDIA T4

Research date: 2026-09-29. Config values below were pulled directly from the raw `config.json`, `generation_config.json`, `tokenizer_config.json` and `model.safetensors.index.json` files on the Hugging Face Hub (main branch) on this date. Parameter counts were recomputed from the config dimensions and cross-checked against the safetensors `total_size` (exact match).

## 1. Vocab size, tokenizer size, and logit width (can 0.5B and 3B be compared token by token?)

### Takeaway
0.5B, 1.5B and 3B all have `vocab_size = 151936`, so their logits have identical width, and they share an identical tokenizer (`len(tokenizer) = 151665`). **7B is different: `vocab_size = 152064`.** A 0.5B draft paired with a 7B target produces logits of width 151936 vs 152064, so the code must slice both to a common width (151665 real tokens is the safe choice) before comparing distributions.

### Cited Findings
- Qwen2.5-0.5B-Instruct `config.json`: `"vocab_size": 151936`, `"tie_word_embeddings": true`, `"torch_dtype": "bfloat16"`, `"bos_token_id": 151643`, `"eos_token_id": 151645` — [config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json)
- Qwen2.5-1.5B-Instruct `config.json`: `"vocab_size": 151936`, `"tie_word_embeddings": true` — [config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json)
- Qwen2.5-3B-Instruct `config.json`: `"vocab_size": 151936`, `"tie_word_embeddings": true` — [config.json](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/config.json)
- Qwen2.5-7B-Instruct `config.json`: `"vocab_size": 152064`, `"tie_word_embeddings": false` — [config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json)
- `tokenizer_config.json` for 3B defines 22 added special tokens with IDs 151643 to 151664: `<|endoftext|>`=151643, `<|im_start|>`=151644, `<|im_end|>`=151645, `<|object_ref_start|>`...`<|file_sep|>` up to 151664 (also `<tool_call>`=151657, `</tool_call>`=151658, FIM tokens 151659-151662). `eos_token` = `<|im_end|>`, `pad_token` = `<|endoftext|>`, `bos_token` = null, `model_max_length` = 131072 — [tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/tokenizer_config.json)
- The `tokenizer_config.json` files of 0.5B, 3B and 7B Instruct are byte-identical (same MD5 hash when downloaded on 2026-09-29) — [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/tokenizer_config.json), [3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/tokenizer_config.json), [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json)
- The Qwen2.5 technical report says the tokenizer has "151,643 regular tokens" and that control tokens were "expanded from 3 to 22" — [Qwen2.5 Technical Report, arXiv 2412.15115](https://arxiv.org/html/2412.15115)

### Inferences
- `len(tokenizer)` = 151,643 regular + 22 special = **151,665** (IDs 0 to 151664). This is derived from the report and the added-token IDs. It was not checked by running `len(tokenizer)`, but the numbers agree.
- In the 0.5B/1.5B/3B models, logit rows 151665 to 151935 (271 rows) are padding rows. In the 7B model the padding rows are 151665 to 152063 (399 rows). They do not correspond to any token. They normally get very low probability, but they are not guaranteed to be exactly zero. Recommendation: slice `logits[..., :151665]` (or `:len(tokenizer)`) for both models before softmax. This makes the widths match for any draft/target pair and stops unused IDs from ever being sampled or counted in acceptance ratios.
- 0.5B to 3B: widths match (151936), so the code works without slicing. 0.5B to 7B: the widths differ (151936 vs 152064), so direct `p/q` elementwise ops would raise a shape error.

### Gaps
- `len(tokenizer)` was not executed live. The value 151665 comes from the added-token IDs plus the report's regular-token count.

## 2. Architecture: tied embeddings, layers, hidden size, heads, KV heads

### Takeaway
0.5B, 1.5B and 3B use tied embeddings (lm_head = embed_tokens). 7B does not. All use GQA with QKV bias, RoPE theta 1e6, RMSNorm eps 1e-6 and SwiGLU. 0.5B and 3B have different head_dim (64 vs 128), which does not matter for speculative decoding because only logits are compared.

### Cited Findings
| Model | layers | hidden | intermediate | Q heads | KV heads | head_dim | tied | vocab_size | max_pos | max_window_layers | sliding_window | use_sliding_window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5B-Instruct | 24 | 896 | 4864 | 14 | 2 | 64 | true | 151936 | 32768 | 21 | 32768 | false |
| 1.5B-Instruct | 28 | 1536 | 8960 | 12 | 2 | 128 | true | 151936 | 32768 | 21 | 32768 | false |
| 3B-Instruct | 36 | 2048 | 11008 | 16 | 2 | 128 | true | 151936 | 32768 | 70 | 32768 | false |
| 7B-Instruct | 28 | 3584 | 18944 | 28 | 4 | 128 | false | 152064 | 32768 | 28 | 131072 | false |

Sources: [0.5B config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json), [1.5B config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json), [3B config](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/config.json), [7B config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json). head_dim = hidden / Q heads, computed.

- The model cards say "Architecture: transformers with RoPE, SwiGLU, RMSNorm, Attention QKV bias and tied word embeddings" for 0.5B/1.5B/3B. They list "Context Length: Full 32,768 tokens and generation 8192 tokens". The 7B card lists "Full 131,072 tokens", but its "current config.json is set for context length up to 32,768 tokens" (YaRN is needed beyond that) — [0.5B card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [3B card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [7B card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Technical report Table 1 lists tied embeddings for 0.5B, 1.5B and 3B, and none for 7B and above. It matches the layer and head numbers above — [arXiv 2412.15115](https://arxiv.org/html/2412.15115)
- All configs: `rope_theta` 1000000.0, `rms_norm_eps` 1e-06, `hidden_act` silu, `architectures` Qwen2ForCausalLM, `model_type` qwen2, `transformers_version` 4.43.1 — configs above.

### Inferences
- KV cache size in fp16 = layers x 2 (K,V) x KV heads x head_dim x 2 bytes:
  - 0.5B: 24x2x2x64x2 = 12,288 B/token (about 12 KiB)
  - 3B: 36x2x2x128x2 = 36,864 B/token (about 36 KiB); 2k tokens is about 72 MiB, so it is negligible on T4.
  - 7B: 28x2x4x128x2 = 57,344 B/token.
- Because of the tied embeddings, 0.5B's embedding matrix (136.1M params) is about 28% of its 494M parameters. The draft's lm_head matmul (896 x 151936) is a noticeable share of its per-token cost.
- License caveat: the 3B card says `license: other` / `qwen-research` (Qwen Research License). 0.5B, 1.5B and 7B are Apache-2.0 ([3B card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [0.5B card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)). This is fine for research, but worth stating in the write-up.

### Gaps
- None material.

## 3. EOS / stop tokens, generation_config defaults, and why they matter for a greedy baseline

### Takeaway
All four Instruct models ship `generation_config.json` with **`do_sample: true`, temperature 0.7, top_p 0.8, top_k 20**, and a repetition_penalty that **differs between draft and target: 1.1 for 0.5B/1.5B, 1.05 for 3B/7B**. The EOS list is `[151645 (<|im_end|>), 151643 (<|endoftext|>)]`. A `model.generate()` baseline that does not explicitly override these settings will sample, and it will apply a repetition penalty. The result is not true greedy and will not match a hand-written greedy or speculative loop.

### Cited Findings
- 3B `generation_config.json`: `"do_sample": true, "eos_token_id": [151645, 151643], "pad_token_id": 151643, "bos_token_id": 151643, "repetition_penalty": 1.05, "temperature": 0.7, "top_p": 0.8, "top_k": 20` — [3B generation_config.json](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/generation_config.json)
- 0.5B `generation_config.json`: the same, except `"repetition_penalty": 1.1` — [0.5B generation_config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/generation_config.json)
- 1.5B: `repetition_penalty` 1.1; 7B: `repetition_penalty` 1.05; all other fields are identical — [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/generation_config.json), [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/generation_config.json)
- `config.json` has a single `eos_token_id: 151645`, while `generation_config.json` has the list `[151645, 151643]`. The tokenizer's `eos_token` is `<|im_end|>` — configs/tokenizer_config above.

### Inferences
- **Greedy baseline:** call `generate(..., do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)`, or pass a fresh `GenerationConfig`. Otherwise:
  - (a) `do_sample=True` is inherited and the baseline is stochastic.
  - (b) Even with `do_sample=False`, `repetition_penalty` is a logits processor that transformers applies regardless of sampling. The "greedy" baseline would then differ from pure argmax over raw logits, and speculative-vs-baseline output-equality checks would fail. This is based on transformers' logits-processor design; the exact current source was not re-read.
  - (c) Transformers typically warns that temperature/top_p/top_k are set while `do_sample=False`.
- The draft and target have different default repetition penalties (1.1 vs 1.05). If the speculative loop ever calls `generate()` on either model, or reuses its generation_config, the two distributions will be distorted differently. Build distributions from raw `model(...).logits` instead.
- Stop criteria in the custom loop should check both 151645 and 151643. In chat mode the Instruct models end turns with `<|im_end|>` (151645).
- `pad_token_id` = 151643 = `<|endoftext|>`. If batching with left padding, the attention mask must be passed. Otherwise pad tokens are indistinguishable from EOS.

### Gaps
- I did not re-read the current transformers source to confirm the exact warning text or that repetition_penalty is applied under greedy. This is standard behaviour but marked unverified here.

## 4. Chat template: default system prompt and add_generation_prompt

### Takeaway
The chat template auto-injects `<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n` when the first message is not a system message. `add_generation_prompt=True` appends `<|im_start|>assistant\n`. The draft and target share the identical template, so their prompts tokenize identically.

### Cited Findings
- Template (non-tool branch): `{%- if messages[0]['role'] == 'system' %} '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' {%- else %} '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'`. Each message renders as `'<|im_start|>' + role + '\n' + content + '<|im_end|>\n'`, and `{%- if add_generation_prompt %} '<|im_start|>assistant\n'`. There is a separate tools branch that appends a `# Tools` block — [3B tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/tokenizer_config.json)
- The tokenizer configs for 0.5B, 3B and 7B are byte-identical, so the template is also identical — same sources as section 1.

### Inferences
- Prompt lengths include about 20+ tokens of default system prompt. Account for this in latency/throughput measurements.
- Without `add_generation_prompt=True`, the model may continue the user turn or emit `<|im_start|>assistant` itself, which distorts acceptance rates.
- `bos_token` is null, so no BOS is prepended. The tokenizer does not add BOS/EOS automatically.

### Gaps
- None.

## 5. Known float16 problems with Qwen2 / Qwen2.5 and mitigations

### Takeaway
The Qwen team officially does not recommend float16 for Qwen2-family models. They cite numerical instability that shows up as `RuntimeError: probability tensor contains either inf, nan or element < 0` or repeated `!!!!` output. The documented root cause, and the fix in their patch, is **overflow of the pre-scaled Q·Kᵀ product in the eager attention path, specifically in the last layer of the 7B model (hidden 3584, 28 heads)**. Qwen2.5-7B-Instruct has exactly those dimensions. I found **no specific report of 0.5B or 3B failing in fp16 on T4**. The main risk sits with 7B. For small models the risk is lower but not ruled out, and it should be tested by checking for inf/NaN in logits.

### Cited Findings
- Qwen docs (Transformers chat page, troubleshooting item "RuntimeError: probability tensor contains either inf, nan or element < 0 or generating repeating !!!!..."): "We don't recommend using `float16` for Qwen2 models or numerical instability may occur, especially for cards without support of fp16 matmul with fp32 accumulate. If you have to use `float16`, consider using [this fork](https://github.com/jklj077/transformers/tree/qwen2-patch) and force `attn_implementation="eager"`." — [Qwen docs v2.0 source](https://qwen.readthedocs.io/en/v2.0/_sources/inference/chat.md.txt)
- The Qwen v2.5 docs say `torch_dtype="auto"` loads bfloat16, or float32 otherwise ("will need double memory"). They add: "You can also pass `torch.bfloat16` or `torch.float16` as `torch_dtype` explicitly." The float16 troubleshooting text above did not appear in the v2.5 page grep — [Qwen docs v2.5 source](https://qwen.readthedocs.io/en/v2.5/_sources/inference/chat.md.txt); [v2.5 chat page](https://qwen.readthedocs.io/en/v2.5/inference/chat.html)
- Latest Qwen docs: "`torch_dtype="auto"` will determine automatically ... For modern devices, the precision determined will be `bfloat16`. If you don't pass `torch_dtype="auto"`, the default data type is `float32`" — [Qwen docs latest, Transformers](https://qwen.readthedocs.io/en/latest/inference/transformers.html)
- The recommended fork's only change (commit 53dba62c, 2024-07-02, "temp patch of manual attention for Qwen2-7B"): `if self.layer_idx == self.config.num_hidden_layers - 1 and self.num_heads == 28 and self.hidden_size == 3584: attn_weights = torch.matmul(query_states / math.sqrt(self.head_dim), key_states.transpose(2, 3))`. It scales Q *before* the matmul, in the last layer only, for 7B-shaped models only — [jklj077/transformers qwen2-patch commit](https://github.com/jklj077/transformers/commit/53dba62c)
- Current transformers `eager_attention_forward` for Qwen2 computes `torch.matmul(query, key_states.transpose(2, 3)) * scaling` (scaling after matmul, so the fp16 overflow risk is present). It then applies `softmax(..., dtype=torch.float32)` (softmax is upcast) and RMSNorm upcasts to float32 (`hidden_states.to(torch.float32)`) — [transformers modeling_qwen2.py (main)](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen2/modeling_qwen2.py)
- PyTorch SDPA math backend: "Keep query, key, value in high precision for accuracy". Half/bf16 q, k and v are converted to float32 unless `allowFP16BF16ReductionMathSDP` is set. It also has "Scale q, k before matmul for stability", with each of q and k scaled by sqrt(scale) — [pytorch attention.cpp (main)](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/attention.cpp)
- Similar fp16 failures in the Qwen2-VL family: fp16 produces "probability tensor contains inf/nan", and `attn_implementation="flash_attention_2"` is reported as a fix — [HF discussion Qwen2-VL-7B #20](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/discussions/20). An inf in attention weights plus the -inf causal mask gives NaN; the proposed fix replaces ±inf before adding the mask — [transformers #35151](https://github.com/huggingface/transformers/issues/35151); [transformers #33294 "Qwen2-VL FP16 inference results in errors or gibberish output"](https://github.com/huggingface/transformers/issues/33294). These are VL models, not the text-only Qwen2.5 models.
- Qwen2.5-7B-Instruct-GPTQ "probability tensor contains inf/nan" report — [QwenLM/Qwen3 #951](https://github.com/QwenLM/Qwen3/issues/951) (quantized, not plain fp16)
- Qwen2.5 K-cache has outlier channels: "per-tensor e4m3 cannot hold Qwen's K outlier channels" — [third-party issue, ThePie88/vLLM-ROCm-Windows #28](https://github.com/ThePie88/vLLM-ROCm-Windows/issues/28) (low-authority source; it indicates large-magnitude K activations)
- Models are released in bf16 (`"torch_dtype": "bfloat16"` in all configs), and the report mentions bfloat16 as the original precision — configs above; [arXiv 2412.15115](https://arxiv.org/html/2412.15115)

### Inferences
- **Mitigations for this project (in order of preference):**
  1. Use `attn_implementation="sdpa"` (the default). On T4 it dispatches to the memory-efficient or math kernel, not the raw fp16 `matmul`-then-scale eager path. The math kernel upcasts to fp32 and pre-scales q/k. The memory-efficient (CUTLASS) kernel is believed to accumulate in fp32 (unverified, see section 6). SDPA is likely safer than `"eager"` in current transformers, because eager scales *after* the fp16 matmul. The Qwen docs' "force eager" advice applied only together with their patched fork.
  2. Guard every forward pass with `torch.isfinite(logits).all()`, and compute softmax/probabilities in float32: `logits.float().softmax(-1)`. For speculative decoding the acceptance ratio p/q must be computed in fp32 in any case, because fp16 probabilities underflow below about 6e-8 (subnormal) or 6e-5 (normal).
  3. If NaN/inf appears (most likely with 7B), fall back to fp32 for that model or upcast attention. Alternatively, test whether `torch.cuda.is_bf16_supported()` bf16 emulation works on T4 (slow; unverified).
- 0.5B and 3B have smaller hidden sizes and none of the 7B-specific signature. I found no report of fp16 failures for them, so they are **probably** fine in fp16. Validate this by comparing fp16 vs fp32 greedy outputs on a few prompts, and by checking the max |logit| and NaN count.
- Even without NaN, fp16 vs bf16 vs fp32 numerics change argmax at near-tie positions. Baseline and speculative runs must use the same dtype and kernel to get exact output-equality in greedy mode.
- T4 tensor cores do support fp16 multiply with fp32 accumulate. The Qwen caveat "cards without fp16 matmul with fp32 accumulate" is therefore probably aimed at other hardware. PyTorch's `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` (default True) permits reduced-precision reductions in split-K GEMMs. Setting it to False is a cheap extra safeguard (behaviour per PyTorch CUDA notes; not re-verified in this session).

### Gaps
- No primary-source report found specifically for Qwen2.5-0.5B-Instruct or Qwen2.5-3B-Instruct producing NaN or garbage in fp16 on T4/V100/Colab/Kaggle. The absence of reports is not proof of safety.
- No Qwen-team statement found that is specific to Qwen2.5 (as opposed to Qwen2) and fp16. The fp16 warning appears in the v2.0 docs, and in my grep it was not present in the v2.5 page.
- I could not confirm whether the 7B overflow also occurs under SDPA's memory-efficient kernel on T4.

## 6. Attention implementations on T4 (sm_75): FlashAttention-2, SDPA backends, sliding-window warnings

### Takeaway
On T4 (compute capability 7.5), PyTorch SDPA's flash backend is **not available** because it needs sm80 to sm121. The **memory-efficient backend is available** (sm50 to sm121), and the math backend is always available. FlashAttention-2 (`flash_attn` package) does not support Turing, so `attn_implementation="flash_attention_2"` will fail. The Qwen2 sliding-window warning is spurious for these configs (`use_sliding_window: false`) and was silenced in transformers in May 2025.

### Cited Findings
- PyTorch `sdp_utils.cpp`: "Flash attention only supports gpu architectures in the range [sm80, sm121]"; "Mem Efficient Attention only supports gpu architectures in the range [sm50, sm121]"; cuDNN MHA also only supports [sm80, sm121] — [pytorch sdp_utils.cpp (main)](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- FlashAttention-2 targets Ampere/Ada/Hopper (A100, RTX 30xx/40xx, H100). For Turing (T4, RTX 2080) there is FlashAttention 1.x, or a separate community "flash-attention-turing" project that supports a subset — [Dao-AILab/flash-attention README](https://github.com/dao-ailab/flash-attention); [PerLod tutorial (secondary)](https://perlod.com/tutorials/flash-attention-hosting/)
- The PyTorch 2.0 blog notes that some SDPA kernels support only sm_80 — [PyTorch blog](https://pytorch.org/blog/out-of-the-box-acceleration/)
- Qwen2 config in transformers: `use_sliding_window: bool = False`; `sliding_window = sliding_window if use_sliding_window else None`. Layers are marked `"sliding_attention"` only if `sliding_window is not None and i >= max_window_layers` — [configuration_qwen2.py (main)](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen2/configuration_qwen2.py). With `use_sliding_window: false` in all four configs, every layer uses full attention.
- The warning "Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered" used to fire even when `use_sliding_window` was false. PR #36316 made it display only when sliding window is actually enabled; merged 2025-05-12 — [transformers PR #36316](https://github.com/huggingface/transformers/pull/36316); related: [transformers #36351](https://github.com/huggingface/transformers/issues/36351), [unsloth #2177](https://github.com/unslothai/unsloth/issues/2177), [DeepSeek-R1-Distill-Qwen-1.5B discussion #27](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/discussions/27)

### Inferences
- On Kaggle T4, keep `attn_implementation="sdpa"`. Do not use `flash_attention_2`: it errors on sm75, and the "flash_attention_2 fixes fp16 NaN" advice from the Qwen2-VL threads cannot be used here.
- If an older transformers version (before about 4.52) is pinned, the sliding-window warning may appear. It is harmless for these configs.
- A custom speculative loop that passes explicit 4D float masks, or non-causal masks for multi-token verification, may push SDPA off the memory-efficient kernel onto the math kernel, depending on alignment and mask dtype. That would be slower but numerically safer (fp32 upcast).

### Gaps
- Not verified: whether the memory-efficient kernel accumulates Q·Kᵀ in fp32 for fp16 inputs on sm75 (believed yes for the xFormers/CUTLASS kernel), and which PyTorch version first made the math-backend fp32 upcast the default. I only confirmed that current main does it.

## 7. Memory footprint (exact parameter counts)

### Takeaway
Qwen2.5-3B = **3,085,938,688 params**: 6.17 GB (5.75 GiB) in fp16, 12.34 GB (11.50 GiB) in fp32. Qwen2.5-0.5B = **494,032,768 params**: 0.99 GB fp16, 1.98 GB fp32. 3B fp16 + 0.5B fp16 is about 7.2 GB and fits easily on a 16 GB T4. **7B in fp16 (15.23 GB) does not fit on one T4** and needs 2xT4 or quantization.

### Cited Findings
- Model cards: 0.5B "Number of Parameters: 0.49B, Non-Embedding: 0.36B"; 1.5B "1.54B / 1.31B"; 3B "3.09B / 2.77B"; 7B "7.61B / 6.53B" — [0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- 3B `model.safetensors.index.json` `"total_size": 6171877376` bytes (bf16, so 3,085,938,688 params) — [3B index](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/model.safetensors.index.json)
- 7B `model.safetensors.index.json` `"total_size": 15231233024` bytes, so 7,615,616,512 params — [7B index](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/model.safetensors.index.json)
- 0.5B single `model.safetensors` file size 988,097,824 bytes (X-Linked-Size header, including the safetensors header) — [0.5B file](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors)
- Qwen docs: without `torch_dtype="auto"` the default is float32, "which will take double the memory" — [Qwen docs latest](https://qwen.readthedocs.io/en/latest/inference/transformers.html)

Exact counts computed from the config dimensions (they match the safetensors totals exactly for 3B and 7B):
| Model | Total params | Embedding params | fp16 bytes | fp16 GiB | fp32 GiB |
|---|---|---|---|---|---|
| 0.5B | 494,032,768 | 136,134,656 | 988,065,536 | 0.92 | 1.84 |
| 1.5B | 1,543,714,304 | 233,373,696 | 3,087,428,608 | 2.88 | 5.75 |
| 3B | 3,085,938,688 | 311,164,928 | 6,171,877,376 | 5.75 | 11.50 |
| 7B | 7,615,616,512 | 544,997,376 (x2, untied) | 15,231,233,024 | 14.19 | 28.37 |

### Inferences
- T4 has 16 GB (about 15 GiB usable on Kaggle). 3B in fp32 (11.5 GiB) plus 0.5B fp32 (1.84 GiB) is about 13.3 GiB before activations and KV. This fits only barely, and the 151936-wide logits held in fp32 over long sequences add up quickly (1 token x 151936 x 4 B = 0.58 MiB per position per model). An fp32 fallback for the target is feasible for short runs only.
- 7B requires `device_map="auto"` across Kaggle's 2xT4 (naive pipeline parallelism, so one GPU is idle at a time, per Qwen docs) or 8-bit/4-bit quantization. Both change the numerics and latency relative to the fp16 3B experiments.
- Logits memory: when verifying k draft tokens, the target returns (k+1) x 151936 logits. In fp16 that is about 0.29 MiB per position, which is negligible. However, `model(input_ids)` over a full prompt returns logits for every prompt position (e.g. 1000 x 151936 x 2 B = 290 MiB). Use `logits_to_keep` / slice only the needed positions when possible.

### Gaps
- T4 usable VRAM on Kaggle (about 15 GiB) comes from general knowledge and was not re-sourced in this session.
