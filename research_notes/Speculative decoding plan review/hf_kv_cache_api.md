# Hugging Face transformers KV cache API and a manual speculative decoding loop (as of 2026-09-29)

Method: I read the source at release tags directly (raw.githubusercontent.com) and the GitHub release notes (GitHub API). Line numbers below refer to the **v5.17.0** tag unless stated otherwise. Main source files:
- cache_utils.py @ v5.17.0: https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py
- generation/utils.py @ v5.17.0: https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py
- generation/candidate_generator.py @ v5.17.0: https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/candidate_generator.py
- generation/configuration_utils.py @ v5.17.0: https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/configuration_utils.py
- models/qwen2/modeling_qwen2.py @ v5.17.0: https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2/modeling_qwen2.py
- masking_utils.py @ v5.17.0: https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/masking_utils.py

## Q1. Latest release, and what changed in the cache system from v4.4x to v5.x

### Takeaway
The latest release is **v5.17.0 (2026-09-09)**. Four changes matter for a hand-written draft/verify/rollback loop:
1. **v4.56** refactored the cache into per-layer objects.
2. **v5.0.0** removed the legacy tuple format for `past_key_values`.
3. **v5.4.0** removed `cache_position` from model forward signatures.
4. **v5.15.0** changed `crop()` to take a *negative number of tokens to remove*. Positive values are now deprecated and will be removed in v5.18.

Kaggle's image pins only `transformers>=5.0.0`, so the version you actually get may land on either side of these changes.

### Cited Findings
- Recent releases: v5.17.0 (2026-09-09), v5.16.1/v5.16.0 (2026-08-26), v5.15.1 (2026-08-19), v5.15.0 (2026-08-10), v5.14.0 (2026-07-15), v5.13.0 (2026-07-03), v5.5.0 (2026-04-02), v5.4.0 (2026-03-27), v5.3.0 (2026-03-04), v5.2.0 (2026-02-16). — [GitHub releases](https://github.com/huggingface/transformers/releases)
- v5.0.0 release notes list "🚨🚨 Remove all traces of legacy cache format (#41378)". I confirmed that `from_legacy_cache` / `to_legacy_cache` no longer exist in v5.0.0 `cache_utils.py`; v4.57.1 still had both. v5.0.0 also made `generate` "delegate default cache initialization to the model" (#41505) and added `__iter__` to DynamicCache (#41569). — [v5.0.0 release](https://github.com/huggingface/transformers/releases/tag/v5.0.0); [cache_utils v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)
- In v5.17, `Cache` is "mostly a list of `CacheLayerMixin` objects, one per model layer". The concrete layer classes are `DynamicLayer`, `DynamicSlidingWindowLayer`, `DynamicIndexedLayer`, `StaticLayer`, `StaticSlidingWindowLayer`, `QuantizedLayer` and several linear-attention/hybrid layers. The cache-level classes are `DynamicCache`, `StaticCache`, `QuantizedCache`, `EncoderDecoderCache`, `MtpCache` and `DFlashCache`. — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- `DynamicCache(ddp_cache_data=None, config=None, offloading=False, offload_only_non_sliding=False)`. If you pass `config`, the constructor creates each layer type from `config.layer_types`. Without `config`, layers are created lazily as `DynamicLayer`. Iterating the cache yields `(keys, values, sliding_window_tensor_or_None)` per layer. — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- Deprecations in v5.17: `get_max_cache_shape`, the `max_cache_len` property and `max_batch_size` all warn "will be removed in version 5.16" (still present in 5.17), with `get_max_length()` / `batch_size` as replacements. — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- v5.4.0: "The `cache_position` argument has been removed from the forward signatures of most major models" (#44181 "🚨 Completely remove cache positions"; #44816 "[generate] Never use `cache_position` anymore in generation"). Qwen2's `modeling_qwen2.py` contains 13 occurrences of `cache_position` in v5.0.0–v5.3.0 and 0 from v5.4.0 on. — [v5.4.0 release](https://github.com/huggingface/transformers/releases/tag/v5.4.0)
- v5.15.0: "🚨 [cache] Cropping can only be done with negative values (#47720)". The same release added "[cache] Allow sliding window layers to be roll-backed for speculative decoding (#47447)" and "[generate] Stop setting the static cache as an attribute to save memory (#47731)". — [v5.15.0 release](https://github.com/huggingface/transformers/releases/tag/v5.15.0)
- v5.16.0: "[docs] Cache crop (#47950)", "Fix sliding window cache index off-by-one on wraparound (#47708)", a Whisper speculative-decoding cache-corruption fix (#48000), and multi-token-prediction speculative decoding via `generate(use_mtp=True)`. — [v5.16.0 release](https://github.com/huggingface/transformers/releases/tag/v5.16.0)
- Kaggle's `kaggle_requirements.txt` (main branch, last commit 2026-09-28) specifies `transformers>=5.0.0`. The image is built on the Colab base `release-colab-external-images_20260917-060051_RC00`. — [Kaggle/docker-python kaggle_requirements.txt](https://github.com/Kaggle/docker-python/blob/main/kaggle_requirements.txt); [Dockerfile.tmpl](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl)

### Inferences
- Because Kaggle pins only `>=5.0.0`, the installed version depends on the image build date. A mid-September 2026 image most likely has 5.15.x–5.17.x, but that is **unverified**. The notebook should print `transformers.__version__` and ideally pin it, for example `pip install transformers==5.17.0`.
- Code written for v4.x (tuple caches, `cache_position=` arguments, `crop(absolute_len)`) is fragile on current Kaggle.

### Gaps
- I did not verify the exact transformers version preinstalled in the current Kaggle GPU image, because the requirement is a lower bound only.
- No v5.18 release exists yet, so exactly how positive `crop` arguments will behave after removal (raise an error or be reinterpreted) is unknown.

## Q2. `DynamicCache.crop`, `get_seq_length`, and how HF assisted generation rolls back the cache

### Takeaway
`crop` still exists, but **its argument meaning depends on the version**:
- **v4.x–v5.14:** `crop(max_length)` takes an absolute target length, or a negative value to remove that many tokens.
- **v5.15+:** `crop(tokens_to_remove)` takes a negative count of tokens to remove. A positive value is a deprecated absolute-length path that logs a warning.
- **`crop(0)` differs by version:** it **empties the cache** in v4.x through v5.13, and is a **no-op** in v5.14+.

The portable rule is `if n_reject > 0: cache.crop(-n_reject)`. HF's own `_assisted_decoding` now calls `outputs.past_key_values.crop(-(candidate_length - n_matches))`, and the draft cache is trimmed with `crop(-tokens_to_remove)`. The old `_crop_past_key_values` helper is gone.

### Cited Findings
- **v5.17 `DynamicLayer.crop`.** The method is decorated `@deprecate_kwarg("max_length", new_name="tokens_to_remove", version="5.18")`. The code:
  - `if tokens_to_remove > 0:` logs a warning ("Calling `crop` with a positive value is deprecated and will be removed in version 5.18...") and treats the value as an absolute size. If that size is at least the current length it returns; otherwise `tokens_to_remove = seq_len - value`.
  - `if tokens_to_remove == 0: return`.
  - Otherwise it runs `self.keys = self.keys[..., : -abs(tokens_to_remove), :]` (same for values).

  — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- `Cache.crop(tokens_to_remove)` loops over all layers. Its docstring says `crop(0)` "will not necessarily always be a no-op": for sliding-window and linear-attention layers it can drop states that are no longer needed. — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- `Cache.is_croppable` is described as "Whether `crop` can put the whole cache back as it was, so a rollback leaves no trace". `CacheLayerMixin.is_croppable = False` and `DynamicLayer.is_croppable = True`. `StaticLayer` defines no `crop` method at all. — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- **v4.57.1 `DynamicLayer.crop(max_length)`.** `if max_length < 0: max_length = seq_len - abs(max_length)`; `if seq_len <= max_length: return`; then slice `[..., :max_length, :]`. So `crop(0)` slices to `[:0]` and **empties the cache**. — [cache_utils.py v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)
- v5.13.0 used `< 0`, the same as v4.57. v5.14.0 and v5.14.1 changed it to `if max_length <= 0:`, which makes `crop(0)` a no-op. A commenter on issue #47433 attributed this to a rebase artifact in PR #47347. That issue also reported that `crop(-5)` on a 3-token cache left 1 stale token in the old code. The maintainer (Cyrilvallez) replied that "cropping to positive values, including zero, will very soon be deprecated and removed" and pointed to PR #47720. — [Issue #47433](https://github.com/huggingface/transformers/issues/47433); [v5.14.0 cache_utils](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/cache_utils.py)
- Official docs (main): "Use crop() to roll back tokens from a cache, such as when a candidate continuation is rejected... Pass the number of tokens to remove as a negative integer", e.g. `past_key_values.crop(-3)`. The docs also say "`crop(0)` does not clear the cache". Sliding-window and linear-attention layers need `activate_past_recording()` before the forward pass; otherwise `crop` raises `RuntimeError` once the window is full. — [HF docs: Cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)
- `get_seq_length(layer_idx=0)`:
  - `DynamicLayer` returns `keys.shape[-2]` as an int, or 0 if the layer is uninitialized.
  - `StaticLayer` returns `self.cumulative_length`, which is a **0-dim tensor**, not an int.
  - The cache-level method returns 0 when `layer_idx >= len(self.layers)`.

  — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- Other cache methods in v5.17: `reset()`, `reorder_cache(beam_idx)`, `batch_repeat_interleave`, `batch_select_indices`, `activate_past_recording()`, `get_max_length()`, `get_mask_sizes()`, `get_query_offset()`. I found no truncation API other than `crop`. — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- **HF target rollback in v5.17 `_assisted_decoding`:**
  - `number_of_tokens_to_crop = candidate_length - n_matches`, then `outputs.past_key_values.crop(-number_of_tokens_to_crop)`.
  - The call is deliberately made even when the value is 0 ("crop will still take care of shrinking back sliding window or linear attention cache layers").
  - After a verify forward over `candidate_length + 1` tokens (the last accepted token plus the drafts), cropping `candidate_length - n_matches` keeps the KV for all accepted drafts. The newly emitted correction/bonus token has **no** KV yet; it is fed as input on the next iteration.

  — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- **HF draft rollback in v5.17** (`AssistedCandidateGenerator._update_past_and_masks`):
  - `target_cache_size = input_ids.shape[-1] - 1 - remove_from_pkv - num_added_tokens`
  - `tokens_to_remove = current - target`
  - `if tokens_to_remove >= 0: crop(-tokens_to_remove)`

  With the defaults this keeps KV for all but the last token of the accepted sequence. — [candidate_generator.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/candidate_generator.py)
- v4.57.1's candidate_generator still had a `_crop_past_key_values(model, past_key_values, max_length)` helper (it calls `past_key_values.crop(max_length)` for Cache objects). In v5.17 there are no occurrences of that name. — [candidate_generator.py v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/candidate_generator.py)

### Inferences
- **Portable rollback idiom** for DynamicCache, on every version from v4.4x to v5.17 and presumably v5.18+:
  ```python
  n_reject = gamma - n_accepted          # tokens of KV that must go
  if n_reject > 0:
      cache.crop(-n_reject)
  assert cache.get_seq_length() == expected_len   # cheap invariant check
  ```
  Never call `crop(-0)`: on v4.x–v5.13 it wipes the entire cache silently. Avoid `crop(absolute_len)` too: it warns on 5.15–5.17 and is slated for removal in 5.18.
- The bookkeeping invariant that matches HF: after a round, the target cache holds KV for `prefix + accepted drafts`, and the bonus/correction token is the pending input for the next forward. The draft cache should be cropped to the same committed length, then fed whatever tokens it has not yet seen (the bonus token, and possibly the last accepted draft token if the draft never processed it). A common off-by-one: when all γ drafts are accepted, the draft model never ran a forward on draft token γ, so its cache is missing that one token.
- Qwen2.5-3B and 0.5B both have `use_sliding_window: false`, so all layers are plain `DynamicLayer` and `activate_past_recording()` is not needed. — [Qwen2.5-3B-Instruct config.json](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)

### Gaps
- Whether v5.18 will make `crop(positive)` raise an error or reinterpret it is not public.

## Q3. `cache_position`, `position_ids` and `attention_mask` for a multi-token forward with an existing cache (Qwen2)

### Takeaway
- **v5.4+:** Qwen2 `forward` has **no `cache_position` parameter**. Passing only `input_ids` and `past_key_values` is correct for batch size 1 without padding: positions are computed as `arange(q_len) + cache.get_seq_length()`, and the causal mask offset also comes from the cache length.
- **v4.4x–v5.3:** `cache_position` exists but is derived the same way when omitted.

So for batch size 1 with no padding, omitting all three arguments is correct in every version, *provided the cache length exactly equals the number of tokens already committed*.

### Cited Findings
- **v5.17 `Qwen2Model.forward` signature:** `(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, **kwargs)`, with no `cache_position`. — [modeling_qwen2.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2/modeling_qwen2.py)
- **Default cache and positions in v5.17:**
  - `if use_cache and past_key_values is None: past_key_values = DynamicCache(config=self.config)`.
  - `if position_ids is None: past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0; position_ids = torch.arange(inputs_embeds.shape[1]) + past_seen_tokens`.

  — [modeling_qwen2.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2/modeling_qwen2.py)
- **Mask construction in v5.17:**
  - `create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids)` calls `_preprocess_mask_arguments`.
  - When a cache is present, `q_offset = past_key_values.get_query_offset(layer_idx)`, which equals `get_seq_length()`, and `kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)`. For `DynamicLayer` this gives `kv_length = seq_len + q_len` and `kv_offset = 0`.
  - The mask is therefore derived from the cache length, not from `position_ids`.
  - A 2D `attention_mask`, if passed, must have shape `(batch, past + q_len)`. A 4D mask is returned as-is.

  — [masking_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/masking_utils.py)
- `generate` keeps a backward-compatibility path that builds `cache_position = arange(seq_len) + past_seen_tokens` only for remote-code models whose forward still declares `cache_position`, and warns that it "will be removed in a future release". — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- The `@merge_with_config_defaults` decorator on `Qwen2Model.forward` fills `use_cache` from `config.use_cache` when it is not passed. — [utils/generic.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/utils/generic.py)

### Inferences
- For batch size 1 with no padding on v5.17: `out = model(input_ids=new_tokens, past_key_values=cache, use_cache=True)` is enough. Explicit `position_ids=torch.arange(L, L+k)[None]` is harmless if it matches. A **wrong** explicit `position_ids` would give wrong RoPE without any error, because the mask ignores it.
- If rollback is done incorrectly (the cache holds stale extra tokens), both RoPE positions and the mask shift together. The output stays plausible but differs from the reference, so the only symptom is a greedy mismatch against the baseline. Assert `cache.get_seq_length() == committed_len` every round.
- Passing `cache_position=` to a v5.4+ Qwen2 model: it lands in `**kwargs`. The SDPA attention function accepts `**kwargs`, so it is probably ignored silently (**unverified**). Do not rely on it; remove it.
- An `attention_mask` of all ones on batch size 1 is also fine, but it must grow by exactly `k` per forward and shrink on rollback. Omitting it is simpler and avoids length-mismatch bugs.

### Gaps
- I did not run code to confirm the exact behaviour of a stray `cache_position` kwarg on v5.17.
- I did not re-read `modeling_qwen2.py` for v4.4x–v5.3 in detail. From memory, when `cache_position` is `None` it was computed as `arange(past_seen, past_seen + q_len)` and `position_ids = cache_position.unsqueeze(0)`; this matches the v5.3 code count but the snippet itself is **unverified**.

## Q4. Does `forward` return logits for all positions?

### Takeaway
Yes, in a manual forward. `Qwen2ForCausalLM.forward(..., logits_to_keep: int | Tensor = 0)` slices `hidden_states[:, -0:]`, which is all positions. Only `generate()` sets `logits_to_keep=1` (and `candidate_length + 1` in assisted decoding). Logits are not upcast to fp32.

### Cited Findings
- `slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep; logits = self.lm_head(hidden_states[:, slice_indices, :])`, with the comment "Only compute necessary logits, and do not upcast them to float if we are not computing the loss". — [modeling_qwen2.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen2/modeling_qwen2.py)
- `generate`: "If the model supports `logits_to_keep` in forward(), set it to 1". `_assisted_decoding` sets `model_inputs["logits_to_keep"] = candidate_length + 1` and then casts `new_logits = outputs.logits[:, -candidate_length-1:].to(torch.float32)`. — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- v5.0.0 release notes: "Add `logits_to_keep` to many older CausalLM models (#41335)". — [v5.0.0 release](https://github.com/huggingface/transformers/releases/tag/v5.0.0)

### Inferences
- In the verify pass (γ+1 tokens), the default returns all γ+1 logits, which is what is needed. Passing `logits_to_keep=gamma+1` would matter only if extra prefix tokens were included in the input.
- In prefill, pass `logits_to_keep=1`. Otherwise the model materializes a `[1, prompt_len, 151936]` fp16 tensor, about 0.3 MB per token, or roughly 300 MB for a 1k-token prompt. That is wasted memory and time on a T4.
- Logits come back in the model dtype (fp16 on T4). Argmax is unaffected by the fp32 cast HF applies, but softmax for sampling-mode acceptance tests should be done in fp32.
- Both Qwen2.5-3B and 0.5B have `vocab_size: 151936` and `tie_word_embeddings: true`, so draft and target logits line up index-for-index. — [3B config](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json); [0.5B config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)

### Gaps
- None significant.

## Q5. StaticCache and torch.compile: can it be rolled back? Status on T4

### Takeaway
In v5.17 `StaticCache` **cannot be cropped**: `StaticLayer` has no `crop` method and `is_croppable` is False. Rollback would require manually decrementing each layer's `cumulative_length` tensor. That looks sound from the code, because the mask offset comes from `cumulative_length` and stale slots get overwritten, but it is an **unverified hack**. For this project DynamicCache is the right choice.

### Cited Findings
- `StaticLayer`:
  - It preallocates `keys`/`values` of shape `[B, H, max_cache_len, D]` and keeps `cumulative_length = torch.tensor(0)`, marked with `torch._dynamo.mark_static_address`.
  - `update()` writes with `index_copy_` at `arange(k) + cumulative_length`, then runs `cumulative_length.add_(k)`.
  - `get_mask_sizes` returns `kv_length = max_cache_len`, so attention always spans the full preallocated length.
  - `get_seq_length()` returns the `cumulative_length` tensor.

  — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- `CacheLayerMixin.reset()` zeroes the keys/values and runs `cumulative_length.zero_()`. `CacheLayerMixin.is_croppable = False`, and only dynamic, linear-attention and hybrid layers define `crop`. — [cache_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
- `create_causal_mask` disables the SDPA `is_causal` skip "if we are compiling and decoding" (`is_compileable and q_length == 1`). — [masking_utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/masking_utils.py)
- Docs: StaticCache "allows to easily compile the decoding stage, but it incurs a waste of tokens in the attention computation". `cache_implementation="static"` in `generate()` "will also turn on automatic compilation of the decoding stage for greedy and sample decoding strategies". — [HF docs: Cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)
- v5.15.0: "[generate] Stop setting the static cache as an attribute to save memory (#47731)". — [v5.15.0 release](https://github.com/huggingface/transformers/releases/tag/v5.15.0)

### Inferences
- Possible manual static rollback (**unverified**): `for layer in cache.layers: layer.cumulative_length.sub_(n_reject)`. Because `q_offset` equals `cumulative_length` and the causal mask hides `kv_idx > q_idx`, stale slots past the offset are masked and later overwritten. This still needs an equivalence test against the DynamicCache path.
- torch.compile with variable query lengths (prefill, a verify of γ+1, a draft of 1) triggers recompiles or needs `dynamic=True`. On a T4 (sm_75, no bf16, fp16 only), the compile overhead plus attention over the full `max_cache_len` probably cancels the gain for short benchmarks. This is my judgment and is **unverified**; I found no T4-specific benchmark.

### Gaps
- No official documentation or benchmark exists for StaticCache rollback in speculative decoding, or for torch.compile support on T4.

## Q6. Known bugs and issues with `DynamicCache.crop` and assisted generation (relevant to Qwen2)

### Takeaway
I found no Qwen2-specific crop or assisted-generation bug. The relevant issues are the version-dependent `crop(0)` / oversized-crop behaviour (#47433, fixed by the v5.15 redesign) and an open issue that `crop` does not roll back linear-attention recurrent states (#49036). The latter does not affect Qwen2, which uses pure attention.

### Cited Findings
- #47433 (closed 2026-07-20): on v5.14-era code, `DynamicCache.crop(0)` "retains the full KV cache" and `crop(-5)` on a 3-token cache retained 1 stale token. The fix direction was PR #47720 (negative-only cropping), shipped in v5.15.0. — [Issue #47433](https://github.com/huggingface/transformers/issues/47433)
- #49036 (open, 2026-09-23): "Cache.crop does not roll back linear-attention recurrent states — block/speculative verification on GDN hybr[id models]". — [Issue #49036](https://github.com/huggingface/transformers/issues/49036)
- The v5.16.0 notes include "Fix sliding window cache index off-by-one on wraparound (#47708)" and Whisper speculative-decoding cache corruption (#48000). Neither applies to Qwen2.5 with sliding window disabled. — [v5.16.0 release](https://github.com/huggingface/transformers/releases/tag/v5.16.0)
- v5.15.0: "Fix assisted decoding for models with EncoderDecoder cache & OlmoHybrid (#47361)". — [v5.15.0 release](https://github.com/huggingface/transformers/releases/tag/v5.15.0)

### Inferences
- In v5.17 `crop(-n)` with `n > seq_len` slices `[..., :-n, :]`, which yields an empty tensor for n > len. The old one-stale-token behaviour is gone, but an over-crop now silently empties the cache. Keep the length assertion.

### Gaps
- My GitHub issue searches were title-limited. There may be issues about Qwen2 fp16 numerical mismatches between batched verify and one-token-at-a-time decoding that I did not find (see the inference in Q8).

## Q7. HF assisted-generation settings: defaults, fixed-γ reference, universal assisted generation

### Takeaway
Current global defaults (v4.45 through v5.17) are `num_assistant_tokens=20`, `num_assistant_tokens_schedule="constant"`, `assistant_confidence_threshold=0.4`. The old `5` / `"heuristic"` defaults were v4.3x–v4.4x. The candidate generator reads γ and the threshold from **`assistant_model.generation_config`**, so for a fair fixed-γ reference set them there:
```python
assistant_model.generation_config.num_assistant_tokens = gamma
assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
assistant_model.generation_config.assistant_confidence_threshold = 0.0
```
Universal assisted generation (different tokenizers) is supported by passing `tokenizer` and `assistant_tokenizer`.

### Cited Findings
- Default history:
  - v4.40.0: `num_assistant_tokens=5`, `num_assistant_tokens_schedule="heuristic"`.
  - v4.45.0 and v4.57.1: `20`, `"constant"`, `assistant_confidence_threshold=0.4`.
  - v5.x: the fields default to `None`, and `_get_default_generation_params()` supplies `num_assistant_tokens: 20`, `num_assistant_tokens_schedule: "constant"`, `assistant_confidence_threshold: 0.4`, `assistant_lookbehind: 10`, `target_lookbehind: 10`.

  — [configuration_utils v4.40.0](https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/generation/configuration_utils.py); [v4.45.0](https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/generation/configuration_utils.py); [v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/configuration_utils.py)
- Schedule documentation:
  - `"heuristic"`: +2 when all speculative tokens are correct, otherwise −1, and the value is "persistent over multiple generation calls with the same assistant model".
  - `"heuristic_transient"`: resets after each call.
  - `"constant"`: unchanged.
  - `assistant_confidence_threshold`: the draft stops early when its confidence is below the threshold. The threshold is adapted during generation using an ROC-based cost (FP cost 25%, FN cost 75%) when sklearn is installed, and it too is "persistent over multiple generation calls".

  — [configuration_utils v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/configuration_utils.py); [candidate_generator.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/candidate_generator.py)
- `AssistedCandidateGenerator.__init__` (v5.17):
  - `self.assistant_generation_config = deepcopy(assistant_model.generation_config)` with global defaults filled in.
  - `self.num_assistant_tokens = self.assistant_generation_config.num_assistant_tokens`.
  - `self.generation_config = deepcopy(generation_config)` (the target's), then `self.generation_config.assistant_confidence_threshold = self.assistant_confidence_threshold`.
  - The draft's own `generate` call is given the **target's** `generation_config` and `logits_processor`.

  v4.57.1 also read `assistant_model.generation_config.num_assistant_tokens`. — [candidate_generator.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/candidate_generator.py); [v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/candidate_generator.py)
- A `ConfidenceCriteria` stopping criterion is added only if `generation_config.assistant_confidence_threshold is not None and > 0`. — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- After an assisted `generate`, if the schedule is `"heuristic"` the final `num_assistant_tokens` is written back into `assistant_model.generation_config`. — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- Draft length per round is `min(num_assistant_tokens, max_length - cur_len - 1)`. — [candidate_generator.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/candidate_generator.py)
- Greedy verification in `_assisted_decoding`: `selected_tokens = new_logits.argmax(-1)` after the logits processors are applied. `n_matches` is the length of the longest matching prefix, and it is decremented by 1 if the candidate hit EOS/max length. Sampling mode with draft logits uses `_speculative_sampling` (Algorithm 1 from arXiv 2211.17192). v5.x adds an optional `assistant_ensemble_weight`. — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- Prompt lookup: `PromptLookupCandidateGenerator(num_output_tokens=10, max_matching_ngram_size=2, ...)` defaults. It is enabled by `prompt_lookup_num_tokens` in `generate`. — [candidate_generator.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/candidate_generator.py)
- Universal assisted generation: `different_tokenizers = all(v is not None for v in (assistant_model, target_tokenizer, assistant_tokenizer))`. It uses `AssistedCandidateGeneratorDifferentTokenizers`, or `UniversalSpeculativeDecodingGenerator` with vocab translation when `do_sample=True`. In that mode the assistant's `repetition_penalty` is forced to `None`. — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)

### Inferences
- A fair fixed-γ HF baseline would be:
  ```python
  model.generate(**inputs, assistant_model=draft, do_sample=False,
                 repetition_penalty=1.0, max_new_tokens=N)
  ```
  with `draft.generation_config.num_assistant_tokens=γ`, `num_assistant_tokens_schedule="constant"` and `assistant_confidence_threshold=0` set beforehand. With the 0.4 default, HF stops drafting early on low-confidence tokens, so its γ is effectively variable.
- Passing `num_assistant_tokens=γ` as a `generate()` kwarg is likely **ignored**, since the candidate generator reads the assistant's config. I inferred this from the code and did not test it.
- If a heuristic schedule is ever used, reset the draft's `generation_config` between benchmark runs, because the value persists.
- The same tokenizer is used here (Qwen2.5 family), so universal assisted generation is not needed.

### Gaps
- I did not verify whether `assistant_confidence_threshold=None` versus `0.0` makes any difference once global defaults are re-applied. `update(defaults_only=True)` may overwrite `None` with 0.4. **Use 0.0, not None.**

## Q8. Does `generate(do_sample=False)` apply `generation_config` defaults such as `repetition_penalty`?

### Takeaway
Yes. `generate` merges `model.generation_config` (loaded from `generation_config.json`) under the user's kwargs. Qwen2.5-3B-Instruct ships `repetition_penalty: 1.05`, and `RepetitionPenaltyLogitsProcessor` is added regardless of `do_sample`. HF greedy output therefore differs from a pure-argmax manual loop unless you pass `repetition_penalty=1.0`. In assisted mode the draft is also run with the target's config and processors.

### Cited Findings
- `_prepare_generation_config` sets priority as "user-defined kwargs or `generation_config` > `self.generation_config` > global default values". It runs `generation_config.update(**self.generation_config.to_dict(), defaults_only=True)`, then the global defaults, then the kwargs. — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- `if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0: processors.append(RepetitionPenaltyLogitsProcessor(...))`. This check is not gated on `do_sample`. — [generation/utils.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py)
- Qwen2.5-3B-Instruct `generation_config.json`: `do_sample: true`, `repetition_penalty: 1.05`, `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`, `eos_token_id: [151645, 151643]`, `pad_token_id: 151643`. Qwen2.5-0.5B-Instruct has the same values except `repetition_penalty: 1.1`. — [3B generation_config.json](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/generation_config.json); [0.5B generation_config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/generation_config.json)
- The v5.17 `GenerationConfig.validate` docstring says warnings about sampling-only flags set while `do_sample=False` fire only for user-set attributes. Temperature/top-p/top-k warpers are only used when sampling. — [configuration_utils v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/configuration_utils.py)
- In assisted generation, the draft's `generate` receives the target's `generation_config` and `logits_processor`, so the draft is penalized with the target's 1.05, not its own 1.1. — [candidate_generator.py v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/candidate_generator.py)

### Inferences
- For "manual greedy == `generate` greedy" tests:
  - Call `generate(..., do_sample=False, repetition_penalty=1.0, max_new_tokens=N)`, or pass a fresh `GenerationConfig`.
  - Also set `eos_token_id` consistently. Qwen uses two EOS ids, and HF stops on either.
  - Plain `do_sample=False` alone still applies the 1.05 penalty.
- The alternative is to implement the repetition penalty in the manual loop on both draft and target logits, so outputs match HF with the defaults.
- Even with identical settings, greedy outputs can diverge rarely in fp16. A verify forward over γ+1 tokens uses a different kernel/reduction order than one-token decoding, so near-tied argmaxes can flip. Compare token agreement rates rather than demanding exact equality on long generations. This is my inference, not verified with a source.

### Gaps
- I found no source quantifying fp16 batched-verify versus sequential-decode divergence rates for Qwen2.5 on a T4.
