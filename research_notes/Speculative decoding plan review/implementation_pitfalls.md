# Speculative decoding from scratch: implementation pitfalls, known bugs, and how to test

Scope: a hand-written draft, verify and accept loop in PyTorch with Hugging Face models (Qwen2.5-3B target, Qwen2.5-0.5B draft, prompt-lookup draft with one-hot q), KV cache rollback, float16 on a T4. Planned tests: (a) chi-square test of the acceptance rule; (b) exact enumeration on tiny random Qwen2 models; (c) greedy exact-match; (d) alpha measured two ways.

Notation warning: Leviathan et al. use p = target and q = draft. Chen et al. (DeepMind) swap them: q = target and p = draft. Their acceptance rule is min(1, q/p) and their residual is (q - p)+ ([Chen et al. 2023, Alg. 2 and "Modified Rejection Sampling"](https://arxiv.org/abs/2302.01318)). Code copied from one paper while reading the other is a common way to invert the ratio. These notes use Leviathan's notation throughout.

---

## 1. Numerical issues in the residual norm(max(0, p - q))

### Takeaway
The residual is mathematically undefined only when p = q on the whole support, and the rejection branch is then reached with probability 0. In finite precision it can still be reached with a near-zero or exactly zero sum. Compute p and q in float32, and guard the normalisation. Reference implementations either clamp to a tiny value (vLLM, which admits "drift") or do no guard at all (HF, feifeibear).

### Cited Findings
- Algorithm 1 of Leviathan et al.: n = min({i-1 | r_i > p_i(x)/q_i(x)} ∪ {γ}). Then p' = p_{n+1}, and if n < γ, p' = norm(max(0, p_{n+1} - q_{n+1})). One token t ~ p' is returned after the n accepted draft tokens — [Leviathan et al. 2023, arXiv 2211.17192, Alg. 1](https://arxiv.org/abs/2211.17192)
- Appendix A.1 shows that the normalising constant of p' is 1 - β, where β = Σ min(p,q). So the residual sum goes to 0 exactly when the acceptance probability goes to 1 — [Leviathan et al., App. A.1](https://arxiv.org/abs/2211.17192)
- vLLM (v0 rejection sampler) clamps the difference with `torch.clamp(difference, min=self._smallest_positive_value)`, where `_smallest_positive_value = torch.finfo(self.probs_dtype).tiny`, and then divides by the sum. The code comment says the recovered distribution is built for all positions, even accepted ones, and that "This causes division-by-zero errors, so we use self._smallest_positive_value to avoid that. This introduces some drift to the distribution." — [vllm v0.6.3 rejection_sampler.py](https://github.com/vllm-project/vllm/blob/v0.6.3/vllm/model_executor/layers/rejection_sampler.py)
- vLLM's test suite has a `draft_and_target_probs_equal=True` case. The test comment says: "Rejection sampling should still work without any NaNs or exceptions" — [vllm v0.6.3 tests/samplers/test_rejection_sampler.py](https://github.com/vllm-project/vllm/blob/v0.6.3/tests/samplers/test_rejection_sampler.py)
- HF transformers `_speculative_sampling` has no epsilon: `p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0); p_prime.div_(p_prime.sum())`, followed by `torch.multinomial`. The acceptance ratio is `p_i / q_i` with no guard either. The target logits are cast first: `new_logits = outputs.logits[...].float()  # .float() is needed to retain precision for later logits manipulations` — [HF transformers v4.46.0 generation/utils.py](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/generation/utils.py)
- feifeibear's `max_fn` is `x_max / sum(x_max)` with no epsilon — [feifeibear/LLMSpeculativeSampling sampling/utils.py](https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/utils.py)
- A NaN row in the logits is a real production failure mode. vLLM issue #53029 reports that an all-NaN logits row was "laundered into an out-of-vocab token id by the rejection sampler", which triggered a device assert far from where the NaN started — [vllm#53029](https://github.com/vllm-project/vllm/issues/53029)

### Inferences
- (Own inference) Recommended guard: compute `d = (p - q).clamp_min(0)` and `s = d.sum()` in float32. If `s <= eps` (for example 1e-12, or `s` not finite), fall back to sampling from p. This is exact in the limit, because the rejection branch then has probability ≈ 1 - β ≈ s. It also avoids the "drift" that vLLM's add-tiny-everywhere approach introduces across the whole vocabulary (vLLM clamps every entry to `tiny`, so with V ≈ 151k entries the drift is small but non-zero).
- (Own inference) Always cast logits to float32 *before* softmax (`logits.float().softmax(-1)`). A float16 softmax over ~151k entries has about 3 decimal digits of precision per entry, and tail probabilities below ~6e-8 underflow to 0 (subnormal floor ~6e-8, normal floor 6.1e-5). The consequences:
  - q(x) can be 0 for a token the draft actually sampled from a float32 distribution, which makes p/q = inf or NaN.
  - The residual can be dominated by rounding noise.
- (Own inference) Compute the ratio as `r * q(x) < p(x)` (multiplication) instead of `r < p(x)/q(x)`. This avoids division by q(x) = 0.
- (Own inference) Use strict `<`. `torch.rand` samples from [0,1), so `r < p/q` accepts with exactly probability min(1, p/q). HF uses `<=`, which only differs on a measure-zero event but accepts r = 0 when p(x) = 0.
- (Own inference) Assert `torch.isfinite` on p and q every step, at least in debug mode. The vLLM issue above shows that downstream code can silently convert NaN into a bogus token id.

### Gaps
- No HF or feifeibear issue was found that reports a NaN from the residual in practice. The search in the feifeibear issue tracker returned nothing, so the frequency of this failure in real runs is unknown.

---

## 2. Temperature, top-k and top-p: must p and q be warped the same way? Greedy handling

### Takeaway
Correctness (output ~ the warped target) requires only two things. The ratio and residual must use the *target distribution you intend to sample from* (the warped p). And they must use *exactly the distribution the draft token was actually drawn from* (whatever q really was). Matching the warping of p and q is an efficiency choice, not a correctness requirement. The common bug is a *mismatch between the q used to sample the draft token and the q used in the ratio*.

### Cited Findings
- Leviathan §2.2 "Standardized Sampling": argmax, top-k, nucleus and temperature "can all easily be cast into standard sampling from an adjusted probability distribution. For example, argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing." After this, "p(x) and q(x) are the distributions from Mp and Mq respectively, adjusted for the sampling method." — [Leviathan et al. §2.2](https://arxiv.org/abs/2211.17192)
- Leviathan: "we always performed the same standardization on the distributions generated by the approximation model as the desired one for the target model (Section 2.2), but further improvements might be obtained by applying different transformations." Their App. A.1 proof holds "for any distributions p(x) and q(x)". So different warping of q is allowed. — [Leviathan et al. §5 and App. A.1](https://arxiv.org/abs/2211.17192)
- Chen et al.: "With standard sampling methods such as nucleus, top-k sampling and adjusting temperature, we can modify the probabilities accordingly before applying this rejection sampling scheme. We have observed that the overall acceptance rate is robust to the exact parameters used." — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)
- LaMDA outputs in Leviathan always go through a Top-40 filter, which "has no effect on argmax, but does have some effect on standard sampling" — [Leviathan et al., footnote 6](https://arxiv.org/abs/2211.17192)
- HF applies the target's `logits_processor` per position, with the correct growing prefix: `new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])` — [HF v4.46.0 utils.py](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/generation/utils.py)
- HF issue #48390 audited 35 logits processors for batched 3-D application over draft slots:
  - 13 crash or corrupt dimensions. For example, `TopPLogitsWarper` hard-codes `scatter(1, ...)`, so vocab ids index the time dimension.
  - 14 "run but silently change semantics" because they read per-slot prefix state (repetition penalty, min-length, no-repeat-ngram, forced-EOS).
  - Lesson: vectorising warpers across draft positions is a bug source. — [transformers#48390](https://github.com/huggingface/transformers/issues/48390)
- feifeibear's `top_k_top_p_filter` modifies `logits` in place (`logits[logits < filter[:, [-1]]] = float('-inf')`) — [feifeibear sampling/utils.py](https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/utils.py)

### Inferences
- (Own inference) Consequences of a mismatch:
  - *q warped differently from p but used consistently*: still exact, only alpha changes. Example: a top-k draft proposes only tokens in its top-k. Any token with p(x) = 0 is always rejected, and target mass outside q's support is recovered through the residual.
  - *q used in the ratio differs from the q the token was sampled from*: for example, the token was sampled at temperature T but the ratio uses the T=1 softmax, or the draft was sampled in float16 but the ratio uses a float32 recompute of a different forward. This biases the output. The chi-square test (a) catches it only if the test harness reproduces this code path, so test through the real sampling function, not a re-implementation.
  - *p unwarped while the "target" is defined as warped*: the output follows the unwarped target, which is a silent quality change.
- (Own inference) Watch for in-place logits mutation (as in feifeibear's filter). If the same logits tensor is later reused for the float32 recompute or for logging, it is already filtered.
- (Own inference) Greedy handling, option 1: represent temperature-0 target and draft as one-hot distributions. Then min(1, p/q) = 1 if the argmaxes agree and 0 otherwise, and the residual is one-hot at the target argmax. This is identical to argmax comparison, which HF uses (`selected_tokens = new_logits.argmax(-1)`; accept while equal). The one-hot path has one problem: it depends on how ties are broken. A tie can occur within float16 resolution; the HF issue in §5 shows two logits both at exactly 24.125. So argmax tie-breaking must be identical in the draft path, the verify path and the baseline.
- (Own inference) Greedy handling, option 2: implement temperature 0 as a separate argmax branch rather than dividing logits by T → 0. Dividing by a tiny T gives inf/NaN.

### Gaps
- No public report was found of a from-scratch repo shipping a q-mismatch bug that was later caught. The claim that it is the most common bug is inference.

---

## 3. Deterministic (one-hot) drafts such as prompt lookup

### Takeaway
With q = δ_x, the Leviathan rule becomes: accept with probability p(x). On rejection, sample from p with x removed and renormalised. HF implements the equivalent "sample y ~ p; accept iff y == x; otherwise output y", which is exactly correct and needs no draft probabilities.

### Cited Findings
- HF assisted decoding uses `_speculative_sampling` only when `do_sample and candidate_logits is not None`. Otherwise ("Case 2", which covers prompt lookup, whose candidate generator returns no logits) it samples `selected_tokens = torch.multinomial(probs, 1)` from the target at every position and keeps candidates up to the first mismatch. The first mismatched sampled token is emitted as the next token — [HF v4.46.0 utils.py, `_assisted_decoding`](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/generation/utils.py)
- The HF feature request #48636 notes that "prompt lookup decoding (no candidate logits)" needs special handling (a `ValueError` guard) for any rule that requires draft logits — [transformers#48636](https://github.com/huggingface/transformers/issues/48636)

### Inferences
- (Own inference, easily verified algebraically) Why Case 2 is exact:
  - P(accept x) = P(y = x) = p(x) = min(1, p(x)/1).
  - Conditional on rejection, y ~ p(·)·1[· ≠ x] / (1 - p(x)). That equals norm(max(0, p - δ_x)), because max(0, p(x) - 1) = 0 and all other entries are unchanged.
  - The same scheme works for any draft with one-hot q.
- (Own inference) Likely bugs:
  - (i) Using a softmax of a "fake" draft logits vector, for example a large constant at x. Then q(x) < 1, and the ratio and residual are both wrong.
  - (ii) After rejection, sampling from the full p (which can re-emit x) instead of p with x excluded. This inflates P(x) to p(x) + (1 - p(x))·p(x).
  - (iii) Treating prompt-lookup acceptance as "accept iff argmax p = x" under sampling. This is the greedy rule and biases sampled outputs.
- (Own inference) When p(x) = 1 in float32 (common for highly predictable tokens), the residual is all zeros. Rejection then has probability 0, but a `<=` comparison with r = 0 has probability 0 anyway. Guard as in §1 regardless.

### Gaps
- No specific implementation was found that got the one-hot rule wrong. The candidate bugs above are inferred failure modes, not reported bugs.

---

## 4. Off-by-one errors, bonus token, KV cache, EOS, max_new_tokens

### Takeaway
Target logits at sequence index t predict token t+1. With a prefix of length L and γ drafts, the verify forward pass over [prefix, x1..xγ] gives γ+1 useful rows: rows L-1 .. L+γ-1. Row L-1+i scores x_{i+1}, and the last row gives the bonus distribution. The EOS and max-length handling at the end of a block is where real libraries have had to add special cases.

### Cited Findings
- Algorithm 1 indexing: p_1..p_{γ+1} = Mp(prefix), …, Mp(prefix + [x1..xγ]). The bonus token is sampled from p_{γ+1} when n = γ — [Leviathan Alg. 1](https://arxiv.org/abs/2211.17192). Chen: "Since the final token of the draft gives us the logits for the next token, if every drafted token is accepted, we can sample from it normally. This gives us a maximum of K+1 tokens per loop." — [Chen et al.](https://arxiv.org/abs/2302.01318)
- HF keeps `candidate_length + 1` target logits: `new_logits = outputs.logits[:, -candidate_length - 1 :]` (and sets `num_logits_to_keep = candidate_length + 1`) — [HF v4.46.0 utils.py](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/generation/utils.py)
- HF EOS and max-length fix inside `_speculative_sampling`: "Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)". If `is_done_candidate and n_matches == candidate_length`, it sets `n_matches -= 1` and drops the bonus token. The same fix exists in the greedy branch. After acceptance, the target cache is cropped with `_crop_past_key_values(..., new_cur_len - 1)` — [HF v4.46.0 utils.py](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/generation/utils.py)
- HF bug #48039 (closed): "Assisted/speculative decoding ignores `stop_strings` completed mid-block, and raises with `assistant_model`". A stop condition completed inside an accepted draft block was missed — [transformers#48039](https://github.com/huggingface/transformers/issues/48039)
- feifeibear's KV-cache version compares `target._prob_history[:, prefix_len + i - 1, j] / approx._prob_history[:, prefix_len + i - 1, j]` for draft token j = x[:, prefix_len + i]. That is the "row t predicts t+1" convention. On rejection it samples `max_fn(target[:, n, :] - approx[:, n, :])` and calls `target_model_cache.rollback(n+1)`. On full acceptance it samples from `target._prob_history[:, -1, :]` and calls `rollback(n+2)`. The draft cache is rolled back with `approx_model_cache.rollback(n+1)` — [feifeibear sampling/speculative_sampling.py](https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/speculative_sampling.py)
- Code-reading observations on the same feifeibear file (own reading of the source, not reported issues). Source: [speculative_sampling.py](https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/speculative_sampling.py) and [utils.py](https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/utils.py):
  - (a) The loop condition is `while prefix.shape[1] < T` with T = seq_len + max_len, and each iteration appends up to γ+1 tokens with no truncation. So output can overshoot max_len by up to γ.
  - (b) There is no EOS check inside accepted draft tokens.
  - (c) `if random_seed: torch.manual_seed(random_seed)` runs *inside* the per-token acceptance loop. Every r is then the same value, which correlates the acceptance decisions, and `random_seed=0` silently disables seeding.
  - (d) `sample()` raises `RuntimeError` whenever token id 0 is sampled (`if (idx_next.item() == 0): raise RuntimeError`).
  - (e) `.item()` inside `sample()` forces a host sync on every sample.

### Inferences
- (Own inference) Draft KV cache after drafting γ tokens: the draft model has *processed* only x_1..x_{γ-1}; x_γ was sampled from the last output but never fed back. What happens next depends on the outcome:
  - All accepted and a bonus token b emitted: the next round must feed [x_γ, b] (2 tokens) to the draft before drafting again. Feeding only b leaves a hole in the cache. Qwen2 with RoPE would then attend with shifted positions, which silently lowers alpha but does not break correctness, because q is whatever the draft actually produced. That makes this bug invisible to the distribution tests and visible only as a lower alpha.
  - Rejection at position n+1: crop the draft cache to the prefix + n accepted tokens and then feed the corrected token.
  - Unit-test cache rollback directly. After each round, compare the draft's and target's next-token logits from the cached path against a fresh no-cache forward on the full current sequence, using a tolerance (see §5 for the expected size of the difference).
- (Own inference) EOS handling: scan the accepted tokens *plus* the correction or bonus token for EOS, and truncate after the first EOS. Also truncate to `max_new_tokens`. Make sure alpha, tokens-per-step and timing statistics only count tokens actually emitted. Otherwise the final block inflates the per-step numbers.
- (Own inference) For HF-style `DynamicCache`, use `cache.crop(n)`. Also check that `cache_position` and `position_ids` agree with the cropped length.

### Gaps
- No public issue was found in the feifeibear tracker via the GitHub issue search (0 results). Bug reports specific to that repo could not be confirmed.

---

## 5. Greedy exactness and non-determinism in fp16/bf16; testing exactness

### Takeaway
Multi-token verification (M = γ+1 rows) and single-token decoding (M = 1) use different GEMM and attention kernel shapes. Their float16/bf16 logits differ at the ULP level, and wherever the top-2 margin is within that noise, greedy outputs diverge. HF maintainers treat this as expected. vLLM has an open RFC calling it a systemic problem. Strict token equality is realistic only in float32 (and even then it is not guaranteed). In float16, the right test compares the verify-pass logits at each emitted position against plain-decode logits at the same position, and classifies each divergence as a near-tie.

### Cited Findings
- HF issue #39421, "Speculative Decoding(do_sample=False) get different outputs": Qwen2.5-32B target with a 7B draft in bf16, greedy, never produced identical outputs. The first divergence showed tied or near-tied logits: the top-2 logits were `[24.125, 24.125]` in one run and `24.125` vs `24.0` in the other. A HF maintainer explained: "in assisted decoding the target model gets a batch of ~5 new tokens every step, unlike greedy decoding where the models sees 1 token only, causing small numerical differences. Same thing as if we compared logits from cached or not-cached generation." Also: "(num_candidates, dim) x (weights_vector) … may trigger different optimized kernels internally in torch (e.g., single vs. batched GEMM)… So yeah, it is expected." The maintainer suggested checking that logits are `allclose` — [transformers#39421](https://github.com/huggingface/transformers/issues/39421)
- vLLM RFC #54506 on greedy verify vs decode parity (FP16, V100):
  - Cause A: split-K GEMMs whose reduction order depends on M. A hand-written F16 GEMM with M=5 vs 5×(M=1) gave "~92–114 of 20480 elements differ by 1–2 ULP".
  - Cause B: torch.compile generates fusion and reduction schedules per shape.
  - Cause C: different execution paths in spec vs non-spec engines.
  - "Real text contains decision points whose top-2 logit margin sits at ULP scale; one 1-ULP difference in one operator flips such a token." Acceptance dropped from ~100% to 72% purely from this.
  - Recommended method: first compare prefill logits bitwise, then "for an accepted prefix, compare the target model's verify-pass logits at position t (computed with M=k+1 rows) against plain decode logits at the same position (M=1)."
  - Warning: random-weight kernel probes both over- and under-predicted real-checkpoint behaviour.
  - Source: [vllm#54506](https://github.com/vllm-project/vllm/issues/54506). The same RFC cites vllm#41758 (ngram spec decode changing temperature-0 output, Qwen3-0.6B, A100).
- vLLM issue #48271: `prompt_logprobs` argmax disagrees with greedy decode for sequences ≥256 tokens, because the RMSNorm kernel's block size depends on num_tokens — [vllm#48271](https://github.com/vllm-project/vllm/issues/48271)
- Cache vs no-cache differences in HF: Llama-3.2-1B float32 showed max |Δlogit| = 3.5e-5 between a full no-cache forward and prefill+decode with cache. Qwen3.5 (GatedDeltaNet) showed 0.125 because of a different recurrent path, independent of dtype — [transformers#46190](https://github.com/huggingface/transformers/issues/46190). Chunked `generate` calls with a compiled forward also diverged from a single call — [transformers#44464](https://github.com/huggingface/transformers/issues/44464)
- Chen et al.: "Even with greedy sampling, a single token deviating due to numerics could result in two sequences diverging wildly… because the different computation graphs lead to different numerics, we cannot not expect identical outputs. However, we expect the samples to come from the same distribution within numerics." They validated with benchmark-metric parity instead (for example XSum ROUGE-2 greedy: 0.157 ArS vs 0.156 SpS) — [Chen et al.](https://arxiv.org/abs/2302.01318)
- vLLM's v0 e2e spec-decode tests (`run_equality_correctness_test`) compare output *token ids* between baseline and spec decoding at temperature 0. Logprobs are checked with `check_logprobs_close` when requested — [vllm v0.6.3 tests/spec_decode/e2e/conftest.py](https://github.com/vllm-project/vllm/blob/v0.6.3/tests/spec_decode/e2e/conftest.py)
- Attention-implementation sensitivity is reported for another Qwen model: "Qwen2VL exhibits significant performance differences under different attention implementations" — [transformers#35749](https://github.com/huggingface/transformers/issues/35749) (title only was reviewed)

### Inferences
- (Own inference) Recommended protocol for test (c):
  1. **Float32, eager or SDPA-math attention, same attention implementation in both runs**: require exact token equality. If this still fails, check whether the failing position has a top-2 margin below ~1e-4 (float32 M-dependence exists but is tiny). Otherwise treat it as a real bug.
  2. **Float16**: at each divergence point, log the target's plain-decode top-2 margin and the verify-pass top-2 margin at the same position. Accept a divergence as a "near-tie" only if |Δlogit| between the two paths ≥ the margin, and the margin is below a threshold fixed in advance (for example < 0.05–0.1 in logit units for fp16, where the ULP at |logit| ≈ 16–32 is 0.0156). Also report how often this happens.
  3. **Teacher-forced check (strongest)**: after the spec-decode run, run the target once, without cache, over the full emitted sequence. For every emitted token, check that it equals the argmax of that no-cache pass, or is a logged near-tie. This checks greedy correctness without requiring trajectories to match.
- (Own inference) Qwen2.5 on a T4 (sm_75) in float16: T4 does not support bf16 natively and FlashAttention-2 does not support Turing. So SDPA will choose the efficient or math kernels, whose choice can differ between q_len = 1 and q_len = γ+1. Pin the attention implementation (`attn_implementation="eager"` or "sdpa") identically for baseline and spec runs.
- (Own inference) float16 overflow: Qwen2 activations can exceed the fp16 range in some layers. If the no-cache float16 forward produces inf/NaN, that is a model/dtype problem, not a spec-decoding bug. Check the baseline's logits for finiteness first.

### Gaps
- No published measurement was found of fp16 verify-vs-decode logit deltas for Qwen2.5-3B on a T4 specifically. The thresholds suggested above are own estimates.

---

## 6. Statistical testing of distributional equivalence

### Takeaway
Use small vocabularies so that exact target probabilities are enumerable and expected cell counts are large. Pair a chi-square goodness-of-fit test with a total-variation (TV) bound whose threshold is derived from the sample size. Include adversarial (p, q) pairs: p = q, disjoint supports, one-hot q, q with zeros where p > 0. vLLM's own test uses a weaker heuristic (relative L2 improvement vs a random reference), so do not copy it as a hypothesis test.

### Cited Findings
- vLLM's `test_rejection_sampling_approximates_target_distribution` uses `vocab_size=10` and sample counts `[10, 100, 1_000, 10_000, 100_000]`. It measures Euclidean `torch.dist(target_probs, rej_sample_probs)` and asserts that the relative improvement in distance to the target exceeds the relative improvement w.r.t. a random reference distribution by `expected_improvement_multiplier = 20`. It also runs a `draft_and_target_probs_equal` case to check for no NaNs — [vllm v0.6.3 tests/samplers/test_rejection_sampler.py](https://github.com/vllm-project/vllm/blob/v0.6.3/tests/samplers/test_rejection_sampler.py)
- Leviathan's proof is per-token: P(x = x') = min(p,q) + (p - min(p,q)) = p(x') — [Leviathan App. A.1](https://arxiv.org/abs/2211.17192). Multi-token correctness follows by applying it sequentially: "By applying this sequentially, we recover the distribution of the target model" — [Chen et al.](https://arxiv.org/abs/2302.01318)
- Standard chi-square validity condition: expected counts per category should be at least ~5 — [SciPy `chisquare` docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html) (stated from memory of the docs; not re-fetched in this session — verify).

### Inferences
- (Own inference) Test (a), synthetic p and q:
  - Setup: K ≈ 8–32 categories, N = 10^5–10^6 samples per (p, q) case. Merge cells with expected count < 5 or choose p with a floor.
  - Use a fixed seed and a pre-set α_test (for example 1e-3). With many (p, q) cases, correct for multiple testing (Bonferroni), or check that the p-values are roughly uniform over many seeds.
  - Also assert TV(empirical, p) < c·√(K/N). The expected TV of an N-sample empirical distribution is roughly ½Σ√(2p(1-p)/(πN)), which is O(√(K/N)). For example, with K = 16 and N = 10^6, the expected TV is about 0.002.
  - Test the full step function (γ drafts, acceptance, residual, bonus), not only the single-token rule. Bugs in the index of the residual row or the bonus row show up only in multi-token sampling.
- (Own inference) Test (b), tiny random Qwen2:
  - Set `vocab_size` small (for example 16–64) in `Qwen2Config`. With 151,936 tokens, enumerating 3 tokens (V³) is impossible and chi-square cells are empty. With V = 16, V³ = 4096 cells: use N ≥ ~5×10^5 or test marginals and conditionals separately (first-token marginal: V cells; pairs: V² cells).
  - Randomly initialised small models give nearly uniform p and q (default `initializer_range` 0.02), so p ≈ q, alpha ≈ 1, and the rejection and residual path is rarely exercised. Scale up `initializer_range` or the LM-head weights, or use temperature < 1, so that p and q are peaked and different. Log the fraction of steps that hit the residual branch and require it to be substantial.
  - Run the test in float32 on CPU for determinism. Exact enumeration must use the same cache and no-cache path as the implementation. Compare against no-cache target forwards (see §5 for how large the path-dependent difference can be).
  - Include γ = 1, γ = 3 and γ larger than the sequence length being tested (so the bonus token lands inside the 3-token window), and include EOS inside the window if EOS handling is part of the step.
- (Own inference) Sequential or conditional check: for 3-token sequences, a single chi-square on V³ cells is weak. Also test P(x2 | x1) for the most frequent x1, where cell counts are large.

### Gaps
- No published standard exists for sample-size or TV thresholds specific to speculative-decoding tests. The numbers above are standard statistics, not taken from a spec-decoding source.

---

## 7. The empirical acceptance-rate estimator

### Takeaway
Leviathan defines α = E[β] with β = Σ_x min(p(x), q(x)), and measured it by evaluating that expectation over 10K tokens generated by the target. Counting accepted/proposed gives a different, biased number if the denominator includes draft tokens that were never evaluated after the first rejection. Σ min(p,q) is meaningful only at positions whose prefix is on the actual output path.

### Cited Findings
- "Definition 3.1. The acceptance rate β_{x<t}, given a prefix x<t, is the probability of accepting x_t ~ q(x_t|x<t) by speculative sampling… If we make the simplifying assumption that the βs are i.i.d., and denote α = E(β), then the number of tokens produced by a single run of Algorithm 1 is a capped geometric variable, with success probability 1-α and cap γ+1." — [Leviathan §3.1](https://arxiv.org/abs/2211.17192)
- "Corollary 3.6. α = 1 - E(D_LK(p,q)) = E(min(p,q))" (the sum over x is implied) — [Leviathan §3](https://arxiv.org/abs/2211.17192)
- Leviathan §4.2: "we evaluated the expectation from Corollary 3.6 on 10K tokens generated by Mp". App. A.3 attributes the gaps between predicted and measured speedup partly to "the simplifying assumption that the βs are i.i.d. being only an approximation" — [Leviathan §4.2, App. A.3](https://arxiv.org/abs/2211.17192)
- vLLM's e2e tests assert a measured acceptance rate with tolerance 1e-2 (`acceptance_rate >= expected_acceptance_rate - 1e-2`) — [vllm v0.6.3 e2e conftest](https://github.com/vllm-project/vllm/blob/v0.6.3/tests/spec_decode/e2e/conftest.py)

### Inferences
- (Own inference) Estimator definitions:
  - `accepted / evaluated`, where "evaluated" is the accepted drafts plus the one rejected draft in each round. Each evaluated position is a Bernoulli(β_t) trial at a context on the output path, so this is a consistent estimate of mean β over those contexts.
  - `accepted / (γ × rounds)` counts unevaluated positions after a rejection as failures. It estimates E[#accepted]/γ = α(1-α^γ)/(γ(1-α)) under i.i.d. That is systematically below α, for example 0.55 vs α = 0.8 at γ = 4. Report it separately as "draft-token efficiency" if at all.
  - Where Σ min(p,q) is computed: at every *evaluated* position (the accepted ones plus the first rejected one), using the same warped p and q as the acceptance rule. Do **not** use positions after the first rejection (their prefix contains a discarded token, so the context is not on the target path), and do not use the bonus position (no q).
  - The two estimators then share the same contexts, and the Σ min(p,q) version has lower variance because it is the conditional expectation of the Bernoulli outcome. They should agree within sampling error. A large gap indicates a bug (for example, a ratio computed from a different q than the one sampled from).
  - The "evaluated" position set is not a uniform sample of target-path positions: which positions get evaluated depends on earlier acceptances within a round. Leviathan instead averaged over *all* tokens generated by Mp. For an α directly comparable to the paper's, compute Σ min(p,q) at every emitted token position (this needs a draft forward on the final sequence), or note the difference.
  - For the prompt-lookup draft, Σ min(p,q) = p(x) at positions where a draft exists. Positions with no lookup match have no proposal and must be excluded, and the fraction of such positions reported separately.
  - Under greedy, β = 1[argmax p = argmax q], so both estimators coincide.

### Gaps
- No published analysis of this specific estimator bias was found. The derivation above is own reasoning from the i.i.d. model in Leviathan §3.1.

---

## 8. Known bugs in popular implementations and how they were detected

### Takeaway
Most publicly reported problems concern numerics (verify vs decode parity), stop conditions inside accepted blocks, NaN handling, and logits-processor semantics across draft positions. Very few concern the core acceptance math. They were detected mostly by greedy token-equality comparisons against the baseline and by inspecting the logits at the first divergence.

### Cited Findings
- HF #39421: greedy assisted output ≠ plain greedy output in bf16. Detected by diffing token sequences and printing the top-5 logits at the first divergent index; closed as expected numerical behaviour — [transformers#39421](https://github.com/huggingface/transformers/issues/39421)
- HF #48039: `stop_strings` completed mid-block were ignored, and it raised with `assistant_model` (closed) — [transformers#48039](https://github.com/huggingface/transformers/issues/48039)
- HF #48390: batched logits processors over draft slots corrupt dimensions or change semantics in 27 of 35 processors (found by an audit during review of #48281, a sampled-DFlash bug that "returned a logits tensor missing the batch dimension") — [transformers#48390](https://github.com/huggingface/transformers/issues/48390)
- HF #32946: StaticCache was incompatible with assisted generation because rejected tokens must be cleared from the cache and StaticCache lacked cropping (closed) — [transformers#32946](https://github.com/huggingface/transformers/issues/32946)
- vLLM #53029: all-NaN logits turned into an out-of-vocab token by the rejection sampler — [vllm#53029](https://github.com/vllm-project/vllm/issues/53029)
- vLLM #54506 / #41758: speculative decoding changes temperature-0 output. Diagnosed by bitwise comparison of verify-pass vs decode logits layer by layer — [vllm#54506](https://github.com/vllm-project/vllm/issues/54506)
- vLLM #54928: "DFlash2 changes greedy Qwen3.8 thinking output at token 30, including K=1 and --enforce-eager" (open; title only reviewed) — [vllm#54928](https://github.com/vllm-project/vllm/issues/54928)
- vLLM's v0 rejection sampler admits distribution "drift" from its tiny-clamp division guard — [vllm rejection_sampler.py](https://github.com/vllm-project/vllm/blob/v0.6.3/vllm/model_executor/layers/rejection_sampler.py)
- feifeibear/LLMSpeculativeSampling: see §4 for code-level issues (per-iteration reseeding, token-0 RuntimeError, max_len overshoot, no EOS handling) — [speculative_sampling.py](https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/speculative_sampling.py), [utils.py](https://github.com/feifeibear/LLMSpeculativeSampling/blob/main/sampling/utils.py)

### Inferences
- (Own inference) The core rejection rule is short and usually correct in popular repos. Bugs cluster in the bookkeeping around it: cache positions, stop conditions, processors, and dtype. Tests should therefore drive the full loop, not just the acceptance function.

### Gaps
- The GitHub issue search for feifeibear/LLMSpeculativeSampling returned 0 results through the API. Community-reported bugs for that repo could not be verified, and the §4 observations come from reading the code only.
- Blog-post errata for from-scratch tutorials were not found in the time budget.

---

## 9. Timing pitfalls on GPU (T4)

### Takeaway
CUDA work is asynchronous. Wall-clock timing must sit behind a synchronisation point (or use CUDA events), with warm-up excluded. Spec-decoding loops contain implicit syncs (`.item()`, Python branching on tensor values, `multinomial` results used for control flow), and these are part of the true cost, so they belong in end-to-end timing.

### Cited Findings
- "A naïve approach may end up timing the kernel launch instead of kernel execution." Use `torch.cuda.synchronize()` or `torch.cuda.Event` timing. Include warm-up iterations to remove JIT, cuDNN autotuning, lazy kernel loading and allocator initialisation costs. GPU clocks vary with temperature and power limits; clocks can be locked via `nvidia-smi`, but the GPU "always retains the ability to decrease the clock rate (throttling)" — [Speechmatics, How to Accurately Time CUDA Kernels in PyTorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch) (also at [lawrencium77 blog](https://lawrencium77.github.io/pytorch/cuda/gpu/2023/03/28/timings.html))
- Per-iteration `torch.cuda.synchronize()` inflates runtimes for fast kernels by creating GPU idle bubbles — [flashinfer-bench#195](https://github.com/flashinfer-ai/flashinfer-bench/issues/195)
- Chen et al. measured the time per loop from TPU profiles and logged the tokens generated per speculative loop to compute speedup and its standard deviation — [Chen et al. §4](https://arxiv.org/abs/2302.01318)

### Inferences
- (Own inference, based on PyTorch's documented asynchronous CUDA semantics: [PyTorch CUDA semantics – asynchronous execution](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)) `.item()`, `.cpu()`, `.tolist()`, `if tensor:` and printing a CUDA tensor all block until the GPU catches up. In a spec-decoding loop, the acceptance decision needs n on the host, so each round has at least one sync. That is inherent, and the baseline has an equivalent per-token sync (EOS check). Time both the spec loop and the baseline end to end with the same sync discipline: `torch.cuda.synchronize()` before starting and before stopping `time.perf_counter()`. Use CUDA events only for per-component breakdowns (draft vs verify time).
- (Own inference) T4 specifics (unverified against a primary source here): the T4 is a 70 W, passively cooled card, so it throttles on clocks under sustained load, and cloud or Colab T4s vary between sessions. To handle this:
  - Interleave baseline and spec runs (ABAB), not all-A then all-B.
  - Repeat over several prompts and seeds, and report the median and IQR.
  - Record `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` during runs.
- (Own inference) Warm-up must cover every distinct shape. The verify pass with γ+1 tokens, and each smaller shape at the end of a sequence, trigger separate kernel selection and allocation.
- (Own inference) Exclude tokenisation and the first prefill from per-token decode timings, or report prefill separately. Make sure both methods produce the same number of new tokens: the γ-overshoot from §4 would otherwise inflate spec throughput.
- (Own inference) Qwen2.5-0.5B and 3B share the Qwen2.5 tokenizer, but the lm_head output dimension (padded vocab) can differ across Qwen2.5 sizes (for example, 151936 vs 152064 in larger models). Before forming p/q, assert that both logits tensors have the same last dimension, or slice both to `len(tokenizer)`. The specific sizes for 0.5B vs 3B were not verified in this session.

### Gaps
- No T4-specific study of clock variance for LLM decoding was found. Claims about PyTorch sync behaviour are based on the PyTorch docs link above without re-fetching the page in this session.
