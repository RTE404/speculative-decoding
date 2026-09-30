# Speculative decoding on a free GPU

A from-scratch PyTorch implementation of speculative decoding ([Leviathan, Kalman and Matias, ICML 2023](https://arxiv.org/abs/2211.17192)), measured on a free Kaggle T4 with Qwen2.5-3B-Instruct as the target. It compares two drafts: Qwen2.5-0.5B-Instruct, and a zero-cost prompt-lookup draft that copies tokens from earlier in the text. It then checks the measured speedups against the paper's theory.

**The finding:** on a T4 in plain PyTorch, a 6× smaller draft model costs about two-thirds of a target step (c ≈ 0.65), because each step's time is mostly fixed software overhead rather than reading weights. So the 0.5B draft gives at most about 1.2×, while the draft that costs nothing gives 1.76× overall and 4.1× on code. The paper's cost model, fed with measured inputs, predicts every result within about 3%.

![Speedup vs draft length](report/20260930-035822/speedup_vs_gamma.png)

## Results

Greedy decoding, 30 prompts (10 each of code, maths and chat), up to 128 new tokens, one Tesla T4, float16. Speedup is tokens per second relative to the 3B target alone (22.0 tokens/s).

| Draft | Best γ | Code | Maths | Chat | **All** | α (code / maths / chat) |
|---|---|---|---|---|---|---|
| 0.5B neural draft | 1 | 1.26× | 1.23× | 1.10× | **1.19×** | 0.98 / 0.92 / 0.73 |
| Copy draft (prompt lookup) | 8 | 4.08× | 1.55× | 1.21× | **1.76×** | 0.88 / 0.50 / 0.32 |

- **The neural draft is capped by its cost.** One 0.5B step takes 29.2 ms against 44.8 ms for the 3B (c = 0.65). Even with 87% of its guesses accepted, drafting more than one token ahead mostly wastes time. On chat, where α is 0.73, γ = 6 runs at 0.65×, slower than the target alone.
- **The copy draft never loses.** A lookup costs 0.017 ms (c = 0.0004), so a failed guess costs nothing, and longer guesses always help. It is strongest where the answer repeats the prompt: the code prompts ask for the full function, so the model rewrites the signature and docstring.
- **Verification is free.** Checking up to 16 tokens in one target pass costs 0.93 to 1.02 of a single step.
- **The theory holds once its inputs are measured.** The paper's formula (Theorem 3.8) is accurate for the neural draft. For the copy draft it overshoots (5.7× predicted against 4.1× on code) because it assumes a proposal every round, while the copy draft finds a match in only about half of them. Adding the measured verification cost and proposal lengths closes the gap to within about 3%.

![Theory vs measurement](report/20260930-035822/theory_vs_measured.png)

Full tables (every γ and task, the three prediction layers, costs, correctness and run conditions): [`report/20260930-035822/README.md`](report/20260930-035822/README.md). Raw data: [`results/20260930-035822/`](results/20260930-035822/).

## Reproducibility

The whole run was repeated in a fresh Kaggle session on a different T4 ([`report/20260930-050746/`](report/20260930-050746/README.md)). The only code changes between the two runs were a variable rename in `run.py` and making the Hugging Face token optional in the notebook; the measurement code was identical.

- That T4 was about 6% slower in absolute terms (baseline 20.6 against 22.0 tokens/s; c = 0.663 against 0.650).
- Every speedup reproduced within **0.4% on average and 1.5% at most**.
- All 351 generated outputs were **token-for-token identical** between the two runs.
- An earlier trial run, with slightly older code, agreed within 1.1% on average.

## Correctness

Speculative decoding must give exactly the target's output. Checked at four levels:

1. **The acceptance rule** (`tests/test_sampling.py`): one million rounds per case against synthetic distributions, including one-hot, nearly equal and disjoint drafts and the bonus token, with a chi-square test.
2. **The whole loop** (`tests/test_tiny_models.py`): tiny random Qwen2 models with a 6-token vocabulary, where the exact distribution over every 3-token output can be listed. Sampled outputs match it, with and without the KV cache. Also tested: greedy exact match, stopping at EOS, a 300-step random walk of cache appends and rollbacks against fresh forward passes, and deliberately planted bugs, which the tests catch.
3. **The real models** (`specdec/exactness.py`): identical greedy output in float32. In float16, every emitted token is the target's top choice when the target re-reads the whole output in one pass.
4. **Every timing run:** 11 of 240 float16 outputs differ from the baseline. All 11 are near-ties, with top-two logit gaps of 0.016 to 0.031, about one float16 rounding step. Verifying several tokens at once uses different matrix kernels, which can flip exact ties.

## Reproduce on Kaggle

1. Create a Kaggle notebook from [`kaggle/run.ipynb`](kaggle/run.ipynb). Set Accelerator to **GPU T4 x2** and Internet to **On**.
2. Add a Kaggle Secret holding a read-only Hugging Face token, and set `SECRET_NAME` in the notebook to its label.
3. Set `ARGS = ""` for the full run (39 minutes on a T4, using under 1 of Kaggle's 30 weekly GPU-hours) or `"--quick"` for a smoke test (about 10 minutes). Use **Save Version → Save & Run All**.

The run writes `results/<run id>/` and `report/<run id>/`. Any machine with a CUDA GPU can run `python -m specdec.run` after `pip install -r requirements.txt`.

## Differences from the paper

- **Hardware and software:** one T4 GPU in PyTorch eager mode, against a TPU-v4 with T5X. Framework overhead is why c is 0.65 here and below 0.05 in the paper.
- **Model sizes:** the draft is 6× smaller than the target, against about 100× in the paper.
- **Decoding:** greedy only in the timing runs. Sampling is tested for correctness but not timed.
- **Scale:** 30 prompts, 128 new tokens, one GPU type, one run plus repeated passes over three prompts.
- **Prompts:** the code prompts ask for the full function, which favours the copy draft.

## Repository layout

| Path | Contents |
|---|---|
| `specdec/` | Implementation: acceptance rule, drafts, KV cache, loop, experiments, report |
| `tests/` | Correctness tests (experiments 1 and 2) |
| `prompts.json` | The 30 prompts, built by `scripts/make_prompts.py` from pinned sources |
| `kaggle/run.ipynb` | The notebook that reproduces everything |
| `results/`, `report/` | Raw results and generated reports per run |
| `speculative-decoding-plan.md` | The project plan, with notes from every step |
| `reports/`, `research_notes/` | Background research behind the plan |

## Licences

Code: MIT (`LICENSE`). Prompts, models and model outputs have their own licences; see [`DATA_LICENSES.md`](DATA_LICENSES.md). In particular, Qwen2.5-3B-Instruct and the outputs in `results/` are for research or evaluation use only.
