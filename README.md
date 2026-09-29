# Speculative decoding on a free GPU

A from-scratch PyTorch implementation of speculative decoding ([Leviathan, Kalman and Matias, ICML 2023](https://arxiv.org/abs/2211.17192)), measured on a free Kaggle T4. It uses Qwen2.5-3B-Instruct as the target model, with two drafts: Qwen2.5-0.5B-Instruct and a zero-cost prompt-lookup draft that copies tokens from the context.

The questions:
1. Does a 0.5B draft speed up the 3B model on a T4 at all, and does the paper's formula explain the result?
2. Does a draft that costs nothing beat the 0.5B neural draft?

**Status:** step 1 of the build order (setup and environment check). No results yet.

## Reproduce on Kaggle

1. Create a Kaggle notebook from `kaggle/run.ipynb`. Set Accelerator to **GPU T4 x2** and Internet to **On**.
2. Add a Kaggle Secret holding a read-only Hugging Face token, and set `SECRET_NAME` in the notebook to its label.
3. Run all cells.

## Repository layout

| Path | Contents |
|---|---|
| `speculative-decoding-plan.md` | The project plan |
| `specdec/` | Implementation |
| `prompts.json` | The 30 prompts, built by `scripts/make_prompts.py` from pinned sources |
| `kaggle/run.ipynb` | The notebook that reproduces everything |
| `results/` | Raw results |
| `reports/`, `research_notes/` | Background research behind the plan |

## Licences

Code: MIT (`LICENSE`). Prompts, models and model outputs have their own licences; see `DATA_LICENSES.md`. In particular, Qwen2.5-3B-Instruct and its outputs are for research or evaluation use only.
