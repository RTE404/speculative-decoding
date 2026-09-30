# Results: run 20260930-050746

Qwen2.5-3B-Instruct target, greedy decoding, 10 prompts per task, up to 128 new tokens. Tesla T4, Tesla T4; torch 2.10.0+cu128, transformers 5.17.0, CUDA 12.8. Code at commit `595ac38`.

## Headline

- **0.5B neural draft**: best setting neural-g1, **1.19×** overall (code 1.26×, maths 1.23×, chat 1.09×).
- **copy draft (prompt lookup)**: best setting copy-g8, **1.76×** overall (code 4.08×, maths 1.55×, chat 1.20×).
- Cost ratio c = 0.663 for the 0.5B draft and 0.0004 for the copy draft. Checking 2 to 16 tokens in one pass costs 0.93 to 1.02 of one target step.

![Speedup vs draft length](speedup_vs_gamma.png)

## Speedup by task (experiment 5)

Tokens per second relative to the target alone, total tokens over total time per task.

| Setting | code | maths | chat | all | tokens/s (all) |
|---|---|---|---|---|---|
| baseline | 1.00× | 1.00× | 1.00× | 1.00× | 20.6 |
| neural-g1 | 1.26× | 1.23× | 1.09× | 1.19× | 24.6 |
| neural-g2 | 1.32× | 1.22× | 1.00× | 1.17× | 24.1 |
| neural-g3 | 1.32× | 1.21× | 0.88× | 1.11× | 22.9 |
| neural-g4 | 1.34× | 1.17× | 0.78× | 1.05× | 21.6 |
| neural-g6 | 1.33× | 1.11× | 0.64× | 0.94× | 19.4 |
| copy-g2 | 2.33× | 1.41× | 1.18× | 1.52× | 31.3 |
| copy-g4 | 3.08× | 1.50× | 1.19× | 1.65× | 34.1 |
| copy-g8 | 4.08× | 1.55× | 1.20× | 1.76× | 36.3 |

## Acceptance

α is accepted draft tokens over evaluated draft tokens (accepted plus the first rejection per round). Match rate is the share of rounds where the draft proposed anything (the copy draft proposes nothing when it finds no match; those rounds are excluded from α). In greedy decoding the two α estimators of the plan are identical by construction, since min(p, q) is 1 or 0.

| Setting | α code | α maths | α chat | match rate | tokens per round |
|---|---|---|---|---|---|
| neural-g1 | 0.98 | 0.92 | 0.73 | 0.99 | 1.86 |
| neural-g2 | 0.98 | 0.92 | 0.73 | 0.99 | 2.60 |
| neural-g3 | 0.98 | 0.92 | 0.72 | 0.99 | 3.22 |
| neural-g4 | 0.98 | 0.92 | 0.72 | 0.99 | 3.72 |
| neural-g6 | 0.98 | 0.92 | 0.73 | 0.99 | 4.57 |
| copy-g2 | 0.83 | 0.47 | 0.30 | 0.60 | 1.50 |
| copy-g4 | 0.85 | 0.49 | 0.31 | 0.56 | 1.65 |
| copy-g8 | 0.88 | 0.50 | 0.32 | 0.53 | 1.75 |

## Theory vs measurement (experiment 6)

Predicted speedup in three layers (see `specdec/theory.py`): **A** the paper's formula with the measured α, γ and c; **B** adds the measured verification cost and the draft tokens actually proposed each round; **C** also uses the tokens each round actually produced, dropping the equal-acceptance assumption. The cost model ignores reading the prompt and Python overhead.

![Theory vs measurement](theory_vs_measured.png)

| Setting | Task | α | A paper | B cost model | C round by round | Measured |
|---|---|---|---|---|---|---|
| neural-g1 | code | 0.98 | 1.19 | 1.24 | 1.24 | 1.26 |
| neural-g1 | maths | 0.92 | 1.15 | 1.20 | 1.20 | 1.23 |
| neural-g1 | chat | 0.73 | 1.04 | 1.08 | 1.08 | 1.09 |
| neural-g1 | all | 0.87 | 1.13 | 1.17 | 1.17 | 1.19 |
| neural-g2 | code | 0.98 | 1.27 | 1.30 | 1.30 | 1.32 |
| neural-g2 | maths | 0.92 | 1.19 | 1.22 | 1.21 | 1.22 |
| neural-g2 | chat | 0.73 | 0.98 | 1.00 | 0.99 | 1.00 |
| neural-g2 | all | 0.88 | 1.14 | 1.16 | 1.16 | 1.17 |
| neural-g3 | code | 0.98 | 1.29 | 1.33 | 1.33 | 1.32 |
| neural-g3 | maths | 0.92 | 1.18 | 1.21 | 1.21 | 1.21 |
| neural-g3 | chat | 0.72 | 0.88 | 0.90 | 0.89 | 0.88 |
| neural-g3 | all | 0.87 | 1.10 | 1.13 | 1.11 | 1.11 |
| neural-g4 | code | 0.98 | 1.31 | 1.34 | 1.33 | 1.34 |
| neural-g4 | maths | 0.92 | 1.16 | 1.18 | 1.17 | 1.17 |
| neural-g4 | chat | 0.72 | 0.79 | 0.80 | 0.79 | 0.78 |
| neural-g4 | all | 0.87 | 1.06 | 1.08 | 1.05 | 1.05 |
| neural-g6 | code | 0.98 | 1.33 | 1.33 | 1.32 | 1.33 |
| neural-g6 | maths | 0.92 | 1.11 | 1.11 | 1.10 | 1.11 |
| neural-g6 | chat | 0.73 | 0.66 | 0.67 | 0.64 | 0.64 |
| neural-g6 | all | 0.88 | 0.98 | 0.98 | 0.94 | 0.94 |
| copy-g2 | code | 0.83 | 2.50 | 2.43 | 2.38 | 2.33 |
| copy-g2 | maths | 0.47 | 1.68 | 1.45 | 1.44 | 1.41 |
| copy-g2 | chat | 0.30 | 1.38 | 1.20 | 1.20 | 1.18 |
| copy-g2 | all | 0.56 | 1.87 | 1.57 | 1.55 | 1.52 |
| copy-g4 | code | 0.85 | 3.71 | 3.43 | 3.25 | 3.08 |
| copy-g4 | maths | 0.49 | 1.92 | 1.57 | 1.56 | 1.50 |
| copy-g4 | chat | 0.31 | 1.44 | 1.23 | 1.22 | 1.19 |
| copy-g4 | all | 0.59 | 2.27 | 1.77 | 1.71 | 1.65 |
| copy-g8 | code | 0.88 | 5.71 | 4.39 | 4.10 | 4.08 |
| copy-g8 | maths | 0.50 | 1.98 | 1.53 | 1.53 | 1.55 |
| copy-g8 | chat | 0.32 | 1.47 | 1.20 | 1.20 | 1.20 |
| copy-g8 | all | 0.62 | 2.56 | 1.80 | 1.75 | 1.76 |

## Costs (experiment 4)

![Verification cost](verify_cost.png)

Target step 48.0 ms, 0.5B draft step 31.8 ms, n-gram lookup 0.019 ms, measured after a 250-token context.

| Tokens checked | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 16 |
|---|---|---|---|---|---|---|---|---|
| Time (target steps) | 1.00 | 0.93 | 0.94 | 0.93 | 0.93 | 0.99 | 1.02 | 1.02 |

## Correctness

- Experiment 3, 128 tokens on one prompt per task: float32 strict exact match **pass**, float16 teacher-forced check **pass** (near-tie threshold 0.1).
- Speculative outputs that differ from the baseline in float16: 11 of 240.
- Teacher-forced check of every differing output: 11 of 11 are explained by near-ties (every token is the target's top choice or within the threshold of it).

## Run conditions

- GPU temperature 71 to 83 °C, SM clock 1170 to 1575 MHz. Configurations were interleaved, so clock changes affect all of them.
- Across 3 repeated passes over one prompt per task, the largest standard deviation of a speedup was 0.007 (copy-g8).
- Peak GPU memory 7.4 GB.
