# Kaggle Notebook Environment and GPU Compatibility Risks (PyTorch + HF transformers on T4 x2), as of late September 2026

Research date: 2026-09-29. Primary sources are the Kaggle/docker-python GitHub repo (releases, commits, PRs, issues), the Kaggle/kaggle-cli repo and docs, the PyTorch dev-discuss mailing list, and Hugging Face docs. Kaggle's own web pages (kaggle.com/docs/*, discussion posts) are rendered with JavaScript and could not be fetched directly. Anything taken only from search-result snippets is marked "(snippet only)".

## 1. Kaggle docker image versions (torch / CUDA / transformers / accelerate), update cadence, pinning

### Takeaway
The latest published Kaggle GPU image is **v170 (released 2026-06-29)**. It ships **torch 2.10.0+cu128**, which has been in place since v167 (March 2026). Transformers is on the **5.x** line (`transformers>=5.0.0` in requirements; v167 pinned 5.0.0), and accelerate is **1.13.0** on the GPU image. An unreleased image built on a new Colab base (**Python 3.13, torch 2.11**) was merged on **2026-09-28** and will probably reach notebooks soon. That means the environment could change mid-project unless it is pinned.

### Cited Findings
- Releases are tagged per variant (CPU/GPU/TPU). Recent ones: v166 (2026-02-17), v167 (2026-03-19/20), v168 (2026-03-20), v169 (2026-05-25), v170 (2026-06-29). Every release note says "This image will be available on Kaggle Notebooks in the next few days", so rollout is staged. — [Kaggle/docker-python releases](https://github.com/Kaggle/docker-python/releases)
- The v170 GPU image is `gcr.io/kaggle-private-byod/python:v170`, digest `bdf9e053…a068`. The CPU image is `gcr.io/kaggle-images/python:v170`. — [v170 releases](https://github.com/Kaggle/docker-python/releases)
- Version history from the release pip-freeze diffs (GPU):
  - v166 (2026-02-17): torch 2.8.0+cu126 → **2.9.0+cu126**; transformers 4.57.1 → **5.2.0**; accelerate 1.11.0 → 1.12.0; tokenizers 0.22.2; safetensors 0.7.0. — [releases](https://github.com/Kaggle/docker-python/releases)
  - v167 (2026-03-19/20): torch 2.9.0+cu126 → **2.10.0+cu128**; torchvision 0.25.0+cu128; transformers 5.2.0 → **5.0.0** (an odd downgrade, as shown in the diff). — [releases](https://github.com/Kaggle/docker-python/releases)
  - v169 (2026-05-25): accelerate 1.12.0 → **1.13.0**; huggingface_hub 1.4.1 → 1.10.1; numpy 2.0.2 → 2.4.6; cuda-bindings 12.9.4 → 13.2.0. — [releases](https://github.com/Kaggle/docker-python/releases)
  - v170 (2026-06-29, GPU): huggingface_hub 1.10.1 → 1.11.0; peft 0.18.1 → 0.19.1; nvidia-cuda-nvcc-cu12 12.5.82 → 12.8.93. The diff also shows numpy 2.4.6 ↔ 2.0.2 and cuda-bindings 13.2.0 ↔ 12.9.4 flipping back, so the diff direction and the resulting numpy version are ambiguous. torch and transformers did not appear in the v170 or v169 diffs, so they were unchanged. — [releases](https://github.com/Kaggle/docker-python/releases)
  - The TPU v170 image differs: accelerate 1.14.0, huggingface_hub 1.21.0, datasets 5.0.0. — [releases](https://github.com/Kaggle/docker-python/releases)
- Users observed torch 2.10.0+cu128 in June 2026 ([issue #1546 comment, 2026-06-14](https://github.com/Kaggle/docker-python/issues/1546)) and `torch: 2.10.0+cpu` on the CPU image on 2026-09-17 ([kaggle-cli #1197](https://github.com/Kaggle/kaggle-cli/issues/1197)). Together these confirm that torch 2.10 is what users are getting as of September 2026.
- Kaggle images are built **on top of Colab runtime images**. The Dockerfile freezes `tensorflow|keras|torch|jax` from the Colab base and then installs `kaggle_requirements.txt` with `uv pip`. So torch/CUDA versions come from Colab, and Kaggle adds packages on top. — [Dockerfile.tmpl](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl)
- `kaggle_requirements.txt` now contains `transformers>=5.0.0` and `torchcodec==0.11.0`. — [kaggle_requirements.txt](https://github.com/Kaggle/docker-python/blob/main/kaggle_requirements.txt)
- **Pending changes, merged but not yet released as of 2026-09-29:**
  - PR #1557 (merged 2026-09-02) moved the base to Colab `20260716-060051`, which brings **torch 2.10 → 2.11** and torchvision 0.25 → 0.26. — [PR #1557](https://github.com/Kaggle/docker-python/pull/1557)
  - PR #1562 (merged 2026-09-28) moved the base to Colab `20260917-060051`. "The new base image moves from Python 3.12 to **Python 3.13**." It also removed tensorflow-io (no cp313 wheel). — [PR #1562](https://github.com/Kaggle/docker-python/pull/1562)
- Update cadence: base-image bumps happened on 2026-02-02, 03-17, 04-27, 06-22, 09-02 and 09-28, with releases roughly every 1–2 months. — [commit history](https://github.com/Kaggle/docker-python/commits/main)
- Pinning via API: `kaggle kernels pull -m` writes a `"docker_image": "gcr.io/kaggle-images/python@sha256:..."` field into `kernel-metadata.json`, and the `.ipynb` keeps `metadata.kaggle.dockerImageVersionId`. This pins the environment when the notebook is pushed back. — [Kaggle staff comment on kaggle-cli #1197](https://github.com/Kaggle/kaggle-cli/issues/1197)

### Inferences
- The project should assume **torch 2.10.0+cu128, Python 3.12, transformers 5.x** today. It could get **torch 2.11 + Python 3.13** after the next image rolls out, which is likely in October 2026 given the "next few days" pattern after a release is tagged.
- If the plan pins a **transformers 4.x** version, it is running against a preinstalled 5.x. Both the downgrade and the API differences matter (see section 3).
- For reproducibility, do three things: (a) in the notebook editor, pin the environment to the original version, or keep the `docker_image` / `dockerImageVersionId` when pushing through the API; (b) print `torch.__version__`, `torch.version.cuda`, `transformers.__version__`, `torch.cuda.get_device_name()` and `torch.cuda.get_arch_list()` in the first cell and save them to the output; (c) pin every pip-installed package exactly.

### Gaps
- I could not see the exact transformers version installed in v170 or v169. The last explicit value in a diff is 5.0.0 (v167), and no later diff shows it changing. Verify at runtime.
- I could not fetch the Kaggle docs page describing the editor's "Environment → Pin to original environment / Always use latest" option (JavaScript-rendered). The option exists per long-standing Kaggle UI, but its exact current wording is unverified.
- It is unknown when a v171 image built on the Python 3.13 / torch 2.11 base will be published.

## 2. Architecture support: PyTorch / CUDA vs T4 (sm_75) and P100 (sm_60); Kaggle incidents

### Takeaway
**T4 (sm_75, Turing) is fully supported** by every current PyTorch build, including cu128, cu129 and CUDA 13.x. **P100 (sm_60) broke on Kaggle in March 2026**, when the image moved to torch 2.10.0+cu128, and Kaggle **retired the P100 on 2026-09-15**. Sessions that request P100 are now switched automatically to T4 x2. For this project the sm_60 issue no longer applies, but "no kernel image" errors will appear if someone pip-installs a CPU-only torch build or one that lacks sm_75.

### Cited Findings
- The PyTorch 2.8 cu128/cu129 builds removed Maxwell and Pascal (sm_50, sm_60). The cu126 builds kept them. — [PyTorch dev-discuss: Maxwell/Pascal removed in CUDA 12.8 and 12.9 builds](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128)
- PyTorch 2.11 dropped Volta (sm_70) from the CUDA 12.8 binaries, so "the minimum GPU version supported by CUDA-12.8+" became Turing. — [dev-discuss: Dropping Volta from CUDA-12.8 for 2.11](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290); [RFC #172352](https://github.com/pytorch/pytorch/issues/172352)
- CUDA 13.0 dropped sm_50–sm_72, so its supported architectures start at Turing (sm_75). — [search summary of PyTorch RFC #190385](https://github.com/pytorch/pytorch/issues/190385) (snippet only)
- **PyTorch 2.14 is the last release with CUDA 12.6 wheels.** From 2.15 there are no cu126 binaries, so Maxwell, Pascal and Volta lose prebuilt support. The notice says "CUDA 13.x does not support compute capabilities below Turing (sm_75)", and **Turing sm_75 is retained in CUDA 13.x builds**. CUDA 12.6 was removed from nightlies on 2026-09-07. — [dev-discuss notice: CUDA 12.6 wheels end at 2.15](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- Kaggle incident: [docker-python issue #1546](https://github.com/Kaggle/docker-python/issues/1546), "Pytorch CUDA P100 GPU Incompatibility", opened **2026-03-20**.
  - Errors: "Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation" and "CUDA error: no kernel image is available for execution on the device". The listed archs were sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
  - A user comment on 2026-06-14 says that on T4 (sm_75) "everything works natively" with torch 2.10.0+cu128. The failure was P100-only. On P100, matmul worked through cuBLAS, but reductions and `.backward()` failed.
  - Workarounds users posted: reinstall torch from the cu126 or cu118 index.
- Kaggle's own PR #1544 (2026-03-19): "Newer version of pytorch is needed for rtx pros / cuda sm_120, but now it no longer works on P100s / cuda sm_60". It marked the PyTorch tests as `p100_exempt`. — [PR #1544](https://github.com/Kaggle/docker-python/pull/1544)
- PR #1560 "Drop P100 support" (merged 2026-09-05) removed the P100 CI stage and made T4x2 "the sole GPU test bed". — [PR #1560](https://github.com/Kaggle/docker-python/pull/1560). A community PR #1561 to restore sm_60 kernels ("best-effort, unverified") is still open. — [issues/PR list](https://github.com/Kaggle/docker-python/pulls)
- Kaggle announcement "Sunsetting the NVIDIA Tesla P100 GPU on September 15, 2026". Its stated reasons are that the hardware is aging, Google Cloud is winding it down, and queues were long. After that date, P100 notebooks are switched automatically to T4x2. — [Kaggle product announcement 735239](https://www.kaggle.com/discussions/product-announcements/735239) (snippet only)

### Inferences
- The T4 is safe on the current stack. The cu128 wheels (torch 2.10/2.11) include sm_75, and CUDA 13 builds keep Turing. There is no announced deprecation of Turing in PyTorch or CUDA as of September 2026.
- The P100 question is moot for new runs, because P100 is gone from Kaggle. Old notebooks or `kernel-metadata.json` files that name `NvidiaTeslaP100` will be silently mapped to T4x2 (see section 7).
- T4 has no bf16 tensor-core support, so fp16 is the right dtype, as the project already plans. This is general knowledge and not sourced here.

### Gaps
- I found no source announcing a future end of sm_75 support in PyTorch or CUDA.

## 3. pip-installing a pinned transformers on Kaggle: conflicts and restarts

### Takeaway
The preinstalled stack is transformers 5.x with huggingface_hub 1.x. Installing an older 4.x transformers **downgrades across a major version**. This can conflict with huggingface_hub 1.x, tokenizers, peft 0.19 and sentence-transformers 5.4, and it must happen **before `transformers` is imported**. "Save & Run All" cannot restart the kernel mid-run.

### Cited Findings
- The installed stack includes transformers>=5.0.0, huggingface_hub 1.11.0 (GPU v170), peft 0.19.1, sentence-transformers 5.4.1 and accelerate 1.13.0. — [v170 / v169 release diffs](https://github.com/Kaggle/docker-python/releases); [kaggle_requirements.txt](https://github.com/Kaggle/docker-python/blob/main/kaggle_requirements.txt)
- Kaggle's image build already runs into resolver fights: pins for protobuf 5.29.5, setuptools<82, fastcore<2 and fury<2, plus notes such as "numba-cuda CPU import crash after base image upgrade" (#1548). — [Dockerfile.tmpl](https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl); [commits](https://github.com/Kaggle/docker-python/commits/main)
- Issue #1546 shows that the notebook runner can have torch already loaded before user code runs. The commenter's fix had to uninstall and reinstall torch *before* importing it, to avoid "the C extension reload problem". — [issue #1546](https://github.com/Kaggle/docker-python/issues/1546)
- Kaggle forum threads say pip installs request a kernel restart that cannot be done during commit (Save & Run All). Suggested workarounds are to install before any import or to use a separate environment. — [Kaggle discussion 300431](https://www.kaggle.com/discussions/getting-started/300431); [Kaggle Q&A 272757](https://www.kaggle.com/discussions/questions-and-answers/272757) (snippet only)

### Inferences
- Safe pattern: make the first cell `!pip install -q "transformers==X" "accelerate==Y" "tokenizers==Z"` (plus huggingface_hub if you go back to 4.x), with no earlier `import transformers`, `import torch` or import of any library that pulls them in. Then assert the versions in the next cell.
- **Do not reinstall torch.** The preinstalled 2.10.0+cu128 works on T4. A plain `pip install torch` could pull a CUDA 13 wheel whose driver requirement may not match Kaggle's host driver (the driver version is unverified).
- If the code was written for transformers 4.x, remember that 5.x changed APIs. For example, the `torch_dtype` argument was renamed to `dtype`, and generation and cache internals changed. This is from general knowledge and **unverified** here. Either pin 4.x consistently or test the code against 5.x. Assisted generation (`assistant_model=`) exists in both, but defaults such as `num_assistant_tokens` and its schedule should be pinned explicitly.
- Transformers 4.x paired with huggingface_hub 1.x may be incompatible, since 4.x releases required `huggingface_hub<1.0`. Unverified; check pip's resolver output.

### Gaps
- I found no Kaggle-specific 2026 report of a transformers 4.x downgrade breaking the image. Evidence is generic.

## 4. T4 x2 specifics: P100 substitution, device_map="auto", CUDA_VISIBLE_DEVICES, clocks, quota

### Takeaway
As of September 2026, the GPU option **is** T4 x2. Single T4 is no longer offered, and P100 is retired. `device_map="auto"` will see **2 GPUs** and may shard a 3B model across them, which adds cross-GPU transfers and distorts latency measurements. Pin everything to one GPU.

### Cited Findings
- Kaggle staff, 2026-09-17: "single T4 is no longer supported—the default T4 GPU pool provisions **2× NVIDIA T4 GPUs (GPU T4 ×2, 30 GB total VRAM)**". The `machine_shape` identifier for T4 x2 is `NvidiaTeslaT4`, and `NvidiaTeslaT4Highmem` is internal/admin-only. — [kaggle-cli #1196](https://github.com/Kaggle/kaggle-cli/issues/1196)
- P100 was retired 2026-09-15, and notebooks that requested it are auto-switched to T4x2. — [Kaggle announcement](https://www.kaggle.com/discussions/product-announcements/735239) (snippet only)
- The kaggle-cli docs list accelerators "as of Sep 2026": NvidiaTeslaT4 (GPU T4 ×2), NvidiaTeslaA100, NvidiaL4, TpuV5E8, TpuV6E8. — [kaggle-cli docs/kernels.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- Weekly GPU quota is 30 hours per week per accelerator type, and sessions can run up to 12 hours for CPU and GPU. — [Kaggle notebooks docs](https://www.kaggle.com/docs/notebooks) (snippet only)

### Inferences
- `device_map="auto"` with accelerate spreads layers across all visible GPUs when the "balanced" logic decides to (the default for multi-GPU). For a clean speculative decoding latency benchmark, use one of these:
  - set `os.environ["CUDA_VISIBLE_DEVICES"]="0"` before importing torch;
  - use `device_map={"": 0}` or `.to("cuda:0")`;
  - deliberately place target and draft on separate GPUs as a named experimental condition, not by accident.
- Quota counts per session-hour, not per GPU. Two T4s cost one hour of the 30h/week quota for each wall-clock hour. This is inferred from "30 hours per week per accelerator type" and **unverified**.
- T4s are 70W, passively cooled cards that can throttle under sustained load. Run warm-up iterations, record `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw` during runs, and report the GPU index. No Kaggle-specific source was found on thermal variance, so this is general knowledge.

### Gaps
- There is no official Kaggle statement on T4 clock or thermal consistency, or on whether one GPU of the pair can be slower than the other.
- The quota accounting for 2 GPUs is not confirmed by an official source in this research.

## 5. Hugging Face downloads on Kaggle: rate limits, token, hf_transfer/Xet, disk

### Takeaway
Qwen2.5 is ungated, so no token is strictly needed. Anonymous downloads are rate-limited **per IP**, and Kaggle sessions probably share NAT IPs, so set `HF_TOKEN` from Kaggle Secrets. `hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER` is **deprecated**; the image's huggingface_hub 1.x uses `hf-xet`, and `HF_XET_HIGH_PERFORMANCE=1` is the replacement. The two models are about 7.5 GB in fp16 safetensors, which fits within Kaggle's disk.

### Cited Findings
- HF rate limits (stated as of September 2025), in 5-minute fixed windows. Anonymous users are limited **per IP address** to 500 API, **3,000 Resolver** and 100 Pages requests. Free logged-in users get 1,000 / 5,000 / 200. HF says anonymous and free limits are "subject to change over time depending on platform health". The page says to always pass `HF_TOKEN`, calling it "the number one reason users get rate limited". Rate-limited requests return 429, and huggingface_hub ≥1.2.0 retries automatically using the `RateLimit` header. — [HF Hub rate limits](https://huggingface.co/docs/hub/rate-limits)
- `HF_HUB_ENABLE_HF_TRANSFER` "is a deprecated environment variable… all file transfers go through the `hf-xet` binary package… `hf_transfer` can't be used anymore". Use `HF_XET_HIGH_PERFORMANCE` instead. — [huggingface_hub env vars](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- The default cache is `HF_HOME=~/.cache/huggingface`, so `/root/.cache/huggingface/hub` on Kaggle. Override it with `HF_HOME` or `HF_HUB_CACHE`. Environment variables are read when huggingface_hub is imported, so set them before importing. `HF_HUB_OFFLINE=1` skips network checks. — [huggingface_hub env vars](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- The image ships hf-xet (1.5.x on TPU v170) and huggingface_hub 1.11.0 on GPU v170. — [releases](https://github.com/Kaggle/docker-python/releases)
- Kaggle disk: `/kaggle/working` has 20 GB that is auto-saved as output. There is additional scratch space outside `/kaggle/working` that is not saved. — [Kaggle notebooks docs](https://www.kaggle.com/docs/notebooks) (snippet only)
- Transformers issue #45797 reports "HF Hub is DOWN on AWS in Xet mode", an example of Xet-path outages. `HF_HUB_DISABLE_XET=1` is the documented opt-out. — [transformers #45797](https://github.com/huggingface/transformers/issues/45797) (title only); [env vars doc](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

### Inferences
- Model sizes: Qwen2.5-3B is about 6.2 GB in bf16 safetensors and 0.5B is about 1 GB, from general knowledge of the HF repos (unverified here). Both fit in the default `/root/.cache`. **Do not** cache into `/kaggle/working`, because that would save several GB of weights as notebook output and eat into the 20 GB.
- If a transformers 4.x downgrade also downgrades huggingface_hub below 1.x, check that hf-xet is still used. Older hubs used hf-xet from 0.32+ (unverified).
- Add retry logic, or use `snapshot_download` with pinned `revision=` commit hashes, so model weights are reproducible.

### Gaps
- There are no measured HF download speeds from Kaggle in 2026.
- The exact size of the scratch disk and `/tmp` on T4 x2 machines was not confirmed. Older forum posts cite roughly 70+ GB, but no 2026 source was found. Check with `df -h` at runtime.

## 6. Qwen2.5 as a Kaggle Model; path layout

### Takeaway
Qwen2.5 is published on Kaggle Models under **qwen-lm/qwen2.5**, with a **Transformers** framework and a `3b-instruct` variation. A `0.5b-instruct` variation very likely exists but was not directly confirmed. Attaching it avoids HF downloads and rate limits.

### Cited Findings
- [kaggle.com/models/qwen-lm/qwen2.5/Transformers/3b-instruct](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/3b-instruct) and [.../Transformers/7b-instruct/1](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/7b-instruct/1) exist (search results). The model description matches the Qwen2.5 card (0.5B–72B). — (snippet only)
- In `kernel-metadata.json`, `model_sources` entries use the format `"username/model-slug/framework/variation-slug/version-number"`, for example `"qwen-lm/qwen2.5/transformers/3b-instruct/1"`. — [kaggle-cli docs/kernels_metadata.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- `kagglehub` 1.0.x is preinstalled and can download model handles programmatically. — [releases](https://github.com/Kaggle/docker-python/releases)

### Inferences
- Attached models conventionally mount at `/kaggle/input/<model-slug>/<framework>/<variation>/<version>/`, for example `/kaggle/input/qwen2.5/transformers/3b-instruct/1/` containing `config.json`, `*.safetensors` and the tokenizer files. This is **unverified** for 2026, since Kaggle has changed mount layouts before. Use `glob` or `os.walk` on `/kaggle/input` to find `config.json` rather than hard-coding the path, or use `kagglehub.model_download("qwen-lm/qwen2.5/transformers/3b-instruct")`, which returns the path.
- Kaggle Model files may differ from the current HF revision (they are snapshots). Record which source was used, and compare against HF commit hashes if exact reproducibility matters. Qwen tokenizer and generation_config changes between snapshots can change outputs.

### Gaps
- The Kaggle model page could not be rendered to confirm the `0.5b-instruct` variation, the version number and file contents.

## 7. Phone verification; `kaggle kernels push` with GPU; pulling outputs; other breakers

### Takeaway
Phone (SMS) verification is still required to unlock GPU and internet. Headless runs through `kaggle kernels push` work with `enable_gpu: true`, `enable_internet: true` and `machine_shape: "NvidiaTeslaT4"`, which gives T4 x2. Outputs are fetched with `kaggle kernels output`. Unknown or retired `machine_shape` values can be substituted silently.

### Cited Findings
- Phone verification: to get GPU or internet access, use the "Get phone verified" link in the notebook settings sidebar. VoIP numbers commonly fail. — [Neuromatch Kaggle tutorial](https://deeplearning.neuromatch.io/tutorials/TechnicalHelp/Tutorial_kaggle.html); [Kaggle product-feedback 506377](https://www.kaggle.com/product-feedback/506377) (snippets only; no 2026 official Kaggle statement fetched)
- `kernel-metadata.json` fields:
  - `id`, `title`, `code_file`, `language`, `kernel_type`
  - `is_private` (default true), `enable_gpu` (default false), `enable_internet` (default false)
  - `machine_shape` ("`NvidiaTeslaT4` for GPU T4 ×2, `NvidiaL4`, or `TpuV5E8`")
  - `dataset_sources`, `competition_sources`, `kernel_sources`, `model_sources`

  Source: [kaggle-cli docs/kernels_metadata.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)
- `kaggle kernels push -p <dir> [--accelerator NvidiaTeslaT4] [-t TIMEOUT_SECONDS] [--no-run]`. Accelerators listed "as of Sep 2026": NvidiaTeslaT4 (GPU T4 ×2, the default GPU), NvidiaTeslaA100, NvidiaL4, TpuV5E8, TpuV6E8. — [kaggle-cli docs/kernels.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- `kaggle kernels output <owner/slug[/version]> -p <dir> [-o] [--file-pattern REGEX] [--page-size up to 200] [--page-token]` downloads the output of the latest run. Outputs are paginated, 20 files by default. — [kaggle-cli docs/kernels.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md). Use `kaggle kernels status <slug>` to poll.
- Known silent failures with push:
  - kaggle-cli #1192 (merged 2026-09-11) adds a warning when a *retired* `machine_shape` is silently substituted on the server side. — cited in [kaggle-cli #1196](https://github.com/Kaggle/kaggle-cli/issues/1196)
  - A `docker_image` pinned in `kernel-metadata.json` (from `kaggle kernels pull -m`) overrides the accelerator image. This caused TPU requests to run on the CPU image (`torch 2.10.0+cpu`). — [kaggle-cli #1197](https://github.com/Kaggle/kaggle-cli/issues/1197)
  - Picking an accelerator in the editor UI only changes the interactive draft until a version is saved. — [kaggle-cli #1196](https://github.com/Kaggle/kaggle-cli/issues/1196)
- Other 2025–2026 breakers seen in the repo:
  - base-image bumps that change numpy between 2.0.2 and 2.4.6;
  - numba-cuda import crashes (#1548);
  - the Python 3.12 → 3.13 move (#1562), which removed packages with no cp313 wheels.

  Sources: [releases](https://github.com/Kaggle/docker-python/releases); [commits](https://github.com/Kaggle/docker-python/commits/main); [PR #1562](https://github.com/Kaggle/docker-python/pull/1562)

### Inferences
- In the first cell, assert `torch.cuda.device_count()==2` and that the device name contains "T4", and fail fast otherwise. This catches silent substitution to a CPU or wrong image. Also assert `torch.cuda.is_available()`, since an image mismatch can silently give `+cpu` torch.
- Do not blindly reuse `kernel-metadata.json` from `pull -m` with an old `docker_image` digest unless that pin is intended. If it is intended, it is the most reliable way to freeze the environment.
- The Python 3.13 image transition is the most likely near-term breaker for pinned wheels. Pin versions that publish cp313 wheels. Current transformers, accelerate and tokenizers do, but verify.

### Gaps
- There is no official 2026 Kaggle statement on phone verification rules; the requirement is corroborated only by tutorials and forum snippets.
- I did not confirm whether `enable_internet` through push also requires phone verification. It very likely does, but this is unverified.
- `kaggle kernels push` timeout defaults and GPU quota accounting for pushed runs are unconfirmed.
