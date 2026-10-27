# GLM-5.3-Flash NoPE Prolog V3 integration

This branch integrates [ops-transformer PR 14225](https://gitcode.com/cann/ops-transformer/pull/14225)
with SFA and the GLM-5.3-Flash KPool indexer. Hardware validation is still required;
the opt-in is disabled by default. CPU tests do not establish kernel correctness,
full-model accuracy, or a performance improvement.

## Source and supported contract

The source is pinned to `b581a9734e30ed43a08dc5f316ce3adbd49d0560` from
`yolic/transformer-readme:1019prolog`, with PR base
`4b92adc974cade0ae510e3b6265320e5f6fbc380`. The PR was open when retrieved.
Its latest tiling check permits Dr=0 only on DAV_2201 (A2/A3), with
`weightQuantMode=2` and `kvCacheQuantMode=3`.

| Item | Required value |
| --- | --- |
| MLA projections | INT8 W8A8_DYNAMIC, NZ weights |
| Activations, normalization, weight_uk | BF16 |
| qk_rope_head_dim / qk_nope_head_dim | 0 / 256 |
| kv_lora_rank | 512 |
| q_lora_rank | 1536 or 2048 |
| Local query heads | 1 through 128 |
| Main KV cache | INT8, packed values and FP32 scales, 528 bytes per token |
| weight / KV / query quantization modes | 2 / 3 / 0 |
| ckvkr_repo_mode / quant_scale_repo_mode / tile_size | 1 / 1 / 128 |
| Context parallelism | Disabled for this integration |

A5/MXFP8, BF16 main KV caches, FP8 main KV caches, and TurboQuant are outside
this PR's Dr=0 contract. An INT8 KV-cache option does not turn MXFP8 or BF16
projection weights into INT8 W8A8 weights.

The fork's `main` at `041768d43c8cb6b2191db8f4c79e45054cb6b2e8` contains the
`torch_npu.npu_mla_prolog_v3` framework call, but no MLA Prolog kernel sources.
The branch therefore imports the V3 implementation and shared Prolog sources
into `csrc/attention/`, using the existing custom-operator build and vendor loading.
It does not create a new Torch operator ABI. The selected build target is
`mla_prolog_v3`; `mla_prolog` is a source dependency, not the operator to build.

`csrc/attention/mla_prolog_v3/UPSTREAM.json` records source paths and SHA256 hashes.
Imported kernel, tiling and inference code matches the pinned PR head, with
LF line endings and trailing whitespace normalized.
The V3 host CMake file additionally compiles the shared tiling implementation.
The original CANN license and copyright headers are preserved. The package
provides `MlaPrologV3` and `aclnnMlaPrologV3WeightNz` through the custom vendor,
so a matching CANN toolkit/compiler and a fresh custom-package build are required.

## Framework behavior

Set `enable_nope_prolog_v3` in `--additional-config` to request the fused path.
Unsupported hardware, projection quantization, dimensions, or cache formats fail
at weight loading rather than silently falling back to native preprocessing.
Successful route selection logs:

```text
NoPE SFA preprocessing: PROLOG_V3 (PR14225, W8A8C8).
```

NoPE supplies rank-two empty sin/cos tensors `[tokens, 0]`. The empty query-rope
output is reshaped with an explicit token count; `view(-1, heads, 0)` is ambiguous.
Cache indices are trimmed to the same token rows as the input, while FULL graph
buckets retain padding rows and `-1` slots.

`query_quant_mode=0` controls the final MLA query. With W8A8, the auxiliary
`query_norm` (`q_c`) is still INT8 with one FP32 dequantization multiplier per
token. The integration computes `(q_c.float() * scale).bfloat16()` before KPool's
query projection. KPool can therefore retain its original BF16 projection weights.
The extra INT8 round trip introduces quantization error and must be included in
full-model accuracy and performance comparisons.

## Build and single-operator validation

Use an A2/A3 container with the repository's matched vLLM/PyTorch/torch_npu
dependencies and a CANN toolkit compatible with the imported PR source.
The tested vLLM source reference for CPU checks is in
`.github/vllm-main-verified.commit`.

```bash
git checkout work/glm53flash-prolog-pr14225
# Use the SOC_VERSION reported for your actual A3 device, for example:
export SOC_VERSION=ascend910_9391
python -m pip install -e . --no-build-isolation

# Set this before starting Python or vLLM, including all worker processes.
export ASCEND_CUSTOM_OPP_PATH="$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export ASCEND_RT_VISIBLE_DEVICES=0
python tools/smoke_nope_prolog_v3.py --tokens 129
python tools/smoke_nope_prolog_v3.py --tokens 129 --graph
```

The smoke test runs the real Prolog and compares its query, dequantized `q_c`,
and unpacked C8 cache against separate projection/normalization operations.
It checks that padding and unused cache rows remain untouched. It has no native
fallback. A failure is a failed check, not evidence that the operator is usable.
The graph check covers one operator capture/replay; full-model graph verification
remains a separate acceptance step.

## Full-model validation

Choose the tensor-parallel size to match the allocated devices and checkpoint.
Start with eager execution, no MTP and no context parallelism:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export TP_SIZE=16
vllm serve /mnt/weight/GLM-5.3-Flash-w8a8 \
  --host 0.0.0.0 --port 8008 --served-model-name glm \
  --trust-remote-code --quantization ascend --dtype bfloat16 \
  --kv-cache-dtype int8 --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel --seed 1024 \
  --max-model-len 8192 --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --additional-config '{"enable_nope_prolog_v3":true}'
```

Restore `ASCEND_RT_VISIBLE_DEVICES` to the allocated device set before this run;
the single-operator command above selected only device 0.

```bash
curl --fail-with-body http://127.0.0.1:8008/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","prompt":"Explain why the sky looks blue.","max_tokens":64,"temperature":0}'
```

Acceptance requires all of the following, recorded with hardware/software versions:

1. Eager model loading and both prefill and decode complete; the route log confirms
   PROLOG_V3, and profiling shows `MlaPrologV3` actually executing.
2. Run the same workload with `enable_nope_prolog_v3=false` as a native control.
   Compare representative task accuracy/logits and performance with identical
   weights, cache dtype and sampling settings; one plausible response is insufficient.
3. Verify chunked prefill, batch padding and decode graph buckets, then MTP draft
   and verification. Confirm query rows and cache indices stay aligned.
4. Complete real CANN compilation and NPU numerical checks before declaring this
   branch ready to merge. No full-model or throughput result is claimed here.

## CPU regression checks

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/ut/attention/test_nope_prolog.py \
  tests/ut/attention/test_sfa_nope_prolog.py \
  tests/ut/attention/test_sfa_nope_forward.py \
  tests/ut/attention/test_sfa_v1.py
bash format.sh ci
```

## Validation recorded on 2026-10-27

- CPU: 119 tests and 11 subtests passed across the four files above, using
  PyTorch 2.10.0 CPU and the repository's verified vLLM source
  `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`.
- `git diff --check`, Python compilation, Ruff, Markdown lint, codespell,
  secret scan, shell lint and the other repository format hooks passed.
- The whole-repository `typos` hook was terminated with exit code -9 in this
  environment. The same hook passed on all changed files to which it applies;
  imported C++ sources are excluded by the repository's existing hook settings.
- Real CANN build, single-operator NPU smoke, full-model inference, accuracy,
  MTP/graph integration and performance: **not run**. No NPU/CANN runtime or
  remote cluster execution connection was available in the editing environment.
