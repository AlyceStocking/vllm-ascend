# Experimental MLA Prolog compatibility adapter

This is an opt-in integration candidate, not a validated GLM deployment or a performance claim. The default `torch_npu` backend preserves existing SFA routing, including native preprocessing for NoPE. The optional `cannbot` backend maps the A5 DSL operator to the same five-output boundary. No model or runner is duplicated.

## Enabling the experimental path

Install the matched CANNBotDSL and operator wheels in a compatible Linux x86_64 Python 3.12 NPU environment. The reviewed operator distribution is `cannbot-arena-net==0.3.0`, with dependency `cannbot-dsl>=0.7.0,<0.8`; its build metadata records DSL 0.7.1 and CANN 9.2.0-beta.2. The distribution version alone does not identify the artifact: the reviewed quantized wheel has SHA256 `c585750272eb8fc1e9e07bc8e3d68a790951c73e518b0cb1173e800dfcdf0d06`. Confirm the actual runtime and matching DSL wheel with the operator provider. This change does not install or vendor either package.

With an otherwise working GLM5.3-Flash MXFP8 configuration, the experimental selection is:

```bash
export VLLM_ASCEND_MLA_PROLOG_BACKEND=cannbot
vllm serve /path/to/GLM-5.3-Flash-w8a8-mxfp8 \
    --enforce-eager \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size <your-validated-TP-size>
```

Retain the other model-specific arguments from your known-working deployment. This is a configuration example, not a tested serving command. Do not enable speculative decoding, context/sequence parallelism, PD, sparse KV offload, or RL weight reloads with this initial adapter. For BF16 main cache validation, omit the FP8 cache setting and keep the original BF16 cache configuration.

Unset the variable, or set it to `torch_npu`, to restore the default. The variable is not sensitive. The opt-in applies only to NoPE layers; existing RoPE layers keep the torch_npu backend. Unsupported explicit selections fail rather than silently reverting after cache mutation.

## Supported interface envelope

| Property | Initial boundary |
| --- | --- |
| Hardware | A5 FP8-optimized profile; the wheel itself targets `dav-3510` |
| Execution | Eager only; `--enforce-eager` required |
| Activations | BF16 before the existing dynamic MXFP8 input quantization |
| Projection and Indexer methods | Exact MXFP8 method, group size 32, scale algorithm 0; no bias or TP input padding |
| Dimensions | NoPE `(Dn,Dr)=(256,0)`, KV rank 512; hidden size 1024..8192 aligned to 256; Q rank 640/1024/1536/2048; local heads 1..128 |
| Normalization | Equal Q/KV epsilon; independent FP32 gamma snapshots |
| Main cache | Contiguous PA_BSND BF16, or experimental packed FP8 E4M3 C8 with 128-element groups and 528-byte slots |
| Page size | 16..1024, aligned to 16, subject to the existing attention backend requirements |
| Query output | BF16; no full-Q quantization or HiF8 path |
| Query norm output | E4M3 values plus per-token/per-32 E8M0 scales for the existing MXFP8 Indexer projection |

Shape admission is not evidence that the wheel contains an AOT specialization for the model. Verify native specialization availability and any JIT behavior on the target machine before treating the configuration as deployable. INT8 C8, TurboQuant, different epsilon values and unsupported shapes are rejected.

## Integration design

The [central operator adapter](../../../vllm_ascend/ops/mla_prolog.py) owns optional imports, preparation, dispatch, cache views and output normalization. Its default backend forwards to `torch_npu.npu_mla_prolog_v3`. Cannbot calls only the public `ops.quant_mla_prolog.quant_mla_prolog`; it does not bypass validation through a private graph/native entry.

The [shared SFA implementation](../../../vllm_ascend/attention/sfa_v1.py) retains the existing preprocessing lifecycle, raw hidden-state input to the Indexer, connector/cache-write ordering and downstream attention. Preparation converts loaded `(in,out)` weights and `(in/64,out,2)` scales into independent ND `(out,in)` weights and `(out,in/64,2)` E8M0 scales. Original projection parameters remain intact. The prepared snapshots increase resident memory and are not refreshed by RL reloads, which are consequently rejected. No weight transpose or repack is performed per token.

The adapter returns `(query, query_rope, query_scale, query_norm, query_norm_scale)`. `query_rope` has explicit shape `[T,N,0]`; Cannbot `qr` is the normalized low-rank Query, not RoPE Query. The [KPool Indexer](../../../vllm_ascend/attention/indexer_kpool.py) can slice and forward an E4M3/E8M0 tuple to its existing MXFP8 linear method without re-quantization. Its tensor-input behavior is unchanged.

Cache adaptation uses storage-preserving views, including byte reinterpretation for packed FP8. A noncontiguous cache is rejected, never silently copied. Failure after a kernel launch is propagated: the adapter does not run native preprocessing again on a potentially modified cache.

## Validation and limitations

The new CPU tests use real CPU torch tensors and mocked operator/NPU calls:

```bash
python -m pytest -q --confcutdir=tests/ut/ops tests/ut/ops/test_mla_prolog_adapter.py
python -m pytest -q --confcutdir=tests/ut/attention tests/ut/attention/test_mla_prolog_compat.py
```

The isolated configuration deliberately avoids unrelated parent test fixtures. The SFA/KPool tests execute extracted production methods with injected dependencies; they do not import the complete vLLM runtime. These tests establish interface, routing and alias contracts only, not true NPU format behavior, vendor execution or numerical correctness. Run the normal repository UT/CI environment as well before merge.

Outstanding hardware checks:

1. Verify the exact wheel, DSL, CANN, torch/torch_npu, vLLM and vllm-ascend revisions, actual weight/scale layouts and AOT coverage.
2. Compare native versus fused Query, dequantized query-norm, cache payload and embedded FP32 scales. Agree tolerances before interpreting the results.
3. Check GLM KPool top-k selection, model logits and representative dataset accuracy, including shared-index layers and page/sequence boundaries.
4. Exercise repeated eager calls with new inputs and slots; verify the original cache storage is updated, including padding slots.
5. Measure memory and performance separately. Independent prepared weights use additional memory, and the vendor checked eager entry currently reads cache indices back to the CPU. No speedup is claimed.
6. Validate torch.compile and ACLGraph capture/replay with mutable-cache semantics separately before relaxing the eager-only gate. A vendor graph entry is not proof of vLLM graph compatibility.

The fused FP32 projection/RMSNorm and cache quantization do not have identical rounding stages to the native BF16 chain. A matching 528-byte cache layout does not establish numerical equivalence.

Before opening an upstream PR, add a numbered release-note fragment per the repository policy. Suggested text: "Added an experimental, opt-in A5 MXFP8 NoPE MLA Prolog compatibility adapter with a shared SFA entry point; hardware correctness and performance validation is required before deployment." No PR number is assigned by this branch-only change.
