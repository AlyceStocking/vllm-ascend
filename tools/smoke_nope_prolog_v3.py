# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2/A3 single-card numerical check for the bundled PR14225 operator.

Run after building/installing vllm-ascend. This executes the real operator;
there is no mock or fallback. It does not constitute full-model validation.
"""

import argparse

import torch
import torch_npu

from vllm_ascend.attention.nope_prolog import dequantize_nope_query
from vllm_ascend.utils import bootstrap_custom_op_env


def run(tokens: int, graph: bool) -> None:
    torch.manual_seed(1024)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    bootstrap_custom_op_env()
    device = "npu:0"
    hidden_size, q_rank, kv_rank, heads, nope_dim = 1024, 1536, 512, 2, 256
    hidden = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16)
    token_x, scale_x = torch_npu.npu_dynamic_quant(hidden)

    def weight(rows, cols):
        value = torch.randint(-8, 9, (rows, cols), device=device, dtype=torch.int8)
        return torch_npu.npu_format_cast(value, 29)

    dq = weight(hidden_size, q_rank)
    dkv = weight(hidden_size, kv_rank)
    uq = weight(q_rank, heads * nope_dim)
    uk = torch.randn(heads, nope_dim, kv_rank, device=device, dtype=torch.bfloat16) * 0.02
    dq_scale = torch.full((1, q_rank), 0.01, device=device)
    dkv_scale = torch.full((1, kv_rank), 0.01, device=device)
    uq_scale = torch.full((1, heads * nope_dim), 0.01, device=device)
    gamma_q = torch.ones(q_rank, device=device, dtype=torch.bfloat16)
    gamma_kv = torch.ones(kv_rank, device=device, dtype=torch.bfloat16)
    rope = hidden.new_empty((tokens, 0))
    blocks = (tokens + 127) // 128 + 1
    # Reserve the final token as graph padding; its slot must not be written.
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    slots[-1] = -1
    cache = torch.full((blocks, 128, 1, 528), 37, device=device, dtype=torch.int8)
    dummy_kr = hidden.new_empty(0)
    clip = torch.ones(1, device=device)
    before = cache.clone()

    def call():
        return torch_npu.npu_mla_prolog_v3(
            token_x=token_x,
            weight_dq=dq,
            weight_uq_qr=uq,
            weight_uk=uk,
            weight_dkv_kr=dkv,
            rmsnorm_gamma_cq=gamma_q,
            rmsnorm_gamma_ckv=gamma_kv,
            rope_sin=rope,
            rope_cos=rope,
            kv_cache=cache,
            kr_cache=dummy_kr,
            cache_index=slots,
            dequant_scale_x=scale_x.reshape(tokens, 1),
            dequant_scale_w_dq=dq_scale,
            dequant_scale_w_uq_qr=uq_scale,
            dequant_scale_w_dkv_kr=dkv_scale,
            k_nope_clip_alpha=clip,
            rmsnorm_epsilon_cq=1e-6,
            rmsnorm_epsilon_ckv=1e-6,
            cache_mode="PA_BSND",
            query_norm_flag=True,
            weight_quant_mode=2,
            kv_cache_quant_mode=3,
            query_quant_mode=0,
            ckvkr_repo_mode=1,
            quant_scale_repo_mode=1,
            tile_size=128,
            qc_qr_scale=1.0,
            kc_scale=1.0,
        )

    result = call()
    if graph:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                call()
        torch.npu.current_stream().wait_stream(stream)
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            result = call()
        captured.replay()
    torch.npu.synchronize()
    query, query_rope, _, query_norm, query_scale = result
    assert query.shape == (tokens, heads, kv_rank)
    assert query_rope.shape == (tokens, heads, 0)
    restored = dequantize_nope_query(query_norm, query_scale, tokens, q_rank, torch.bfloat16)

    def project(x, w, scale, per_token_scale):
        return torch_npu.npu_quant_matmul(
            x, w, scale.reshape(-1), pertoken_scale=per_token_scale.reshape(-1), output_dtype=torch.bfloat16
        )

    expected_qc = torch_npu.npu_rms_norm(project(token_x, dq, dq_scale, scale_x), gamma_q, epsilon=1e-6)[0]
    quantized_qc, qc_scale = torch_npu.npu_dynamic_quant(expected_qc)
    expected_q = project(quantized_qc, uq, uq_scale, qc_scale).reshape(tokens, heads, nope_dim)
    expected_query = torch.einsum("tnd,ndl->tnl", expected_q, uk)
    expected_kv = torch_npu.npu_rms_norm(project(token_x, dkv, dkv_scale, scale_x), gamma_kv, epsilon=1e-6)[0]
    # Numerical thresholds are smoke thresholds, not a model accuracy claim.
    torch.testing.assert_close(restored.float(), expected_qc.float(), rtol=0.03, atol=0.06)
    torch.testing.assert_close(query.float(), expected_query.float(), rtol=0.05, atol=0.08)
    rows = cache.view(-1, 528)
    valid_rows = rows[: tokens - 1]
    scales = valid_rows[:, 512:].contiguous().view(torch.float32)
    restored_kv = (valid_rows[:, :512].float().reshape(-1, 4, 128) * scales.unsqueeze(-1)).reshape(-1, 512)
    torch.testing.assert_close(restored_kv, expected_kv[: tokens - 1].float(), rtol=0.03, atol=0.06)
    torch.testing.assert_close(rows[tokens - 1 :], before.view(-1, 528)[tokens - 1 :], rtol=0, atol=0)
    print(f"PASS: Dr=0 W8A8C8, tokens={tokens}, graph={graph}, query/q_c/KV numerical checks, padding preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be >= 2 (one valid token and one padded token)")
    run(args.tokens, args.graph)
