# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Dr=0 contract delivered by ops-transformer PR 14225.

Keep these checks independent of the runtime so CPU tests can exercise the
operator boundary without mocking an engine or loading an NPU extension.
"""

import torch


def nope_prolog_unsupported_reasons(
    *,
    device_type: str,
    cache_dtype: torch.dtype | None,
    dynamic_int8_projections: bool,
    activation_dtype: torch.dtype,
    hidden_size: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    num_heads: int,
) -> list[str]:
    reasons = []
    if device_type not in ("A2", "A3"):
        reasons.append("NoPE Prolog V3 requires A2/A3; PR14225 does not enable A5/MXFP8 Dr=0.")
    if cache_dtype != torch.int8:
        reasons.append("NoPE Prolog V3 requires --kv-cache-dtype int8 (W8A8C8).")
    if not dynamic_int8_projections:
        reasons.append("NoPE Prolog V3 requires W8A8_DYNAMIC int8 fused_qkv_a_proj and q_proj weights.")
    if activation_dtype != torch.bfloat16:
        reasons.append("NoPE Prolog V3 requires bfloat16 activations and normalization weights.")
    if hidden_size not in (1024, 2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192):
        reasons.append(f"NoPE Prolog V3 does not support hidden_size={hidden_size}.")
    if q_lora_rank not in (1536, 2048):
        reasons.append("NoPE Prolog V3 requires q_lora_rank in {1536, 2048}.")
    if kv_lora_rank != 512 or qk_nope_head_dim != 256:
        reasons.append("NoPE Prolog V3 requires kv_lora_rank=512 and qk_nope_head_dim=256.")
    if not 1 <= num_heads <= 128:
        reasons.append("NoPE Prolog V3 requires 1 <= local_num_heads <= 128.")
    return reasons


def dequantize_nope_query(
    query: torch.Tensor,
    scale: torch.Tensor | None,
    num_tokens: int,
    q_lora_rank: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore Prolog's per-token INT8 query_norm for the KPool projection.

    query_quant_mode=0 describes the final MLA query, not query_norm. With
    weight_quant_mode=2, query_norm is INT8 with one FP32 multiplier per row.
    Multiply in FP32 before casting; casting the quantized bytes alone changes
    the indexer's query and therefore its selected tokens.
    """
    if query.dtype != torch.int8 or query.shape != (num_tokens, q_lora_rank):
        raise RuntimeError("NoPE Prolog query_norm must be INT8 [input_tokens, q_lora_rank], including graph padding.")
    if scale is None or scale.dtype != torch.float32 or scale.shape not in ((num_tokens,), (num_tokens, 1)):
        raise RuntimeError("NoPE Prolog query_norm requires one FP32 dequantization scale per input token.")
    if query.device != scale.device:
        raise RuntimeError("NoPE Prolog query_norm and its scales must be on the same device.")
    return (query.float() * scale.reshape(num_tokens, 1)).to(dtype)
