# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.nope_prolog import dequantize_nope_query, nope_prolog_unsupported_reasons


def _contract(**overrides):
    values = dict(
        device_type="A3",
        cache_dtype=torch.int8,
        dynamic_int8_projections=True,
        activation_dtype=torch.bfloat16,
        hidden_size=6144,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        num_heads=8,
    )
    return values | overrides


@pytest.mark.parametrize("device_type", ["A2", "A3"])
@pytest.mark.parametrize("q_lora_rank", [1536, 2048])
def test_dr0_operator_contract(device_type, q_lora_rank):
    assert not nope_prolog_unsupported_reasons(**_contract(device_type=device_type, q_lora_rank=q_lora_rank))


@pytest.mark.parametrize(
    "override,match",
    [
        ({"device_type": "A5"}, "A2/A3"),
        ({"cache_dtype": None}, "int8"),
        ({"cache_dtype": torch.float8_e4m3fn}, "int8"),
        ({"dynamic_int8_projections": False}, "W8A8_DYNAMIC"),
        ({"activation_dtype": torch.float16}, "bfloat16"),
        ({"hidden_size": 1000}, "hidden_size"),
        ({"q_lora_rank": 1024}, "q_lora_rank"),
        ({"kv_lora_rank": 256}, "kv_lora_rank"),
        ({"qk_nope_head_dim": 128}, "qk_nope_head_dim"),
        ({"num_heads": 0}, "local_num_heads"),
        ({"num_heads": 129}, "local_num_heads"),
    ],
)
def test_unsupported_dr0_modes_are_rejected(override, match):
    assert any(match in reason for reason in nope_prolog_unsupported_reasons(**_contract(**override)))


@pytest.mark.parametrize("column_scale", [False, True])
def test_query_norm_scales_each_row_before_bf16_projection(column_scale):
    query = torch.tensor([[2, -4, 0], [10, -3, 127], [-128, 1, 5]], dtype=torch.int8)
    scale = torch.tensor([0.25, 0.1, 2.0], dtype=torch.float32)
    if column_scale:
        scale = scale[:, None]
    result = dequantize_nope_query(query, scale, 3, 3, torch.bfloat16)
    expected = torch.tensor([[0.5, -1.0, 0], [1.0, -0.3, 12.7], [-256.0, 2.0, 10.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # The final row represents graph padding and must not disappear.
    assert result.shape == (3, 3)
    projection = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.bfloat16)
    torch.testing.assert_close(result @ projection.T, expected @ projection.T, rtol=0, atol=0)


@pytest.mark.parametrize(
    "query,scale",
    [
        (torch.zeros(3, 8), torch.ones(3)),
        (torch.zeros(2, 8, dtype=torch.int8), torch.ones(2)),
        (torch.zeros(3, 8, dtype=torch.int8), None),
        (torch.zeros(3, 8, dtype=torch.int8), torch.ones(2)),
        (torch.zeros(3, 8, dtype=torch.int8), torch.ones(3, 2)),
        (torch.zeros(3, 8, dtype=torch.int8), torch.ones(3, dtype=torch.bfloat16)),
    ],
)
def test_invalid_query_norm_cannot_reach_kpool(query, scale):
    with pytest.raises(RuntimeError, match="NoPE Prolog query_norm"):
        dequantize_nope_query(query, scale, 3, 8, torch.bfloat16)
