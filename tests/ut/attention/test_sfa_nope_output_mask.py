# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.attention.sfa_v1 as sfa


@pytest.mark.parametrize("a5", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("actual_tokens", [0, 3, 4])
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_nope_output_mask_preserves_valid_rows(a5, dtype, actual_tokens, graph_mode):
    capacity, heads, dim = 4, 2, 8
    query = torch.zeros(capacity, heads, dim, dtype=dtype)
    cache = torch.zeros(1, 4, 1, dim, dtype=dtype)
    indices = torch.zeros(capacity, 1, 1, dtype=torch.int32)
    raw = torch.arange(capacity * heads * dim, dtype=torch.float32).reshape(capacity, heads, dim).to(dtype)
    raw[actual_tokens:] = float("nan")
    before = raw.clone()
    metadata = SimpleNamespace(
        num_actual_tokens=actual_tokens,
        smla_metadata=torch.empty(1) if a5 else None,
        smla_topk_length=torch.ones(capacity, dtype=torch.int32),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        block_size=4,
        query_start_loc=torch.tensor([0, actual_tokens], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
    )
    with (
        patch.object(sfa, "get_forward_context", return_value=SimpleNamespace(cudagraph_runtime_mode=graph_mode)),
        patch.object(sfa, "sparse_flash_mla", return_value=(raw,), create=True),
        patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention", return_value=(raw,), create=True),
        patch.object(torch, "arange", wraps=torch.arange) as arange,
    ):
        output = sfa.sparse_mla(query, cache, indices, metadata, 1.0)

    torch.testing.assert_close(output[:actual_tokens], before[:actual_tokens])
    assert torch.count_nonzero(output[actual_tokens:]) == 0
    torch.testing.assert_close(raw, before, equal_nan=True)
    if graph_mode == CUDAGraphMode.NONE and actual_tokens == capacity:
        assert output is raw
        arange.assert_not_called()
    else:
        arange.assert_called_once()


@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_nope_graph_mask_uses_device_lengths_even_with_unpadded_host_count(graph_mode):
    # Model graph capture at full capacity, then a replay with one real row.
    # Host-side metadata remains at the capture value; device lengths change.
    query = torch.zeros(4, 1, 8)
    cache = torch.zeros(1, 4, 1, 8)
    indices = torch.zeros(4, 1, 1, dtype=torch.int32)
    raw = torch.ones_like(query)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        smla_metadata=None,
        block_size=4,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
    )
    with (
        patch.object(sfa, "get_forward_context", return_value=SimpleNamespace(cudagraph_runtime_mode=graph_mode)),
        patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention", return_value=(raw,), create=True),
    ):
        torch.testing.assert_close(sfa.sparse_mla(query, cache, indices, metadata, 1.0), raw)
        metadata.query_start_loc[-1] = 1
        raw[1:] = float("nan")
        output = sfa.sparse_mla(query, cache, indices, metadata, 1.0)
    torch.testing.assert_close(output[:1], raw[:1])
    assert torch.count_nonzero(output[1:]) == 0
