# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.attention.sfa_v1 as sfa
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod


def _impl():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_nope_prolog_v3 = True
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl.c8_cache_dtype = torch.int8
    impl.q_lora_rank = 1536
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.local_num_heads = 2
    impl.has_indexer = True
    impl.indexer = SimpleNamespace(wq_b=SimpleNamespace(weight=torch.empty(2, 1536, dtype=torch.bfloat16)))
    impl._quant_type = AscendW8A8DynamicLinearMethod
    method = SimpleNamespace(quant_method=AscendW8A8DynamicLinearMethod())
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=torch.empty(1024, 2048, dtype=torch.int8),
        weight_scale=torch.ones(2048),
        quant_method=method,
        input_size=1024,
    )
    impl.q_proj = SimpleNamespace(
        weight=torch.empty(1536, 512, dtype=torch.int8), weight_scale=torch.ones(512), quant_method=method
    )
    impl.W_UK_T = torch.empty(2, 256, 512, dtype=torch.bfloat16)
    impl.q_a_layernorm = SimpleNamespace(weight=torch.ones(1536, dtype=torch.bfloat16), variance_epsilon=1e-6)
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(512, dtype=torch.bfloat16), variance_epsilon=1e-6)
    impl.sfa_qsfa_tile_size = 128
    impl.sfa_qsfa_kr_cache_dummy = torch.empty(0, dtype=torch.bfloat16)
    impl.sfa_qsfa_k_nope_clip_alpha = torch.ones(1)
    for name in ("weight_dq", "weight_uq_qr", "weight_dkv_kr"):
        setattr(impl, name, torch.empty(1, dtype=torch.int8))
    for name in ("dequant_scale_w_dq", "dequant_scale_w_uq_qr", "dequant_scale_w_dkv_kr"):
        setattr(impl, name, torch.ones(1))
    return impl


def test_nope_opt_in_accepts_dense_kpool_projection(monkeypatch):
    impl = _impl()
    monkeypatch.setattr(sfa, "get_ascend_device_type", lambda: SimpleNamespace(name="A3"))
    impl._process_weights_for_fused_prolog_v3 = MagicMock()
    assert impl._resolve_preprocess_type(torch.bfloat16) is PreprocessType.PROLOG_V3
    impl._process_weights_for_fused_prolog_v3.assert_called_once()


def test_nope_opt_in_rejects_a5_instead_of_silent_native_fallback(monkeypatch):
    impl = _impl()
    monkeypatch.setattr(sfa, "get_ascend_device_type", lambda: SimpleNamespace(name="A5"))
    impl._process_weights_for_fused_prolog_v3 = MagicMock()
    with pytest.raises(ValueError, match="A2/A3"):
        impl._resolve_preprocess_type(torch.bfloat16)
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


@pytest.mark.parametrize("tokens", [1, 4, 129])
def test_nope_prolog_passes_empty_rope_packed_cache_and_dense_query(monkeypatch, tokens):
    impl = _impl()
    hidden = torch.zeros(tokens, 1024, dtype=torch.bfloat16)
    cache = torch.empty(2, 128, 1, 528, dtype=torch.int8)
    slots = torch.arange(tokens, dtype=torch.int64)
    slots[-1] = -1
    query_norm = torch.full((tokens, 1536), 4, dtype=torch.int8)
    scales = torch.arange(1, tokens + 1, dtype=torch.float32).reshape(tokens, 1) / 8
    prolog = MagicMock(
        return_value=(
            torch.zeros(tokens, 2, 512, dtype=torch.bfloat16),
            torch.empty(tokens, 2, 0, dtype=torch.bfloat16),
            None,
            query_norm,
            scales,
        )
    )
    monkeypatch.setattr(sfa.torch_npu, "npu_mla_prolog_v3", prolog)
    monkeypatch.setattr(
        sfa.torch_npu,
        "npu_dynamic_quant",
        lambda x: (torch.zeros_like(x, dtype=torch.int8), torch.ones(tokens)),
    )
    returned_hidden, query, query_rope, q_c = impl._sfa_preprocess_prolog_v3(hidden, (cache,), None, None, slots)
    assert returned_hidden is hidden
    assert query.shape == (tokens, 2, 512)
    assert query_rope.shape == (tokens, 2, 0)
    torch.testing.assert_close(q_c, (query_norm.float() * scales).bfloat16(), rtol=0, atol=0)
    args = prolog.call_args.kwargs
    assert args["rope_sin"].shape == args["rope_cos"].shape == (tokens, 0)
    assert args["kv_cache"] is cache
    assert args["cache_index"].dtype == torch.int64
    assert args["cache_index"].tolist() == slots.tolist()
    assert args["weight_quant_mode"] == 2
    assert args["kv_cache_quant_mode"] == 3
    assert args["query_quant_mode"] == 0
    assert args["ckvkr_repo_mode"] == args["quant_scale_repo_mode"] == 1
    assert args["tile_size"] == 128


@pytest.mark.parametrize("graph_mode,expected_tokens", [(CUDAGraphMode.NONE, 3), (CUDAGraphMode.FULL, 4)])
def test_forward_aligns_prolog_slots_after_nope_token_trimming(monkeypatch, graph_mode, expected_tokens):
    impl = _impl()
    impl.g_proj = impl.layerwise_kv_cache_hook = None
    impl.preprocess_type = PreprocessType.PROLOG_V3
    hidden = torch.ones(4, 1024, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2, -1], dtype=torch.int32),
        num_actual_tokens=3,
        cos=None,
        sin=None,
    )
    impl._compose_sfa_kv_cache = MagicMock(return_value=(torch.empty(2, 128, 1, 528, dtype=torch.int8),))
    impl._get_sfa_kv_slot_mapping = lambda _: metadata.slot_mapping
    impl._get_indexer_attn_metadata = lambda: None
    impl._get_parallel_forward_context = MagicMock(
        return_value=SimpleNamespace(actual_seq_lengths_query=None, actual_seq_lengths_key=None)
    )
    monkeypatch.setattr(sfa, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=graph_mode))
    monkeypatch.setattr(sfa, "wait_for_kv_layer_from_connector", lambda _: None)

    class ReachedProlog(Exception):
        pass

    def inspect_prolog(**kwargs):
        assert kwargs["hidden_states"].shape[0] == expected_tokens
        assert kwargs["slot_mapping"].shape == (expected_tokens,)
        assert kwargs["slot_mapping"].dtype == torch.int64
        torch.testing.assert_close(kwargs["slot_mapping"], metadata.slot_mapping[:expected_tokens].long())
        raise ReachedProlog

    impl._sfa_preprocess_prolog_v3 = inspect_prolog
    with pytest.raises(ReachedProlog):
        impl.forward("model.layers.0.self_attn", hidden, (), metadata, output=torch.empty_like(hidden))
