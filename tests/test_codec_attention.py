from dataclasses import replace

import mlx.core as mx
import numpy as np

from personaplex_mlx.mimi.mimi import mimi_202407
from personaplex_mlx.mimi.transformer import Attention
from personaplex_mlx.mimi.kv_cache import KVCache


def test_codec_attention_does_not_see_future_frame():
    cfg = replace(mimi_202407(8).transformer, d_model=8, num_heads=2)
    attention = Attention(cfg)
    prefix = mx.random.normal((1, 1, 8))
    a = mx.concatenate([prefix, mx.zeros((1, 1, 8))], axis=1)
    b = mx.concatenate([prefix, mx.ones((1, 1, 8)) * 10], axis=1)
    first = attention(a, KVCache(4, 2))[:, 0]
    second = attention(b, KVCache(4, 2))[:, 0]
    mx.eval(first, second)
    np.testing.assert_allclose(np.array(first), np.array(second), atol=1e-6)


def test_codec_feedforward_matches_upstream_exact_gelu():
    import torch
    from personaplex_mlx.mimi.transformer import MlpNoGating
    cfg = replace(mimi_202407(8).transformer, d_model=8, dim_feedforward=16)
    mlp = MlpNoGating(cfg)
    x = mx.random.normal((1, 2, 8)) * 3
    mx.eval(x, mlp.parameters())
    with torch.no_grad():
        hidden = torch.nn.functional.linear(torch.tensor(np.array(x)), torch.tensor(np.array(mlp.linear1.weight)))
        ref = torch.nn.functional.linear(torch.nn.functional.gelu(hidden), torch.tensor(np.array(mlp.linear2.weight)))
    got = mlp(x)
    mx.eval(got)
    np.testing.assert_allclose(np.array(got), ref.numpy(), atol=1e-6, rtol=1e-5)
