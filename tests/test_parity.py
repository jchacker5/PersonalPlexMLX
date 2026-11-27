"""Parity of the MLX port against the upstream PyTorch implementation on a tiny random model.

Greedy decoding must produce identical tokens through the system prompt (precomputed voice
embeddings + text), a KV ring-buffer wrap-around and several KV buckets. This exercises the
weight remapping, the 8-codebook depformer cut, the batched prompt prefill and the compiled
frame step.
"""

import json

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from moshi.models import loaders
from moshi.models.lm import LMGen, LMModel
from personaplex_mlx import gen as gen_mod
from personaplex_mlx.convert import TensorSource, convert_lm
from personaplex_mlx.gen import PersonaPlexGen
from personaplex_mlx.loader import load_lm
from personaplex_mlx.model import PersonaPlexConfig

TINY = dict(
    dim=64, num_heads=4, num_layers=2, context=40, text_card=100, card=2048,
    depformer_dim=32, depformer_num_heads=2, depformer_num_layers=2,
)


def _upstream(seed=0):
    torch.manual_seed(seed)
    kw = dict(loaders._lm_kwargs)
    kw["dep_q"] = 16
    kw.update(TINY)
    kw["depformer_dim_feedforward"] = int(4.125 * TINY["depformer_dim"])
    m = LMModel(device="cpu", dtype=torch.float32, **kw)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("alpha"):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 / p.shape[-1] ** 0.5 if p.dim() > 1 else 1))
    return m.eval()


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w")
    up = _upstream()
    save_file({k: v.contiguous() for k, v in up.state_dict().items()}, str(tmp / "src.safetensors"))
    cfg = PersonaPlexConfig(**TINY)
    q = {"bits_temporal": 16, "bits_depformer": 16, "bits_other": 16, "group_size": 64}
    out = tmp / "mlx"
    out.mkdir()
    convert_lm(TensorSource(str(tmp / "src.safetensors")), out, cfg, q, dtype=mx.float32)
    (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "quantization": q}))
    return up, load_lm(out)


def _run_upstream(up, voice, text, user):
    torch.manual_seed(0)
    g = LMGen(up, device="cpu", use_sampling=False, audio_silence_frame_cnt=6, sample_rate=24000, frame_rate=12.5)
    g.voice_prompt_embeddings = torch.tensor(voice["embeddings"])
    g.voice_prompt_cache = torch.tensor(voice["cache"])
    g.text_prompt_tokens = text
    g.streaming_forever(1)
    g.reset_streaming()
    with torch.no_grad():
        g.step_system_prompts(mimi=None)
        outs = []
        for u in user:
            o = g.step(torch.tensor(u).view(1, 8, 1))
            outs.append(None if o is None else o[0, :, 0].numpy())
    return outs


def _run_mlx(lm, voice, text, user, **kw):
    g = PersonaPlexGen(lm, greedy=True, **kw)
    g.load_prompt(text, voice_embeddings=voice["embeddings"], voice_ring=voice["cache"])
    return [None if (o := g.step(np.array(u)[None])) is None else o[0] for u in user]


def _inputs(cfg, n_voice=5, n_text=8, n_user=60, seed=1):
    rng = np.random.default_rng(seed)
    voice = {
        "embeddings": rng.normal(size=(n_voice, 1, 1, cfg["dim"])).astype(np.float32),
        "cache": rng.integers(0, cfg["card"], size=(1, 17, 4)).astype(np.int64),
    }
    voice["cache"][:, 0] = rng.integers(0, cfg["text_card"], size=(1, 4))
    text = [int(t) for t in rng.integers(4, cfg["text_card"], size=n_text)]
    user = rng.integers(0, cfg["card"], size=(n_user, 8))
    return voice, text, user


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("bucket", [256, 16])
def test_greedy_parity(models, compile, bucket, monkeypatch):
    monkeypatch.setattr(gen_mod, "KV_BUCKET", bucket)
    up, lm = models
    voice, text, user = _inputs(TINY)
    ref = _run_upstream(up, voice, text, user)
    got = _run_mlx(lm, voice, text, user, compile=compile)
    assert len(ref) == len(got)
    n_compared = 0
    for i, (r, g) in enumerate(zip(ref, got)):
        assert (r is None) == (g is None), i
        if r is not None:
            np.testing.assert_array_equal(r[:9], g, err_msg=f"frame {i}")
            n_compared += 1
    assert n_compared > TINY["context"], "test must cover a KV ring wrap-around"


def test_prompt_snapshot_restore(models):
    _, lm = models
    voice, text, user = _inputs(TINY)
    g = PersonaPlexGen(lm, greedy=True)
    snap = g.load_prompt(text, voice_embeddings=voice["embeddings"], voice_ring=voice["cache"])
    a = [g.step(np.array(u)[None]) for u in user[:20]]
    g.restore(snap)
    b = [g.step(np.array(u)[None]) for u in user[:20]]
    for x, y in zip(a, b):
        assert (x is None and y is None) or np.array_equal(x, y)
