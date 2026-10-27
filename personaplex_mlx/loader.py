"""Loading converted PersonalPlexMLX checkpoints (see `convert.py`)."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .convert import MIMI_FILE, TOKENIZER_FILE
from .model import PersonaPlexConfig, PersonaPlexLM, quantize_model


def load_lm(path: str | Path, dtype=None) -> PersonaPlexLM:
    path = Path(path)
    meta = json.loads((path / "config.json").read_text())
    cfg = PersonaPlexConfig.from_dict(meta)
    model = PersonaPlexLM(cfg)
    quantize_model(model, meta["quantization"])
    weights = {}
    for f in sorted(path.glob("model-*.safetensors")):
        weights.update(mx.load(str(f)))
    if dtype is None:
        dtype = next(v.dtype for k, v in weights.items() if k.endswith("norm1.weight"))
    model.load_weights(list(weights.items()), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def load_mimi(path: str | Path, num_codebooks: int = 8):
    from .mimi import Mimi, mimi_202407

    mimi = Mimi(mimi_202407(num_codebooks))
    mimi.load_pytorch_weights(str(Path(path) / MIMI_FILE), strict=True)
    return mimi


def load_tokenizer(path: str | Path):
    import sentencepiece

    return sentencepiece.SentencePieceProcessor(str(Path(path) / TOKENIZER_FILE))


def list_voices(path: str | Path) -> list[str]:
    return sorted(p.stem for p in (Path(path) / "voices").iterdir() if p.suffix in (".safetensors", ".wav"))


def load_voice(path: str | Path, name: str) -> dict:
    """Returns {"embeddings", "cache"} for converted presets or {"wav": path} for recordings."""
    vdir = Path(path) / "voices"
    st = vdir / f"{name}.safetensors"
    if st.exists():
        d = mx.load(str(st))
        return {
            "embeddings": np.array(d["embeddings"].astype(mx.float32)),
            "cache": np.array(d["cache"]).astype(np.int64),
        }
    wav = vdir / f"{name}.wav"
    if wav.exists():
        return {"wav": str(wav)}
    raise FileNotFoundError(f"voice {name!r} not found in {vdir}; available: {list_voices(path)}")
