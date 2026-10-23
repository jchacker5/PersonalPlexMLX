"""Convert nvidia/personaplex-7b-v1 to a quantized MLX checkpoint.

The source checkpoint is a single 16.7 GB bf16 safetensors file. By default it is *streamed*:
each tensor is fetched with an HTTP range request, remapped, quantized and appended to the
output shard, so neither the full source file nor the full float model ever sits on disk or in
memory. A local `--source` file is read the same way.

    python -m personaplex_mlx.convert --out weights/personaplex-q6 --bits 6
"""

import argparse
import json
import os
import re
import shutil
import struct
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from .model import PersonaPlexConfig, quant_spec

DEFAULT_REPO = "nvidia/personaplex-7b-v1"
LM_FILE = "model.safetensors"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOKENIZER_FILE = "tokenizer_spm_32k_3.model"
VOICES_FILE = "voices.tgz"
SHARD_BYTES = 2 << 30

_DTYPES = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, None),
    "F32": (np.float32, None),
    "I64": (np.int64, None),
}


class TensorSource:
    """Lazy reader for a safetensors file, local path or HTTP URL (via range requests)."""

    def __init__(self, path_or_url: str):
        self.loc = path_or_url
        self.remote = path_or_url.startswith("http")
        n = struct.unpack("<Q", self._read(0, 8))[0]
        self.header = json.loads(self._read(8, n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def _read(self, start: int, length: int) -> bytes:
        if not self.remote:
            with open(self.loc, "rb") as f:
                f.seek(start)
                return f.read(length)
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    self.loc, headers={"Range": f"bytes={start}-{start + length - 1}"}
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != length:
                    raise IOError(f"short read {len(data)} != {length}")
                return data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError

    def names(self) -> list[str]:
        return list(self.header)

    def get(self, name: str) -> mx.array:
        info = self.header[name]
        start, end = info["data_offsets"]
        np_dtype, mx_view = _DTYPES[info["dtype"]]
        arr = np.frombuffer(self._read(self.base + start, end - start), dtype=np_dtype).reshape(info["shape"])
        a = mx.array(arr)
        return a.view(mx_view) if mx_view is not None else a


def lm_weight_plan(names: list[str], cfg: PersonaPlexConfig) -> list[tuple[str, str, Callable]]:
    """Maps upstream LMModel tensor names to PersonaPlexMLX names.

    Returns (source_name, target_name, transform) triples. Depformer slices >= cfg.dep_q (the
    user-stream codebooks) are dropped.
    """
    n = cfg.dep_q_checkpoint
    ident = lambda w: w  # noqa: E731
    alpha = lambda w: w.reshape(-1)  # noqa: E731
    chunk = lambda s: (lambda w: w.reshape(n, -1, w.shape[-1])[s])  # noqa: E731
    plan = []
    for name in names:
        if m := re.fullmatch(r"emb\.(\d+)\.weight", name):
            plan.append((name, f"audio_embs.{m[1]}.emb.weight", ident))
        elif name == "text_emb.weight":
            plan.append((name, "text_emb.emb.weight", ident))
        elif name == "text_linear.weight":
            plan.append((name, name, ident))
        elif name == "out_norm.alpha":
            plan.append((name, "out_norm.weight", alpha))
        elif m := re.fullmatch(r"transformer\.layers\.(\d+)\.(.+)", name):
            i, rest = m[1], m[2]
            rest = rest.replace("in_proj_weight", "in_proj.weight").replace(".alpha", ".weight")
            plan.append((name, f"layers.{i}.{rest}", alpha if name.endswith(".alpha") else ident))
        elif m := re.fullmatch(r"depformer_in\.(\d+)\.weight", name):
            if int(m[1]) < cfg.dep_q:
                plan.append((name, f"depformer.slices.{m[1]}.linear_in.weight", ident))
        elif name == "depformer_text_emb.weight":
            plan.append((name, "depformer.slices.0.emb.emb.weight", ident))
        elif m := re.fullmatch(r"depformer_emb\.(\d+)\.weight", name):
            s = int(m[1]) + 1
            if s < cfg.dep_q:
                plan.append((name, f"depformer.slices.{s}.emb.emb.weight", ident))
        elif m := re.fullmatch(r"linears\.(\d+)\.weight", name):
            if int(m[1]) < cfg.dep_q:
                plan.append((name, f"depformer.slices.{m[1]}.linear_out.weight", ident))
        elif m := re.fullmatch(r"depformer\.layers\.(\d+)\.self_attn\.(in_proj_weight|out_proj\.weight)", name):
            proj = "in_proj" if m[2] == "in_proj_weight" else "out_proj"
            for s in range(cfg.dep_q):
                plan.append((name, f"depformer.slices.{s}.layers.{m[1]}.self_attn.{proj}.weight", chunk(s)))
        elif m := re.fullmatch(r"depformer\.layers\.(\d+)\.(norm[12])\.alpha", name):
            for s in range(cfg.dep_q):
                plan.append((name, f"depformer.slices.{s}.layers.{m[1]}.{m[2]}.weight", alpha))
        elif m := re.fullmatch(r"depformer\.layers\.(\d+)\.gating\.(\d+)\.(linear_in|linear_out)\.weight", name):
            if int(m[2]) < cfg.dep_q:
                plan.append((name, f"depformer.slices.{m[2]}.layers.{m[1]}.gating.{m[3]}.weight", ident))
        else:
            raise ValueError(f"unexpected tensor in checkpoint: {name}")
    return plan


class ShardWriter:
    def __init__(self, out: Path, prefix: str = "model"):
        self.out, self.prefix = out, prefix
        self.buf: dict[str, mx.array] = {}
        self.size = 0
        self.files: list[str] = []
        self.index: dict[str, str] = {}

    def add(self, name: str, arr: mx.array):
        mx.eval(arr)
        self.buf[name] = arr
        self.size += arr.nbytes
        if self.size >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        fname = f"{self.prefix}-{len(self.files):05d}.safetensors"
        mx.save_safetensors(str(self.out / fname), self.buf)
        self.index.update({k: fname for k in self.buf})
        self.files.append(fname)
        self.buf, self.size = {}, 0


def convert_lm(source: TensorSource, out: Path, cfg: PersonaPlexConfig, q: dict, dtype=mx.bfloat16, workers: int = 4):
    plan = lm_weight_plan(source.names(), cfg)
    by_source: dict[str, list[tuple[str, Callable]]] = {}
    for src, dst, fn in plan:
        by_source.setdefault(src, []).append((dst, fn))
    writer = ShardWriter(out)
    order = sorted(by_source, key=lambda s: source.header[s]["data_offsets"][0])
    total = sum(source.header[s]["data_offsets"][1] - source.header[s]["data_offsets"][0] for s in order)
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        for src, w in zip(order, pool.map(source.get, order)):
            for dst, fn in by_source[src]:
                t = fn(w)
                spec = quant_spec(dst, t.shape, q)
                if spec is None:
                    writer.add(dst, t.astype(dtype))
                else:
                    wq, scales, biases = mx.quantize(t.astype(dtype), group_size=spec[1], bits=spec[0])
                    stem = dst.removesuffix(".weight")
                    writer.add(f"{stem}.weight", wq)
                    writer.add(f"{stem}.scales", scales)
                    writer.add(f"{stem}.biases", biases)
            info = source.header[src]["data_offsets"]
            done += info[1] - info[0]
            rate = done / max(time.time() - t0, 1e-6) / 1e6
            print(f"\r  {done / 1e9:6.2f}/{total / 1e9:.2f} GB  {rate:5.0f} MB/s", end="", flush=True)
    writer.flush()
    print()
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": writer.index}, f, indent=1)


def convert_voices(tgz: Path, out: Path):
    """Converts upstream voice prompt files (*.pt torch pickles) to safetensors."""
    vdir = out / "voices"
    vdir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(tgz) as tf:
        tf.extractall(tmp, filter="data")
        files = sorted(Path(tmp).rglob("*.pt")) + sorted(Path(tmp).rglob("*.wav"))
        for p in files:
            if p.suffix == ".wav":
                shutil.copy(p, vdir / p.name)
                continue
            import torch  # only needed for the one-off voice conversion

            state = torch.load(p, map_location="cpu", weights_only=True)
            emb = state["embeddings"].float().numpy()
            cache = state["cache"].long().numpy().astype(np.int32)
            mx.save_safetensors(
                str(vdir / f"{p.stem}.safetensors"),
                {"embeddings": mx.array(emb).astype(mx.bfloat16), "cache": mx.array(cache)},
            )
        print(f"  {len(files)} voices")


def resolve_remote(repo: str, filename: str, token: str | None) -> str:
    """Resolves a (gated) HF file to its pre-signed download URL, which needs no auth header."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    return get_hf_file_metadata(hf_hub_url(repo, filename), token=token).location


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--hf-repo", default=DEFAULT_REPO)
    p.add_argument("--source", help="local model.safetensors (default: stream from the HF repo)")
    p.add_argument("--bits", type=int, default=6, help="temporal transformer bits (4, 5, 6, 8 or 16)")
    p.add_argument("--bits-depformer", type=int, default=8)
    p.add_argument("--bits-other", type=int, default=8, help="embeddings and text head")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-lm", action="store_true", help="only fetch mimi, tokenizer and voices")
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = PersonaPlexConfig()
    q = {
        "bits_temporal": args.bits,
        "bits_depformer": args.bits_depformer,
        "bits_other": args.bits_other,
        "group_size": args.group_size,
    }
    with open(out / "config.json", "w") as f:
        json.dump({**cfg.to_dict(), "quantization": q, "source": args.hf_repo}, f, indent=1)

    print("fetching mimi + tokenizer")
    for fname in (MIMI_FILE, TOKENIZER_FILE):
        hf_hub_download(args.hf_repo, fname, local_dir=out, token=token)
    print("fetching voices")
    with tempfile.TemporaryDirectory() as tmp:
        tgz = hf_hub_download(args.hf_repo, VOICES_FILE, local_dir=tmp, token=token)
        convert_voices(Path(tgz), out)

    if not args.skip_lm:
        if args.source:
            src = TensorSource(args.source)
        else:
            src = TensorSource(resolve_remote(args.hf_repo, LM_FILE, token))
        print(f"converting LM ({q})")
        convert_lm(src, out, cfg, q, workers=args.workers)
    shutil.rmtree(out / ".cache", ignore_errors=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
