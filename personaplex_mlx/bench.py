"""Per-frame latency benchmark at full PersonaPlex-7B size.

With no `--model`, weights are random (no download needed): latency depends only on shapes and
quantization, not on weight values. Real time requires < 80 ms per frame (12.5 Hz).

    python -m personaplex_mlx.bench --bits 6 --bits-depformer 8
    python -m personaplex_mlx.bench --model weights/personaplex-q6
"""

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .gen import PersonaPlexGen
from .model import PersonaPlexConfig, PersonaPlexLM, quantize_model

FRAME_MS = 80.0


def random_model(q: dict, dep_q: int = 8) -> PersonaPlexLM:
    model = PersonaPlexLM(PersonaPlexConfig(dep_q=dep_q))
    model.set_dtype(mx.bfloat16)
    quantize_model(model, q)
    # Materialize layer by layer so the float weights never all coexist in memory.
    for _, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.QuantizedLinear, nn.Embedding, nn.QuantizedEmbedding)):
            mx.eval(m.parameters())
    mx.eval(model.parameters())
    return model


def time_frames(gen: PersonaPlexGen, pos: int, n: int, rng) -> list[float]:
    gen.pos, gen.offset = pos, pos + 1
    times = []
    for _ in range(n):
        user = rng.integers(0, 2048, size=(gen.B, 8))
        t0 = time.perf_counter()
        gen.step(user)
        times.append((time.perf_counter() - t0) * 1e3)
        gen.pos = pos  # stay in the same KV bucket
    return times


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="converted checkpoint dir (default: random weights)")
    p.add_argument("--bits", type=int, default=6)
    p.add_argument("--bits-depformer", type=int, default=8)
    p.add_argument("--bits-other", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--dep-q", type=int, default=8, help="16 reproduces the upstream (uncut) depformer")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--positions", default="0,1500,2999", help="KV positions (conversation length in frames)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-kv-buckets", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    q = {"bits_temporal": args.bits, "bits_depformer": args.bits_depformer, "bits_other": args.bits_other, "group_size": args.group_size}
    t0 = time.perf_counter()
    if args.model:
        from .loader import load_lm

        model = load_lm(args.model)
    else:
        model = random_model(q, args.dep_q)
    load_s = time.perf_counter() - t0
    gen = PersonaPlexGen(model, batch_size=args.batch_size, compile=not args.no_compile, kv_buckets=not args.no_kv_buckets)
    positions = [int(x) for x in args.positions.split(",")]
    t0 = time.perf_counter()
    for pos in positions:
        time_frames(gen, pos, 3, np.random.default_rng(0))  # compile + warm
    warm_s = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    results = {}
    for pos in positions:
        t = np.array(time_frames(gen, pos, args.frames, rng))
        results[pos] = {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90))}
    weights_gb = sum(v.nbytes for _, v in nn.utils.tree_flatten(model.parameters())) / 1e9
    report = {
        "device": mx.device_info().get("device_name"),
        "quant": q if not args.model else "checkpoint",
        "dep_q": model.cfg.dep_q,
        "batch_size": args.batch_size,
        "compile": not args.no_compile,
        "kv_buckets": not args.no_kv_buckets,
        "weights_gb": round(weights_gb, 2),
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        "load_s": round(load_s, 1),
        "warmup_s": round(warm_s, 1),
        "frames": results,
    }
    if args.json:
        print(json.dumps(report))
        return
    print(f"{report['device']}  quant={report['quant']}  dep_q={report['dep_q']}  B={args.batch_size}  "
          f"compile={report['compile']}  weights={weights_gb:.2f} GB  peak={report['peak_mem_gb']} GB")
    for pos, r in results.items():
        rt = FRAME_MS / r["median_ms"]
        print(f"  kv_pos={pos:5d}  median {r['median_ms']:6.1f} ms  p90 {r['p90_ms']:6.1f} ms  "
              f"{rt:4.2f}x real time {'OK' if r['p90_ms'] < FRAME_MS else 'TOO SLOW'}")


if __name__ == "__main__":
    main()
