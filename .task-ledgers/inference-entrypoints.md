# Finish inference entry points
Goal: Finish offline WAV and local browser streaming on existing MLX port, then validate real weights when authenticated.
Checkout /Users/jchacker5/code/PersonalPlexMLX; branch mlx; starting head dc56a76.
Constraints: No main changes or NVIDIA submissions. Preserve licenses. No secrets or weights committed. Check disk before conversion; retain 5 GiB floor.
Design: Shared 24 kHz/1920-sample streaming session. Offline resamples mono audio and preserves duration. aiohttp server binds loopback, one session, bounded PCM WebSocket; explicit start/stop microphone page. Codecs reset per session; prompt state restored.
Done: Read generator, loader, upstream offline behavior, Mimi streaming APIs; approved model access visible in Arc.
Done: Implemented shared runtime, WAV runner, local server/client. Audio/server tests pass. CLI OAuth authorized explicitly by user; authenticated model HEAD succeeds. Bounded converter read-ahead test passes.
Done: Fixed worker-thread MLX tensor creation (regression test), strict Mimi loading now ignores only unused residual codebooks. Real codec loads and produces finite 1920-sample frames.
Now: Converting real q4 weights with 2 workers; log /tmp/personalplex-conversion.log. Running opt-in real-codec parity versus upstream Torch.
Next: Run suite and CLI checks, convert weights if access succeeds, offline quality/latency check, publish branch with honest WIP boundaries.

Validation update:
- Converted real checkpoint successfully; assets weights/personaplex-q4 (about 4.6 GiB), ignored by Git.
- Fixed missing causal mask and approximate GELU in vendored Mimi; focused regressions pass and real-codec encoder/shared-code decoder parity passes (five speech frames).
- No audible-quality acceptance yet. Full WAV command loaded model/prompt and entered generation, but was intentionally interrupted after disk fell below 5 GiB. No final WAV or timing receipt exists.
- Disk now about 2.1 GiB free; swap about 18 GiB. Stopped all task-owned inference/diagnostic processes; no unrelated cleanup or process termination.
- Remaining blocker: recover disk headroom above 5 GiB before heavy inference. Then run offline assistant + service, inspect/listen to WAV, verify real PCM WebSocket generation, and user microphone/speaker acceptance.
Now: Final lightweight suite, package/CLI checks, publish implementation on mlx with explicit remaining acceptance boundary.

Final verification:
- PERSONALPLEX_MODEL=weights/personaplex-q4 .venv/bin/python -m pytest tests -q: 14 passed in 6.22s, including real codec parity.
- Both module and installed CLI help paths work; client JS passes node --check.
- uv build --wheel succeeded; verified client asset inclusion, added explicit vendored Mimi LICENSE package data.
- Full WAV runtime/latency and live microphone acceptance remain blocked by disk headroom; about 1.7 GiB free. Preserve local converted weights for continuation.
- Publishing implementation and README on mlx, keeping PR #1 within user's public fork and main untouched.
