# PersonalPlexMLX

**An independent Apple Silicon inference port of NVIDIA PersonaPlex, built with MLX.**

This project explores a practical systems problem: how to run a 7B full-duplex speech model
on a base M4 MacBook with 32 GB of unified memory. It adds a quantized model port, a bounded
streaming checkpoint converter, an offline WAV runner, and a localhost microphone interface.

**Status: experimental.** Tiny-model language-model parity and real-weight codec checks pass.
Completed end-to-end measurements and audio examples are documented in [the demo report](demos/README.md).
Microphone/speaker acceptance and broad audio-quality evaluation remain separate work.

## Engineering contributions

- **Reduced inference work:** generate the eight agent audio codebooks; omit the eight
  user-codebook predictions that upstream replaces with microphone input. Greedy parity is
  tested against upstream on a tiny random model, including cache wraparound.
- **Compiled frame execution:** combine temporal transformer, text sampling, depformer and
  audio sampling into one compiled MLX language-model call per frame.
- **Prompt prefill and reuse:** batch teacher-forced voice/text prompts, then snapshot and
  restore conversation state.
- **Memory-aware conversion:** fetch the 16.7 GB source checkpoint by HTTP ranges and quantize
  tensors incrementally. Bound download read-ahead and keep MLX operations on the calling thread
  to avoid stream-affinity errors. Source weights never need to occupy disk in full.
- **Mixed precision and attention buckets:** use separate bit widths for temporal, depformer,
  and embedding/head weights; bucket attention lengths over a fixed temporal KV ring.
- **Codec correctness:** retain the required Mimi codebooks with strict loading, enforce causal
  attention inside audio chunks, and match upstream's exact GELU activation.

The port and its tools live in [`personaplex_mlx/`](personaplex_mlx/). The original PyTorch
implementation in [`moshi/`](moshi/) is retained as a reference.

## Quick start

Requires Apple Silicon, macOS, Python 3.12, and [uv](https://docs.astral.sh/uv/).
The measured development machine is an Apple M4 with 32 GB RAM. Allow approximately 4.6 GiB
for converted assets plus working space; retain at least 5 GiB of free disk before inference.
These are development observations, not a minimum-hardware guarantee.

```bash
git clone https://github.com/jchacker5/PersonalPlexMLX.git
cd PersonalPlexMLX
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[convert,server]"
```

Accept access terms for [NVIDIA's model](https://huggingface.co/nvidia/personaplex-7b-v1)
in your Hugging Face account, then authenticate locally. A read token is sufficient for downloads;
review the permissions if you choose the CLI's browser authorization flow.

```bash
hf auth login
personalplex-convert --out weights/personaplex-q4 --bits 4 --bits-depformer 4
```

The default embedding/head precision is 8-bit. Model weights are downloaded and converted
locally; they are not included in this repository.

### Generate a WAV

```bash
personalplex-offline --model weights/personaplex-q4 --voice NATF2 \
  --input-wav assets/test/input_assistant.wav \
  --output-wav output/assistant.wav --output-text output/assistant.json
```

Input audio is resampled to 24 kHz mono. The output has the input's duration; the JSON contains
text and frame timing. Timing includes the codec and language model and excludes model loading
and prompt prefill. See [the demo report](demos/README.md) for measured scope and limitations.

### Talk locally

```bash
personalplex-server --model weights/personaplex-q4 --voice NATF2 --port 8999
```

Open `http://127.0.0.1:8999` in a current Chromium browser. Use headphones, click **Start
conversation**, and grant microphone access. **Stop** releases the microphone.
The server binds to loopback, allows one conversation, rejects cross-origin requests, and uses
raw float32 PCM over a local WebSocket. The client stops when audio falls behind.
Audio is not saved or sent to an external inference service by this client.

Both commands support `--text-prompt`, `--seed`, `--greedy`, and `--no-compile`.

## Verification

```bash
uv pip install -e ".[dev,server]" -e ./moshi
python -m pytest tests -q
# With locally downloaded assets, also run real-codec verification:
PERSONALPLEX_MODEL=weights/personaplex-q4 python -m pytest tests -q
```

Tests cover tiny-model greedy parity against upstream PyTorch, compiled/eager execution,
cache wraparound, prompt restoration, bounded conversion and bfloat16 thread handling,
audio framing/resampling, causal codec attention, GELU matching, and WebSocket validation.
The opt-in real-codec test compares continuous encoder features and shared-code decoded audio
on five speech frames. It does not establish full-model quantization parity or human sound quality.

## Tradeoffs and remaining work

4-bit quantization reduces memory traffic but can affect speech quality. Random-weight
language-model timing is not proof of end-to-end real-time conversation. The earlier development
benchmark reported roughly 60–75 ms per language-model frame at 4-bit on this M4; treat it as
historical, model-only evidence. Use the reproducible real-weight results in the demo report
for the current complete pipeline.

Remaining evaluation includes longer conversations, repeated timing runs, listening comparisons
across precision levels, and microphone/speaker acceptance. This repository is an inference port,
not a newly trained foundation model.

## Credits and licenses

Maintained by [jchacker5](https://github.com/jchacker5). This is an independent community project,
not an official NVIDIA or Kyutai release and not endorsed by either organization.

- **NVIDIA:** [PersonaPlex model, research and original implementation](https://github.com/NVIDIA/personaplex).
- **Kyutai:** [Moshi and Mimi](https://github.com/kyutai-labs/moshi), including the vendored MLX codec.
- **Apple:** MLX and the cache implementation credited in the vendored source.

Code is MIT; upstream source notices and license files are retained, including the
[vendored codec license](personaplex_mlx/mimi/LICENSE). Weights have separate terms in the
[NVIDIA model repository](https://huggingface.co/nvidia/personaplex-7b-v1).
See the [preserved upstream README](docs/UPSTREAM_README.md) for research citation and original usage.
