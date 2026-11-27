# Real-weight demo and measurements

These examples use the locally converted NVIDIA PersonaPlex checkpoint on an Apple M4 with
32 GB of unified memory: temporal transformer and depformer at 4 bits, embeddings/head at 8 bits.
They demonstrate completed offline inference. The complete pipeline is currently **slower than
real time** on this machine. They are not evidence of accepted microphone/speaker behavior.

## Assistant example

- [Input audio](../assets/test/input_assistant.wav), supplied with NVIDIA's original repository.
- [Generated audio — 40 seconds](assistant-q4.wav).
- [Generated text and timing](assistant-q4.json).
- [Run configuration and input hash](assistant-receipt.json).

The WAV is the full output, without editorial cuts. It is finite, non-silent, 24 kHz mono audio
with the same duration as the input. This is a structural validation; subjective sound quality
and comparative listening across quantization levels remain unaccepted.

Generation took **103.51 seconds for 40 seconds of audio**: real-time factor **2.59**
(lower is faster; real time requires less than 1). Median frame time was **205.03 ms** and
95th percentile **277.50 ms**; a live 12.5 Hz pipeline has an 80 ms frame budget.

Timing includes streaming encoding, language-model generation and audio decoding. Model loading
and prompt prefill are excluded. Later attention buckets can compile during the measured loop.
This is one run on a shared development machine, not a controlled multi-run benchmark.

```bash
personalplex-offline --model weights/personaplex-q4 --voice NATF2 \
  --input-wav assets/test/input_assistant.wav \
  --output-wav output/assistant.wav --output-text output/assistant.json
```

## Second run and memory

A second, 20-second [input](../assets/test/input_service.wav) with the NATM1 voice and a
CitySan customer-service role produced [this output](service-q4.wav) and
[text/timing](service-q4.json). The exact role prompt is recorded in the
[execution receipt](service-receipt.json); it is a different role from the upstream repair-service
prompt, so this is a timing/conditioning example rather than a task-accuracy benchmark.

Generation took **52.24 seconds** (real-time factor **2.61**), with median frame time
**204.41 ms** and 95th percentile **281.99 ms**. Including loading, the process took **57.27 seconds**.
Peak MLX allocation was **7.61 GiB**; peak process RSS was **4.36 GiB**. These are overlapping,
different accounting measures, not additive totals. Environment: Python 3.12.12, MLX 0.32.4,
macOS 27.0.1. Both runs used seed 42424242 and sampling, not greedy decoding.

## Reproduce a memory receipt

The wrapper below invokes the same offline CLI and records environment versions, input hash,
quantization settings, wall time including loading, peak MLX allocation and process peak RSS.
MLX allocation and RSS measure different things; neither is the machine's total memory usage.

```bash
python scripts/record_demo.py --receipt output/receipt.json \
  --model weights/personaplex-q4 --voice NATF2 \
  --input-wav assets/test/input_assistant.wav \
  --output-wav output/assistant.wav --output-text output/assistant.json
```

## Local server check

A real-weight local HTTP/WebSocket check processed eight PCM frames and returned eight finite
1920-sample audio frames. See the [receipt](websocket-receipt.json). This validates the server
inference path; it does not substitute for browser microphone/speaker acceptance.

## Interpretation

The earlier 60–75 ms figures were historical language-model-only random-weight measurements.
This complete real-weight run does not establish real-time performance. The current engineering
priority is profiling the codec, compilation transitions and frame dispatch, followed by repeated
runs and listening comparisons. No throughput claim here extrapolates to other Apple chips.

Generated output comes from NVIDIA PersonaPlex; upstream model/codec credit and separate model
terms are linked in the repository README. No checkpoint weights are redistributed.
