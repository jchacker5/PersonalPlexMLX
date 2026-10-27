"""Generate agent audio from a WAV using a locally converted MLX checkpoint."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .runtime import Conversation, DEFAULT_PROMPT, FRAME_SIZE, SAMPLE_RATE, audio_frames, read_audio


def generate_audio(session, pcm):
    started = time.perf_counter()
    output, text, latencies = [], [], []
    frames = (len(pcm) + FRAME_SIZE - 1) // FRAME_SIZE
    if not frames:
        raise ValueError('Input audio is empty')
    # Flush the generator's delayed codebooks so the final input frames are not lost.
    def inputs():
        yield from audio_frames(pcm)
        for _ in range(session.max_delay):
            yield np.zeros(FRAME_SIZE, dtype=np.float32)
    for frame in inputs():
        tick = time.perf_counter()
        audio, piece = session.process(frame)
        latencies.append((time.perf_counter() - tick) * 1000)
        if audio is not None:
            output.append(audio)
            text.append(piece)
    if not output:
        raise RuntimeError('No output audio generated')
    audio = np.concatenate(output)[:len(pcm)]
    audio = np.pad(audio, (0, max(0, len(pcm) - len(audio))))
    elapsed = time.perf_counter() - started
    return audio, ''.join(text), {
        'frames': frames, 'seconds': elapsed,
        'real_time_factor': elapsed / (len(pcm) / SAMPLE_RATE),
        'frame_ms_median': float(np.median(latencies)),
        'frame_ms_p95': float(np.percentile(latencies, 95)),
    }


def add_model_args(parser):
    parser.add_argument('--model', required=True, help='Converted checkpoint directory')
    parser.add_argument('--voice', default='NATF2', help='Converted voice preset name, without extension')
    parser.add_argument('--text-prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--seed', type=int, default=42424242)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--no-compile', action='store_true')


def make_conversation(args):
    return Conversation(args.model, args.voice, args.text_prompt, args.seed,
                        args.greedy, compile=not args.no_compile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument('--input-wav', required=True)
    parser.add_argument('--output-wav', required=True)
    parser.add_argument('--output-text', required=True, help='JSON transcript and timing report')
    args = parser.parse_args()
    pcm = read_audio(args.input_wav)
    session = make_conversation(args)
    audio, text, stats = generate_audio(session, pcm)
    for path in (args.output_wav, args.output_text):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, audio, SAMPLE_RATE, subtype='PCM_16')
    Path(args.output_text).write_text(json.dumps({'text': text, 'stats': stats}, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
