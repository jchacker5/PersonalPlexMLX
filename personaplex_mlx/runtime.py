"""Shared, single-conversation streaming pipeline for the WAV and browser runners."""
from math import gcd
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .gen import PersonaPlexGen
from .loader import load_lm, load_mimi, load_tokenizer, load_voice

SAMPLE_RATE = 24000
FRAME_SIZE = 1920
DEFAULT_PROMPT = 'You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.'


def read_audio(path: str | Path) -> np.ndarray:
    pcm, rate = sf.read(path, dtype='float32', always_2d=True)
    pcm = pcm.mean(axis=1)
    if not pcm.size or not np.isfinite(pcm).all():
        raise ValueError('Audio must be nonempty and finite')
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        pcm = resample_poly(pcm, SAMPLE_RATE // divisor, rate // divisor)
    return np.clip(pcm, -1, 1).astype(np.float32)


def audio_frames(pcm: np.ndarray):
    for start in range(0, len(pcm), FRAME_SIZE):
        frame = pcm[start:start + FRAME_SIZE]
        yield np.pad(frame, (0, FRAME_SIZE - len(frame))).astype(np.float32)


def validate_frame(pcm: np.ndarray) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.shape != (FRAME_SIZE,) or not np.isfinite(pcm).all():
        raise ValueError(f'Expected {FRAME_SIZE} finite mono samples at {SAMPLE_RATE} Hz')
    return np.clip(pcm, -1, 1)


class Conversation:
    """Owns mutable codec/generator state; use from one worker at a time."""
    def __init__(self, model_path, voice='NATF2', text_prompt=DEFAULT_PROMPT,
                 seed=42424242, greedy=False, compile=True):
        mx.random.seed(seed)
        self.model = load_lm(model_path)
        self.mimi = load_mimi(model_path)
        self.tokenizer = load_tokenizer(model_path)
        self.gen = PersonaPlexGen(self.model, greedy=greedy, compile=compile)
        self.max_delay = self.gen.max_delay
        prompt = text_prompt.strip()
        if prompt and not (prompt.startswith('<system>') and prompt.endswith('<system>')):
            prompt = f'<system> {prompt} <system>'
        tokens = self.tokenizer.encode(prompt) if prompt else None
        preset = load_voice(model_path, voice)
        self.mimi.reset_all()
        kwargs = {}
        if 'embeddings' in preset:
            kwargs = dict(voice_embeddings=preset['embeddings'], voice_ring=preset['cache'])
        else:
            codes = []
            for frame in audio_frames(read_audio(preset['wav'])):
                code = self.mimi.encode_step(mx.array(frame[None, None, :]))
                mx.eval(code)
                codes.append(np.array(code)[0].T)
            kwargs['voice_tokens'] = np.concatenate(codes, axis=0)
        self.prompt_state = self.gen.load_prompt(tokens, **kwargs)
        # Compile the initial attention bucket without changing the saved prompt.
        self.gen.warmup(all_buckets=False)
        self.reset()

    def reset(self):
        self.gen.restore(self.prompt_state)
        self.mimi.reset_all()

    def process(self, pcm):
        pcm = validate_frame(pcm)
        codes = self.mimi.encode_step(mx.array(pcm[None, None, :]))
        mx.eval(codes)
        codes = np.array(codes)
        if codes.shape != (1, 8, 1):
            raise RuntimeError(f'Unexpected Mimi frame shape: {codes.shape}')
        tokens = self.gen.step(codes[:, :, 0])
        if tokens is None:
            return None, ''
        audio = self.mimi.decode_step(mx.array(tokens[:, 1:, None], dtype=mx.int32))
        mx.eval(audio)
        pcm_out = np.array(audio, dtype=np.float32).reshape(-1)
        if pcm_out.shape != (FRAME_SIZE,) or not np.isfinite(pcm_out).all():
            raise RuntimeError('Mimi produced an invalid audio frame')
        token = int(tokens[0, 0])
        text = self.tokenizer.id_to_piece(token).replace('▁', ' ') if token > 3 else ''
        return pcm_out, text
