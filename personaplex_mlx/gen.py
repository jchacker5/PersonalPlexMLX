"""Streaming generation for PersonaPlex in MLX.

Token bookkeeping (the delayed-codebook ring buffer and "provided" flags) is a faithful port of
`LMGen` in `moshi/moshi/models/lm.py`, kept on the host in numpy because it is tiny. Everything
that touches weights runs as a single compiled MLX function per frame, so there is exactly one
host/GPU synchronization per 80 ms frame.

The system prompt (voice prompt, silence, text prompt, silence) is fully teacher-forced: every
token the model would otherwise sample is overwritten by a provided one, so its only lasting
effect is the KV cache. `load_prompt` therefore records the prompt inputs on the host and runs
the temporal transformer over all of them in one batched `prefill`, instead of hundreds of
sequential steps. The resulting state can be snapshotted and restored for instant persona
switching.
"""

from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import numpy as np

from .model import PersonaPlexLM

TEXT_PAD = 3  # upstream `zero_text_code`, the text token used while prompting
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)
KV_BUCKET = 256


def sample_top_k(logits: mx.array, temp: float, top_k: int) -> mx.array:
    """logits: [B, V] -> [B] int32. temp <= 0 means greedy."""
    logits = logits.astype(mx.float32)
    if temp <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    logits = logits / temp
    V = logits.shape[-1]
    if 0 < top_k < V:
        kth = mx.partition(logits, V - top_k, axis=-1)[..., V - top_k : V - top_k + 1]
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits, axis=-1).astype(mx.int32)


@dataclass
class PromptState:
    """Snapshot of everything `PersonaPlexGen` needs to resume right after a system prompt."""

    cache: list
    ring: np.ndarray
    provided: np.ndarray
    offset: int
    pos: int


class PersonaPlexGen:
    def __init__(
        self,
        model: PersonaPlexLM,
        batch_size: int = 1,
        temp_audio: float = 0.8,
        temp_text: float = 0.7,
        top_k_audio: int = 250,
        top_k_text: int = 25,
        greedy: bool = False,
        compile: bool = True,
        kv_buckets: bool = True,
        audio_silence_frames: int = 6,
    ):
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        self.B = batch_size
        self.delays = np.array(cfg.delays)
        self.max_delay = int(self.delays.max())
        self.CT = self.max_delay + 3
        self.dep_q = cfg.dep_q
        self.n_user = cfg.n_q - cfg.dep_q
        self.silence_frames = audio_silence_frames
        self.kv_buckets = kv_buckets
        self.initial = np.array([cfg.text_card] + [cfg.card] * cfg.n_q, dtype=np.int64)
        if greedy:
            temp_audio = temp_text = 0.0
        self._sample_text = partial(sample_top_k, temp=temp_text, top_k=top_k_text)
        self._sample_audio = partial(sample_top_k, temp=temp_audio, top_k=top_k_audio)
        self._compile = compile
        self._frame_fns: dict[int, object] = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.ring = np.full((self.B, self.cfg.num_codebooks, self.CT), -2, dtype=np.int64)
        self.provided = np.zeros((self.B, self.cfg.num_codebooks, self.CT), dtype=bool)
        self.offset = 0  # upstream LMGen offset (one ahead of the transformer position)
        self.pos = 0  # number of temporal transformer steps taken
        self.cache = self.model.make_cache(self.B, dtype=self._cache_dtype())

    def _cache_dtype(self):
        return self.model.out_norm.weight.dtype

    def snapshot(self) -> PromptState:
        return PromptState(list(self.cache), self.ring.copy(), self.provided.copy(), self.offset, self.pos)

    def restore(self, s: PromptState):
        # Copy the cache so that streaming (which updates it in place) leaves the snapshot intact.
        self.cache = [(mx.array(k), mx.array(v)) for k, v in s.cache]
        self.ring, self.provided = s.ring.copy(), s.provided.copy()
        self.offset, self.pos = s.offset, s.pos

    # ------------------------------------------------------- token bookkeeping
    def _prepare(self, user=None, agent=None, text=None):
        """Port of LMGen.prepare_step_input. Inputs are [B, 8] / [B, 8] / int or [B]."""
        CT, o = self.CT, self.offset
        if user is not None:
            for q in range(self.n_user):
                k = 1 + self.dep_q + q
                wp = (o + self.delays[k]) % CT
                self.ring[:, k, wp] = user[:, q]
                self.provided[:, k, wp] = True
        if agent is not None:
            for q in range(self.dep_q):
                k = 1 + q
                wp = (o + self.delays[k]) % CT
                self.ring[:, k, wp] = agent[:, q]
                self.provided[:, k, wp] = True
        if text is not None:
            wp = (o + self.delays[0]) % CT
            self.ring[:, 0, wp] = text
            self.provided[:, 0, wp] = True
        for k, d in enumerate(self.delays):
            if o <= d:
                self.ring[:, k, o % CT] = self.initial[k]
                self.provided[:, k, o % CT] = True
        if o == 0:
            self.ring[:, :, 0] = self.initial
            self.offset += 1
            return None
        ip, tp = (o - 1) % CT, o % CT
        return self.ring[:, :, ip].copy(), self.ring[:, :, tp].copy(), self.provided[:, :, tp].copy(), ip, tp

    def _process(self, text_tok: np.ndarray, audio_tok: np.ndarray, ip: int, tp: int):
        """Port of the bookkeeping half of LMGen.process_transformer_output."""
        self.provided[:, :, ip] = False
        self.ring[:, 0, tp] = np.where(self.provided[:, 0, tp], self.ring[:, 0, tp], text_tok)
        sl = slice(1, self.dep_q + 1)
        self.ring[:, sl, tp] = np.where(self.provided[:, sl, tp], self.ring[:, sl, tp], audio_tok)
        if self.offset <= self.max_delay:
            self.offset += 1
            return None
        idx = (self.offset - self.max_delay + self.delays[: self.dep_q + 1]) % self.CT
        out = self.ring[:, np.arange(self.dep_q + 1), idx]
        self.offset += 1
        return out

    # ---------------------------------------------------------------- compute
    def _frame(self, kv_len, tokens, forced_text, text_prov, forced_audio, audio_prov, pos, cache):
        model = self.model
        x = model.embed(tokens[:, :, None])
        h, cache = model.temporal_step(x, cache, pos, kv_len)
        text_tok = self._sample_text(model.text_linear(h)[:, 0])
        dep_text = mx.where(text_prov, forced_text, text_tok)
        audio = model.depformer(h, dep_text, forced_audio, audio_prov, self._sample_audio)
        return text_tok, audio, cache

    def _frame_fn(self, kv_len: int):
        fn = self._frame_fns.get(kv_len)
        if fn is None:
            fn = partial(self._frame, kv_len)
            if self._compile:
                fn = mx.compile(fn, inputs=mx.random.state, outputs=mx.random.state)
            self._frame_fns[kv_len] = fn
        return fn

    def _kv_len(self, pos: int) -> int:
        C = self.cfg.context
        if not self.kv_buckets:
            return C
        return min(C, -(-(pos + 1) // KV_BUCKET) * KV_BUCKET)

    def _run_model(self, inp, target, prov):
        fn = self._frame_fn(self._kv_len(self.pos))
        cache, self.cache = self.cache, None
        text_tok, audio, cache = fn(
            mx.array(inp, dtype=mx.int32),
            mx.array(target[:, 0], dtype=mx.int32),
            mx.array(prov[:, 0]),
            mx.array(target[:, 1 : self.dep_q + 1], dtype=mx.int32),
            mx.array(prov[:, 1 : self.dep_q + 1]),
            mx.array(self.pos, dtype=mx.int32),
            cache,
        )
        self.cache = cache
        mx.eval(text_tok, audio, cache)
        self.pos += 1
        return np.array(text_tok), np.array(audio)

    def step(self, user_tokens: np.ndarray) -> np.ndarray | None:
        """Feed one frame of user audio tokens [B, 8].

        Returns [B, 9] (text token, 8 agent audio tokens) for an earlier frame once the
        delays are filled, else None.
        """
        prepared = self._prepare(user=np.asarray(user_tokens).reshape(self.B, self.n_user))
        if prepared is None:
            return None
        inp, target, prov, ip, tp = prepared
        text_tok, audio = self._run_model(inp, target, prov)
        return self._process(text_tok, audio, ip, tp)

    def warmup(self, all_buckets: bool = True):
        """Compile the frame function (for every KV bucket) so the first real frames are on time."""
        s = self.snapshot()
        C = self.cfg.context
        lens = sorted({self._kv_len(p) for p in range(0, C, KV_BUCKET)} | {C}) if all_buckets else [KV_BUCKET]
        for L in lens:
            self.pos = L - 1
            self._run_model(
                np.tile(self.initial, (self.B, 1)),
                np.tile(self.initial, (self.B, 1)),
                np.zeros((self.B, self.cfg.num_codebooks), dtype=bool),
            )
        self.restore(s)

    # ---------------------------------------------------------------- prompts
    def load_prompt(
        self,
        text_tokens: list[int] | None,
        voice_embeddings: np.ndarray | None = None,
        voice_ring: np.ndarray | None = None,
        voice_tokens: np.ndarray | None = None,
    ) -> PromptState:
        """Run the system prompt (voice, silence, text, silence) as one batched prefill.

        Voice can be given as precomputed upstream embeddings `[T, B, 1, dim]` plus the saved
        token ring (`voice_ring`, as stored in the upstream `.pt` files), or as Mimi tokens of
        a voice recording `[T, 8]`. Returns a snapshot that `restore` can reuse later.
        """
        self.reset()
        B = self.B
        chunks: list[mx.array] = []
        rows: list[np.ndarray] = []

        def flush_rows():
            if rows:
                chunks.append(self.model.embed(mx.array(np.stack(rows, axis=-1), dtype=mx.int32)))
                rows.clear()

        def record(user=None, agent=None, text=None, embedding=None):
            prepared = self._prepare(user, agent, text)
            if prepared is None:
                if embedding is None:
                    return  # upstream `step` consumes offset 0 without running the model
                # upstream `step_embeddings` retries until the model can run
                prepared = self._prepare(user, agent, text)
            inp, target, prov, ip, tp = prepared
            if not prov[:, : self.dep_q + 1].all():
                raise RuntimeError("system prompt step has unforced outputs; cannot prefill")
            if embedding is None:
                rows.append(inp)
            else:
                flush_rows()
                chunks.append(embedding)
            self._process(target[:, 0], target[:, 1 : self.dep_q + 1], ip, tp)

        sine = np.tile(SINE_TOKENS, (B, 1))
        silence = np.tile(SILENCE_TOKENS, (B, 1))
        if voice_embeddings is not None:
            init = np.tile(self.initial, (B, 1))
            for e in voice_embeddings:
                record(
                    user=init[:, 1 : 1 + self.n_user],
                    agent=init[:, 1 + self.n_user :],
                    text=TEXT_PAD,
                    embedding=mx.array(e).reshape(B, 1, -1),
                )
            if voice_ring is not None:
                self.ring[:] = voice_ring
        elif voice_tokens is not None:
            for t in voice_tokens:
                record(user=sine, agent=np.tile(t, (B, 1)), text=TEXT_PAD)
        for _ in range(self.silence_frames):
            record(user=sine, agent=silence, text=TEXT_PAD)
        for tok in text_tokens or []:
            record(user=sine, agent=silence, text=tok)
        for _ in range(self.silence_frames):
            record(user=sine, agent=silence, text=TEXT_PAD)
        flush_rows()

        x = mx.concatenate([c.astype(self._cache_dtype()) for c in chunks], axis=1)
        T = x.shape[1]
        _, self.cache = self.model.prefill(x, self.cache, 0)
        mx.eval(self.cache)
        self.pos = T
        return self.snapshot()
