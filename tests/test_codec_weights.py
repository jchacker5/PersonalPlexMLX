"""Opt-in codec parity check: PERSONALPLEX_MODEL=/path/to/converted/checkpoint pytest ..."""
import os

import mlx.core as mx
import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('PERSONALPLEX_MODEL'), reason='Requires downloaded model assets')
def test_real_codec_streaming_matches_upstream():
    import torch
    from moshi.models.loaders import get_mimi
    from personaplex_mlx.convert import MIMI_FILE
    from personaplex_mlx.loader import load_mimi
    from personaplex_mlx.runtime import read_audio, audio_frames
    from pathlib import Path
    path = Path(os.environ['PERSONALPLEX_MODEL'])
    torch.set_num_threads(2)
    upstream = get_mimi(path / MIMI_FILE, 'cpu')
    port = load_mimi(path)
    port.reset_all()
    upstream.streaming_forever(1)
    all_frames = list(audio_frames(read_audio(Path(__file__).parents[1] / 'assets/test/input_assistant.wav')))
    first_speech = next(i for i, frame in enumerate(all_frames) if np.sqrt(np.mean(frame ** 2)) > .02)
    frames = all_frames[first_speech:first_speech + 5]
    with torch.no_grad():
        for frame in frames:
            ref_latent = upstream._encode_to_unquantized_latent(torch.from_numpy(frame[None, None, :]))
            got_latent = port.encoder.step(mx.array(frame[None, None, :]))
            got_latent = port.encoder_transformer(got_latent, cache=port.encoder_cache)[0]
            got_latent = port.downsample.step(got_latent)
            mx.eval(got_latent)
            np.testing.assert_allclose(np.array(got_latent), ref_latent.numpy(), atol=1e-3, rtol=1e-3)
            # Tiny floating-point changes can select different low-energy residual codes,
            # especially on silence. Compare continuous encoder outputs and shared-code
            # decoding rather than requiring identical discrete residual codebooks.
            ref_codes = upstream.quantizer.encode(ref_latent)
            got_codes = mx.array(ref_codes.numpy())
            ref_audio = upstream.decode(ref_codes).numpy()
            got_audio = port.decode_step(got_codes)
            mx.eval(got_audio)
            np.testing.assert_allclose(np.array(got_audio), ref_audio, atol=2e-3, rtol=2e-3)
            assert np.sqrt(np.mean((np.array(got_audio) - ref_audio) ** 2)) < 2e-4
