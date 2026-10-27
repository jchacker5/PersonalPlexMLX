import numpy as np
import pytest


def test_audio_resampling_and_padding(tmp_path):
    import soundfile as sf
    from personaplex_mlx.runtime import read_audio, audio_frames
    p = tmp_path / 'stereo.wav'
    sf.write(p, np.ones((4410, 2), dtype=np.float32) * .25, 44100)
    pcm = read_audio(p)
    assert pcm.shape == (2400,)
    frames = list(audio_frames(pcm))
    assert len(frames) == 2
    assert all(f.shape == (1920,) for f in frames)
    assert np.all(frames[-1][480:] == 0)


def test_frame_validation():
    from personaplex_mlx.runtime import validate_frame
    with pytest.raises(ValueError):
        validate_frame(np.zeros(1919))
    with pytest.raises(ValueError):
        validate_frame(np.full(1920, np.nan))
    assert validate_frame(np.full(1920, 2.)).max() == 1.


def test_offline_preserves_duration_and_flushes_delays():
    from personaplex_mlx.offline import generate_audio
    class Session:
        max_delay = 2
        def __init__(self): self.calls = 0
        def process(self, frame):
            self.calls += 1
            if self.calls <= 2: return None, ''
            return np.full(1920, .2, dtype=np.float32), ' hi'
    s = Session()
    pcm, text, stats = generate_audio(s, np.zeros(2400, dtype=np.float32))
    assert pcm.shape == (2400,)
    assert np.allclose(pcm, .2)
    assert s.calls == 4
    assert text == ' hi hi'
    assert stats['frames'] == 2
