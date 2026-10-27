from concurrent.futures import ThreadPoolExecutor


def test_converter_bounds_read_ahead():
    from personaplex_mlx.convert import bounded_map
    consumed = []
    def names():
        for n in range(20):
            consumed.append(n)
            yield n
    with ThreadPoolExecutor(2) as pool:
        output = bounded_map(pool, lambda n: n * 2, names(), 2)
        assert next(output) == 0
        assert len(consumed) <= 3
        assert list(output) == list(range(2, 40, 2))


def test_bfloat_quantization_keeps_mlx_on_calling_thread(tmp_path):
    import mlx.core as mx
    import torch
    from safetensors.torch import save_file
    from personaplex_mlx.convert import TensorSource, convert_lm
    from personaplex_mlx.model import PersonaPlexConfig
    src = tmp_path / 'source.safetensors'
    save_file({'text_linear.weight': torch.ones((64, 64), dtype=torch.bfloat16)}, src)
    out = tmp_path / 'out'
    out.mkdir()
    q = dict(bits_temporal=4, bits_depformer=4, bits_other=4, group_size=64)
    convert_lm(TensorSource(str(src)), out, PersonaPlexConfig(), q, workers=2)
    weights = mx.load(str(out / 'model-00000.safetensors'))
    restored = mx.dequantize(weights['text_linear.weight'], weights['text_linear.scales'],
                             weights['text_linear.biases'], bits=4, group_size=64)
    assert mx.allclose(restored, mx.ones((64, 64))).item()
