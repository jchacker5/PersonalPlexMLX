import asyncio

import numpy as np
from aiohttp import web, ClientSession


def test_websocket_stream_rejects_other_origin_and_bad_audio():
    from personaplex_mlx.server import create_app
    class Session:
        def reset(self): pass
        def process(self, frame): return frame * .5, ' hello'
    async def run():
        runner = web.AppRunner(create_app(Session()))
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        try:
            async with ClientSession() as client:
                r = await client.get(base + '/health')
                assert (await r.json())['sample_rate'] == 24000
                r = await client.get(base + '/ws', headers={'Origin': 'https://evil.example'})
                assert r.status == 403
                async with client.ws_connect(base + '/ws', origin=base) as ws:
                    assert (await ws.receive_json())['type'] == 'ready'
                    r = await client.get(base + '/ws', headers={'Origin': base})
                    assert r.status == 409
                    await ws.send_bytes(np.ones(1920, dtype='<f4').tobytes())
                    assert (await ws.receive_json())['text'] == ' hello'
                    audio = await ws.receive()
                    assert np.allclose(np.frombuffer(audio.data, dtype='<f4'), .5)
                    assert (await ws.receive_json())['type'] == 'frame'
                    await ws.send_bytes(b'bad')
                    assert (await ws.receive_json())['type'] == 'error'
        finally:
            await runner.cleanup()
    asyncio.run(run())
