"""Local-only, one-conversation PCM WebSocket server for PersonalPlexMLX."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web

from .offline import add_model_args, make_conversation
from .runtime import FRAME_SIZE, SAMPLE_RATE, validate_frame


def create_app(session, executor=None):
    busy = False
    executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='personalplex')

    @web.middleware
    async def local_only(request, handler):
        # Prevent DNS rebinding and drive-by cross-origin microphone sessions.
        if request.url.host not in ('127.0.0.1', 'localhost', '::1'):
            raise web.HTTPForbidden(text='Localhost only')
        origin = request.headers.get('Origin')
        if origin and origin != f'{request.scheme}://{request.host}':
            raise web.HTTPForbidden(text='Same-origin requests only')
        response = await handler(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    async def index(request):
        return web.Response(text=(Path(__file__).parent / 'web/index.html').read_text(), content_type='text/html')

    async def health(request):
        return web.json_response({'service': 'personalplex-mlx', 'sample_rate': SAMPLE_RATE,
                                  'frame_size': FRAME_SIZE, 'busy': busy})

    async def websocket(request):
        nonlocal busy
        if busy:
            raise web.HTTPConflict(text='A conversation is already active')
        busy = True
        ws = web.WebSocketResponse(max_msg_size=FRAME_SIZE * 4 + 1, heartbeat=20, compress=False)
        pending = None
        loop = asyncio.get_running_loop()
        async def compute(fn, *args):
            nonlocal pending
            pending = loop.run_in_executor(executor, fn, *args)
            result = await asyncio.shield(pending)
            pending = None
            return result
        try:
            await ws.prepare(request)
            await compute(session.reset)
            await ws.send_json({'type': 'ready', 'sample_rate': SAMPLE_RATE, 'frame_size': FRAME_SIZE})
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    try:
                        if len(message.data) != FRAME_SIZE * 4:
                            raise ValueError('Send exactly one 1920-sample float32 PCM frame')
                        frame = validate_frame(np.frombuffer(message.data, dtype='<f4'))
                        audio, text = await compute(session.process, frame)
                        if audio is not None:
                            await ws.send_json({'type': 'text', 'text': text})
                            await ws.send_bytes(np.asarray(audio, dtype='<f4').tobytes())
                        await ws.send_json({'type': 'frame'})
                    except ValueError as exc:
                        await ws.send_json({'type': 'error', 'message': str(exc)})
                        await ws.close(code=1003)
                        break
                elif message.type == WSMsgType.TEXT:
                    await ws.close(code=1003, message=b'Binary PCM only')
                    break
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            # Do not let a disconnect hand mutable MLX state to a new session mid-frame.
            try:
                if pending is not None:
                    await asyncio.shield(pending)
            finally:
                busy = False
        return ws

    async def cleanup(app):
        executor.shutdown(wait=True)

    app = web.Application(middlewares=[local_only], client_max_size=FRAME_SIZE * 4)
    app.router.add_get('/', index)
    app.router.add_get('/health', health)
    app.router.add_get('/ws', websocket)
    app.on_cleanup.append(cleanup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    parser.add_argument('--port', type=int, default=8999)
    args = parser.parse_args()
    print('Loading model and preparing voice prompt…', flush=True)
    # Keep model loading, lazy cache arrays and inference on the same MLX thread.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='personalplex')
    try:
        session = executor.submit(make_conversation, args).result()
        web.run_app(create_app(session, executor), host='127.0.0.1', port=args.port, access_log=None)
    finally:
        executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
