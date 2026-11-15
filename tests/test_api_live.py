"""WebSocket API and live UDP intake (real sockets on localhost)."""

import asyncio
import concurrent.futures
import json
import socket
import time

import numpy as np
import pytest
from synthetic import raw_packet, stub_predict, write_session
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from trifi.api import Api
from trifi.app import LiveApp, serve_replay
from trifi.config import Config
from trifi.inference import InferenceWorker, ModelInfo


def free_port(kind=socket.SOCK_STREAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


# ── API ──────────────────────────────────────────────────────────────────────

def test_token_required_and_messages_validated():
    modes = []

    async def scenario():
        cfg = Config(api_token="s3cret")
        api = Api(cfg, hello=lambda: {"node_ids": [1, 2, 3]}, on_set_mode=modes.append)
        server = await api.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            with pytest.raises(InvalidStatus) as exc:
                async with connect(f"ws://127.0.0.1:{port}/"):
                    pass
            assert exc.value.response.status_code == 401
            with pytest.raises(InvalidStatus):
                async with connect(f"ws://127.0.0.1:{port}/?token=wrong"):
                    pass

            async with connect(f"ws://127.0.0.1:{port}/?token=s3cret") as ws:
                hello = json.loads(await ws.recv())
                assert hello["type"] == "hello" and hello["schema"] == 1
                for bad in ('{"type":"set_op_mode","mode":9}', '{"type":"set_op_mode","mode":"1"}',
                            '{"type":"set_op_mode","mode":true}', "not json", '{"type":"reboot"}'):
                    await ws.send(bad)
                    assert json.loads(await ws.recv())["type"] == "error"
                await ws.send('{"type":"set_op_mode","mode":2}')
                api.broadcast({"type": "rescue_ai", "schema": 1})
                assert json.loads(await ws.recv())["type"] == "rescue_ai"
        finally:
            server.close()
            await server.wait_closed()

    run(scenario())
    assert modes == [2]


def test_replay_api_rejects_mode_changes():
    async def scenario():
        api = Api(Config(), hello=lambda: {})
        server = await api.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.recv()
                await ws.send('{"type":"set_op_mode","mode":1}')
                assert "replay" in json.loads(await ws.recv())["error"]
        finally:
            server.close()
            await server.wait_closed()

    run(scenario())


def test_oversized_message_closes_connection():
    async def scenario():
        api = Api(Config(), hello=lambda: {})
        server = await api.start("127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.recv()
                await ws.send("x" * 10_000)
                with pytest.raises(ConnectionClosed):
                    await ws.recv()
        finally:
            server.close()
            await server.wait_closed()

    run(scenario())


# ── Live intake under load (Phase 2 gate: < 1% drops at 3 x 100 Hz) ─────────

def slow_predict(window: np.ndarray) -> np.ndarray:
    time.sleep(0.03)                     # slower than a real CPU ResNet-18 pass
    return stub_predict(window)


@pytest.mark.parametrize("seconds", [3.0])
def test_live_intake_keeps_up_while_inference_is_slow(seconds):
    cfg = Config(udp_host="127.0.0.1", udp_port=free_port(socket.SOCK_DGRAM),
                 ws_host="127.0.0.1", ws_port=free_port())
    worker = InferenceWorker(ModelInfo(path=None, n_classes=2),
                             executor=concurrent.futures.ThreadPoolExecutor(1),
                             predict_fn=slow_predict)
    app = LiveApp(cfg, worker=worker)
    rate, nodes = 100, (1, 2, 3)
    packets = [raw_packet(n, i + 1, np.full(64, 20.0) + (i % 7), -50)
               for i in range(int(seconds * rate)) for n in nodes]

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(app.run(stop))
        await asyncio.sleep(0.3)
        loop = asyncio.get_running_loop()

        def blast():
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                start = time.perf_counter()
                for k, p in enumerate(packets):
                    due = start + (k // len(nodes)) / rate
                    while time.perf_counter() < due:
                        time.sleep(0.001)
                    s.sendto(p, ("127.0.0.1", cfg.udp_port))

        await loop.run_in_executor(None, blast)
        await asyncio.sleep(0.5)
        stop.set()
        await task

    run(scenario())
    received = app.packets
    assert received >= 0.99 * len(packets), f"{received}/{len(packets)} received"
    frames = sum(app.pipeline.parser.stats[n].frames for n in nodes)
    assert frames == received
    assert all(app.pipeline.parser.stats[n].lost <= 0.01 * seconds * rate for n in nodes)
    assert sum(n.ml_prob > 0 or len(n.prob_hist) > 0 for n in app.pipeline.nodes.values()) == 3


# ── Replay to the dashboard ──────────────────────────────────────────────────

def test_serve_replay_streams_session(tmp_path):
    cfg = Config(ws_host="127.0.0.1", ws_port=free_port())
    session = write_session(tmp_path / "s.jsonl.gz", cfg, seconds=12.0)
    received = []

    async def scenario():
        task = asyncio.create_task(serve_replay(cfg, session, stub_predict, 2, speed=50.0))
        await asyncio.sleep(0.3)
        async with connect(f"ws://127.0.0.1:{cfg.ws_port}/") as ws:
            hello = json.loads(await ws.recv())
            assert hello["type"] == "hello" and hello["live"] is False
            try:
                async for msg in ws:
                    received.append(json.loads(msg))
            except ConnectionClosed:
                pass
        return await task

    count = run(scenario())
    assert count == 60                                   # 12 s at 5 Hz
    assert received and all(m["replay"] and m["schema"] == 1 for m in received)
