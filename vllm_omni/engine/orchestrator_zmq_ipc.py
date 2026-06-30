"""ZMQ IPC between APIServer and the orchestrator worker process."""

from __future__ import annotations

import pickle
import queue
import threading
import time
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio
from vllm.logger import init_logger

from vllm_omni.engine.messages import EngineQueueMessage

logger = init_logger(__name__)

READY_FILENAME = "ready"
WORKER_INIT_FILENAME = "worker_init.pkl"


def ipc_paths(ipc_dir: str) -> dict[str, str]:
    base = Path(ipc_dir).resolve()
    return {
        "req": f"ipc://{base / 'orch-req'}",
        "out": f"ipc://{base / 'orch-out'}",
        "rpc": f"ipc://{base / 'orch-rpc'}",
    }


def _pickle_msg(msg: EngineQueueMessage) -> bytes:
    return pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)


def _unpickle_msg(data: bytes) -> EngineQueueMessage:
    msg = pickle.loads(data)
    if not isinstance(msg, EngineQueueMessage):
        raise TypeError(f"expected EngineQueueMessage, got {type(msg)}")
    return msg


class OrchestratorZmqServer:
    """Worker-side ZMQ endpoints used directly by the orchestrator event loop."""

    def __init__(self, ipc_dir: str) -> None:
        self._ipc_dir = ipc_dir
        self._paths = ipc_paths(ipc_dir)
        self._ctx = zmq.asyncio.Context.instance()
        self._req = self._ctx.socket(zmq.PULL)
        self._req.bind(self._paths["req"])
        self._out = self._ctx.socket(zmq.PUSH)
        self._out.bind(self._paths["out"])
        self._rpc = self._ctx.socket(zmq.PUSH)
        self._rpc.bind(self._paths["rpc"])
        logger.info("[OrchestratorZmqServer] bound on %s", ipc_dir)

    async def recv_request(self) -> EngineQueueMessage:
        data = await self._req.recv()
        return _unpickle_msg(data)

    async def recv_request_nowait(self) -> EngineQueueMessage | None:
        if not await self._req.poll(timeout=0):
            return None
        data = await self._req.recv()
        return _unpickle_msg(data)

    async def send_output(self, msg: EngineQueueMessage) -> None:
        payload = _pickle_msg(msg)
        await self._out.send(payload, copy=False)

    async def send_rpc(self, msg: EngineQueueMessage) -> None:
        payload = _pickle_msg(msg)
        await self._rpc.send(payload, copy=False)

    def close(self) -> None:
        for sock in (self._req, self._out, self._rpc):
            try:
                sock.close(linger=0)
            except Exception:
                pass


class OrchestratorZmqSender:
    """Send-only client for in-worker callbacks (must not connect out/rpc)."""

    def __init__(self, ipc_dir: str) -> None:
        self._paths = ipc_paths(ipc_dir)
        self._ctx = zmq.Context.instance()
        self._req = self._ctx.socket(zmq.PUSH)
        self._req.connect(self._paths["req"])

    def send(self, msg: EngineQueueMessage) -> None:
        payload = _pickle_msg(msg)
        self._req.send(payload, copy=False)

    def close(self) -> None:
        try:
            self._req.close(linger=0)
        except Exception:
            pass


class OrchestratorZmqClient:
    """APIServer-side client for the orchestrator worker."""

    def __init__(self, ipc_dir: str) -> None:
        self._ipc_dir = ipc_dir
        self._paths = ipc_paths(ipc_dir)
        self._ctx = zmq.Context.instance()
        self._req = self._ctx.socket(zmq.PUSH)
        self._req.connect(self._paths["req"])
        self._out = self._ctx.socket(zmq.PULL)
        self._out.connect(self._paths["out"])
        self._rpc = self._ctx.socket(zmq.PULL)
        self._rpc.connect(self._paths["rpc"])
        self._rpc_buffer: queue.Queue[EngineQueueMessage] = queue.Queue()
        self._rpc_thread = threading.Thread(target=self._drain_rpc, name="orch-zmq-rpc-drain", daemon=True)
        self._closed = False
        self._rpc_thread.start()

    def _drain_rpc(self) -> None:
        poller = zmq.Poller()
        poller.register(self._rpc, zmq.POLLIN)
        while not self._closed:
            events = dict(poller.poll(200))
            if self._rpc not in events:
                continue
            self._rpc_buffer.put(_unpickle_msg(self._rpc.recv()))

    def send(self, msg: EngineQueueMessage) -> None:
        payload = _pickle_msg(msg)
        self._req.send(payload, copy=False)

    def recv_output(self, timeout: float = 0.001) -> EngineQueueMessage | None:
        ready = self._out.poll(int(timeout * 1000))
        if not ready:
            return None
        data = self._out.recv()
        return _unpickle_msg(data)

    def recv_rpc(self, timeout: float | None = None) -> EngineQueueMessage:
        if timeout is None:
            return self._rpc_buffer.get()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            try:
                return self._rpc_buffer.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed = True
        for sock in (self._req, self._out, self._rpc):
            try:
                sock.close(linger=0)
            except Exception:
                pass


def write_worker_init(ipc_dir: str, model: str, kwargs: dict[str, Any]) -> None:
    path = Path(ipc_dir) / WORKER_INIT_FILENAME
    payload = {"model": model, "kwargs": kwargs}
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def read_worker_init(ipc_dir: str) -> tuple[str, dict[str, Any]]:
    path = Path(ipc_dir) / WORKER_INIT_FILENAME
    payload = pickle.loads(path.read_bytes())
    return payload["model"], payload["kwargs"]


def mark_ready(ipc_dir: str) -> None:
    (Path(ipc_dir) / READY_FILENAME).write_text(str(time.time()))


def wait_ready(ipc_dir: str, timeout_s: float) -> None:
    ready = Path(ipc_dir) / READY_FILENAME
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready.is_file():
            return
        time.sleep(0.2)
    raise TimeoutError(f"Orchestrator worker did not become ready within {timeout_s}s ({ipc_dir})")


def make_ipc_dir(prefix: str = "vllm_omni_orch_ipc_") -> str:
    import tempfile

    return tempfile.mkdtemp(prefix=prefix)


def cleanup_ipc_dir(ipc_dir: str) -> None:
    import shutil

    try:
        shutil.rmtree(ipc_dir, ignore_errors=True)
    except Exception:
        pass
