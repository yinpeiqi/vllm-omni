"""Regression tests for orchestrator ZMQ IPC."""

from __future__ import annotations

import asyncio
import threading

import pytest

from vllm_omni.engine.messages import OutputMessage
from vllm_omni.engine.orchestrator_zmq_ipc import (
    OrchestratorZmqClient,
    OrchestratorZmqSender,
    OrchestratorZmqServer,
    cleanup_ipc_dir,
    make_ipc_dir,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_send_only_client_does_not_steal_outputs() -> None:
    """In-worker send-only clients must not connect out/rpc and steal API messages."""
    ipc_dir = make_ipc_dir(prefix="test_zmq_ipc_")
    try:
        server = OrchestratorZmqServer(ipc_dir)
        in_worker_sender = OrchestratorZmqSender(ipc_dir)
        api_client = OrchestratorZmqClient(ipc_dir)

        async def _send_output() -> None:
            await server.send_output(
                OutputMessage(
                    request_id="req-1",
                    stage_id=0,
                    engine_outputs=OmniRequestOutput(request_id="req-1"),
                    finished=True,
                )
            )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_send_output())
        finally:
            loop.close()

        msg = api_client.recv_output(timeout=1.0)
        assert msg is not None
        assert msg.request_id == "req-1"
        assert msg.finished is True
        assert api_client.recv_output(timeout=0.01) is None
    finally:
        in_worker_sender.close()
        api_client.close()
        server.close()
        cleanup_ipc_dir(ipc_dir)


def test_full_client_steals_outputs_without_consumer() -> None:
    """Document the failure mode when a second full client connects but never recv_output."""
    ipc_dir = make_ipc_dir(prefix="test_zmq_ipc_")
    try:
        server = OrchestratorZmqServer(ipc_dir)
        ghost_client = OrchestratorZmqClient(ipc_dir)
        api_client = OrchestratorZmqClient(ipc_dir)

        async def _send_many() -> None:
            for i in range(20):
                await server.send_output(
                    OutputMessage(
                        request_id=f"req-{i}",
                        stage_id=0,
                        engine_outputs=OmniRequestOutput(request_id=f"req-{i}"),
                        finished=True,
                    )
                )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_send_many())
        finally:
            loop.close()

        received = []
        deadline = threading.Event()

        def _drain_api() -> None:
            while len(received) < 20 and not deadline.wait(0.05):
                msg = api_client.recv_output(timeout=0.05)
                if msg is not None:
                    received.append(msg.request_id)

        t = threading.Thread(target=_drain_api, daemon=True)
        t.start()
        t.join(timeout=2.0)
        deadline.set()

        assert len(received) < 20, "ghost full client should steal some output messages"
    finally:
        ghost_client.close()
        api_client.close()
        server.close()
        cleanup_ipc_dir(ipc_dir)
