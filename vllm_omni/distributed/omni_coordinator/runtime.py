# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lifecycle wrapper around :class:`OmniCoordinator`.

``OmniCoordinatorRuntime`` spawns the coordinator as an independent process
(matching vLLM's DPCoordinator pattern). Physical isolation prevents GIL
contention and makes direct-object coupling impossible.

The ROUTER address is later handed to :class:`OmniMasterServer` so it can be
published to registering replicas; the PUB address is handed to the
``MembershipController``, which constructs its :class:`OmniCoordClientForHub`
against it.
"""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.connection
import weakref

from vllm.utils.network_utils import get_open_ports_list

logger = logging.getLogger(__name__)


def _shutdown_proc(proc: multiprocessing.Process) -> None:
    """Best-effort process termination for weakref finalizer."""
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2)


class OmniCoordinatorRuntime:
    """Own one :class:`OmniCoordinator` running in a child process.

    Constructor spawns the process; :meth:`close` tears it down.
    The class deliberately does not expose the coordinator instance —
    callers consume it only via ZMQ through :class:`OmniCoordClientForStage`
    and :class:`OmniCoordClientForHub`.
    """

    def __init__(
        self,
        *,
        host: str,
        heartbeat_timeout: float,
    ) -> None:
        if not host:
            raise ValueError("host must be a non-empty string")
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be positive")

        router_port, pub_port = get_open_ports_list(count=2)
        self.router_address: str = f"tcp://{host}:{router_port}"
        self.pub_address: str = f"tcp://{host}:{pub_port}"

        self._closed = False

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        from .omni_coordinator_proc import OmniCoordinatorProc

        self._proc: multiprocessing.Process = ctx.Process(
            target=OmniCoordinatorProc.run,
            kwargs={
                "router_zmq_addr": self.router_address,
                "pub_zmq_addr": self.pub_address,
                "heartbeat_timeout": heartbeat_timeout,
                "ready_pipe": child_conn,
            },
            daemon=True,
            name="OmniCoordinator",
        )
        self._proc.start()
        child_conn.close()

        ready = multiprocessing.connection.wait([parent_conn, self._proc.sentinel], timeout=30)
        if not ready:
            self._proc.terminate()
            self._proc.join(timeout=5)
            raise RuntimeError("OmniCoordinator process failed to start within 30s")

        try:
            status = parent_conn.recv()
        except EOFError:
            raise RuntimeError("OmniCoordinator process died during startup") from None
        finally:
            parent_conn.close()

        if status != "ready":
            raise RuntimeError(f"OmniCoordinator unexpected status: {status}")

        self._finalizer = weakref.finalize(self, _shutdown_proc, self._proc)

        logger.info(
            "[OmniCoordinatorRuntime] Started (pid=%d router=%s pub=%s heartbeat_timeout=%.1fs)",
            self._proc.pid,
            self.router_address,
            self.pub_address,
            heartbeat_timeout,
        )

    def close(self) -> None:
        """Tear down the coordinator process. Idempotent."""
        if self._closed:
            return
        self._closed = True
        _shutdown_proc(self._proc)
        self._finalizer.detach()
