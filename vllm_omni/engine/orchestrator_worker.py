"""Orchestrator worker process entrypoint."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from vllm.logger import init_logger

logger = init_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run orchestrator and stage workers in a dedicated process.")
    parser.add_argument("--ipc-dir", required=True, help="Directory for ZMQ IPC sockets and init payload.")
    args = parser.parse_args()

    os.environ["VLLM_OMNI_ORCHESTRATOR_WORKER"] = "1"
    os.environ["VLLM_OMNI_ORCHESTRATOR_IPC_DIR"] = args.ipc_dir

    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.engine.orchestrator_zmq_ipc import mark_ready, read_worker_init

    model, kwargs = read_worker_init(args.ipc_dir)
    logger.info("[OrchestratorWorker] starting model=%s ipc_dir=%s", model, args.ipc_dir)

    engine = AsyncOmniEngine(model=model, **kwargs)
    mark_ready(args.ipc_dir)
    logger.info("[OrchestratorWorker] ready")

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("[OrchestratorWorker] signal %s, shutting down", signum)
        try:
            engine.shutdown()
        except Exception:
            logger.exception("[OrchestratorWorker] shutdown failed")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while engine.is_alive():
        time.sleep(1.0)

    logger.error("[OrchestratorWorker] orchestrator thread exited unexpectedly")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
