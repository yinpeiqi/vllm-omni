"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies,
subsequent attempts to collect output raise RuntimeError.
"""

import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(ipc_client, mocker: MockerFixture, *, worker_alive: bool = True) -> AsyncOmniEngine:
    """Create an APIServer-side AsyncOmniEngine bypassing __init__."""
    engine = object.__new__(AsyncOmniEngine)
    engine._ipc_client = ipc_client
    engine._is_orchestrator_worker = False
    engine._worker_proc = mocker.MagicMock(
        poll=mocker.MagicMock(return_value=None if worker_alive else 0),
    )
    return engine


def test_try_get_output_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Draining remaining results then hitting an empty queue with a dead
    orchestrator must raise RuntimeError so callers know the pipeline is gone."""
    mock_client = mocker.MagicMock()
    mock_client.recv_output.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        None,
    ]

    engine = _make_engine(mock_client, mocker, worker_alive=True)

    assert engine.try_get_output().request_id == "r1"

    engine._worker_proc.poll.return_value = 0

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output()


@pytest.mark.asyncio
async def test_try_get_output_async_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Same scenario as above but for the async variant."""
    mock_client = mocker.MagicMock()
    mock_client.recv_output.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        None,
    ]

    engine = _make_engine(mock_client, mocker, worker_alive=True)

    assert (await engine.try_get_output_async()).request_id == "r1"

    engine._worker_proc.poll.return_value = 0

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        await engine.try_get_output_async()


def test_fatal_error_message_surfaces_through_try_get_output(mocker: MockerFixture):
    """When the orchestrator thread crashes, it enqueues a fatal error message.

    ``try_get_output`` must return this message so the caller
    (``OmniBase._handle_output_message``) can detect the fatal flag.
    """
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    mock_client = mocker.MagicMock()
    mock_client.recv_output.return_value = fatal_msg

    engine = _make_engine(mock_client, mocker, worker_alive=False)

    msg = engine.try_get_output()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True
    assert "crashed" in msg.error


@pytest.mark.asyncio
async def test_fatal_error_message_surfaces_through_try_get_output_async(mocker: MockerFixture):
    """Async variant of the fatal error message test."""
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    mock_client = mocker.MagicMock()
    mock_client.recv_output.return_value = fatal_msg

    engine = _make_engine(mock_client, mocker, worker_alive=False)

    msg = await engine.try_get_output_async()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True
