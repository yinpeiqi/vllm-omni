from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    OutputMessage,
    ShutdownRequestMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _build_terminal_empty_output,
    _infer_stage_audio_sample_rate,
)
from vllm_omni.engine.orchestrator_zmq_ipc import (
    OrchestratorZmqClient,
    OrchestratorZmqServer,
    cleanup_ipc_dir,
    make_ipc_dir,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class OrchestratorFixture:
    orchestrator: Orchestrator
    ipc_client: OrchestratorZmqClient
    ipc_dir: str
    thread: threading.Thread
    result_future: concurrent.futures.Future[None]


class FakeStageClient:
    def __init__(
        self,
        *,
        stage_type: str = "llm",
        final_output: bool = False,
        final_output_type: str = "text",
        next_inputs: list[dict] | None = None,
        engine_input_source: list[int] | None = None,
        is_comprehension: bool = False,
        model_stage: str | None = None,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> None:
        self.stage_id = 0
        self.replica_id = 0
        self.stage_type = stage_type
        self.final_output = final_output
        self.final_output_type = final_output_type
        self.default_sampling_params = SamplingParams(max_tokens=1)
        self.requires_multimodal_data = False
        self.engine_input_source = list(engine_input_source or [0])
        self.is_comprehension = is_comprehension
        self.model_stage = model_stage
        self.next_inputs = list(next_inputs or [])
        self.custom_process_input_func = None
        self._kv_sender_info = dict(kv_sender_info) if kv_sender_info is not None else None
        self.add_request_calls: list[tuple] = []
        self.abort_calls: list[list[str]] = []
        self.collective_rpc_calls: list[tuple[str, float | None, tuple[Any, ...], dict[str, Any]]] = []
        self.shutdown_calls = 0
        # Thread-safe queues: tests push from pytest thread, orch reads in its loop.
        self.outputs_queue: queue.Queue[Any] = queue.Queue()
        self._output_queue: queue.Queue[Any] = queue.Queue()

    # Orchestrator-facing interface.
    async def add_request_async(self, *args, **kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        while True:
            try:
                outputs = self.outputs_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if isinstance(outputs, Exception):
                raise outputs
            return outputs

    async def get_diffusion_output_async(self):
        while True:
            try:
                return self._output_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.001)

    def get_diffusion_output_nowait(self):
        try:
            return self._output_queue.get_nowait()
        except queue.Empty:
            return None

    def set_engine_outputs(self, outputs) -> None:
        return None

    def process_engine_inputs(self, source_outputs, prompt=None, streaming_context=None):
        return list(self.next_inputs)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))

    async def collective_rpc_async(
        self,
        *,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_kwargs = dict(kwargs or {})
        self.collective_rpc_calls.append((method, timeout, args, normalized_kwargs))
        return {
            "supported": False,
            "todo": True,
            "reason": f"{self.__class__.__name__}.collective_rpc_async is not implemented yet",
        }

    def get_kv_sender_info(self) -> dict[str, Any] | None:
        if self._kv_sender_info is None:
            return None
        return dict(self._kv_sender_info)

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    # Test helpers for seeding fake stage outputs.
    def push_engine_core_outputs(self, outputs) -> None:
        self.outputs_queue.put_nowait(outputs)

    def push_diffusion_output(self, output) -> None:
        self._output_queue.put_nowait(output)


def test_terminal_empty_audio_output_uses_stage_sample_rate() -> None:
    final_stage = FakeStageClient(final_output=True, final_output_type="audio")
    final_stage.sample_rate = 44100
    final_pool = SimpleNamespace(stage_client=final_stage, _stage_vllm_config=None)

    terminal_output = _build_terminal_empty_output(
        "req-1",
        final_output_type="audio",
        audio_sample_rate=_infer_stage_audio_sample_rate(final_pool),
    )

    assert terminal_output.outputs[0].multimodal_output["sr"] == 44100


class FakeCollectiveRpcStageClient(FakeStageClient):
    def __init__(self, *args, rpc_result: Any = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rpc_result = rpc_result

    async def collective_rpc_async(
        self,
        *,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_kwargs = dict(kwargs or {})
        self.collective_rpc_calls.append((method, timeout, args, normalized_kwargs))
        return self.rpc_result


class FakeOutputProcessor:
    def __init__(self, *, request_outputs: list[object] | None = None) -> None:
        self.request_outputs = list(request_outputs or [])
        self.add_request_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.abort_calls: list[list[str]] = []

    def add_request(self, *args, **kwargs) -> None:
        self.add_request_calls.append((args, kwargs))
        return None

    def process_outputs(self, *_args, **_kwargs):
        return SimpleNamespace(
            request_outputs=list(self.request_outputs),
            reqs_to_abort=[],
        )

    def abort_requests(self, request_ids, internal: bool = False):
        self.abort_calls.append(request_ids)
        return request_ids

    def update_scheduler_stats(self, _scheduler_stats) -> None:
        return None


class SequenceOutputProcessor(FakeOutputProcessor):
    """Return one canned request output per ``process_outputs`` call."""

    def __init__(self, *, request_outputs: list[object]) -> None:
        super().__init__(request_outputs=request_outputs)
        self._next_index = 0

    def process_outputs(self, *_args, **_kwargs):
        index = min(self._next_index, len(self.request_outputs) - 1)
        self._next_index += 1
        return SimpleNamespace(
            request_outputs=[self.request_outputs[index]],
            reqs_to_abort=[],
        )


def _sampling_params(max_tokens: int = 4) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens)


def _engine_core_outputs(tag: str, timestamp: float) -> SimpleNamespace:
    return SimpleNamespace(outputs=[tag], timestamp=timestamp, scheduler_stats=None)


def _build_request_output(
    request_id: str,
    *,
    token_ids: list[int] | None = None,
    prompt_token_ids: list[int] | None = None,
    finished: bool = True,
    text: str = "test",
) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=list(token_ids or [1, 2]),
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop" if finished else None,
        stop_reason=None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=list(prompt_token_ids or [10, 11]),
        prompt_logprobs=None,
        outputs=[completion],
        finished=finished,
        metrics=None,
        lora_request=None,
    )


def _build_stage_pools(
    stage_clients: list[list[FakeStageClient]],
    *,
    output_processors: list[FakeOutputProcessor] | None = None,
    stage_vllm_configs: list[object] | None = None,
) -> list[StagePool]:
    """Build StagePool list from per-stage replica lists.

    ``stage_clients[i]`` is the list of FakeStageClient replicas for stage i.
    """
    num_stages = len(stage_clients)
    if output_processors is None:
        output_processors = [FakeOutputProcessor() for _ in stage_clients]
    if stage_vllm_configs is None:
        stage_vllm_configs = [SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)) for _ in stage_clients]

    pools: list[StagePool] = []
    for stage_id in range(num_stages):
        clients = stage_clients[stage_id]
        if clients[0].stage_type == "diffusion":
            pools.append(StagePool(stage_id, clients[0]))
        else:
            pools.append(
                StagePool(
                    stage_id,
                    clients,
                    output_processor=output_processors[stage_id],
                    stage_vllm_config=stage_vllm_configs[stage_id],
                )
            )
    return pools


def _build_harness(
    stage_clients: list[object],
    *,
    output_processors: list[object] | None = None,
    stage_vllm_configs: list[object] | None = None,
    async_chunk: bool = False,
    stage_pools: list[StagePool] | None = None,
) -> OrchestratorFixture:
    """Build an Orchestrator test harness.

    Accepts either pre-built ``stage_pools`` or flat lists of single-replica
    clients/processors.
    """
    if stage_pools is None:
        # Wrap flat lists into per-stage single-replica lists.
        nested_clients = [[c] for c in stage_clients]
        stage_pools = _build_stage_pools(
            nested_clients,
            output_processors=output_processors,
            stage_vllm_configs=stage_vllm_configs,
        )

    ready_future: concurrent.futures.Future[tuple[Orchestrator, str, OrchestratorZmqClient]] = (
        concurrent.futures.Future()
    )
    result_future: concurrent.futures.Future[None] = concurrent.futures.Future()

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            ipc_dir = make_ipc_dir(prefix="test_orch_ipc_")
            zmq_ipc = OrchestratorZmqServer(ipc_dir)
            ipc_client = OrchestratorZmqClient(ipc_dir)
            orchestrator = Orchestrator(
                zmq_ipc,
                stage_pools=stage_pools,
                async_chunk=async_chunk,
            )
            ready_future.set_result((orchestrator, ipc_dir, ipc_client))
            try:
                await orchestrator.run()
            finally:
                zmq_ipc.close()
                cleanup_ipc_dir(ipc_dir)

        try:
            loop.run_until_complete(_run())
            result_future.set_result(None)
        except Exception as exc:
            result_future.set_exception(exc)
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    thread = threading.Thread(target=_runner, daemon=True, name="test-orchestrator")
    thread.start()

    orchestrator, ipc_dir, ipc_client = ready_future.result(timeout=5)
    return OrchestratorFixture(
        orchestrator=orchestrator,
        ipc_client=ipc_client,
        ipc_dir=ipc_dir,
        thread=thread,
        result_future=result_future,
    )


async def _shutdown_orchestrator(orchestrator_fixture: OrchestratorFixture) -> None:
    orchestrator_fixture.ipc_client.send(ShutdownRequestMessage())
    await asyncio.to_thread(orchestrator_fixture.thread.join, 5)
    if orchestrator_fixture.thread.is_alive():
        raise AssertionError("Timed out waiting for orchestrator thread shutdown")
    orchestrator_fixture.result_future.result(timeout=0)


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for predicate")
        await asyncio.sleep(0.01)


async def _get_output_message(orchestrator_fixture: OrchestratorFixture, *, timeout: float = 2.0) -> OutputMessage:
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            msg = orchestrator_fixture.ipc_client.recv_output(timeout=0.001)
        except Exception:
            msg = None
        if msg is None:
            await asyncio.sleep(0.01)
            continue
        if isinstance(msg, OutputMessage):
            return msg


async def _get_rpc_message(
    orchestrator_fixture: OrchestratorFixture,
    *,
    timeout: float = 2.0,
) -> CollectiveRPCResultMessage:
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator rpc output")
        try:
            return orchestrator_fixture.ipc_client.recv_rpc(timeout=0.05)
        except queue.Empty:
            await asyncio.sleep(0.01)


async def _enqueue_add_request(
    orchestrator_fixture: OrchestratorFixture,
    *,
    request_id: str,
    prompt,
    original_prompt,
    sampling_params_list,
    final_stage_id: int,
    final_output_stage_ids: list[int] | None = None,
) -> None:
    orchestrator_fixture.ipc_client.send(
        StageSubmissionMessage(
            type="add_request",
            request_id=request_id,
            prompt=prompt,
            original_prompt=original_prompt,
            output_prompt_text=None,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            final_output_stage_ids=final_output_stage_ids,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.perf_counter(),
        )
    )


async def _enqueue_abort_request(orchestrator_fixture: OrchestratorFixture, request_ids: list[str]) -> None:
    orchestrator_fixture.ipc_client.send(AbortRequestMessage(request_ids=request_ids))


@pytest.fixture
def orchestrator_factory():
    fixtures: list[OrchestratorFixture] = []

    def _factory(*args, **kwargs) -> OrchestratorFixture:
        fixture = _build_harness(*args, **kwargs)
        fixtures.append(fixture)
        return fixture

    yield _factory

    for fixture in fixtures:
        if fixture.thread.is_alive():
            try:
                fixture.ipc_client.send(ShutdownRequestMessage())
            except Exception:
                pass
            fixture.thread.join(timeout=5)
        try:
            fixture.ipc_client.close()
        except Exception:
            pass
        cleanup_ipc_dir(fixture.ipc_dir)


# ---------------------------------------------------------------------------
# Existing single-replica tests (adapted to StagePool interface)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_two_stage_llm(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8, 9]}],
    )
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-llm", token_ids=[3, 4], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-llm", token_ids=[10, 11], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-llm", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-llm",
            prompt=request,
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_engine_core_outputs(_engine_core_outputs("stage0-raw", 1.0))

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        stage1_request = stage1.add_request_calls[0][0]
        assert stage1_request.request_id == "req-llm"
        assert stage1_request.prompt_token_ids == [7, 8, 9]

        stage1.push_engine_core_outputs(_engine_core_outputs("stage1-raw", 2.0))

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-llm"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-llm"
        assert "req-llm" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_single_stage_diffusion(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-diff",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-diff",
                images=[],
                final_output_type="image",
            )
        )

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-diff"
        assert output_msg.stage_id == 0
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-diff"
        assert "req-diff" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_single_stage_diffusion_streaming_forwards_intermediate_chunks(orchestrator_factory) -> None:
    """Intermediate diffusion chunks (finished=False) reach the frontend before the final chunk."""
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-stream",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-stream",
                images=[],
                final_output_type="image",
                custom_output={"chunk": 0},
                finished=False,
            )
        )
        stage0.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-stream",
                images=[],
                final_output_type="image",
                custom_output={"chunk": 1},
                finished=True,
            )
        )

        output_msgs: list[OutputMessage] = []
        deadline = time.monotonic() + 2.0
        while not output_msgs or not output_msgs[-1].finished:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"Timed out waiting for finished orchestrator output, got {len(output_msgs)} message(s)"
                )
            try:
                msg = orchestrator_fixture.ipc_client.recv_output(timeout=0.001)
            except Exception:
                msg = None
            if msg is None:
                await asyncio.sleep(0.01)
                continue
            if isinstance(msg, OutputMessage):
                output_msgs.append(msg)

        assert [msg.request_id for msg in output_msgs] == ["req-stream", "req-stream"]
        assert [msg.finished for msg in output_msgs] == [False, True]
        assert [msg.engine_outputs.finished for msg in output_msgs] == [False, True]
        assert [msg.engine_outputs.custom_output["chunk"] for msg in output_msgs] == [0, 1]
        await _wait_for(lambda: "req-stream" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_llm_to_diffusion(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-img", token_ids=[3, 4], finished=True)]),
        FakeOutputProcessor(),
    ]
    orchestrator_fixture = orchestrator_factory([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-img", prompt_token_ids=[1, 2, 3])
    params = OmniDiffusionSamplingParams()
    original_prompt = {"prompt": "draw a fox"}

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-img",
            prompt=request,
            original_prompt=original_prompt,
            sampling_params_list=[_sampling_params(), params],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_engine_core_outputs(_engine_core_outputs("stage0-raw", 1.0))

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        assert stage1.add_request_calls[0] == ("req-img", original_prompt, params)

        stage1.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-img",
                images=[],
                final_output_type="image",
            )
        )

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-img"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-img"
        assert "req-img" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_async_chunk(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-async", token_ids=[1], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-async", token_ids=[20, 21], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory(
        [stage0, stage1],
        output_processors=processors,
        async_chunk=True,
    )
    request = SimpleNamespace(request_id="req-async", prompt_token_ids=[1, 2, 3, 4])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-async",
            prompt=request,
            original_prompt={"prompt": "hello async"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        prewarmed_request = stage1.add_request_calls[0][0]
        assert prewarmed_request.request_id == "req-async"
        assert prewarmed_request.prompt_token_ids
        assert all(token_id == 0 for token_id in prewarmed_request.prompt_token_ids)

        stage1.push_engine_core_outputs(_engine_core_outputs("stage1-final", 3.0))

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-async"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert "req-async" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_audio_only_request_skips_non_requested_final_output_stage(orchestrator_factory) -> None:
    """Audio-only clients should not receive thinker text over the output channel."""
    stage0 = FakeStageClient(stage_type="llm", final_output=True, final_output_type="text")
    stage1 = FakeStageClient(stage_type="llm", final_output=True, final_output_type="audio")
    processors = [
        SequenceOutputProcessor(
            request_outputs=[_build_request_output("req-audio-only", token_ids=[1], finished=False)]
        ),
        FakeOutputProcessor(request_outputs=[]),
    ]
    orchestrator_fixture = orchestrator_factory([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-audio-only", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-audio-only",
            prompt=request,
            original_prompt={"prompt": "hello audio"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
            final_output_stage_ids=[1],
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_engine_core_outputs(_engine_core_outputs("stage0-partial", 1.0))

        with pytest.raises(AssertionError, match="Timed out waiting for orchestrator output"):
            await _get_output_message(orchestrator_fixture, timeout=0.2)
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_shutdown(orchestrator_factory) -> None:
    stages = [
        FakeStageClient(stage_type="llm", final_output=False),
        FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image"),
    ]
    orchestrator_fixture = orchestrator_factory(stages)

    await _shutdown_orchestrator(orchestrator_fixture)

    assert not orchestrator_fixture.thread.is_alive()
    for stage in stages:
        assert stage.shutdown_calls == 1


@pytest.mark.asyncio
async def test_run_abort(orchestrator_factory) -> None:
    stages = [
        FakeStageClient(stage_type="llm", final_output=False),
        FakeStageClient(stage_type="llm", final_output=True),
    ]
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-abort", token_ids=[1], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-abort", token_ids=[2], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory(stages, output_processors=processors)
    request = SimpleNamespace(request_id="req-abort", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-abort",
            prompt=request,
            original_prompt={"prompt": "cancel me"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stages[0].add_request_calls) == 1)

        await _enqueue_abort_request(orchestrator_fixture, ["req-abort"])
        await _wait_for(lambda: bool(stages[0].abort_calls))

        assert stages[0].abort_calls == [["req-abort"]]
        assert stages[1].abort_calls == []
        assert "req-abort" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


# ---------------------------------------------------------------------------
# Multi-replica tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_replica_round_robin_distribution(orchestrator_factory) -> None:
    """Two replicas at stage-0, single replica at stage-1.

    Send two requests — they should land on different stage-0 replicas
    (round-robin), then both forward to the single stage-1 replica.
    """
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8]}],
    )

    proc0 = FakeOutputProcessor(request_outputs=[_build_request_output("req-0", token_ids=[3], finished=True)])
    proc1 = FakeOutputProcessor(request_outputs=[_build_request_output("req-0", token_ids=[10], finished=True)])

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        output_processors=[proc0, proc1],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )

    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        # Request 0 → should land on replica 0 (RR starts at 0)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "hello 0"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)
        assert len(stage0_r1.add_request_calls) == 0

        # Request 1 → should land on replica 1 (RR advances)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-1",
            prompt=SimpleNamespace(request_id="req-1", prompt_token_ids=[5, 6]),
            original_prompt={"prompt": "hello 1"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 1)
        assert len(stage0_r0.add_request_calls) == 1  # unchanged

        # Complete req-0 at stage-0 replica-0 → should forward to stage-1
        stage0_r0.push_engine_core_outputs(_engine_core_outputs("s0r0-raw", 1.0))
        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        assert stage1.add_request_calls[0][0].request_id == "req-0"

        # Complete req-0 at stage-1 → final output
        proc1.request_outputs = [_build_request_output("req-0", token_ids=[10], finished=True)]
        stage1.push_engine_core_outputs(_engine_core_outputs("s1-raw", 2.0))
        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-0"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert "req-0" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_simultaneous_multi_replica_stage_outputs_all_forwarded(orchestrator_factory) -> None:
    """When multiple replica queues wake in one wait window, none may be dropped."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8]}],
    )

    proc0 = SequenceOutputProcessor(
        request_outputs=[
            _build_request_output("req-0", token_ids=[3], finished=True),
            _build_request_output("req-1", token_ids=[4], finished=True),
        ]
    )
    proc1 = FakeOutputProcessor(request_outputs=[_build_request_output("req-0", token_ids=[10], finished=True)])

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        output_processors=[proc0, proc1],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "hello 0"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)

        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-1",
            prompt=SimpleNamespace(request_id="req-1", prompt_token_ids=[5, 6]),
            original_prompt={"prompt": "hello 1"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 1)

        stage0_r0.push_engine_core_outputs(_engine_core_outputs("s0r0-raw", 1.0))
        stage0_r1.push_engine_core_outputs(_engine_core_outputs("s0r1-raw", 1.1))

        await _wait_for(lambda: len(stage1.add_request_calls) == 2, timeout=5.0)
        forwarded_ids = {call[0].request_id for call in stage1.add_request_calls}
        assert forwarded_ids == {"req-0", "req-1"}
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_abort_broadcasts_to_all_replicas(orchestrator_factory) -> None:
    """Abort must be sent to every replica across all stages."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)

    proc0 = FakeOutputProcessor()
    proc1 = FakeOutputProcessor()

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        output_processors=[proc0, proc1],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-abort-mr",
            prompt=SimpleNamespace(request_id="req-abort-mr", prompt_token_ids=[1]),
            original_prompt={"prompt": "cancel"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)

        await _enqueue_abort_request(orchestrator_fixture, ["req-abort-mr"])
        await _wait_for(lambda: bool(stage0_r0.abort_calls))

        assert stage0_r0.abort_calls == [["req-abort-mr"]]
        assert stage0_r1.abort_calls == []
        assert stage1.abort_calls == []
        assert "req-abort-mr" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_shutdown_all_replicas(orchestrator_factory) -> None:
    """Shutdown must shut down every replica across all stages."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    await _shutdown_orchestrator(orchestrator_fixture)

    assert not orchestrator_fixture.thread.is_alive()
    for client in [stage0_r0, stage0_r1, stage1]:
        assert client.shutdown_calls == 1


@pytest.mark.asyncio
async def test_stage_pool_submit_update_reuses_existing_binding() -> None:
    """A request admitted to one replica must keep using that replica on updates."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0_r0, stage0_r1],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    req0_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )
    req1_state = OrchestratorRequestState(
        request_id="req-1",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    await pool.submit_initial("req-0", req0_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]))
    await pool.submit_update("req-0", req0_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[3]))
    await pool.submit_initial("req-1", req1_state, SimpleNamespace(request_id="req-1", prompt_token_ids=[4, 5]))
    await pool.submit_update("req-1", req1_state, SimpleNamespace(request_id="req-1", prompt_token_ids=[6]))

    assert pool.get_bound_replica_id("req-0") == 0
    assert pool.get_bound_replica_id("req-1") == 1
    assert len(stage0_r0.add_request_calls) == 2
    assert len(stage0_r1.add_request_calls) == 2
    assert stage0_r0.add_request_calls[0][0].request_id == "req-0"
    assert stage0_r0.add_request_calls[1][0].request_id == "req-0"
    assert stage0_r1.add_request_calls[0][0].request_id == "req-1"
    assert stage0_r1.add_request_calls[1][0].request_id == "req-1"


@pytest.mark.asyncio
async def test_stage_pool_submit_update_refreshes_output_processor_state() -> None:
    output_processor = FakeOutputProcessor()

    class AssertingStageClient(FakeStageClient):
        async def add_request_async(self, *args, **kwargs) -> None:
            if len(self.add_request_calls) == 1:
                prompts = [call_kwargs["prompt"] for _, call_kwargs in output_processor.add_request_calls]
                assert prompts == ["seg-1", "seg-2"]
            await super().add_request_async(*args, **kwargs)

    stage0 = AssertingStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=output_processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    req_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    await pool.submit_initial(
        "req-0",
        req_state,
        SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
        prompt_text="seg-1",
    )
    await pool.submit_update(
        "req-0",
        req_state,
        SimpleNamespace(request_id="req-0", prompt_token_ids=[3], resumable=True),
        prompt_text="seg-2",
    )

    assert len(output_processor.add_request_calls) == 2
    assert output_processor.add_request_calls[1][1]["prompt"] == "seg-2"


@pytest.mark.asyncio
async def test_handle_streaming_update_passes_prompt_text_to_stage_pool() -> None:
    class RecordingPool:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        async def submit_update(self, request_id, req_state, request, *, prompt_text=None) -> int:
            self.calls.append((request_id, prompt_text))
            return 0

    pool = RecordingPool()
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = False
    orchestrator.request_states = {
        "req-stream": OrchestratorRequestState(
            request_id="req-stream",
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
    }
    orchestrator.stage_pools = [pool]

    await orchestrator._handle_streaming_update(
        StageSubmissionMessage(
            type="streaming_update",
            request_id="req-stream",
            prompt=SimpleNamespace(request_id="req-stream", prompt_token_ids=[1], resumable=True),
            original_prompt={"prompt": "segment-2"},
            output_prompt_text="segment-2",
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.perf_counter(),
        )
    )

    assert pool.calls == [("req-stream", "segment-2")]
    assert orchestrator.request_states["req-stream"].streaming.enabled is True


@pytest.mark.asyncio
async def test_stage_pool_submit_initial_rolls_back_output_processor_when_client_submit_fails() -> None:
    class FailingStageClient(FakeStageClient):
        async def add_request_async(self, *args, **kwargs) -> None:
            raise RuntimeError("submit failed")

    class TrackingOutputProcessor(FakeOutputProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.added_request_ids: list[str] = []
            self.removed_request_ids: list[str] = []

        def add_request(self, request, *_args, **_kwargs) -> None:
            self.added_request_ids.append(request.request_id)

        def remove_request(self, request_id: str) -> None:
            self.removed_request_ids.append(request_id)

    client = FailingStageClient(stage_type="llm", final_output=False)
    output_processor = TrackingOutputProcessor()
    pool = StagePool(
        0,
        [client],
        output_processor=output_processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    req_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        await pool.submit_initial("req-0", req_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]))

    assert output_processor.added_request_ids == ["req-0"]
    assert output_processor.removed_request_ids == ["req-0"]
    assert pool.get_bound_replica_id("req-0") is None


@pytest.mark.asyncio
async def test_stage_pool_abort_requests_logs_when_binding_is_missing(caplog) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    target_logger = logging.getLogger("vllm_omni.engine.stage_pool")
    target_logger.addHandler(caplog.handler)
    prev_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        await pool.abort_requests(["missing-req"])
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prev_level)

    assert not stage0.abort_calls
    assert "abort: no live binding for req=missing-req in stage-0" in caplog.text


@pytest.mark.asyncio
async def test_collective_rpc_ignores_invalid_stage_ids(orchestrator_factory, caplog) -> None:
    stage0 = FakeCollectiveRpcStageClient(stage_type="llm", final_output=True, rpc_result={"stage": 0})
    stage1 = FakeCollectiveRpcStageClient(stage_type="llm", final_output=True, rpc_result={"stage": 1})
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        target_logger = logging.getLogger("vllm_omni.engine.orchestrator")
        target_logger.addHandler(caplog.handler)
        prev_level = target_logger.level
        target_logger.setLevel(logging.WARNING)
        try:
            orchestrator_fixture.ipc_client.send(
                CollectiveRPCRequestMessage(
                    rpc_id="rpc-1",
                    method="list_loras",
                    timeout=None,
                    args=(),
                    kwargs={},
                    stage_ids=[99, 1],
                )
            )

            msg = await _get_rpc_message(orchestrator_fixture)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(prev_level)

        assert msg.type == "collective_rpc_result"
        assert msg.rpc_id == "rpc-1"
        assert msg.stage_ids == [1]
        assert msg.results == [{"stage": 1}]
        assert not stage0.collective_rpc_calls
        assert len(stage1.collective_rpc_calls) == 1
        assert "collective_rpc: ignoring invalid stage_id 99" in caplog.text
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_cfg_companion_inherits_parent_affinity(orchestrator_factory) -> None:
    """CFG companions should be routed to the same stage-0 replica as their parent."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        # Consume replica-0 first so the parent request binds to replica-1.
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="warmup",
            prompt=SimpleNamespace(request_id="warmup", prompt_token_ids=[0]),
            original_prompt={"prompt": "warmup"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)

        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="parent",
            prompt=SimpleNamespace(request_id="parent", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "parent"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 1)

        orchestrator_fixture.ipc_client.send(
            AddCompanionRequestMessage(
                companion_id="parent-neg",
                parent_id="parent",
                role="negative",
                prompt=SimpleNamespace(request_id="parent-neg", prompt_token_ids=[9]),
                companion_prompt_text={"prompt": "negative"},
                sampling_params_list=[_sampling_params()],
            )
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 2)

        assert stage_pools[0].get_bound_replica_id("parent") == 1
        assert stage_pools[0].get_bound_replica_id("parent-neg") == 1
        assert len(stage0_r0.add_request_calls) == 1
        assert stage0_r1.add_request_calls[0][0].request_id == "parent"
        assert stage0_r1.add_request_calls[1][0].request_id == "parent-neg"
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_drain_replica_once_drains_multiple_batches_until_empty() -> None:
    poll_calls = 0

    class BatchPool:
        stage_type = "llm"

        def try_poll_llm_raw_output_nowait(self, replica_id: int):
            nonlocal poll_calls
            poll_calls += 1
            if poll_calls <= 3:
                return SimpleNamespace(outputs=[SimpleNamespace(request_id="req-0")], scheduler_stats=None)
            return None

        async def process_llm_raw_outputs(self, *args, **kwargs):
            return []

    pool = BatchPool()
    orchestrator = object.__new__(Orchestrator)
    orchestrator._shutdown_event = asyncio.Event()
    orchestrator._stat_logger = None
    orchestrator._stage_replica_to_engine_idx = {}
    orchestrator.request_states = {}
    orchestrator.stage_pools = [pool]

    async def _noop_kv_ready(*args, **kwargs) -> None:
        return None

    async def _noop_handle_processed(*args, **kwargs) -> None:
        return None

    orchestrator._handle_kv_ready_raw_outputs = _noop_kv_ready
    orchestrator._handle_processed_outputs = _noop_handle_processed

    handled = await orchestrator._drain_replica_once(0, pool, 0)

    assert handled is True
    assert poll_calls == 4


def test_iter_drain_replicas_prefers_deeper_queues() -> None:
    class QueueDepthPool:
        def __init__(self, *, final_output: bool, depths: dict[int, int]) -> None:
            self.final_output = final_output
            self.stage_type = "llm"
            self._depths = depths

        def live_replica_ids(self):
            return sorted(self._depths)

        def replica_outputs_queue_size(self, replica_id: int) -> int:
            return self._depths[replica_id]

    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [
        QueueDepthPool(final_output=False, depths={0: 10, 1: 50}),
        QueueDepthPool(final_output=True, depths={0: 5, 1: 20}),
    ]

    order = [(stage_id, replica_id) for stage_id, _pool, replica_id in orchestrator._iter_drain_replicas()]

    assert order == [(1, 1), (1, 0), (0, 1), (0, 0)]


def test_orchestrator_does_not_re_introduce_global_stats_throttle() -> None:
    """Regression: each (stage, replica) must independently publish its wrapped
    vllm:* stats when its scheduler emits non-None scheduler_stats.

    A previous version of Orchestrator carried a global self._last_stats_ts /
    _stats_interval_s gate around _stat_logger.record(). Because
    OmniSchedulerMixin.make_stats() already throttles at 1 Hz per scheduler
    (one per (stage, replica)), the extra global gate starved every replica
    other than the first to emit within each second — their {stage, replica}
    gauges/counters went stale.

    The fix removed the global gate entirely; the only signal needed is
    'this replica's scheduler emitted non-None scheduler_stats'. This test
    fails loudly if someone reintroduces the global throttle.
    """
    import inspect

    from vllm_omni.engine.orchestrator import Orchestrator

    source = inspect.getsource(Orchestrator)
    assert "_last_stats_ts" not in source, (
        "Orchestrator must not gate stat recording on a global timestamp. "
        "OmniSchedulerMixin.make_stats() already throttles per scheduler "
        "(per (stage, replica)); an outer global gate starves all but the "
        "first replica to emit within each 1s window."
    )
    assert "_stats_interval_s" not in source
    assert "raw_outputs.scheduler_stats is not None" in source, (
        "Orchestrator must gate stat recording solely on "
        "raw_outputs.scheduler_stats being non-None — the per-scheduler 1Hz "
        "throttle in OmniSchedulerMixin.make_stats() is the only gate needed."
    )
