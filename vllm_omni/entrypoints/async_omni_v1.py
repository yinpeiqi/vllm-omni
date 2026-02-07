"""
AsyncOmni V1 - Refactored async orchestrator using AsyncOmniEngine.

This is the new implementation that uses AsyncOmniEngine (which manages
StageAsyncCoreClient instances) instead of OmniStage with worker processes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Iterable, Sequence
from dataclasses import asdict
from pprint import pformat
from typing import TYPE_CHECKING, Any

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.log_utils import OrchestratorMetrics
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.tokenizers import TokenizerLike
    from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams

logger = init_logger(__name__)


class AsyncOmniV1(EngineClient):
    """Asynchronous unified entry point for multi-stage pipelines using AsyncOmniEngine.

    This is the V1 refactored version that uses AsyncOmniEngine instead of
    OmniStage workers. It provides the same interface as AsyncOmni but with
    a cleaner architecture.

    Args:
        model: Model name or path to load.
        stage_configs: Optional list of stage configurations. If None, loads from model.
        stage_configs_path: Optional path to YAML file containing stage configurations.
        stage_init_timeout: Timeout for stage initialization (seconds).
        log_requests: Whether to log requests.
        enable_stats: Whether to enable statistics logging.
        async_chunk: Whether to use async chunk mode (parallel stage execution).
        output_modalities: List of output modalities.
        **kwargs: Additional keyword arguments.

    Example:
        >>> async_omni = AsyncOmniV1(model="Qwen/Qwen2.5-Omni-7B")
        >>> async for output in async_omni.generate(
        ...     prompt="Hello",
        ...     request_id="req-1",
        ...     sampling_params_list=[SamplingParams(), SamplingParams()]
        ... ):
        ...     print(output)
    """

    def __init__(
        self,
        model: str,
        stage_configs: list[Any] | None = None,
        stage_configs_path: str | None = None,
        stage_init_timeout: int = 300,
        log_requests: bool = True,
        enable_stats: bool = False,
        async_chunk: bool = False,
        output_modalities: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.log_requests = log_requests
        self.enable_stats = enable_stats
        self.async_chunk = async_chunk
        self.output_modalities = output_modalities or []

        # Initialize AsyncOmniEngine
        self.engine = AsyncOmniEngine(
            model=model,
            stage_configs=stage_configs,
            stage_configs_path=stage_configs_path,
            stage_init_timeout=stage_init_timeout,
            log_requests=log_requests,
            **kwargs,
        )

        # Pause/resume control
        self._pause_cond: asyncio.Condition = asyncio.Condition()
        self._paused: bool = False

        # Request state tracking
        self.request_states: dict[str, ClientRequestState] = {}
        self.output_handler: asyncio.Task | None = None

        # Get default sampling params from stages
        self.default_sampling_params_list = [
            self.engine.get_stage_client(i).default_sampling_params
            for i in range(self.engine.num_stages)
        ]

        logger.info(
            f"[AsyncOmniV1] Initialized with {self.engine.num_stages} stages for model {model}"
        )

    @property
    def num_stages(self) -> int:
        """Get the number of stages."""
        return self.engine.num_stages

    # ==================== Generate Method ====================

    async def generate(
        self,
        prompt: OmniPromptType,
        request_id: str,
        sampling_params_list: Sequence[OmniSamplingParams] | None = None,
        *,
        output_modalities: list[str] | None = None,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Generate outputs for the given prompt asynchronously.

        Coordinates multi-stage pipeline execution. Processes the prompt through
        all stages in the pipeline and yields outputs as they become available.

        Args:
            prompt: Prompt to process. Can be a text string, token IDs,
                or multimodal prompt.
            request_id: Unique identifier for this request
            sampling_params_list: List of SamplingParams, one for each stage.
                Must have the same length as the number of stages.
                If None, uses default sampling params for each stage.
            output_modalities: Optional list of output modalities.

        Yields:
            OmniRequestOutput objects as they are produced by each stage.

        Raises:
            ValueError: If sampling_params_list has incorrect length.
        """
        # Wait until generation is resumed if the engine is paused
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        logger.debug(f"[AsyncOmniV1] generate() called for request {request_id}")

        try:
            # Start output handler on the first call to generate()
            self._run_output_handler()

            # Use default sampling params if not provided
            if sampling_params_list is None:
                sampling_params_list = self.default_sampling_params_list

            if len(sampling_params_list) != self.num_stages:
                raise ValueError(
                    f"Expected {self.num_stages} sampling params, got {len(sampling_params_list)}"
                )

            # Track per-request metrics
            wall_start_ts = time.time()
            req_start_ts: dict[int, float] = {}

            # Determine the final stage for E2E stats
            final_stage_id_for_e2e = get_final_stage_id_for_e2e(
                output_modalities,
                self.output_modalities,
                [self.engine.get_stage_client(i) for i in range(self.num_stages)],
            )

            # Create metrics tracker
            metrics = OrchestratorMetrics(
                self.num_stages,
                self.enable_stats,
                wall_start_ts,
            )

            # Create request state
            req_state = ClientRequestState(request_id)
            req_state.metrics = metrics
            self.request_states[request_id] = req_state

            # Add request to stage 0
            await self.engine.add_request(
                stage_id=0,
                request_id=request_id,
                prompt=prompt,
                params=sampling_params_list[0],
            )
            metrics.stage_first_ts[0] = time.time()
            req_start_ts[request_id] = time.time()

            logger.info(
                f"[AsyncOmniV1] Entering scheduling loop: stages={self.num_stages}, "
                f"final_stage={final_stage_id_for_e2e}"
            )

            # Process results based on mode
            if self.async_chunk:
                # Async chunk mode: parallel stage execution
                stage_queues = {stage_id: asyncio.Queue() for stage_id in range(self.num_stages)}
                req_state.stage_queues = stage_queues
                async for output in self._process_async_results(
                    request_id,
                    prompt,
                    sampling_params_list,
                    req_state,
                    metrics,
                    final_stage_id_for_e2e,
                    req_start_ts,
                    wall_start_ts,
                ):
                    yield output
            else:
                # Sequential mode: stages execute one after another
                async for output in self._process_sequential_results(
                    request_id,
                    req_state,
                    metrics,
                    final_stage_id_for_e2e,
                    req_start_ts,
                    wall_start_ts,
                    sampling_params_list,
                    prompt,
                ):
                    yield output

            logger.debug(f"[AsyncOmniV1] Request {request_id} completed")

            # Log summary
            try:
                summary = metrics.build_and_log_summary(final_stage_id_for_e2e)
                logger.info("[Summary] %s", pformat(summary, sort_dicts=False))
            except Exception as e:
                logger.exception(f"[AsyncOmniV1] Failed to build/log summary: {e}")
            finally:
                self.request_states.pop(request_id, None)

        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            logger.info(f"[AsyncOmniV1] Request {request_id} aborted.")
            raise

    # ==================== Processing Methods ====================

    async def _process_sequential_results(
        self,
        request_id: str,
        req_state: ClientRequestState,
        metrics: OrchestratorMetrics,
        final_stage_id_for_e2e: int,
        req_start_ts: dict[int, float],
        wall_start_ts: float,
        sampling_params_list: list[OmniSamplingParams],
        prompt: Any,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process results sequentially: stage 0 → stage 1 → stage 2 → ..."""
        for stage_id in range(final_stage_id_for_e2e + 1):
            finished = False
            while not finished:
                # Wait for result from this stage
                result = await req_state.queue.get()
                assert stage_id == req_state.stage_id

                # Process the result
                engine_outputs, finished, output_to_yield = self._process_single_result(
                    result,
                    stage_id,
                    metrics,
                    req_start_ts,
                    wall_start_ts,
                    final_stage_id_for_e2e,
                )

                if output_to_yield:
                    yield output_to_yield

            # Stage finished, prepare next stage
            if not isinstance(engine_outputs, list):
                engine_outputs = [engine_outputs]

            # Set outputs for cross-stage data flow
            self.engine.set_stage_engine_outputs(stage_id, engine_outputs)

            # Forward to next stage if there is one
            next_stage_id = stage_id + 1
            if next_stage_id <= final_stage_id_for_e2e:
                # Process inputs for next stage from current stage outputs
                next_inputs = self.engine.process_stage_engine_inputs(
                    stage_id=next_stage_id,
                    stage_list=[self.engine.get_stage_client(i) for i in range(self.num_stages)],
                    prompt=prompt,
                )

                # Add request to next stage
                for next_input in next_inputs:
                    await self.engine.add_request(
                        stage_id=next_stage_id,
                        request_id=request_id,
                        prompt=next_input,
                        params=sampling_params_list[next_stage_id],
                    )
                    metrics.stage_first_ts[next_stage_id] = time.time()

                logger.debug(f"[AsyncOmniV1] Forwarded request {request_id} to stage {next_stage_id}")

    async def _process_async_results(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_list: list[OmniSamplingParams],
        req_state: ClientRequestState,
        metrics: OrchestratorMetrics,
        final_stage_id_for_e2e: int,
        req_start_ts: dict[int, float],
        wall_start_ts: float,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Process results asynchronously: stages can run in parallel."""
        all_stages_finished = {stage_id: False for stage_id in range(final_stage_id_for_e2e + 1)}
        submit_flag = True

        while not all(all_stages_finished.values()):
            for stage_id in range(final_stage_id_for_e2e + 1):
                if all_stages_finished[stage_id]:
                    continue

                try:
                    result = req_state.stage_queues[stage_id].get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)
                    continue

                # Process the result
                engine_outputs, finished, output_to_yield = self._process_single_result(
                    result,
                    stage_id,
                    metrics,
                    req_start_ts,
                    wall_start_ts,
                    final_stage_id_for_e2e,
                )

                # Submit to all downstream stages on first output from stage 0
                if submit_flag and stage_id == 0:
                    submit_flag = False
                    # TODO: Implement async chunk logic for parallel stage execution
                    # This requires special handling of prompt token IDs
                    pass

                all_stages_finished[stage_id] = finished

                if output_to_yield:
                    yield output_to_yield

    def _process_single_result(
        self,
        result: dict[str, Any],
        stage_id: int,
        metrics: OrchestratorMetrics,
        req_start_ts: dict[int, float],
        wall_start_ts: float,
        final_stage_id_for_e2e: int,
    ) -> tuple[Any, bool, OmniRequestOutput | None]:
        """Process a single result from a stage.

        Returns:
            engine_outputs: The decoded outputs.
            finished: Whether the stage processing is finished.
            output_to_yield: An OmniRequestOutput to yield, or None.
        """
        req_id = result.get("request_id")

        # Check for errors
        if "error" in result:
            logger.error(f"[AsyncOmniV1] Stage {stage_id} error on request {req_id}: {result['error']}")
            raise RuntimeError(result)

        # Get engine outputs
        engine_outputs = result.get("engine_outputs")
        if isinstance(engine_outputs, list) and len(engine_outputs) > 0:
            engine_outputs = engine_outputs[0]

        finished = getattr(engine_outputs, "finished", False)

        # Mark last output time
        metrics.stage_last_ts[stage_id] = max(metrics.stage_last_ts[stage_id] or 0.0, time.time())

        # Process metrics
        try:
            _m = result.get("metrics")
            if _m is not None and finished:
                if hasattr(_m, "__dict__"):
                    _m = asdict(_m)
                metrics.on_stage_metrics(stage_id, req_id, _m)
        except Exception as e:
            logger.exception(f"[AsyncOmniV1] Failed to process metrics for stage {stage_id}, req {req_id}: {e}")

        logger.debug(f"[AsyncOmniV1] Stage-{stage_id} completed request {req_id}")

        # Determine if this is a final output stage
        stage_client = self.engine.get_stage_client(stage_id)
        output_to_yield = None

        if stage_client.final_output:
            logger.debug(f"[AsyncOmniV1] Request {req_id} finalized at stage-{stage_id}")

            # Finalize request metrics
            try:
                rid_key = str(req_id)
                if stage_id == final_stage_id_for_e2e and rid_key not in metrics.e2e_done and finished:
                    metrics.on_finalize_request(
                        stage_id,
                        req_id,
                        req_start_ts.get(req_id, wall_start_ts),
                    )
            except Exception as e:
                logger.exception(f"[AsyncOmniV1] Finalize request handling error: {e}")

            # Construct output to yield
            images = []
            if stage_client.final_output_type == "image":
                if isinstance(engine_outputs, OmniRequestOutput) and engine_outputs.images:
                    images = engine_outputs.images
                elif hasattr(engine_outputs, "images") and engine_outputs.images:
                    images = engine_outputs.images

            if stage_client.final_output_type == "image":
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage_client.final_output_type,
                    request_output=engine_outputs,
                    images=images,
                )
            else:
                output_to_yield = OmniRequestOutput(
                    stage_id=stage_id,
                    final_output_type=stage_client.final_output_type,
                    request_output=engine_outputs,
                )

        return engine_outputs, finished, output_to_yield

    # ==================== Output Handler ====================

    def _run_output_handler(self) -> None:
        """Start the output handler if not already running."""
        if self.output_handler is not None:
            return

        request_states = self.request_states
        engine = self.engine

        async def output_handler():
            """Background coroutine that collects outputs from all stages."""
            try:
                while True:
                    idle = True

                    # Poll all stages for outputs
                    for stage_id in range(engine.num_stages):
                        stage_client = engine.get_stage_client(stage_id)

                        # Try to get output from this stage (non-blocking)
                        try:
                            outputs = await asyncio.wait_for(
                                stage_client.get_output_async(),
                                timeout=0.001
                            )
                        except asyncio.TimeoutError:
                            continue

                        idle = False

                        # Process each output
                        for output in outputs.outputs:
                            req_id = output.request_id
                            req_state = request_states.get(req_id)

                            if req_state is None:
                                logger.debug(
                                    f"[AsyncOmniV1] Request may have been aborted; "
                                    f"dropping output for req {req_id} at stage-{stage_id}"
                                )
                                continue

                            # Construct result dict
                            result = {
                                "request_id": req_id,
                                "stage_id": stage_id,
                                "engine_outputs": [output],
                                "metrics": None,  # TODO: Add metrics
                            }

                            # Put result into appropriate queue
                            if hasattr(req_state, "stage_queues") and stage_id in req_state.stage_queues:
                                await req_state.stage_queues[stage_id].put(result)
                            else:
                                await req_state.queue.put(result)
                                req_state.stage_id = stage_id

                    if idle:
                        await asyncio.sleep(0.001)  # Avoid CPU overload when idle
                    else:
                        await asyncio.sleep(0)  # Yield to other coroutines

            except Exception as e:
                logger.exception("[AsyncOmniV1] output_handler failed.")
                # Propagate error to all requests
                for req_state in request_states.values():
                    error_msg = {"request_id": req_state.request_id, "error": str(e)}
                    if hasattr(req_state, "stage_queues"):
                        for queue in req_state.stage_queues.values():
                            await queue.put(error_msg)
                    else:
                        await req_state.queue.put(error_msg)
                self.output_handler = None  # Allow restart

        self.output_handler = asyncio.create_task(output_handler())
        logger.debug("[AsyncOmniV1] Output handler started")

    # ==================== Control Methods ====================

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort request(s)."""
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)

        # Abort in all stages
        for stage_id in range(self.engine.num_stages):
            stage_client = self.engine.get_stage_client(stage_id)
            await stage_client.abort_requests_async(request_ids)

        # Remove from request states
        for req_id in request_ids:
            self.request_states.pop(req_id, None)

        if self.log_requests:
            logger.info(f"[AsyncOmniV1] Aborted request(s) {','.join(request_ids)}")

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        """Pause generation."""
        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        # TODO: Implement request draining if wait_for_inflight_requests

        if clear_cache:
            for stage_id in range(self.engine.num_stages):
                await self.engine.reset_prefix_cache(stage_id=stage_id)
                await self.engine.reset_mm_cache(stage_id=stage_id)

    async def resume_generation(self) -> None:
        """Resume generation."""
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()

    async def is_paused(self) -> bool:
        """Check if paused."""
        async with self._pause_cond:
            return self._paused

    # ==================== Properties ====================

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        """Check if the engine is stopped."""
        return self.errored

    @property
    def errored(self) -> bool:
        """Check if any stage has errored."""
        return self.engine.errored

    @property
    def dead_error(self) -> BaseException:
        """Get the dead error."""
        return EngineDeadError()

    # ==================== EngineClient Interface ====================

    async def get_vllm_config(self) -> VllmConfig:
        """Get vllm config from the first stage."""
        return self.engine.get_stage_vllm_config(0)

    async def get_model_config(self) -> Any:
        """Get model config from the first stage."""
        vllm_config = await self.get_vllm_config()
        return vllm_config.model_config if vllm_config else None

    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get input preprocessor."""
        return None  # TODO: Implement if needed

    async def get_tokenizer(self) -> TokenizerLike:
        """Get tokenizer from the first stage."""
        return self.engine.get_stage_tokenizer(0)

    async def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return await self.engine.is_tracing_enabled()

    async def do_log_stats(self) -> None:
        """Log statistics."""
        await self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Check engine health."""
        await self.engine.check_health()

    # ==================== Shutdown ====================

    def shutdown(self) -> None:
        """Shutdown the engine."""
        logger.info("[AsyncOmniV1] Shutting down")
        self.engine.shutdown()
        if self.output_handler is not None:
            self.output_handler.cancel()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass
