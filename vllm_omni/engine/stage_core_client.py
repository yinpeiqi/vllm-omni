"""
Stage Engine Core Client for vLLM-Omni V1 architecture.

Directly inherits from vLLM's SyncMPClient to reuse EngineCore architecture.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import fcntl

import time
from typing import TYPE_CHECKING, Any, Literal

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import SyncMPClient
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.executor import Executor
from vllm_omni.entrypoints.omni_stage import _resolve_worker_cls
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor

from vllm_omni.entrypoints.stage_utils import (
    SHUTDOWN_TASK,
    OmniStageTaskType,
    _to_dict,
    is_profiler_task,
    maybe_dump_to_shm,
    set_stage_devices,
)
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from vllm_omni.distributed.omni_connectors import (
    get_stage_connector_config,
    initialize_orchestrator_connectors,
)

from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.stage_utils import _to_dict
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

if TYPE_CHECKING:
    from vllm.inputs import TextPrompt
    from vllm.v1.engine import EngineCoreOutput

    from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


class StageMPClient(SyncMPClient):
    """Stage client that inherits from vLLM's SyncMPClient.

    Fully reuses SyncMPClient.__init__ for:
    - ZMQ setup, sockets
    - launch_core_engines() -> EngineCoreProc
    - outputs_queue, output_queue_thread
    - All utility methods (shutdown, get_output, abort_requests, etc.)

    TODO (from OmniStage):
    - [ ] Connectors: build_stage_connectors(), try_recv_via_connector() for cross-stage data transfer
    - [ ] Device setup: set_stage_devices(), device locking for parallel init
    - [ ] Profiler control: start_profile(), stop_profile() via PROFILER_START/STOP tasks
    - [ ] Batching: max_batch_size, batch_timeout for batch collection
    - [ ] Shared Memory IPC: shm_threshold_bytes, maybe_dump_to_shm() for large data
    - [ ] Metrics: make_request_stats(), make_stage_stats(), token counting
    - [ ] Ray Actor backend: worker_backend="ray", start_ray_actor()
    - [ ] Async engine: AsyncOmniLLM, AsyncOmniDiffusion support
    - [ ] Global request_id injection: _inject_global_id() for cross-stage ID consistency
    - [ ] Diffusion stage: OmniDiffusion engine support (currently raises NotImplementedError)
    """

    def __init__(
        self,
        stage_config: Any,
        model: str,
        stage_init_timeout: int = 300,
    ):
        # Parse stage config first
        self._stage_config = stage_config
        self._stage_id: int = stage_config.stage_id
        self._stage_type: Literal["llm", "diffusion"] = getattr(
            stage_config, "stage_type", "llm"
        )

        if self._stage_type == "diffusion":
            # TODO: Support diffusion via OmniDiffusion engine
            raise NotImplementedError(
                "Diffusion not supported with EngineCore. Use V0 architecture."
            )

        logger.info(f"[StageMPClient] Stage-{self._stage_id} stage_config: {stage_config}")
        # Engine args
        self._engine_args_raw = stage_config.engine_args

        # TODO: Extract model_stage, engine_output_type from engine_args
        # self._model_stage = getattr(stage_config.engine_args, "model_stage", None)
        # self._engine_output_type = getattr(stage_config.engine_args, "engine_output_type", None)

        # Runtime config
        self._runtime_cfg = getattr(stage_config, "runtime", {})
        self._requires_multimodal_data = getattr(
            self._runtime_cfg, "requires_multimodal_data", False
        )

        # TODO: Batching config from runtime
        # self._max_batch_size = int(getattr(self._runtime_cfg, "max_batch_size", 1) or 1)
        # self._batch_timeout = batch_timeout

        # Input/output config
        self._engine_input_source: list[int] = getattr(
            stage_config, "engine_input_source", []
        )
        self._final_output: bool = getattr(stage_config, "final_output", False)
        self._final_output_type: str | None = getattr(
            stage_config, "final_output_type", None
        )

        # TODO: is_comprehension flag
        # self._is_comprehension = getattr(stage_config, "is_comprehension", False)

        # Default sampling params
        default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
        SPClass = SamplingParams if self._stage_type == "llm" else OmniDiffusionSamplingParams
        self._default_sampling_params: OmniSamplingParams = SPClass(**default_sp)

        # Custom input processing
        if hasattr(stage_config, "custom_process_input_func"):
            mod_path, fn_name = stage_config.custom_process_input_func.rsplit(".", 1)
            self._custom_process_input_func = getattr(
                importlib.import_module(mod_path), fn_name
            )
        else:
            self._custom_process_input_func = None

        # Engine outputs (set by orchestrator)
        self._engine_outputs: Any = None

        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
        # IMPORTANT: Ensure vLLM's internal multiprocessing workers (e.g., GPUARWorker /
        # GPUARModelRunner) are spawned with a fork-safe method.
        # Mooncake / gRPC / RDMA and CUDA/NCCL can deadlock under fork-with-threads.
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            logger.info("[Stage] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
        # Best-effort: also force python mp start method in this stage process.
        # This may raise if already set; that's fine.
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # TODO: Connectors for cross-stage data transfer
        # self._connectors: dict[tuple[str, str], OmniConnectorBase] | None = None
        # if connectors_config:
        #     self._connectors = build_stage_connectors(stage_id, connectors_config)

        # TODO: Shared memory IPC
        # self._shm_threshold_bytes = shm_threshold_bytes

        # TODO: Metrics aggregation
        # self._agg_total_tokens = 0
        # self._agg_total_gen_time_ms = 0.0
        # self._batch_seq = 0

        # TODO: Device setup with locking
        # set_stage_devices(self._stage_id, self._runtime_cfg.get("devices"), device_type)

        # Build VllmConfig
        engine_args_dict = _to_dict(self._engine_args_raw)
        engine_args_dict["model"] = model

        if self._stage_type != "diffusion":
            _resolve_worker_cls(engine_args_dict)

        logger.info(f"[StageMPClient] Stage-{self._stage_id} engine_args_dict: {engine_args_dict}")

        # Device mapping
        device_type = None
        try:
            from vllm_omni.platforms import current_omni_platform

            device_type = current_omni_platform.device_type
            set_stage_devices(self._stage_id, self._runtime_cfg.get("devices"), device_type=device_type)
            logger.info(f"[StageMPClient] Stage-{self._stage_id} set devices for {device_type}, runtime devices: {self._runtime_cfg.get('devices')}")
        except Exception as e:
            logger.warning("Device setup failed: %s", e)

        # Sequential initialization on the same device to avoid memory calculation errors
        # when multiple instances start simultaneously
        # For TP/PP/DP/SP, we need to lock ALL devices that will be used by this stage
        lock_files = []
        try:
            # Get all parallel sizes from engine_args or parallel_config (defaults to 1)
            if "parallel_config" in engine_args_dict:
                parallel_config = engine_args_dict["parallel_config"]
                tensor_parallel_size = parallel_config.get("tensor_parallel_size", 1)
                pipeline_parallel_size = parallel_config.get("pipeline_parallel_size", 1)
                data_parallel_size = parallel_config.get("data_parallel_size", 1)
                prefill_context_parallel_size = parallel_config.get("prefill_context_parallel_size", 1)
                sequence_parallel_size = parallel_config.get("sequence_parallel_size", 1)
                cfg_parallel_size = parallel_config.get("cfg_parallel_size", 1)
            else:
                tensor_parallel_size = engine_args_dict.get("tensor_parallel_size", 1)
                pipeline_parallel_size = engine_args_dict.get("pipeline_parallel_size", 1)
                data_parallel_size = engine_args_dict.get("data_parallel_size", 1)
                prefill_context_parallel_size = engine_args_dict.get("prefill_context_parallel_size", 1)
                sequence_parallel_size = 1  # not use in omni model
                cfg_parallel_size = 1  # not used in omni model

            # Calculate total number of devices needed for this stage
            # For a single stage worker:
            # - TP: splits model across devices (always needed)
            # - PP: splits layers across pipeline stages, but each stage uses TP devices
            # - DP: replicates model, but each replica uses TP devices
            # - PCP: context parallelism, typically uses TP devices
            # - SP: sequence parallelism, typically uses TP devices
            # - CFG: Classifier-Free Guidance parallelism for diffusion models
            # The number of devices per stage is determined by TP * PP * DP * PCP * SP * CFG size
            # (PP/DP/PCP are higher-level parallelism that don't add devices per stage)
            num_devices_per_stage = (
                tensor_parallel_size
                * pipeline_parallel_size
                * data_parallel_size
                * prefill_context_parallel_size
                * sequence_parallel_size
                * cfg_parallel_size
            )

            # Get physical device IDs from device control env var (e.g., CUDA_VISIBLE_DEVICES)
            # After set_stage_devices, this env var is set to physical device(s)
            device_control_env = current_omni_platform.device_control_env_var
            visible_devices_str = os.environ.get(device_control_env)
            physical_devices = []

            if visible_devices_str:
                try:
                    physical_devices = [int(x.strip()) for x in visible_devices_str.split(",") if x.strip()]
                except (ValueError, IndexError):
                    pass

            if not physical_devices:
                # Fallback: use logical device count if device control env var not set
                num_devices = current_omni_platform.get_device_count()
                physical_devices = list(range(num_devices))

            # Determine which devices will be used (min of devices per stage and available devices)
            num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
            devices_to_lock = physical_devices[:num_devices_to_lock]

            # Sort devices_to_lock to prevent deadlock (all processes acquire locks in same order)
            devices_to_lock = sorted(devices_to_lock)

            logger.debug(
                "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d, CFG=%d; will lock %d devices: %s",
                tensor_parallel_size,
                pipeline_parallel_size,
                data_parallel_size,
                prefill_context_parallel_size,
                sequence_parallel_size,
                cfg_parallel_size,
                num_devices_to_lock,
                devices_to_lock,
            )

            # Acquire exclusive locks for all devices using fcntl.flock
            # Locks are automatically released when process dies
            wait_start = time.time()
            acquired_lock_fds = []  # Store file descriptors to keep locks alive

            for device_id in devices_to_lock:
                lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
                lock_acquired = False

                while not lock_acquired:
                    try:
                        # Open or create the lock file
                        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)

                        # Try to acquire exclusive lock (non-blocking first)
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            # Successfully acquired lock - write PID
                            os.ftruncate(lock_fd, 0)  # Clear file
                            os.write(lock_fd, f"{os.getpid()}\n".encode())
                            os.fsync(lock_fd)  # Ensure written to disk
                            lock_acquired = True
                            acquired_lock_fds.append(lock_fd)
                            logger.debug("Acquired exclusive lock for device %s", device_id)
                        except BlockingIOError:
                            # Lock is held by another process
                            os.close(lock_fd)

                            # Check if we've been waiting too long
                            if time.time() - wait_start > stage_init_timeout:
                                logger.warning(
                                    "Timeout waiting for device %s initialization lock, proceeding anyway",
                                    device_id,
                                )
                                break

                            # Wait a bit before retrying
                            time.sleep(0.1)
                    except OSError as e:
                        # Other error - log and continue without lock
                        logger.debug(
                            "Failed to acquire lock for device %s: %s, continuing anyway",
                            device_id,
                            e,
                        )
                        try:
                            os.close(lock_fd)
                        except (OSError, NameError):
                            pass
                        break

            lock_files = acquired_lock_fds
        except Exception as e:
            logger.debug(
                "[Stage-%s] Failed to set up sequential initialization lock: %s",
                self._stage_id,
            e,
        )
        # Init engine based on stage_type
        if engine_args_dict.get("async_chunk", False):
            logger.debug("[Stage-%s] Async chunk enabled, injecting connectors config", self._stage_id)
            stage_connector_spec = {}
            # TODO
            # for v in connectors_config.values():
            #     stage_connector_spec = dict(v.get("spec", {}))
            #     break
            # engine_args["stage_connector_spec"] = stage_connector_spec
            # engine_args["stage_id"] = self._stage_id

        # Move the omni LLM to here:
        
        # Load stage configurations
        self.config_path = resolve_model_config_path(model)

        # Initialize connectors
        # TODO: pass shm_threshold_bytes from runtime config
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend="multi_process", shm_threshold_bytes=65536
        )

        engine_args = OmniEngineArgs(**engine_args_dict)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.LLM_CLASS
        )
        # TODO: other vllm configs, e.g., compilation configs, connector, ...
        print(vllm_config)
        executor_class = Executor.get_class(vllm_config)
        try:
            if self._stage_type == "diffusion":
                raise NotImplementedError("Diffusion not supported with EngineCore. Use V0 architecture.")
                # engine_args.pop("model_stage", None)
                # engine_args.pop("model", None)
                # stage_engine = OmniDiffusion(
                #     model=model,
                #     stage_id=self._stage_id,
                #     engine_input_source=stage_payload.get("engine_input_source", []),
                #     **engine_args,
                # )
            else:
                logger.info(f"[StageMPClient] Stage-{self._stage_id} initializing EngineCore")
                # Call super().__init__ - starts EngineCore, ZMQ, outputs_queue, etc.
                super().__init__(vllm_config, executor_class, log_stats=False)
        finally:
            # Release all locks by closing file descriptors
            # Locks are automatically released when file descriptors are closed
            # or when process dies
            for lock_fd in lock_files:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                    logger.debug("Released initialization lock (fd=%s)", lock_fd)
                except (OSError, ValueError):
                    pass
        logger.info(f"[StageMPClient] Stage-{self._stage_id} EngineCore running")



        # TODO: Store references that OmniStage exposes
        # self._vllm_config = vllm_config
        # self._tokenizer = ...  # from EngineCore
        # self._input_preprocessor = ...  # from EngineCore
        # self._is_tracing_enabled = False

    # ==================== Stage Properties ====================

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def stage_type(self) -> Literal["llm", "diffusion"]:
        return self._stage_type

    @property
    def final_output(self) -> bool:
        return self._final_output

    @property
    def final_output_type(self) -> str | None:
        return self._final_output_type

    @property
    def default_sampling_params(self) -> OmniSamplingParams:
        return self._default_sampling_params

    @property
    def engine_outputs(self) -> Any:
        return self._engine_outputs

    @property
    def engine_input_source(self) -> list[int]:
        return self._engine_input_source

    @property
    def requires_multimodal_data(self) -> bool:
        return self._requires_multimodal_data

    @property
    def engine_args(self) -> Any:
        return self._engine_args_raw

    # ==================== Overrides ====================

    def add_request(self, request: EngineCoreRequest | dict[str, Any]) -> None:
        """Add request - supports both EngineCoreRequest and task dict."""
        if isinstance(request, dict):
            # TODO: Inject global_request_id into additional_information
            # for cross-stage ID consistency (see OmniStage.submit)
            request = self._task_to_request(request)
        logger.info(f"[StageMPClient] Stage-{self._stage_id} adding request: {request}")
        super().add_request(request)

    # ==================== Stage Methods ====================

    def submit(self, task: dict[str, Any]) -> None:
        """V0 compatibility alias."""
        self.add_request(task)

    def try_collect(self) -> dict[str, Any] | None:
        """Non-blocking get output (V0 compatibility).

        TODO: Handle shared memory outputs (engine_outputs_shm)
        TODO: Include metrics in output
        """
        try:
            outputs = self.outputs_queue.get_nowait()
            logger.info(f"[StageMPClient] Stage-{self._stage_id} got output: {outputs}")
            if isinstance(outputs, Exception):
                return {"error": str(outputs)}
            if not outputs.outputs:
                return None
            out = outputs.outputs[0]
            return {
                "request_id": out.request_id,
                "stage_id": self._stage_id,
                "engine_outputs": [out],
                # TODO: Add metrics
                # "metrics": make_request_stats(...),
            }
        except Exception:
            return None

    def set_engine_outputs(self, engine_outputs: "EngineCoreOutput") -> None:
        """Set engine outputs (called by orchestrator)."""
        self._engine_outputs = engine_outputs

    # TODO: Profiler control methods
    # def start_profile(self) -> None:
    #     """Start profiling via call_utility."""
    #     self.call_utility("profile", True)
    #
    # def stop_profile(self) -> dict:
    #     """Stop profiling and return results."""
    #     # Note: OmniStage sends PROFILER_STOP task and waits for response
    #     self.call_utility("profile", False)
    #     return {}

    def process_engine_inputs(
        self,
        stage_list: list[Any],
        prompt: "OmniTokensPrompt | TextPrompt | None" = None,
    ) -> list["OmniTokensPrompt | TextPrompt"]:
        """Process inputs from upstream stages.

        TODO: Support connector-based data transfer (try_recv_via_connector)
        """
        from vllm_omni.inputs.data import OmniTokensPrompt

        if self._custom_process_input_func is not None:
            logger.info(f"[StageMPClient] Stage-{self._stage_id} using custom process input function")
            return self._custom_process_input_func(
                stage_list, self._engine_input_source, prompt, self._requires_multimodal_data
            )

        if not self._engine_input_source:
            raise ValueError(f"engine_input_source empty for stage {self._stage_id}")

        source_id = self._engine_input_source[0]
        source_outputs = stage_list[source_id].engine_outputs

        if not isinstance(prompt, list):
            prompt = [prompt]

        mm_data = {
            so.request_id: p.get("multi_modal_data")
            for so, p in zip(source_outputs, prompt)
        }

        logger.info(f"[StageMPClient] Stage-{self._stage_id} processing engine inputs: {source_outputs}")
        return [
            OmniTokensPrompt(
                prompt_token_ids=so.outputs[0].token_ids,
                multi_modal_data=mm_data[so.request_id] if self._requires_multimodal_data else None,
            )
            for so in source_outputs
        ]

    # ==================== Internal ====================

    def _task_to_request(self, task: dict[str, Any]) -> EngineCoreRequest:
        """Convert task dict to EngineCoreRequest.

        TODO: Handle connector-based input resolution (try_recv_via_connector)
        TODO: Track rx_metrics (rx_decode_time_ms, rx_transfer_bytes)
        """
        request_id = str(task.get("request_id", ""))
        engine_inputs = task.get("engine_inputs", {})
        sampling_params = task.get("sampling_params", self._default_sampling_params)

        # Extract from engine_inputs
        if isinstance(engine_inputs, dict):
            prompt_token_ids = engine_inputs.get("prompt_token_ids")
            mm_features = engine_inputs.get("mm_features")
            prompt_embeds = engine_inputs.get("prompt_embeds")
        elif hasattr(engine_inputs, "prompt_token_ids"):
            prompt_token_ids = engine_inputs.prompt_token_ids
            mm_features = getattr(engine_inputs, "mm_features", None)
            prompt_embeds = getattr(engine_inputs, "prompt_embeds", None)
        else:
            prompt_token_ids = None
            mm_features = None
            prompt_embeds = None

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            prompt_embeds=prompt_embeds,
        )
