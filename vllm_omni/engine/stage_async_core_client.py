"""
Stage Async Engine Core Client for vLLM-Omni V1 architecture.

Directly inherits from vLLM's AsyncMPClient to reuse EngineCore architecture.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import fcntl
import time
from typing import TYPE_CHECKING, Any, Literal

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.executor import Executor

from vllm_omni.entrypoints.omni_stage import _resolve_worker_cls
from vllm_omni.entrypoints.stage_utils import _to_dict, set_stage_devices
from vllm_omni.entrypoints.utils import resolve_model_config_path
from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

if TYPE_CHECKING:
    from vllm.inputs import TextPrompt
    from vllm.v1.engine import EngineCoreOutput
    from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


class StageAsyncCoreClient(AsyncMPClient):
    """Stage async client that inherits from vLLM's AsyncMPClient.

    Fully reuses AsyncMPClient.__init__ for:
    - ZMQ setup, sockets
    - launch_core_engines() -> EngineCoreProc
    - outputs_queue, output_queue_task
    - All utility methods (shutdown, get_output_async, abort_requests_async, etc.)

    This is the async version of StageMPClient, designed for use with AsyncOmniEngine.
    """

    def __init__(
        self,
        stage_config: Any,
        model: str,
        stage_init_timeout: int = 300,
    ):
        """Create an async EngineCore client for a single stage."""
        logger.info(
            "[StageAsyncCoreClient] Initializing stage client: model=%s, stage_id=%s, timeout=%s",
            model,
            getattr(stage_config, "stage_id", None),
            stage_init_timeout,
        )

        # -------- Stage metadata (public fields) --------
        self.stage_config = stage_config
        self.stage_id: int = stage_config.stage_id
        self.stage_type: Literal["llm", "diffusion"] = getattr(stage_config,
                                                               "stage_type",
                                                               "llm")
        if self.stage_type == "diffusion":
            raise NotImplementedError(
                "Diffusion not supported with EngineCore. Use V0 architecture."
            )

        # Engine args and derived hints
        self.engine_args = stage_config.engine_args
        self.model_stage = getattr(self.engine_args, "model_stage", None)
        self.engine_output_type = getattr(self.engine_args, "engine_output_type",
                                          None)
        self.is_comprehension = getattr(stage_config, "is_comprehension", False)

        # Runtime config
        self.runtime_cfg = getattr(stage_config, "runtime", {})
        self.requires_multimodal_data = getattr(self.runtime_cfg,
                                                "requires_multimodal_data",
                                                False)

        # Input/output config
        self.engine_input_source: list[int] = getattr(stage_config,
                                                      "engine_input_source",
                                                      [])
        self.final_output: bool = getattr(stage_config, "final_output", False)
        self.final_output_type: str | None = getattr(stage_config,
                                                     "final_output_type", None)

        # Default sampling params
        default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
        SPClass = (
            SamplingParams if self.stage_type == "llm" else OmniDiffusionSamplingParams
        )
        self.default_sampling_params: OmniSamplingParams = SPClass(**default_sp)

        # Custom input processing
        if hasattr(stage_config, "custom_process_input_func"):
            mod_path, fn_name = stage_config.custom_process_input_func.rsplit(".", 1)
            self.custom_process_input_func = getattr(
                importlib.import_module(mod_path), fn_name
            )
        else:
            self.custom_process_input_func = None

        # Engine outputs (set by orchestrator)
        self.engine_outputs: Any = None

        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()

        # Set multiprocessing method to spawn
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            logger.info("[StageAsync] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # Build VllmConfig
        engine_args_dict = _to_dict(self.engine_args)
        engine_args_dict["model"] = model

        if self.stage_type != "diffusion":
            _resolve_worker_cls(engine_args_dict)

        logger.info(
            f"[StageAsyncCoreClient] Stage-{self.stage_id} engine_args_dict: {engine_args_dict}"
        )

        # Device mapping
        device_type = None
        current_omni_platform = None
        try:
            from vllm_omni.platforms import current_omni_platform as _cop

            # Stash for later use (device count, env var)
            current_omni_platform = _cop
            device_type = _cop.device_type
            set_stage_devices(
                self.stage_id,
                self.runtime_cfg.get("devices"),
                device_type=device_type,
            )
            logger.info(
                f"[StageAsyncCoreClient] Stage-{self.stage_id} set devices for {device_type}, "
                f"runtime devices: {self.runtime_cfg.get('devices')}"
            )
        except Exception as e:
            logger.warning("Device setup failed: %s", e)

        # Sequential initialization with device locking
        lock_files = []
        try:
            if current_omni_platform is None:
                from vllm_omni.platforms import current_omni_platform as _cop

                current_omni_platform = _cop
            # Get parallel sizes
            if "parallel_config" in engine_args_dict:
                parallel_config = engine_args_dict["parallel_config"]
                tensor_parallel_size = parallel_config.get("tensor_parallel_size", 1)
                pipeline_parallel_size = parallel_config.get("pipeline_parallel_size", 1)
                data_parallel_size = parallel_config.get("data_parallel_size", 1)
                prefill_context_parallel_size = parallel_config.get(
                    "prefill_context_parallel_size", 1
                )
                sequence_parallel_size = parallel_config.get("sequence_parallel_size", 1)
                cfg_parallel_size = parallel_config.get("cfg_parallel_size", 1)
            else:
                tensor_parallel_size = engine_args_dict.get("tensor_parallel_size", 1)
                pipeline_parallel_size = engine_args_dict.get(
                    "pipeline_parallel_size", 1
                )
                data_parallel_size = engine_args_dict.get("data_parallel_size", 1)
                prefill_context_parallel_size = engine_args_dict.get(
                    "prefill_context_parallel_size", 1
                )
                sequence_parallel_size = 1
                cfg_parallel_size = 1

            # Calculate total devices needed
            num_devices_per_stage = (
                tensor_parallel_size
                * pipeline_parallel_size
                * data_parallel_size
                * prefill_context_parallel_size
                * sequence_parallel_size
                * cfg_parallel_size
            )

            # Get physical device IDs
            device_control_env = current_omni_platform.device_control_env_var
            visible_devices_str = os.environ.get(device_control_env)
            physical_devices = []

            if visible_devices_str:
                try:
                    physical_devices = [
                        int(x.strip())
                        for x in visible_devices_str.split(",")
                        if x.strip()
                    ]
                except (ValueError, IndexError):
                    pass

            if not physical_devices:
                num_devices = current_omni_platform.get_device_count()
                physical_devices = list(range(num_devices))

            # Determine devices to lock
            num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
            devices_to_lock = sorted(physical_devices[:num_devices_to_lock])

            logger.debug(
                "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d, CFG=%d; "
                "will lock %d devices: %s",
                tensor_parallel_size,
                pipeline_parallel_size,
                data_parallel_size,
                prefill_context_parallel_size,
                sequence_parallel_size,
                cfg_parallel_size,
                num_devices_to_lock,
                devices_to_lock,
            )

            # Acquire locks
            wait_start = time.time()
            acquired_lock_fds = []

            for device_id in devices_to_lock:
                lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
                lock_acquired = False

                while not lock_acquired:
                    try:
                        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            os.ftruncate(lock_fd, 0)
                            os.write(lock_fd, f"{os.getpid()}\n".encode())
                            os.fsync(lock_fd)
                            lock_acquired = True
                            acquired_lock_fds.append(lock_fd)
                            logger.debug(
                                "Acquired exclusive lock for device %s", device_id
                            )
                        except BlockingIOError:
                            os.close(lock_fd)
                            if time.time() - wait_start > stage_init_timeout:
                                logger.warning(
                                    "Timeout waiting for device %s initialization lock, "
                                    "proceeding anyway",
                                    device_id,
                                )
                                break
                            time.sleep(0.1)
                    except OSError as e:
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
                self.stage_id,
                e,
            )

        # Load stage configurations
        self.config_path = resolve_model_config_path(model)

        # Initialize connectors
        self.omni_transfer_config, self.connectors = (
            initialize_orchestrator_connectors(
                self.config_path, worker_backend="multi_process", shm_threshold_bytes=65536
            )
        )

        # Create engine config
        engine_args = OmniEngineArgs(**engine_args_dict)
        vllm_config = engine_args.create_engine_config(
            usage_context=UsageContext.LLM_CLASS
        )
        executor_class = Executor.get_class(vllm_config)

        try:
            if self.stage_type == "diffusion":
                raise NotImplementedError(
                    "Diffusion not supported with EngineCore. Use V0 architecture."
                )
            else:
                logger.info(
                    f"[StageAsyncCoreClient] Stage-{self.stage_id} initializing EngineCore"
                )
                # Call super().__init__ - starts EngineCore, ZMQ, outputs_queue, etc.
                super().__init__(vllm_config, executor_class, log_stats=False)
        finally:
            # Release all locks
            for lock_fd in lock_files:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                    logger.debug("Released initialization lock (fd=%s)", lock_fd)
                except (OSError, ValueError):
                    pass

        logger.info(
            f"[StageAsyncCoreClient] Stage-{self.stage_id} EngineCore running"
        )

    # ==================== Overrides ====================

    async def add_request_async(
        self, request: EngineCoreRequest | dict[str, Any]
    ) -> None:
        """Add request - supports both EngineCoreRequest and task dict."""
        logger.info(
            f"[StageAsyncCoreClient] Stage-{self.stage_id} adding request: {request.request_id if isinstance(request, EngineCoreRequest) else request.get('request_id', 'N/A')}"
        )
        await super().add_request_async(request)

    # ==================== Stage Methods ====================

    def set_engine_outputs(self, engine_outputs: "EngineCoreOutput") -> None:
        """Set engine outputs (called by orchestrator)."""
        self.engine_outputs = engine_outputs

    def process_engine_inputs(
        self,
        stage_list: list[Any],
        prompt: "OmniTokensPrompt | TextPrompt | None" = None,
    ) -> list["OmniTokensPrompt | TextPrompt"]:
        """Process inputs from upstream stages."""
        from vllm_omni.inputs.data import OmniTokensPrompt

        if self.custom_process_input_func is not None:
            logger.info(
                f"[StageAsyncCoreClient] Stage-{self.stage_id} using custom process input function"
            )
            return self.custom_process_input_func(
                stage_list,
                self.engine_input_source,
                prompt,
                self.requires_multimodal_data,
            )

        if not self.engine_input_source:
            raise ValueError(f"engine_input_source empty for stage {self.stage_id}")

        source_id = self.engine_input_source[0]
        source_outputs = stage_list[source_id].engine_outputs

        if not isinstance(prompt, list):
            prompt = [prompt]

        mm_data = {
            so.request_id: p.get("multi_modal_data")
            for so, p in zip(source_outputs, prompt)
        }

        # logger.info(
        #     f"[StageAsyncCoreClient] Stage-{self.stage_id} processing engine inputs: {source_outputs}"
        # )
        return [
            OmniTokensPrompt(
                prompt_token_ids=so.outputs[0].token_ids,
                multi_modal_data=(
                    mm_data[so.request_id] if self.requires_multimodal_data else None
                ),
            )
            for so in source_outputs
        ]
