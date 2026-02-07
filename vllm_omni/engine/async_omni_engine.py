"""
Async Omni Engine for vLLM-Omni V1 architecture.

Manages multiple stages with StageAsyncCoreClient.
Does NOT inherit from any class. Only implements add_request functionality.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.engine.stage_async_core_client import StageAsyncCoreClient
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)
from omegaconf import OmegaConf

from vllm.config import VllmConfig
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


class AsyncOmniEngine:
    """Async Omni Engine managing multiple stages.

    Does NOT inherit from any class. Only implements add_request functionality
    for multi-stage execution.

    Args:
        model: Model name or path
        stage_configs: List of stage configurations. If None, loads from model.
        stage_configs_path: Path to YAML file with stage configs. If None, loads from model.
        stage_init_timeout: Timeout for stage initialization (seconds)
        log_requests: Whether to log requests
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        model: str,
        stage_configs: list[Any] | None = None,
        stage_configs_path: str | None = None,
        stage_init_timeout: int = 300,
        log_requests: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.stage_init_timeout = stage_init_timeout
        self.log_requests = log_requests

        logger.info(f"[AsyncOmniEngine] Initializing with model {model}")
        logger.info(f"[AsyncOmniEngine] stage_configs: {stage_configs}")
        logger.info(f"[AsyncOmniEngine] stage_configs_path: {stage_configs_path}")
        logger.info(f"[AsyncOmniEngine] stage_init_timeout: {stage_init_timeout}")
        logger.info(f"[AsyncOmniEngine] log_requests: {log_requests}")
        logger.info(f"[AsyncOmniEngine] kwargs: {kwargs}")
        ### base engine args
        tokenizer = kwargs.get("tokenizer", None)

        base_engine_args = {"tokenizer": tokenizer} if tokenizer is not None else None
        # Load stage configurations from YAML
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model, base_engine_args=base_engine_args)
            if not self.stage_configs:
                default_stage_cfg = self._create_default_diffusion_stage_cfg(kwargs)
                self.stage_configs = OmegaConf.create(default_stage_cfg)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path, base_engine_args=base_engine_args)

        self.num_stages = len(self.stage_configs)

        logger.info(
            f"[AsyncOmniEngine] Initializing with {self.num_stages} stages for model {model}"
        )

        # Initialize stage clients
        self.stage_clients: list[StageAsyncCoreClient] = []
        self.stage_input_processors: list[OmniInputProcessor] = []
        self.stage_output_processors: list[MultimodalOutputProcessor] = []
        self.stage_tokenizers: list[TokenizerLike] = []
        self.stage_vllm_configs: list[VllmConfig] = []

        # Initialize connectors (for cross-stage data transfer like KV cache)
        self.connectors: dict[tuple[str, str], Any] = {}

        self._initialize_stages()
        self._initialize_connectors()

        logger.info(f"[AsyncOmniEngine] Initialization complete")

    def _initialize_stages(self) -> None:
        """Initialize all stage clients and their processors."""
        for stage_id, stage_cfg in enumerate(self.stage_configs):
            logger.info(f"[AsyncOmniEngine] Initializing stage {stage_id}")

            # Create stage client
            stage_client = StageAsyncCoreClient(
                stage_config=stage_cfg,
                model=self.model,
                stage_init_timeout=self.stage_init_timeout,
            )

            # Get vllm_config from EngineCoreClient (available after StageAsyncCoreClient init).
            vllm_config = stage_client.vllm_config

            # Initialize tokenizer the same way as AsyncOmniLLM.
            # If skip_tokenizer_init is enabled, tokenizer is intentionally omitted.
            if vllm_config.model_config.skip_tokenizer_init:
                tokenizer = None
            else:
                tokenizer = cached_tokenizer_from_config(
                    model_config=vllm_config.model_config
                )

            # Determine engine output type
            engine_output_type = getattr(
                stage_cfg.engine_args, "engine_output_type", None
            )

            # Create input processor
            stage_input_processor = OmniInputProcessor(
                vllm_config=vllm_config,
                tokenizer=tokenizer,
            )

            # Create output processor
            stage_output_processor = MultimodalOutputProcessor(
                tokenizer=tokenizer,
                log_stats=False,
                engine_core_output_type=engine_output_type,
            )

            # Store references
            self.stage_clients.append(stage_client)
            self.stage_input_processors.append(stage_input_processor)
            self.stage_output_processors.append(stage_output_processor)
            self.stage_tokenizers.append(tokenizer)
            self.stage_vllm_configs.append(vllm_config)

            logger.info(f"[AsyncOmniEngine] Stage {stage_id} initialized")

    def _initialize_connectors(self) -> None:
        """Initialize connectors for cross-stage data transfer.

        Connectors are used for efficient data transfer between stages,
        such as KV cache transfer, shared memory communication, etc.
        """
        try:
            from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors
            from vllm_omni.entrypoints.utils import resolve_model_config_path

            config_path = resolve_model_config_path(self.model)

            # Initialize connectors with default settings
            omni_transfer_config, connectors = initialize_orchestrator_connectors(
                config_path,
                worker_backend="multi_process",
                shm_threshold_bytes=65536,
            )

            self.omni_transfer_config = omni_transfer_config
            self.connectors = connectors

            if connectors:
                logger.info(
                    f"[AsyncOmniEngine] Initialized {len(connectors)} connectors for cross-stage transfer"
                )
            else:
                logger.debug("[AsyncOmniEngine] No connectors configured")

        except Exception as e:
            logger.warning(f"[AsyncOmniEngine] Failed to initialize connectors: {e}")
            self.omni_transfer_config = None
            self.connectors = {}

    # ==================== Stage Access Methods ====================

    def get_stage_client(self, stage_id: int) -> StageAsyncCoreClient:
        """Get the stage client for a specific stage."""
        if stage_id < 0 or stage_id >= self.num_stages:
            raise ValueError(
                f"Invalid stage_id {stage_id}, must be in [0, {self.num_stages})"
            )
        return self.stage_clients[stage_id]

    def get_stage_input_processor(self, stage_id: int) -> OmniInputProcessor:
        """Get the input processor for a specific stage."""
        if stage_id < 0 or stage_id >= self.num_stages:
            raise ValueError(
                f"Invalid stage_id {stage_id}, must be in [0, {self.num_stages})"
            )
        return self.stage_input_processors[stage_id]

    def get_stage_output_processor(
        self, stage_id: int
    ) -> MultimodalOutputProcessor:
        """Get the output processor for a specific stage."""
        if stage_id < 0 or stage_id >= self.num_stages:
            raise ValueError(
                f"Invalid stage_id {stage_id}, must be in [0, {self.num_stages})"
            )
        return self.stage_output_processors[stage_id]

    def get_stage_tokenizer(self, stage_id: int) -> TokenizerLike:
        """Get the tokenizer for a specific stage."""
        if stage_id < 0 or stage_id >= self.num_stages:
            raise ValueError(
                f"Invalid stage_id {stage_id}, must be in [0, {self.num_stages})"
            )
        return self.stage_tokenizers[stage_id]

    def get_stage_vllm_config(self, stage_id: int) -> VllmConfig:
        """Get the vllm config for a specific stage."""
        if stage_id < 0 or stage_id >= self.num_stages:
            raise ValueError(
                f"Invalid stage_id {stage_id}, must be in [0, {self.num_stages})"
            )
        return self.stage_vllm_configs[stage_id]

    # ==================== Add Request Methods ====================

    async def add_request(
        self,
        stage_id: int,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: dict[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> None:
        """Add a request to a specific stage.

        This method handles:
        1. Injecting global_request_id into additional_information
        2. Processing connector data if available
        3. Processing inputs through input_processor
        4. Adding request to the stage client

        Args:
            stage_id: Stage to add the request to
            request_id: Unique identifier for the request
            prompt: Input prompt (can be EngineCoreRequest or PromptType)
            params: Sampling or pooling parameters
            arrival_time: Request arrival time
            lora_request: Optional LoRA request
            tokenization_kwargs: Optional tokenization arguments
            trace_headers: Optional trace headers
            priority: Request priority
            prompt_text: Optional prompt text
        """
        stage_client = self.get_stage_client(stage_id)
        stage_input_processor = self.get_stage_input_processor(stage_id)
        stage_output_processor = self.get_stage_output_processor(stage_id)

        # Step 1: Inject global_request_id into additional_information
        # This allows workers to use the global ID for cross-stage operations like KV transfer
        def _inject_global_id(target_ein):
            """Inject global request ID into engine inputs."""
            if isinstance(target_ein, dict):
                if "additional_information" not in target_ein:
                    target_ein["additional_information"] = {}

                if target_ein["additional_information"] is None:
                    target_ein["additional_information"] = {}

                if isinstance(target_ein["additional_information"], dict):
                    # Wrap in list because OmniInputProcessor requires Tensor or list values
                    target_ein["additional_information"]["global_request_id"] = [str(request_id)]

        # Inject global_request_id if prompt is dict-like
        if isinstance(prompt, dict):
            _inject_global_id(prompt)
        elif isinstance(prompt, list):
            for item in prompt:
                _inject_global_id(item)

        # Step 2: Check if we need to receive data from connector
        # For stages > 0, we might need to receive KV cache or other data from upstream stages
        if stage_id > 0 and hasattr(self, 'connectors') and self.connectors:
            # Check if there's a connector for the edge (prev_stage -> this_stage)
            prev_stage_id = stage_id - 1
            connector_key = (str(prev_stage_id), str(stage_id))
            connector = self.connectors.get(connector_key)

            if connector:
                try:
                    # Try to receive data from connector
                    # This is typically KV cache data for cross-stage transfer
                    logger.debug(
                        f"[AsyncOmniEngine] Stage {stage_id} checking connector "
                        f"for data from stage {prev_stage_id}"
                    )
                    # Note: Connector receive is typically handled in the worker process
                    # Here we just log that a connector exists
                except Exception as e:
                    logger.warning(
                        f"[AsyncOmniEngine] Failed to process connector data for stage {stage_id}: {e}"
                    )

        # Step 3: Process inputs if needed
        if not isinstance(prompt, EngineCoreRequest):
            if stage_input_processor is None:
                raise ValueError(
                    f"Stage {stage_id} has no input processor, "
                    "cannot process non-EngineCoreRequest prompts"
                )
            if prompt_text is not None:
                # Keep consistent with vLLM AsyncLLM: prompt_text is only accepted
                # when passing an EngineCoreRequest directly.
                raise ValueError("should only provide prompt_text with EngineCoreRequest")

            processed = stage_input_processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )

            # Our OmniInputProcessor implementation may return either:
            # - EngineCoreRequest (vLLM style), or
            # - (prompt_str, EngineCoreRequest) (legacy omni style).
            if isinstance(processed, tuple) and len(processed) == 2:
                prompt_text_from_proc, request = processed
                prompt_text = prompt_text_from_proc
            else:
                request = processed
                prompt_text = prompt
        else:
            request = prompt

        # Align with vLLM AsyncLLM: assign a stable internal request id and ensure
        # request.external_req_id is set (OutputProcessor requires it).
        stage_input_processor.assign_request_id(request)
        # TODO: hack here
        request.request_id = request_id

        # Step 4: Register request in OutputProcessor (this process) so that
        # EngineCoreOutputs can be converted to RequestOutput via process_outputs().
        # Must be done before EngineCore starts returning outputs.
        stage_output_processor.add_request(
            request=request,
            prompt=prompt_text,
            parent_req=None,
            request_index=0,
            queue=None,
        )

        # Step 5: Add request to stage client (EngineCore, separate process)
        await stage_client.add_request_async(request)

        if self.log_requests:
            logger.info(f"[AsyncOmniEngine] Added request {request_id} to stage {stage_id}")

    async def get_output_async(self, stage_id: int) -> list[Any]:
        """Get processed outputs for a stage.

        Pulls EngineCoreOutputs from the stage (EngineCoreClient), then runs the
        stage's OutputProcessor to convert them into RequestOutput objects.

        This mirrors vLLM's AsyncLLM output_handler behavior, but scoped per stage.
        """
        stage_client = self.get_stage_client(stage_id)
        stage_output_processor = self.get_stage_output_processor(stage_id)

        # 1) Pull EngineCoreOutputs from EngineCore.
        outputs = await stage_client.get_output_async()

        # 2) Process EngineCoreOutputs -> RequestOutput.
        processed = stage_output_processor.process_outputs(
            outputs.outputs,  # type: ignore[arg-type]
            getattr(outputs, "timestamp", None),
            None,
        )

        # 3) Abort any reqs that finished due to stop strings.
        if getattr(processed, "reqs_to_abort", None):
            await stage_client.abort_requests_async(processed.reqs_to_abort)

        # Best-effort: propagate scheduler stats if available.
        try:
            if hasattr(outputs, "scheduler_stats"):
                stage_output_processor.update_scheduler_stats(outputs.scheduler_stats)
        except Exception:
            pass

        return list(getattr(processed, "request_outputs", []) or [])

    # ==================== Multi-Stage Orchestration Methods ====================

    def set_stage_engine_outputs(self, stage_id: int, engine_outputs: Any) -> None:
        """Set engine outputs for a stage (for cross-stage data flow).

        Args:
            stage_id: Stage to set outputs for
            engine_outputs: Outputs to set
        """
        stage_client = self.get_stage_client(stage_id)
        stage_client.set_engine_outputs(engine_outputs)

    def process_stage_engine_inputs(
        self,
        stage_id: int,
        stage_list: list[Any],
        prompt: Any = None,
    ) -> list[Any]:
        """Process engine inputs for a stage from upstream stages.

        Args:
            stage_id: Stage to process inputs for
            stage_list: List of all stage clients
            prompt: Optional original prompt

        Returns:
            Processed inputs for the stage
        """
        stage_client = self.get_stage_client(stage_id)
        return stage_client.process_engine_inputs(stage_list, prompt)

    # ==================== Shutdown ====================

    def shutdown(self) -> None:
        """Shutdown all stage clients."""
        logger.info("[AsyncOmniEngine] Shutting down all stages")
        for stage_id, stage_client in enumerate(self.stage_clients):
            try:
                stage_client.shutdown()
                logger.info(f"[AsyncOmniEngine] Stage {stage_id} shut down")
            except Exception as e:
                logger.warning(
                    f"[AsyncOmniEngine] Failed to shutdown stage {stage_id}: {e}"
                )

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass
