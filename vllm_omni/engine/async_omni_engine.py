"""
Async Omni Engine for vLLM-Omni V1 architecture.

Manages multiple stages with StageAsyncCoreClient.
Does NOT inherit from any class. Only implements add_request functionality.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_omni.engine.stage_async_core_client import StageAsyncCoreClient
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)

if TYPE_CHECKING:
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

        # Load stage configurations
        if stage_configs is None:
            if stage_configs_path is not None:
                stage_configs = load_stage_configs_from_yaml(stage_configs_path)
            else:
                config_path = resolve_model_config_path(model)
                stage_configs = load_stage_configs_from_model(config_path)

        if not stage_configs:
            raise ValueError("No stage configurations found")

        self.stage_configs = stage_configs
        self.num_stages = len(stage_configs)

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

            # Get tokenizer and vllm_config from stage client
            try:
                # Use collective_rpc to get tokenizer synchronously
                # Note: This is a blocking call during initialization
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    pass

                if loop:
                    # We're in an async context, but collective_rpc is sync
                    # We'll get these later in an async method
                    tokenizer = None
                    vllm_config = stage_client.vllm_config
                else:
                    # Not in async context, safe to call sync method
                    tokenizer_list = stage_client.collective_rpc("get_tokenizer", timeout=30)
                    tokenizer = tokenizer_list[0] if tokenizer_list else None
                    vllm_config = stage_client.vllm_config
            except Exception as e:
                logger.warning(
                    f"[AsyncOmniEngine] Failed to get tokenizer/config for stage {stage_id}: {e}"
                )
                tokenizer = None
                vllm_config = None

            # Determine engine output type
            engine_output_type = getattr(
                stage_cfg.engine_args, "engine_output_type", None
            )

            # Create input processor
            if vllm_config is not None and tokenizer is not None:
                stage_input_processor = OmniInputProcessor(
                    vllm_config=vllm_config,
                    tokenizer=tokenizer,
                )
            else:
                stage_input_processor = None

            # Create output processor
            if tokenizer is not None:
                stage_output_processor = MultimodalOutputProcessor(
                    tokenizer=tokenizer,
                    log_stats=False,
                    engine_core_output_type=engine_output_type,
                )
            else:
                stage_output_processor = None

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
            request = stage_input_processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
            )
        else:
            request = prompt

        # Step 4: Add request to stage client
        await stage_client.add_request_async(request)

        if self.log_requests:
            logger.info(f"[AsyncOmniEngine] Added request {request_id} to stage {stage_id}")

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
