"""Shared runtime base for diffusion stage clients."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType


class StageDiffusionClientBase(ABC):
    """Shared stage-aware behavior for diffusion clients."""

    stage_type: str = "diffusion"
    replica_id: int = 0
    is_comprehension: bool = False

    def _initialize_stage_client(self, metadata: StageMetadata, *, batch_size: int) -> None:
        self.stage_id = metadata.stage_id
        self.replica_id = metadata.replica_id
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.requires_multimodal_data = metadata.requires_multimodal_data
        self.custom_process_input_func = metadata.custom_process_input_func
        self.engine_input_source = metadata.engine_input_source
        self.batch_size = batch_size

        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task] = {}
        self._shutting_down = False
        self._engine_dead = False

    def _normalize_profile_rpc_args(self, method: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if method != "profile":
            return args

        args_list = list(args)
        is_start = args_list[0] if args_list else True
        profile_prefix = args_list[1] if len(args_list) > 1 else None
        if is_start and profile_prefix is None:
            args_list.append(self._default_profile_prefix())
        return tuple(args_list)

    def _default_profile_prefix(self) -> str:
        return f"stage_{self.stage_id}_rep_{self.replica_id}_diffusion_{int(time.time())}"

    @abstractmethod
    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None: ...

    @abstractmethod
    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None: ...

    @abstractmethod
    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None: ...

    @abstractmethod
    async def abort_requests_async(self, request_ids: list[str]) -> None: ...

    @abstractmethod
    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any: ...

    @abstractmethod
    def check_health(self) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...
