from __future__ import annotations

from typing import Any

import pytest

from vllm_omni.diffusion.stage_diffusion_client_base import StageDiffusionClientBase
from vllm_omni.engine.stage_init_utils import StageMetadata

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu]


class _ConcreteStageDiffusionClient(StageDiffusionClientBase):
    def __init__(self, metadata: StageMetadata, *, batch_size: int = 1) -> None:
        self._initialize_stage_client(metadata, batch_size=batch_size)

    async def add_request_async(
        self,
        request_id: str,
        prompt: Any,
        sampling_params: Any,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        return None

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[Any],
        sampling_params: Any,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        return None

    def get_diffusion_output_nowait(self):
        return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        return None

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return None

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _make_metadata() -> StageMetadata:
    return StageMetadata(
        stage_id=3,
        stage_type="diffusion",
        engine_output_type=None,
        is_comprehension=False,
        requires_multimodal_data=True,
        engine_input_source=[1, 2],
        final_output=True,
        final_output_type="image",
        default_sampling_params={},
        custom_process_input_func=str.upper,
        model_stage=None,
        runtime_cfg=None,
        replica_id=7,
    )


def test_initialize_stage_client_sets_shared_metadata_and_state() -> None:
    client = _ConcreteStageDiffusionClient(_make_metadata(), batch_size=4)

    assert client.stage_id == 3
    assert client.replica_id == 7
    assert client.stage_type == "diffusion"
    assert client.final_output is True
    assert client.final_output_type == "image"
    assert client.requires_multimodal_data is True
    assert client.custom_process_input_func is str.upper
    assert client.engine_input_source == [1, 2]
    assert client.batch_size == 4
    assert client._tasks == {}
    assert client._shutting_down is False
    assert client._engine_dead is False


def test_normalize_profile_rpc_args_injects_default_prefix() -> None:
    client = _ConcreteStageDiffusionClient(_make_metadata())

    args = client._normalize_profile_rpc_args("profile", (True,))

    assert args[0] is True
    assert len(args) == 2
    assert args[1].startswith("stage_3_rep_7_diffusion_")


def test_normalize_profile_rpc_args_preserves_existing_values() -> None:
    client = _ConcreteStageDiffusionClient(_make_metadata())

    explicit = client._normalize_profile_rpc_args("profile", (True, "custom-prefix"))
    non_profile = client._normalize_profile_rpc_args("list_loras", (1, 2))

    assert explicit == (True, "custom-prefix")
    assert non_profile == (1, 2)
