# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(
    request_id: str,
    *,
    request_ids: list[str] | None = None,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[f"prompt_{request_id}"],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_ids=request_ids or [request_id],
        request_id=request_id,
    )


def _make_engine() -> DiffusionEngine:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.scheduler = RequestScheduler()
    engine.scheduler.initialize(SimpleNamespace())
    engine.executor = SimpleNamespace(shutdown=Mock())
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._out_queue = {}
    engine._closed = False
    engine.abort_queue = queue.Queue()
    engine._loop_started = False
    engine.stop_event = None
    engine.worker_thread = None
    return engine


def test_close_completes_pending_async_waiters() -> None:
    engine = _make_engine()
    event_loop = asyncio.new_event_loop()
    try:
        future = event_loop.create_future()
        engine._out_queue["pending-req"] = future

        engine.close()

        assert future.done()
        assert future.result().error == "DiffusionEngine is closed."
    finally:
        event_loop.close()


def test_abort_parent_request_id_aborts_batched_scheduler_request() -> None:
    engine = _make_engine()
    request = _make_request("batch-parent", request_ids=["batch-parent-0", "batch-parent-1"])
    sched_req_id = engine.scheduler.add_request(request)

    engine.abort("batch-parent")
    engine._process_aborts_queue()

    state = engine.scheduler.get_request_state(sched_req_id)
    assert state is not None
    assert state.status == DiffusionRequestStatus.FINISHED_ABORTED


def test_close_rejects_late_async_requests() -> None:
    engine = _make_engine()
    event_loop = asyncio.new_event_loop()
    try:
        engine.main_loop = event_loop
        engine.close()

        with pytest.raises(RuntimeError, match="closed"):
            engine.add_request(_make_request("late-req"))
    finally:
        event_loop.close()


def test_close_resets_loop_started_for_dead_worker_thread() -> None:
    engine = _make_engine()
    engine._loop_started = True
    engine.worker_thread = SimpleNamespace(is_alive=Mock(return_value=False))

    engine.close()

    assert engine._loop_started is False
