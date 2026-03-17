"""
Remote stage marker for vLLM-Omni single-stage deployment mode.

``RemoteStageClient`` is a lightweight dataclass that carries the
``OmniMasterServer`` reference and stage metadata needed by
``AsyncOmniEngine._attach_llm_stage`` to:

  1. Retrieve the pre-allocated ZMQ bind addresses for the remote engine.
  2. Wait for the remote engine's READY signal *after* the
     ``StageEngineCoreClient`` has bound its sockets (so the engine can
     connect and complete model loading).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.engine.omni_core_engine import OmniMasterServer


@dataclass
class RemoteStageClient:
    """Lightweight marker carrying context for attaching a remote stage.

    Not an engine client itself – ``AsyncOmniEngine._attach_llm_stage``
    uses this to create a ``StageEngineCoreClient`` with the pre-allocated
    addresses from ``omni_master_server``, then waits for READY.
    """

    omni_master_server: OmniMasterServer
    stage_id: int
    stage_init_timeout: int
