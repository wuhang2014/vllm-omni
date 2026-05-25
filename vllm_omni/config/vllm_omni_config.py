# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consolidated top-level configuration for the vllm-omni engine.

``VllmOmniConfig`` replaces the kwargs-dict plumbing that previously threaded
fields across ``OmniBase`` → ``AsyncOmniEngine`` → ``_resolve_stage_configs``
→ ``build_vllm_config``.  It is a single typed object built once after stage
resolution and consumed by all downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.config.diffusion_config import DiffusionConfig


@dataclass
class VllmOmniConfig:
    """Canonical, fully-resolved configuration for the entire vllm-omni engine.

    Built once during ``AsyncOmniEngine.__init__`` after stage config
    resolution.  All downstream consumers receive this object instead of
    reaching into kwargs dicts to extract individual fields.

    Invariant: every field is populated / defaulted by the time construction
    completes.  No more ``getattr(cfg, "some_field", None)`` fallbacks.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # Core identity
    # ═══════════════════════════════════════════════════════════════════════
    model: str = ""
    model_arch: str | None = None

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline / deploy (resolved stage configs)
    # ═══════════════════════════════════════════════════════════════════════
    async_chunk: bool = False

    # ═══════════════════════════════════════════════════════════════════════
    # Orchestrator lifecycle
    # ═══════════════════════════════════════════════════════════════════════
    stage_init_timeout: int = 300
    init_timeout: int = 600

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-stage communication
    # ═══════════════════════════════════════════════════════════════════════
    shm_threshold_bytes: int = 65536
    batch_timeout: int = 10

    # ═══════════════════════════════════════════════════════════════════════
    # Cluster / backend
    # ═══════════════════════════════════════════════════════════════════════
    worker_backend: str = "multi_process"
    ray_address: str | None = None

    # ═══════════════════════════════════════════════════════════════════════
    # Config file references
    # ═══════════════════════════════════════════════════════════════════════
    stage_configs_path: str | None = None
    deploy_config_path: str | None = None
    stage_overrides: dict[str, Any] | None = None

    # ═══════════════════════════════════════════════════════════════════════
    # Master / coordinator
    # ═══════════════════════════════════════════════════════════════════════
    master_address: str | None = None
    master_port: int | None = None
    dp_size_local: int = 1
    lb_policy: str = "random"
    heartbeat_timeout: float = 30.0

    # ═══════════════════════════════════════════════════════════════════════
    # Headless / single-stage mode
    # ═══════════════════════════════════════════════════════════════════════
    single_stage_mode: bool = False
    single_stage_id_filter: int | None = None

    # ═══════════════════════════════════════════════════════════════════════
    # Output
    # ═══════════════════════════════════════════════════════════════════════
    output_modalities: list[str] = field(default_factory=list)
    log_stats: bool = False

    # ═══════════════════════════════════════════════════════════════════════
    # Tokenizer (from CLI, may be None if auto-resolved later)
    # ═══════════════════════════════════════════════════════════════════════
    tokenizer: str | None = None

    # ═══════════════════════════════════════════════════════════════════════
    # Diffusion config (built once from kwargs, shared across stages)
    # ═══════════════════════════════════════════════════════════════════════
    diffusion: DiffusionConfig | None = None
    diffusion_batch_size: int = 1

    # ═══════════════════════════════════════════════════════════════════════
    # TTS defaults
    # ═══════════════════════════════════════════════════════════════════════
    tts_batch_max_items: int = 32

    # ═══════════════════════════════════════════════════════════════════════
    # Observability
    # ═══════════════════════════════════════════════════════════════════════
    enable_ar_profiler: bool = False

    # ═══════════════════════════════════════════════════════════════════════
    # PD disaggregation
    # ═══════════════════════════════════════════════════════════════════════
    pd_config: dict[str, Any] | None = None

    @classmethod
    def from_kwargs(
        cls,
        model: str,
        kwargs: dict[str, Any],
        *,
        diffusion_config: DiffusionConfig | None = None,
    ) -> VllmOmniConfig:
        """Build ``VllmOmniConfig`` from the raw kwargs dict flow.

        Consolidates what was previously popped/extracted at three layers:
        ``OmniBase.__init__``, ``AsyncOmniEngine.__init__``, and
        ``_resolve_stage_configs``.
        """
        return cls(
            model=model,
            model_arch=kwargs.get("model_arch"),
            async_chunk=bool(kwargs.get("async_chunk", False)),
            stage_init_timeout=int(kwargs.get("stage_init_timeout", 300)),
            init_timeout=int(kwargs.get("init_timeout", 600)),
            shm_threshold_bytes=int(kwargs.get("shm_threshold_bytes", 65536)),
            batch_timeout=int(kwargs.get("batch_timeout", 10)),
            worker_backend=str(kwargs.get("worker_backend", "multi_process")),
            ray_address=kwargs.get("ray_address"),
            stage_configs_path=kwargs.get("stage_configs_path"),
            deploy_config_path=kwargs.get("deploy_config"),
            stage_overrides=kwargs.get("stage_overrides"),
            master_address=kwargs.get("omni_master_address"),
            master_port=kwargs.get("omni_master_port"),
            dp_size_local=int(kwargs.get("omni_dp_size_local") or 1),
            lb_policy=str(kwargs.get("omni_lb_policy") or "random"),
            heartbeat_timeout=float(kwargs.get("omni_heartbeat_timeout") or 30.0),
            single_stage_mode=bool(kwargs.get("single_stage_mode", False)),
            single_stage_id_filter=kwargs.get("stage_id") if isinstance(kwargs.get("stage_id"), int) else None,
            output_modalities=kwargs.get("output_modalities") or [],
            log_stats=bool(kwargs.get("log_stats", False)),
            tokenizer=kwargs.get("tokenizer"),
            diffusion=diffusion_config or (DiffusionConfig.from_kwargs(kwargs) if kwargs else None),
            diffusion_batch_size=int(kwargs.get("diffusion_batch_size", 1)),
            tts_batch_max_items=int(kwargs.get("tts_batch_max_items", 32)),
            enable_ar_profiler=bool(kwargs.get("enable_ar_profiler", False)),
            pd_config=kwargs.get("pd_config"),
        )
