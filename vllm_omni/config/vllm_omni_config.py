"""
Unified configuration for vLLM-Omni.

Defines the two frozen dataclasses that represent the fully-resolved
configuration consumed by the runtime:

- ``StageResolvedConfig`` — pre-built config for one logical stage
  (VllmConfig for LLM, OmniDiffusionConfig for diffusion, plus
  metadata, connectors, runtime overrides).

- ``VllmOmniConfig`` — the top-level frozen config that aggregates
  all stages, orchestrator settings, and pipeline topology.

These are built by ``OmniEngineArgs.create_omni_config()``
(see :mod:`vllm_omni.engine.arg_utils`) and consumed by
:class:`~vllm_omni.engine.async_omni_engine.AsyncOmniEngine` and
:class:`~vllm_omni.engine.orchestrator.Orchestrator`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.diffusion.data import OmniDiffusionConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_omni.engine.stage_init_utils import StageMetadata


# ---------------------------------------------------------------------------
# StageResolvedConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResolvedConfig:
    """Pre-built, immutable config for one logical stage.

    Downstream code reads from this object — it never rebuilds
    ``VllmConfig``, ``OmniDiffusionConfig``, or ``StageMetadata``.
    The factory (:meth:`OmniEngineArgs.create_omni_config`) performs
    all construction up-front.

    For LLM stages ``vllm_config`` / ``executor_class`` are set and
    ``diffusion_config`` is ``None``.  For diffusion stages the
    opposite holds.
    """

    stage_id: int
    """Logical stage identifier (matches ``PipelineConfig`` ordering)."""

    stage_type: Literal["llm", "diffusion"]
    """Backend type discriminator."""

    # ── LLM stages only ────────────────────────────────────────────

    vllm_config: VllmConfig | None = None
    """Pre-built upstream ``VllmConfig`` for LLM stages (``None`` for diffusion)."""

    executor_class: type | None = None
    """vLLM executor class for LLM stages (``None`` for diffusion)."""

    # ── Diffusion stages only ──────────────────────────────────────

    diffusion_config: OmniDiffusionConfig | None = None
    """Pre-built ``OmniDiffusionConfig`` for diffusion stages (``None`` for LLM)."""

    # ── Metadata (pre-extracted — see  :class:`StageMetadata`) ─────

    metadata: StageMetadata | None = None
    """Pre-extracted :class:`~vllm_omni.engine.stage_init_utils.StageMetadata`."""

    # ── Runtime overrides ──────────────────────────────────────────

    num_replicas: int = 1
    """Number of identical replicas for this stage."""

    runtime: dict[str, Any] | None = None
    """Per-stage runtime config (``devices``, etc.)."""

    # ── Connectors (for async chunking) ────────────────────────────

    stage_connector_spec: dict[str, Any] | None = None
    """Connector spec for async-chunk stage-to-stage transfer."""

    omni_kv_connector: tuple[dict[str, Any] | None, str | None, str | None] = (
        None,
        None,
        None,
    )
    """KV connector info: ``(config_dict, from_stage, to_stage)``."""

    # ── Prompt expansion ───────────────────────────────────────────

    prompt_expand_func: Callable[..., Any] | None = None
    """Optional function that expands a prompt before stage submission."""


# ---------------------------------------------------------------------------
# VllmOmniConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmOmniConfig:
    """Fully-resolved, immutable configuration for the entire omni runtime.

    Built once at the entrypoint (:meth:`OmniEngineArgs.create_omni_config`)
    and consumed by every downstream component — no more scattered kwargs,
    no more re-building config per stage.
    """

    model: str
    """Model name or local path."""

    pipeline: PipelineConfig | None = None
    """Frozen pipeline topology from the pipeline registry."""

    stages: tuple[StageResolvedConfig, ...] = ()
    """Pre-built per-stage configs, one entry per logical stage."""

    # ── Orchestrator-level settings ────────────────────────────────

    async_chunk: bool = False
    """Whether async (interleaved) chunking is enabled."""

    stage_init_timeout: int = 300
    """Timeout in seconds for a single stage to initialise."""

    init_timeout: int = 600
    """Total orchestrator startup timeout in seconds."""

    shm_threshold_bytes: int = 65536
    """Byte threshold below which shared memory is used for transfer."""

    batch_timeout: int = 10
    """Batch collection timeout in seconds."""

    worker_backend: str = "multi_process"
    """Backend for stage workers (``"multi_process"`` or ``"ray"``)."""

    log_stats: bool = False
    """Whether to emit per-request pipeline metrics."""

    # ── PD disagg ──────────────────────────────────────────────────

    pd_config: dict[str, Any] | None = None
    """Prefill-Decode disaggregation info: ``{"pd_pair": ..., "bootstrap_addr": ...}``."""

    omni_transfer_config: Any = None
    """Loaded omni transfer config (connector definitions for KV transfer)."""

    # ── Prompt expansion ───────────────────────────────────────────

    prompt_expand_func: Callable[..., Any] | None = None
    """Global prompt-expand function (from the first stage that defines one)."""

    # ── Convenience accessors ──────────────────────────────────────

    @property
    def num_stages(self) -> int:
        """Number of logical stages in the pipeline."""
        return len(self.stages)

    def get_stage(self, index: int) -> StageResolvedConfig:
        """Return the config for the stage at *index*."""
        return self.stages[index]

    def stage_type(self, index: int) -> str:
        """Return ``"llm"`` or ``"diffusion"`` for the stage at *index*."""
        return self.stages[index].stage_type

    def is_single_stage(self) -> bool:
        """Return ``True`` when the pipeline has exactly one stage."""
        return len(self.stages) == 1

    def __post_init__(self) -> None:
        """Validate invariants after construction."""
        if not self.model:
            raise ValueError("VllmOmniConfig.model must be a non-empty string")
        for i, stage in enumerate(self.stages):
            if stage.stage_type not in ("llm", "diffusion"):
                raise ValueError(f"Stage {i}: stage_type must be 'llm' or 'diffusion', got {stage.stage_type!r}")
            if stage.stage_type == "llm" and stage.vllm_config is None:
                raise ValueError(f"Stage {i}: LLM stage must have vllm_config set")
            if stage.stage_type == "diffusion" and stage.diffusion_config is None:
                raise ValueError(f"Stage {i}: diffusion stage must have diffusion_config set")


# ---------------------------------------------------------------------------
# Factory: build_vllm_omni_config
# ---------------------------------------------------------------------------


def build_vllm_omni_config(
    model: str,
    *,
    engine_args: Any = None,
    stage_init_timeout: int = 300,
    init_timeout: int = 600,
    shm_threshold_bytes: int = 65536,
    batch_timeout: int = 10,
    worker_backend: str = "multi_process",
    log_stats: bool = False,
    deploy_config: str | None = None,
    stage_overrides: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> VllmOmniConfig:
    """Build a resolved ``VllmOmniConfig`` from engine arguments.

    This is the single factory that replaces the old scattered config
    construction (``_resolve_stage_configs``, ``build_vllm_config``,
    ``build_diffusion_config``, Phase 2 injection loop, etc.).

    Called by :meth:`OmniEngineArgs.create_omni_config` and by offline
    callers that have an ``OmniEngineArgs`` instance.
    """
    # ── Resolve stage configs from pipeline registry + deploy YAML ──
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_omni_kv_config_for_stage,
    )
    from vllm_omni.engine.stage_init_utils import (
        build_diffusion_config,
        build_engine_args_dict,
        build_vllm_config,
        compute_replica_layout,
        extract_stage_metadata,
        get_stage_connector_spec,
        load_omni_transfer_config_for_model,
    )
    from vllm_omni.entrypoints.utils import _convert_dataclasses_to_dict, load_stage_configs_from_model

    # Build CLI overrides from engine args (replace kwargs dict)
    if engine_args is not None:
        cli_overrides = _convert_dataclasses_to_dict(
            {
                f.name: getattr(engine_args, f.name)
                for f in engine_args.__dataclass_fields__.values()
                if getattr(engine_args, f.name) is not None
            }
        )
    else:
        cli_overrides = _convert_dataclasses_to_dict(dict(kwargs))

    # Apply stage_overrides to cli_overrides
    if stage_overrides:
        for stage_id_str, overrides in stage_overrides.items():
            for key, val in overrides.items():
                cli_overrides[f"stage_{stage_id_str}_{key}"] = val

    # Load and merge stage configs
    stage_configs = load_stage_configs_from_model(
        model,
        base_engine_args=cli_overrides,
        deploy_config_path=deploy_config,
        stage_overrides=stage_overrides,
    )

    if not stage_configs:
        # No pipeline config found — build a default single-stage config.
        # Default to diffusion only for diffusion/know models; LLM-only
        # models get an empty config (caller handles the fallback).
        worker_type = getattr(engine_args, "worker_type", None) if engine_args is not None else None
        if worker_type in ("ar", "generation", None):
            # LLM or unknown — return empty config, legacy path handles it
            return VllmOmniConfig(
                model=model,
                async_chunk=False,
                stage_init_timeout=stage_init_timeout,
                init_timeout=init_timeout,
                shm_threshold_bytes=shm_threshold_bytes,
                batch_timeout=batch_timeout,
                worker_backend=worker_backend,
                log_stats=log_stats,
            )

        # Diffusion fallback
        from omegaconf import OmegaConf

        default_cfg = OmegaConf.create(
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "engine_args": cli_overrides,
                "final_output": True,
                "runtime": {"devices": cli_overrides.get("devices")},
            }
        )
        metadata = extract_stage_metadata(default_cfg)
        diffusion_config = build_diffusion_config(model, default_cfg, metadata)

        return VllmOmniConfig(
            model=model,
            stages=(
                StageResolvedConfig(
                    stage_id=0,
                    stage_type="diffusion",
                    diffusion_config=diffusion_config,
                    metadata=metadata,
                    num_replicas=1,
                ),
            ),
            async_chunk=False,
            stage_init_timeout=stage_init_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            batch_timeout=batch_timeout,
            worker_backend=worker_backend,
            log_stats=log_stats,
        )

    # Resolve async_chunk: CLI override wins over deploy YAML.
    # Use kwargs.get without default so None = "not set by user" and
    # deploy YAML value is used via the is None fallback below.
    async_chunk: bool | None = kwargs.get("async_chunk")

    # ── Pre-build per-stage configs ──────────────────────────────────
    replicas_per_stage, replica_devices_map = compute_replica_layout(stage_configs)
    omni_transfer_config = load_omni_transfer_config_for_model(
        model, getattr(stage_configs[0], "_config_path", None) if stage_configs else None
    )

    resolved_stages: list[StageResolvedConfig] = []
    prompt_expand_func = None

    for stage_idx, stage_cfg in enumerate(stage_configs):
        metadata = extract_stage_metadata(stage_cfg)
        stage_id = metadata.stage_id

        if metadata.prompt_expand_func is not None:
            prompt_expand_func = metadata.prompt_expand_func

        stage_connector_spec = get_stage_connector_spec(
            omni_transfer_config=omni_transfer_config,
            stage_id=stage_id,
            async_chunk=async_chunk,
        )
        omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, stage_id)

        num_replicas = replicas_per_stage[stage_idx] if stage_idx < len(replicas_per_stage) else 1
        runtime = getattr(stage_cfg, "runtime", None)
        if runtime is not None and hasattr(runtime, "get"):
            runtime = dict(runtime)
        elif runtime is not None:
            runtime = vars(runtime) if hasattr(runtime, "__dict__") else {}

        if metadata.stage_type == "diffusion":
            diffusion_config = build_diffusion_config(model, stage_cfg, metadata)
            vllm_config = None
            executor_class = None
        else:
            engine_args_dict = build_engine_args_dict(
                stage_cfg,
                model,
                stage_connector_spec=stage_connector_spec,
                cli_tokenizer=getattr(engine_args, "tokenizer", None) if engine_args is not None else None,
            )
            # Inject KV connector
            omni_conn_cfg, omni_from, omni_to = omni_kv_connector
            if omni_conn_cfg:
                omni_kv = engine_args_dict.get("omni_kv_config") or {}
                if not isinstance(omni_kv, dict):
                    omni_kv = dict(omni_kv)
                omni_kv["connector_config"] = omni_conn_cfg
                omni_kv["omni_from_stage"] = omni_from
                omni_kv["omni_to_stage"] = omni_to
                omni_kv.setdefault("stage_id", stage_id)
                engine_args_dict["omni_kv_config"] = omni_kv
            vllm_config, executor_class = build_vllm_config(
                stage_cfg,
                model,
                stage_connector_spec=stage_connector_spec,
                engine_args_dict=engine_args_dict,
            )
            diffusion_config = None

        resolved_stages.append(
            StageResolvedConfig(
                stage_id=stage_id,
                stage_type=metadata.stage_type,
                vllm_config=vllm_config,
                executor_class=executor_class,
                diffusion_config=diffusion_config,
                metadata=metadata,
                num_replicas=num_replicas,
                runtime=runtime,
                stage_connector_spec=stage_connector_spec,
                omni_kv_connector=omni_kv_connector,
                prompt_expand_func=prompt_expand_func,
            )
        )

    # ── Detect PD config ─────────────────────────────────────────────
    # Use the original OmegaConf stage_configs which carry is_prefill_only /
    # is_decode_only flags — StageResolvedConfig does not expose those.
    pd_config = _detect_pd_config_from_omega_conf(stage_configs)

    # ── Resolve async_chunk from stage 0 if not explicit ─────────────
    if async_chunk is None and resolved_stages:
        async_chunk = bool(getattr(getattr(stage_configs[0], "engine_args", None), "async_chunk", False))

    return VllmOmniConfig(
        model=model,
        stages=tuple(resolved_stages),
        async_chunk=bool(async_chunk) if async_chunk is not None else False,
        stage_init_timeout=stage_init_timeout,
        init_timeout=init_timeout,
        shm_threshold_bytes=shm_threshold_bytes,
        batch_timeout=batch_timeout,
        worker_backend=worker_backend,
        log_stats=log_stats,
        pd_config=pd_config,
        omni_transfer_config=omni_transfer_config,
        prompt_expand_func=prompt_expand_func,
    )


def _detect_pd_config_from_omega_conf(
    stage_configs: list,
) -> dict[str, Any] | None:
    """Detect PD disaggregation from the original OmegaConf stage configs.

    OmegaConf objects carry ``is_prefill_only`` / ``is_decode_only`` flags
    that ``StageResolvedConfig`` does not expose.  Use them here so PD
    disagg survives the move to the unified config path.
    """
    try:
        from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin

        pd_pair = PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stage_configs)
        if pd_pair is None:
            return None
        return {"pd_pair": pd_pair}
    except Exception as exc:
        logger = _get_config_logger()
        logger.warning(
            "[build_vllm_omni_config] PD detection failed: %s. PD disaggregation disabled.",
            exc,
        )
        return None


def _get_config_logger():
    from vllm.logger import init_logger

    return init_logger("vllm_omni.config.vllm_omni_config")
