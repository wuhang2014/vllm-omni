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
from dataclasses import dataclass, field
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
# Ad-hoc config stubs (replaces anonymous type() calls)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefaultDiffusionCfg:
    """Minimal stub config used when no pipeline is registered for the model.

    Provides the attributes that ``extract_stage_metadata`` and
    ``build_diffusion_config`` expect without an OmegaConf roundtrip.
    """

    stage_id: int = 0
    stage_type: str = "diffusion"
    engine_args: dict[str, Any] = field(default_factory=dict)
    final_output: bool = True
    runtime: dict[str, Any] = field(default_factory=dict)


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

    legacy_stage_configs: Any = None
    """Original OmegaConf stage configs from ``load_stage_configs_from_model``.
    Preserved for backward-compat with code that accesses OmegaConf attributes
    (``stage.engine_args.model_stage``, ``stage.is_prefill_only``, etc.) via
    :attr:`OmniBase.stage_configs`.  Not used by the new init path.
    """

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
# Factory helpers
# ---------------------------------------------------------------------------

_ORCHESTRATOR_KWARG_DEFAULTS: dict[str, Any] = {
    "stage_init_timeout": 300,
    "init_timeout": 600,
    "shm_threshold_bytes": 65536,
    "batch_timeout": 10,
    "worker_backend": "multi_process",
    "log_stats": False,
}


def _build_cli_overrides(
    engine_args: Any,
    kwargs: dict[str, Any],
    stage_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build CLI overrides dict from engine args or raw kwargs.

    When *engine_args* is a dataclass instance, extracts all non-None field
    values.  Otherwise falls back to *kwargs*.  Stage-level overrides are
    applied as ``stage_<id>_<key>`` prefixed keys.
    """
    # defer import to avoid circular dependency with vllm_omni.entrypoints.utils
    from vllm_omni.entrypoints.utils import _convert_dataclasses_to_dict

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

    if stage_overrides:
        for stage_id_str, overrides in stage_overrides.items():
            for key, val in overrides.items():
                cli_overrides[f"stage_{stage_id_str}_{key}"] = val

    return cli_overrides


def _build_default_diffusion_config(
    model: str,
    cli_overrides: dict[str, Any],
    engine_args: Any,
    **orchestrator_kwargs: Any,
) -> VllmOmniConfig:
    """Build a single-stage diffusion ``VllmOmniConfig`` when no pipeline is registered.

    Only entered when ``load_stage_configs_from_model`` returns no stages.
    Returns an empty LLM config (for pure-AR workers) when ``engine_args.worker_type``
    is ``"ar"`` or ``"generation"``; otherwise builds a single diffusion stage.
    """
    # defer import to avoid circular dependency with vllm_omni.engine.stage_init_utils
    from vllm_omni.engine.stage_init_utils import build_diffusion_config, extract_stage_metadata

    worker_type = getattr(engine_args, "worker_type", None) if engine_args is not None else None
    if worker_type in ("ar", "generation"):
        return VllmOmniConfig(
            model=model,
            async_chunk=False,
            **{k: orchestrator_kwargs.get(k, v) for k, v in _ORCHESTRATOR_KWARG_DEFAULTS.items()},
        )

    # Build a stub config that provides the attributes
    # extract_stage_metadata / build_diffusion_config expect.
    default_cfg = DefaultDiffusionCfg(
        engine_args=cli_overrides,
        runtime={"devices": cli_overrides.get("devices")},
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
        **{k: orchestrator_kwargs.get(k, v) for k, v in _ORCHESTRATOR_KWARG_DEFAULTS.items()},
    )


def _resolve_stages(
    model: str,
    stage_configs: list,
    engine_args: Any,
    async_chunk: bool | None,
    omni_transfer_config: Any,
    replicas_per_stage: list[int],
    replica_devices_map: dict[int, list[str]],
) -> tuple[list[StageResolvedConfig], Callable[..., Any] | None]:
    """Build ``StageResolvedConfig`` for every stage in *stage_configs*.

    Handles both LLM and diffusion stages — building ``VllmConfig``,
    ``OmniDiffusionConfig``, KV connectors, and runtime metadata.

    Returns ``(resolved_stages, prompt_expand_func)``.
    """
    # NOTE: These imports are deferred to avoid circular dependency with
    # vllm_omni.engine.stage_init_utils (which imports OmniEngineArgs
    # from vllm_omni.engine.arg_utils, and arg_utils imports VllmOmniConfig
    # from this module).
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_omni_kv_config_for_stage,
    )
    from vllm_omni.engine.stage_init_utils import (
        build_diffusion_config,
        build_engine_args_dict,
        build_vllm_config,
        extract_stage_metadata,
        get_stage_connector_spec,
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

    return resolved_stages, prompt_expand_func


def _resolve_async_chunk(
    explicit_cli_value: bool | None,
    resolved_stages: list,
    stage_configs: list,
) -> bool:
    """Resolve the final ``async_chunk`` value with clear precedence.

    Precedence (highest to lowest):
    1. Explicit CLI override (*explicit_cli_value* not ``None``).
       Passed through from ``kwargs.get("async_chunk")`` where ``None`` means
       "the user did not type ``--async-chunk`` or ``--no-async-chunk``".
    2. Deploy YAML / pipeline value (already baked into stage_configs via
       ``merge_pipeline_deploy`` and resolved in ``StageConfigFactory``).
    3. Stage 0 ``engine_args.async_chunk`` (OmegaConf fallback from legacy
       stage config loading).
    4. Default: ``False``.
    """
    if explicit_cli_value is not None:
        return bool(explicit_cli_value)

    if resolved_stages:
        return bool(
            getattr(
                getattr(stage_configs[0], "engine_args", None),
                "async_chunk",
                False,
            )
        )

    return False


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

    Delegates to the following helpers:

    - :func:`_build_cli_overrides` — convert engine args → flat dict
    - :func:`_build_default_diffusion_config` — fallback when no pipeline found
    - :func:`_resolve_stages` — build per-stage ``StageResolvedConfig``
    - :func:`_resolve_async_chunk` — resolve async_chunk precedence
    """
    from vllm_omni.engine.stage_init_utils import (
        compute_replica_layout,
        load_omni_transfer_config_for_model,
    )
    from vllm_omni.entrypoints.utils import load_stage_configs_from_model

    # 1. Build CLI overrides dict
    cli_overrides = _build_cli_overrides(engine_args, kwargs, stage_overrides)

    # 2. Load and merge stage configs
    stage_configs = load_stage_configs_from_model(
        model,
        base_engine_args=cli_overrides,
        deploy_config_path=deploy_config,
        stage_overrides=stage_overrides,
    )

    # 3. Handle missing pipeline — delegate to default-diffusion builder
    if not stage_configs:
        return _build_default_diffusion_config(
            model=model,
            cli_overrides=cli_overrides,
            engine_args=engine_args,
            stage_init_timeout=stage_init_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            batch_timeout=batch_timeout,
            worker_backend=worker_backend,
            log_stats=log_stats,
        )

    # 4. Resolve async_chunk: CLI override wins over deploy YAML.
    #    Use kwargs.get without default so None = "not set by user".
    explicit_async_chunk: bool | None = kwargs.get("async_chunk")

    # 5. Compute replica layout and transfer config
    replicas_per_stage, replica_devices_map = compute_replica_layout(stage_configs)
    omni_transfer_config = load_omni_transfer_config_for_model(
        model, getattr(stage_configs[0], "_config_path", None) if stage_configs else None
    )

    # 6. Build per-stage resolved configs
    resolved_stages, prompt_expand_func = _resolve_stages(
        model=model,
        stage_configs=stage_configs,
        engine_args=engine_args,
        async_chunk=explicit_async_chunk,
        omni_transfer_config=omni_transfer_config,
        replicas_per_stage=replicas_per_stage,
        replica_devices_map=replica_devices_map,
    )

    # 7. Detect PD disagg from OmegaConf stage configs
    pd_config = _detect_pd_config_from_omega_conf(stage_configs)

    # 8. Resolve async_chunk (explicit CLI → stage 0 fallback → False)
    async_chunk = _resolve_async_chunk(explicit_async_chunk, resolved_stages, stage_configs)

    # 9. Assemble and return
    return VllmOmniConfig(
        model=model,
        stages=tuple(resolved_stages),
        async_chunk=async_chunk,
        stage_init_timeout=stage_init_timeout,
        init_timeout=init_timeout,
        shm_threshold_bytes=shm_threshold_bytes,
        batch_timeout=batch_timeout,
        worker_backend=worker_backend,
        log_stats=log_stats,
        pd_config=pd_config,
        omni_transfer_config=omni_transfer_config,
        prompt_expand_func=prompt_expand_func,
        legacy_stage_configs=stage_configs,
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
    except (ImportError, AttributeError, TypeError) as exc:
        logger = _get_config_logger()
        logger.debug(
            "[build_vllm_omni_config] PD detection failed: %s. PD disaggregation disabled.",
            exc,
        )
        return None


def _get_config_logger():
    from vllm.logger import init_logger

    return init_logger("vllm_omni.config.vllm_omni_config")
