"""
Unified configuration for vLLM-Omni.

Two frozen dataclasses plus helper functions consumed by
:meth:`~vllm_omni.engine.arg_utils.OmniEngineArgs.create_omni_config`,
the single factory that replaces
all scattered config building, kwargs partitioning, OmegaConf
roundtrips, and heuristic default-resolution.

- :class:`StageResolvedConfig` — pre-built immutable per-stage config
- :class:`VllmOmniConfig` — top-level frozen runtime configuration
- :meth:`~vllm_omni.engine.arg_utils.OmniEngineArgs.create_omni_config` — the single factory

Built once at the entrypoint and consumed by
:class:`~vllm_omni.engine.async_omni_engine.AsyncOmniEngine`,
:class:`~vllm_omni.engine.orchestrator.Orchestrator`, and
:class:`~vllm_omni.entrypoints.omni_base.OmniBase`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_omni.config.stage_config import PipelineConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.engine.arg_utils import OmniEngineArgs

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# StageResolvedConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResolvedConfig:
    """Pre-built, immutable config for one logical stage.

    All construction (``VllmConfig``, ``OmniDiffusionConfig``,
    connector resolution, prompt expansion) happens inside
    :meth:`OmniEngineArgs.create_omni_config`.  Downstream code
    reads attributes — never rebuilds.
    """

    stage_id: int
    """Logical stage identifier (matches pipeline ordering)."""

    stage_type: Literal["llm", "diffusion"]
    """Backend type discriminator."""

    # ── LLM stages only ──────────────────────────────────────────

    vllm_config: VllmConfig | None = None
    """Pre-built upstream ``VllmConfig`` for LLM stages (``None`` for diffusion)."""

    executor_class: type | None = None
    """vLLM executor class for LLM stages (``None`` for diffusion)."""

    # ── Diffusion stages only ────────────────────────────────────

    diffusion_config: OmniDiffusionConfig | None = None
    """Pre-built ``OmniDiffusionConfig`` for diffusion stages (``None`` for LLM)."""

    # ── Per-stage attributes (formerly on StageMetadata) ─────────

    engine_output_type: str | None = None
    """Optional output type specification for the engine."""

    is_comprehension: bool = False
    """True when this stage owns the tokenizer (comprehension stage)."""

    requires_multimodal_data: bool = False
    """True when this stage requires multimodal data preprocessing."""

    engine_input_source: list[int] = field(default_factory=list)
    """Stage IDs that feed into this stage.  Used for PD detection and
    KV routing."""

    final_output: bool = False
    """True when this stage produces the final pipeline output."""

    final_output_type: str | None = None
    """Output modality of the stage (``"text"``, ``"audio"``, ``"image"``, …)."""

    default_sampling_params: Any = None
    """Pre-extracted default sampling params (``OmniSamplingParams`` or dict)."""

    custom_process_input_func: Callable[..., Any] | None = None
    """Optional function that preprocesses input before stage submission."""

    model_stage: str | None = None
    """Stage type identifier (e.g. ``"thinker"`` or ``"talker"``)."""

    model_arch: str | None = None
    """Model architecture name from pipeline topology."""

    cfg_kv_collect_func: Callable[..., Any] | None = None
    """Optional CFG KV collection function for diffusion stages."""

    # ── Runtime overrides ────────────────────────────────────────

    num_replicas: int = 1
    """Number of identical replicas for this stage."""

    runtime: dict[str, Any] | None = None
    """Per-stage runtime config (``devices``, etc.)."""

    # ── Connectors ───────────────────────────────────────────────

    stage_connector_spec: dict[str, Any] | None = None
    """Connector spec for async-chunk stage-to-stage transfer."""

    omni_kv_connector: tuple[dict[str, Any] | None, str | None, str | None] = (
        None,
        None,
        None,
    )
    """KV connector info: ``(config_dict, from_stage, to_stage)``."""

    # ── Prompt expansion ─────────────────────────────────────────

    prompt_expand_func: Callable[..., Any] | None = None
    """Optional function that expands a prompt before stage submission."""

    # ── PD disaggregation ────────────────────────────────────────

    is_prefill_only: bool = False
    """True when this stage is a dedicated prefill-only stage (PD disagg)."""

    is_decode_only: bool = False
    """True when this stage is a dedicated decode-only stage (PD disagg)."""


# ---------------------------------------------------------------------------
# VllmOmniConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmOmniConfig:
    """Fully-resolved, immutable runtime configuration.

    Built once by :meth:`OmniEngineArgs.create_omni_config`.  Consumed by
    :class:`~vllm_omni.engine.async_omni_engine.AsyncOmniEngine`,
    :class:`~vllm_omni.engine.orchestrator.Orchestrator`, and
    :class:`~vllm_omni.entrypoints.omni_base.OmniBase`.
    """

    model: str
    """Model name or local path."""

    stages: tuple[StageResolvedConfig, ...] = ()
    """Pre-built per-stage configs, one entry per logical stage."""

    # ── Convenience: direct access for single-diffusion-stage models ──

    diffusion_config: OmniDiffusionConfig | None = None
    """Top-level ``OmniDiffusionConfig`` for single-stage diffusion models
    or the DiT stage of a multi-stage pipeline. ``None`` for pure-LLM
    pipelines."""

    pipeline: PipelineConfig | None = None
    """Frozen pipeline topology from the registry (for introspection)."""

    # ── Orchestrator ─────────────────────────────────────────────

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

    # ── PD disagg ────────────────────────────────────────────────

    pd_config: dict[str, Any] | None = None
    """Prefill-Decode disaggregation info:
    ``{"pd_pair": ..., "bootstrap_addr": ...}``."""

    omni_transfer_config: Any = None
    """Loaded omni transfer config (connector definitions for KV transfer)."""

    # ── Prompt expansion ─────────────────────────────────────────

    prompt_expand_func: Callable[..., Any] | None = None
    """Global prompt-expand function (from the first stage that defines one)."""

    # ── Convenience ──────────────────────────────────────────────

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
# Minimal stage-cfg protocol object (replaces StageConfig for
# build_engine_args_dict / build_vllm_config / build_diffusion_config)
# ---------------------------------------------------------------------------


class _PerStageCfg:
    """Lightweight object that provides the attributes expected by
    ``build_engine_args_dict``, ``build_vllm_config``, and
    ``build_diffusion_config``.

    Replaces the removed ``StageConfig`` / OmegaConf roundtrip.
    """

    __slots__ = ("stage_id", "stage_type", "engine_args", "default_sampling_params")
    # Fields that build_engine_args_dict read via getattr.
    # They are set explicitly as instance attributes below.

    def __init__(
        self,
        stage_id: int,
        stage_type: str,
        engine_args: dict[str, Any],
        default_sampling_params: dict[str, Any] | None = None,
    ) -> None:
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.engine_args = engine_args
        self.default_sampling_params = default_sampling_params or {}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _resolve_dotted_func(path: str | None) -> Any:
    """Resolve a dotted import string to a callable, or return None."""
    if not path:
        return None
    if callable(path):
        return path
    try:
        mod_path, fn_name = path.rsplit(".", 1)
        return getattr(__import__(mod_path, fromlist=[fn_name]), fn_name)
    except Exception:
        return None


def _parse_stage_overrides(raw: str | dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Parse *stage_overrides* from a JSON string or dict into a typed dict.

    Returns ``{}`` when *raw* is ``None`` or unparsable.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid stage_overrides JSON, ignoring: %s", raw)
        return {}


def _resolve_stages(
    model: str,
    stage_overrides: dict[str, dict[str, Any]],
    engine_args: OmniEngineArgs,
    async_chunk: bool,
    omni_transfer_config: Any,
) -> tuple[list[StageResolvedConfig], OmniDiffusionConfig | None, Callable[..., Any] | None]:
    """Build :class:`StageResolvedConfig` for every stage described in *stage_overrides*.

    Handles both LLM and diffusion stages — calling the building blocks
    from :mod:`vllm_omni.engine.stage_init_utils`.

    Returns:
        ``(resolved_stages, top_level_diffusion_config, prompt_expand_func)``.
    """
    # Defer imports to avoid circular dependency.
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_omni_kv_config_for_stage,
    )
    from vllm_omni.engine.stage_init_utils import (
        build_diffusion_config,
        build_engine_args_dict,
        build_vllm_config,
        get_stage_connector_spec,
    )

    resolved_stages: list[StageResolvedConfig] = []
    prompt_expand_func = None
    top_level_diffusion_config: OmniDiffusionConfig | None = None

    # Sort stage IDs to ensure consistent ordering.
    sorted_stage_ids = sorted(stage_overrides.keys(), key=int)

    for stage_id_str in sorted_stage_ids:
        stage_id = int(stage_id_str)
        data = stage_overrides[stage_id_str]

        stage_type: str = data.get("stage_type", "llm")
        per_stage_engine_args: dict[str, Any] = {
            **{f.name: getattr(engine_args, f.name) for f in engine_args.__dataclass_fields__.values()},
            **data.get("engine_args", {}),
        }
        # Remove None values — downstream builders rely on defaults.
        per_stage_engine_args = {k: v for k, v in per_stage_engine_args.items() if v is not None}
        # Override with stage_id from topology.
        per_stage_engine_args["stage_id"] = stage_id

        default_sp: dict[str, Any] = data.get("default_sampling_params", {})
        runtime: dict[str, Any] = data.get("runtime", {})
        num_replicas: int = runtime.get("num_replicas", data.get("num_replicas", 1))

        # Per-stage cfg protocol object.
        stage_cfg = _PerStageCfg(
            stage_id=stage_id,
            stage_type=stage_type,
            engine_args=per_stage_engine_args,
            default_sampling_params=default_sp,
        )

        # prompt_expand_func from stage_overrides data
        pef = data.get("prompt_expand_func")
        if pef and prompt_expand_func is None:
            if callable(pef):
                prompt_expand_func = pef
            elif isinstance(pef, str):
                _mod, _fn = pef.rsplit(".", 1)
                prompt_expand_func = getattr(importlib.import_module(_mod), _fn)

        stage_connector_spec = get_stage_connector_spec(
            omni_transfer_config=omni_transfer_config,
            stage_id=stage_id,
            async_chunk=async_chunk,
        )
        omni_kv_connector = resolve_omni_kv_config_for_stage(omni_transfer_config, stage_id)

        if stage_type == "diffusion":
            diffusion_config = build_diffusion_config(model, stage_cfg, stage_cfg)
            if top_level_diffusion_config is None:
                top_level_diffusion_config = diffusion_config
            vllm_config = None
            executor_class = None
        else:
            engine_args_dict = build_engine_args_dict(
                stage_cfg,
                model,
                stage_connector_spec=stage_connector_spec,
                cli_tokenizer=getattr(engine_args, "tokenizer", None),
            )
            # Inject KV connector config into engine args.
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
                stage_type=stage_type,
                vllm_config=vllm_config,
                executor_class=executor_class,
                diffusion_config=diffusion_config,
                engine_output_type=data.get("engine_output_type"),
                is_comprehension=data.get("is_comprehension", False),
                requires_multimodal_data=data.get("requires_multimodal_data", False),
                engine_input_source=data.get("input_sources", data.get("engine_input_source", [])),
                final_output=data.get("final_output", False),
                final_output_type=data.get("final_output_type"),
                default_sampling_params=data.get("default_sampling_params"),
                custom_process_input_func=_resolve_dotted_func(data.get("custom_process_input_func")),
                model_stage=data.get("model_stage") or per_stage_engine_args.get("model_stage"),
                model_arch=per_stage_engine_args.get("model_arch") or data.get("model_arch"),
                cfg_kv_collect_func=_resolve_dotted_func(data.get("cfg_kv_collect_func")),
                num_replicas=num_replicas,
                runtime=runtime,
                stage_connector_spec=stage_connector_spec,
                omni_kv_connector=omni_kv_connector,
                prompt_expand_func=prompt_expand_func,
                is_prefill_only=data.get("is_prefill_only", False),
                is_decode_only=data.get("is_decode_only", False),
            )
        )

    return resolved_stages, top_level_diffusion_config, prompt_expand_func


def _detect_pd_config(
    resolved_stages: list[StageResolvedConfig],
) -> dict[str, Any] | None:
    """Detect PD disaggregation from resolved :class:`StageResolvedConfig` objects.

    Uses ``is_prefill_only`` / ``is_decode_only`` flags and
    :class:`~vllm_omni.entrypoints.pd_utils.PDDisaggregationMixin`.
    """
    try:
        from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin

        # Build lightweight wrappers for the PD mixin.
        class _PDStage:
            __slots__ = ("stage_id", "is_prefill_only", "is_decode_only", "engine_input_source")

            def __init__(self, s: StageResolvedConfig) -> None:
                self.stage_id = s.stage_id
                self.is_prefill_only = s.is_prefill_only
                self.is_decode_only = s.is_decode_only
                self.engine_input_source = s.engine_input_source

        wrappers = [_PDStage(s) for s in resolved_stages]
        pd_pair = PDDisaggregationMixin.detect_pd_separation_from_stage_configs(wrappers)
        if pd_pair is None:
            return None
        return {"pd_pair": pd_pair}
    except (ImportError, AttributeError, TypeError) as exc:
        logger.debug(
            "[build_vllm_omni_config] PD detection failed: %s. PD disaggregation disabled.",
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Factory — public API
# ---------------------------------------------------------------------------


