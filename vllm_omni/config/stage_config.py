# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage configuration system for vLLM-Omni."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

from vllm_omni.config.yaml_util import load_yaml_config, to_dict

logger = init_logger(__name__)

_STAGE_OVERRIDE_PATTERN = re.compile(r"^stage_(\d+)_(.+)$")


def build_stage_runtime_overrides(
    stage_id: int,
    cli_overrides: dict[str, Any],
    *,
    internal_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Build per-stage runtime overrides from global and ``stage_<id>_*`` kwargs.

    ``internal_keys`` defaults to the union of
    ``arg_utils.internal_blacklist_keys()`` and ``arg_utils.SHARED_FIELDS``
    so that neither orchestrator-only fields nor shared-pipeline fields
    (``model`` / ``stage_configs_path`` / ``log_stats`` / ``stage_id``) leak
    into a stage's per-stage runtime overrides — the orchestrator sets those
    uniformly for every stage, they are not per-stage knobs. Callers can
    pass an explicit set for tests or specialized flows.
    """
    if internal_keys is None:
        from vllm_omni.engine.arg_utils import SHARED_FIELDS, internal_blacklist_keys

        internal_keys = internal_blacklist_keys() | SHARED_FIELDS

    result: dict[str, Any] = {}

    for key, value in cli_overrides.items():
        if value is None or key in internal_keys:
            continue

        match = _STAGE_OVERRIDE_PATTERN.match(key)
        if match is not None:
            override_stage_id = int(match.group(1))
            param_name = match.group(2)
            if override_stage_id == stage_id and param_name not in internal_keys:
                result[param_name] = value
            continue

        result[key] = value

    return result


def strip_parent_engine_args(
    kwargs: dict[str, Any],
    *,
    parent_fields: dict[str, dataclasses.Field],
    keep_keys: set[str] | frozenset[str] = frozenset(),
    strip_keys: set[str] | frozenset[str] = frozenset(),
    no_warn_keys: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], list[str]]:
    """Strip parent ``EngineArgs`` fields before merging into stage YAML."""
    overridden: list[str] = []
    result: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in strip_keys:
            continue

        if key not in parent_fields or key in keep_keys:
            result[key] = value
            continue

        field_def = parent_fields[key]
        if field_def.default is not dataclasses.MISSING:
            default = field_def.default
        elif field_def.default_factory is not dataclasses.MISSING:
            default = field_def.default_factory()
        else:
            default = dataclasses.MISSING

        if default is dataclasses.MISSING or value is None:
            continue

        if dataclasses.is_dataclass(default) and not isinstance(default, type):
            default = asdict(default)

        if value != default and key not in no_warn_keys:
            overridden.append(key)

    return result, sorted(overridden)


class StageType(str, Enum):
    """Type of processing stage in the Omni pipeline."""

    # TODO(@lishunyang12): remove once all models migrate to StageExecutionType
    LLM = "llm"
    DIFFUSION = "diffusion"


class StageExecutionType(str, Enum):
    """Merged StageType + WorkerType — 3 combinations today."""

    LLM_AR = "llm_ar"
    LLM_GENERATION = "llm_generation"
    DIFFUSION = "diffusion"


@dataclass(frozen=True)
class StagePipelineConfig:
    """Fixed topology for one stage (frozen, not user-configurable)."""

    stage_id: int
    model_stage: str
    execution_type: StageExecutionType = StageExecutionType.LLM_AR
    input_sources: tuple[int, ...] = ()
    final_output: bool = False
    final_output_type: str | None = None
    owns_tokenizer: bool = False
    requires_multimodal_data: bool = False
    hf_config_name: str | None = None
    engine_output_type: str | None = None
    model_arch: str | None = None
    sampling_constraints: dict[str, Any] = field(default_factory=dict)
    custom_process_input_func: str | None = None
    custom_process_next_stage_input_func: str | None = None
    # Alternates picked by ``merge_pipeline_deploy`` based on ``deploy.async_chunk``.
    async_chunk_process_next_stage_input_func: str | None = None
    sync_process_input_func: str | None = None
    prompt_expand_func: str | None = None
    cfg_kv_collect_func: str | None = None
    omni_kv_config: dict[str, Any] | None = None
    # Model subdirectory indirections: for multi-component HF repos where the
    # stage's config/tokenizer lives in a subdirectory (e.g. GLM-Image's AR
    # config is in ``vision_language_encoder/``).  Consumed at stage-init time
    # by ``stage_init_utils._resolve_model_tokenizer_paths``.
    model_subdir: str | None = None
    tokenizer_subdir: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline topology for a model (frozen)."""

    model_type: str
    model_arch: str = ""
    stages: tuple[StagePipelineConfig, ...] = ()
    # HF architecture aliases: used by StageConfigFactory when the model's
    # HF config reports a generic model_type that collides with a different
    # model (e.g. MiMo Audio reports model_type="qwen2"). The factory
    # matches ``hf_config.architectures[*]`` against this tuple to route
    # to the correct pipeline. Leave empty for models with unique model_type.
    hf_architectures: tuple[str, ...] = ()
    # Diffusers pipeline class name: for models that ship a ``model_index.json``
    # (no root ``config.json``), the ``_class_name`` field is matched against
    # this value to auto-detect the pipeline.  Only needed for diffusers-style
    # multi-component repos (e.g. GLM-Image).  ``None`` = not a diffusers model.
    diffusers_class_name: str | None = None

    def get_stage(self, stage_id: int) -> StagePipelineConfig | None:
        """Look up a stage by its ID."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def validate(self) -> list[str]:
        """Return list of topology errors (empty if valid)."""
        errors: list[str] = []
        if not self.stages:
            errors.append("Pipeline has no stages defined")
            return errors
        stage_ids = [s.stage_id for s in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("Duplicate stage IDs found")
        stage_id_set = set(stage_ids)
        for stage in self.stages:
            for src in stage.input_sources:
                if src not in stage_id_set:
                    errors.append(f"Stage {stage.stage_id} references non-existent input source {src}")
                if src == stage.stage_id:
                    errors.append(f"Stage {stage.stage_id} references itself")
        if not any(not s.input_sources for s in self.stages):
            errors.append("No entry point (stage with empty input_sources)")
        return errors


class _LazyPipelineRegistry:
    """Dict-like registry that lazy-loads pipelines from the central declaration.

    In-tree pipelines are declared once in
    ``vllm_omni/config/pipeline_registry.py`` as
    ``model_type -> (module_path, variable_name)`` entries; the module is
    imported only when the pipeline is first looked up. This mirrors the
    pattern in ``vllm/model_executor/models/registry.py`` and addresses
    https://github.com/vllm-project/vllm-omni/issues/2887 (item 4): having
    every registration in one file makes a missing entry easy to spot.

    Out-of-tree / dynamic registrations via ``register_pipeline()`` are stored
    directly in ``_loaded`` and take precedence over the lazy-map entry with
    the same ``model_type``.

    The class exposes the subset of ``dict`` operations the rest of this
    module relies on (``__contains__``, ``__getitem__``, ``__setitem__``,
    ``get``, ``keys``, ``values``, ``items``, ``__iter__``), so existing call
    sites don't need to change.
    """

    def __init__(self) -> None:
        self._loaded: dict[str, PipelineConfig] = {}
        # Populated lazily to avoid a circular import at module init time.
        self._lazy_map: dict[str, tuple[str, str]] | None = None

    def _get_lazy_map(self) -> dict[str, tuple[str, str]]:
        if self._lazy_map is None:
            from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

            self._lazy_map = _OMNI_PIPELINES
        return self._lazy_map

    def _load_lazy(self, model_type: str) -> PipelineConfig | None:
        entry = self._get_lazy_map().get(model_type)
        if entry is None:
            return None
        module_path, var_name = entry
        import importlib

        try:
            module = importlib.import_module(module_path)
            pipeline = getattr(module, var_name, None)
            if pipeline is None:
                logger.error(
                    "Pipeline variable %r not found in module %r (registered for %r)",
                    var_name,
                    module_path,
                    model_type,
                )
                return None
            errors = pipeline.validate()
            if errors:
                logger.warning("Pipeline %s has issues: %s", pipeline.model_type, errors)
            self._loaded[model_type] = pipeline
            return pipeline
        except Exception:
            logger.exception("Failed to import pipeline module %r for %r", module_path, model_type)
            return None

    def __contains__(self, model_type: str) -> bool:
        if model_type in self._loaded:
            return True
        return model_type in self._get_lazy_map()

    def __getitem__(self, model_type: str) -> PipelineConfig:
        if model_type in self._loaded:
            return self._loaded[model_type]
        pipeline = self._load_lazy(model_type)
        if pipeline is None:
            raise KeyError(model_type)
        return pipeline

    def get(self, model_type: str, default: PipelineConfig | None = None) -> PipelineConfig | None:
        if model_type in self._loaded:
            return self._loaded[model_type]
        pipeline = self._load_lazy(model_type)
        return pipeline if pipeline is not None else default

    def __setitem__(self, model_type: str, pipeline: PipelineConfig) -> None:
        self._loaded[model_type] = pipeline

    def __delitem__(self, model_type: str) -> None:
        """Remove a dynamically-registered pipeline.

        Only the dynamic-cache side of the registry can be mutated; the
        central declarative registry is immutable at runtime. Calling ``del``
        on a model_type that only exists in the central registry raises
        ``KeyError``.
        """
        if model_type in self._loaded:
            del self._loaded[model_type]
            return
        if model_type in self._get_lazy_map():
            raise KeyError(
                f"{model_type!r} is declared in the central pipeline_registry and "
                "cannot be removed at runtime. Edit "
                "vllm_omni/config/pipeline_registry.py to delete a built-in entry."
            )
        raise KeyError(model_type)

    def keys(self) -> set[str]:
        return set(self._get_lazy_map().keys()) | set(self._loaded.keys())

    def _safe_get(self, key: str) -> PipelineConfig | None:
        try:
            return self[key]
        except Exception:
            logger.warning("Skipping pipeline %r because it failed to load.", key)
        return None

    def values(self):
        # Iterating forces a lazy import for each pipeline; failures are logged and skipped.
        for key in self.keys():
            if (p := self._safe_get(key)) is not None:
                yield p

    def items(self):
        for key in self.keys():
            if (p := self._safe_get(key)) is not None:
                yield key, p

    def __iter__(self):
        return iter(self.keys())


_PIPELINE_REGISTRY = _LazyPipelineRegistry()


def register_pipeline(pipeline: PipelineConfig) -> None:
    """Register a pipeline config dynamically.

    In-tree pipelines are declared in ``pipeline_registry._OMNI_PIPELINES``
    and loaded lazily; calling ``register_pipeline`` is only needed for
    out-of-tree plugins or tests that build a ``PipelineConfig`` at runtime.
    A dynamic registration overrides the central-registry entry with the same
    ``model_type``.
    """
    errors = pipeline.validate()
    if errors:
        logger.warning("Pipeline %s has issues: %s", pipeline.model_type, errors)
    _PIPELINE_REGISTRY[pipeline.model_type] = pipeline


_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"


@dataclass
class StageDeployConfig:
    """Per-stage deployment knobs.

    Only fields whose value legitimately varies across stages of the same
    pipeline live here (e.g. ``max_num_seqs`` on thinker vs talker,
    ``devices`` for GPU placement). Pipeline-wide settings
    (``trust_remote_code``, ``distributed_executor_backend``, ``dtype``,
    ``quantization``, prefix/chunked prefill, DP/PP sizes) are declared at
    the top level of ``DeployConfig`` and propagated to every stage.
    """

    # === Omni fields ===
    # Stage identity and Omni runtime placement.
    stage_id: int
    devices: str | None = None
    num_replicas: int = 1

    # Inter-stage connector wiring and request defaults.
    output_connectors: dict[str, str] | None = None
    input_connectors: dict[str, str] | None = None
    default_sampling_params: dict[str, Any] | None = None
    subtalker_sampling_params: dict[str, Any] | None = None

    # === vLLM EngineArgs fields ===
    # Parallelism and scheduler/memory capacity.
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None

    # Execution, scheduling, and KV/cache behavior.
    enforce_eager: bool | None = None
    async_scheduling: bool | None = None
    disable_hybrid_kv_cache_manager: bool | None = None
    mm_processor_cache_gb: float | None = None

    # Compilation, profiling, tokenizer/config parsing, and model loading.
    compilation_config: dict[str, Any] | None = None
    profiler_config: dict[str, Any] | None = None
    skip_mm_profiling: bool | None = None
    enable_flashinfer_autotune: bool | None = None
    config_format: str | None = None
    load_format: str | None = None
    tokenizer_mode: str | None = None

    # Pass-through vLLM EngineArgs fields that are not represented above.
    engine_extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployConfig:
    """Loaded from deploy/<model>.yaml — the only config file users edit.

    Top-level fields (``trust_remote_code``, ``distributed_executor_backend``,
    ``dtype``, ``quantization``, ``enable_prefix_caching``,
    ``enable_chunked_prefill``, ``data_parallel_size``,
    ``pipeline_parallel_size``) are pipeline-wide: they apply uniformly to
    every stage. Fields that legitimately vary per stage live in the
    individual ``StageDeployConfig`` entries under ``stages:``.
    """

    async_chunk: bool = True
    connectors: dict[str, Any] | None = None
    edges: list[dict[str, Any]] | None = None
    stages: list[StageDeployConfig] = field(default_factory=list)
    platforms: dict[str, Any] | None = None
    # Overrides the auto-detected pipeline registry key for structural variants.
    pipeline: str | None = None

    # === Pipeline-wide engine settings (applied uniformly to every stage) ===
    trust_remote_code: bool | None = None
    distributed_executor_backend: str | None = None
    dtype: str | None = None
    quantization: str | None = None
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    data_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None


_STAGE_RESERVED_KEYS = frozenset(
    {
        "stage_id",
        "devices",
        "num_replicas",
        "output_connectors",
        "input_connectors",
        "default_sampling_params",
        "engine_extras",
        "engine_args",
        "runtime",
    }
)

# Fields on StageDeployConfig that are populated from engine_args dict
_STAGE_DEPLOY_FIELDS = {f.name: f for f in fields(StageDeployConfig) if f.name not in _STAGE_RESERVED_KEYS}


def _parse_stage_deploy(stage_data: dict[str, Any]) -> StageDeployConfig:
    """Parse a single stage entry from deploy YAML into StageDeployConfig."""
    # Get the non-reserved keys for this stage
    flat_args = {k: v for k, v in stage_data.items() if k not in _STAGE_RESERVED_KEYS}
    runtime_cfg = dict(stage_data.get("runtime", {}))
    devices = runtime_cfg.get("devices", stage_data.get("devices"))
    num_replicas = runtime_cfg.get("num_replicas", stage_data.get("num_replicas", 1))

    if "engine_args" in stage_data:
        for k, v in stage_data["engine_args"].items():
            existing = flat_args.get(k)
            # If we have multiple dictionaries, merge recursively.
            if isinstance(v, dict) and isinstance(existing, dict):
                flat_args[k] = _get_recursively_merged_dict(existing, v)
            else:
                flat_args[k] = v

    kwargs: dict[str, Any] = {
        "stage_id": stage_data["stage_id"],
        "devices": devices,
        "num_replicas": int(num_replicas),
    }
    for name, f in _STAGE_DEPLOY_FIELDS.items():
        if name in flat_args:
            kwargs[name] = flat_args.pop(name)

    kwargs["output_connectors"] = stage_data.get("output_connectors")
    kwargs["input_connectors"] = stage_data.get("input_connectors")
    kwargs["default_sampling_params"] = stage_data.get("default_sampling_params")
    kwargs["engine_extras"] = flat_args
    return StageDeployConfig(**kwargs)


_DEEP_MERGE_KEYS = frozenset({"default_sampling_params", "subtalker_sampling_params", "engine_extras", "engine_args"})


def _deep_merge_stage(base: dict, overlay: dict) -> dict:
    """Deep-merge ``_DEEP_MERGE_KEYS`` so thin overlays don't drop base keys."""
    # Deep merge _DEEP_MERGE_KEYS recursively
    base_merge_dict = {k: v for k, v in base.items() if k in _DEEP_MERGE_KEYS}
    overlay_merge_dict = {k: v for k, v in overlay.items() if k in _DEEP_MERGE_KEYS}

    # Get the merge dict; priority is base < overlay < merged sub
    merged_subdict = _get_recursively_merged_dict(original=base_merge_dict, update=overlay_merge_dict)
    merged_dict = {**base, **overlay, **merged_subdict}
    return merged_dict


def _get_recursively_merged_dict(original: dict, update: dict) -> dict:
    """Recursively merge two dicts, returning a new dict."""
    merged = original.copy()
    for k, update_v in update.items():
        orig_v = merged.get(k)
        if isinstance(orig_v, dict) and isinstance(update_v, dict):
            merged[k] = _get_recursively_merged_dict(orig_v, update_v)
        else:
            if orig_v is not None and (isinstance(orig_v, dict) != isinstance(update_v, dict)):
                logger.warning(
                    "Deep-merge key %r has non-dict value (base=%s, overlay=%s); "
                    "overlay will fully replace base instead of merging.",
                    k,
                    type(orig_v).__name__,
                    type(update_v).__name__,
                )

            merged[k] = update_v
    return merged


def _merge_stage_lists(
    base_stages: list[dict[str, Any]] | None,
    overlay_stages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge two ``stages:`` lists by ``stage_id`` (overlay wins per field)."""
    by_id: dict[int, dict[str, Any]] = {s["stage_id"]: s for s in (base_stages or [])}
    for overlay_stage in overlay_stages or []:
        sid = overlay_stage["stage_id"]
        if sid in by_id:
            by_id[sid] = _deep_merge_stage(by_id[sid], overlay_stage)
        else:
            by_id[sid] = overlay_stage
    return list(by_id.values())


def _merge_platforms(
    base: dict[str, Any] | None,
    overlay: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Deep-merge two ``platforms:`` blocks per-platform, per-stage_id."""
    if not base and not overlay:
        return None
    base = base or {}
    overlay = overlay or {}
    merged: dict[str, Any] = {}
    for plat in set(base) | set(overlay):
        bp = base.get(plat) or {}
        op = overlay.get(plat) or {}
        merged_plat = {**bp, **{k: v for k, v in op.items() if k != "stages"}}
        merged_plat["stages"] = _merge_stage_lists(bp.get("stages"), op.get("stages"))
        merged[plat] = merged_plat
    return merged


def resolve_deploy_yaml(path: str | Path) -> dict[str, Any]:
    """Load a deploy YAML with optional ``base_config`` inheritance."""
    raw_dict = to_dict(load_yaml_config(path))

    base_path = raw_dict.pop("base_config", None)
    if base_path is None:
        return raw_dict

    # Resolve relative to the overlay file's directory
    base_path = Path(path).parent / base_path
    base_dict = resolve_deploy_yaml(base_path)

    # Merge top-level scalars: overlay wins. ``stages:`` and ``platforms:``
    # are deep-merged below so an overlay can layer on top of the base.
    merged = {
        **base_dict,
        **{k: v for k, v in raw_dict.items() if k not in ("stages", "platforms")},
    }
    merged["stages"] = _merge_stage_lists(base_dict.get("stages"), raw_dict.get("stages"))
    merged_platforms = _merge_platforms(base_dict.get("platforms"), raw_dict.get("platforms"))
    if merged_platforms is not None:
        merged["platforms"] = merged_platforms

    return merged


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a deploy YAML (with optional base_config inheritance)."""
    raw_dict = resolve_deploy_yaml(path)

    stages = [_parse_stage_deploy(s) for s in raw_dict.get("stages", [])]

    kwargs: dict[str, Any] = {
        "async_chunk": raw_dict.get("async_chunk", True),
        "connectors": raw_dict.get("connectors", None),
        "edges": raw_dict.get("edges", None),
        "stages": stages,
        "platforms": raw_dict.get("platforms", None),
        "pipeline": raw_dict.get("pipeline", None),
    }
    # Pipeline-wide engine settings: only set if explicitly present in YAML
    # so the DeployConfig dataclass defaults take effect otherwise.
    for name in (
        "trust_remote_code",
        "distributed_executor_backend",
        "dtype",
        "quantization",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "data_parallel_size",
        "pipeline_parallel_size",
    ):
        if name in raw_dict:
            kwargs[name] = raw_dict[name]
    return DeployConfig(**kwargs)


# Pipeline-wide DeployConfig fields that are propagated to every stage's
# engine args during merge. These live at top level of the deploy YAML.
_PIPELINE_WIDE_ENGINE_FIELDS: tuple[str, ...] = (
    "trust_remote_code",
    "distributed_executor_backend",
    "dtype",
    "quantization",
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "data_parallel_size",
    "pipeline_parallel_size",
)


def _auto_detect_model_type(model: str, trust_remote_code: bool = True) -> tuple[str | None, Any]:
    """Auto-detect model_type from model directory.

    Args:
        model: Model name or path.
        trust_remote_code: Whether to trust remote code for HF config loading.

    Returns:
        Tuple of (model_type, hf_config). Both may be None on failure.
    """
    try:
        from vllm.transformers_utils.config import get_config

        hf_config = get_config(model, trust_remote_code=trust_remote_code)
        return hf_config.model_type, hf_config
    except Exception as e:
        logger.debug("`get_config` failed for %s; Falling back to raw config.json path", e)

    # Fallback: read config.json directly for custom model types that
    # are not registered with transformers (e.g. qwen3_tts).
    try:
        from vllm.transformers_utils.config import get_hf_file_to_dict

        config_dict = get_hf_file_to_dict("config.json", model, revision=None)
        if config_dict:
            if "model_type" in config_dict:
                return config_dict["model_type"], None
            # VoxCPM2-style configs use singular ``architecture`` rather
            # than HF's standard ``model_type`` / ``architectures``. Accept
            # it as a fallback so the pipeline registry can still match.
            if "architecture" in config_dict and isinstance(config_dict["architecture"], str):
                return config_dict["architecture"], None
    except Exception as e:
        logger.debug("Failed to auto-detect model type for %s: %s", model, e)

    # Fallback for diffusers-style models: check model_index.json.
    # Some models (e.g. GLM-Image) have no root config.json but ship a
    # model_index.json with _class_name that maps to a pipeline key via
    # PipelineConfig.diffusers_class_name.
    try:
        from vllm.transformers_utils.config import get_hf_file_to_dict

        model_index = get_hf_file_to_dict("model_index.json", model, revision=None)
        if model_index and "_class_name" in model_index:
            class_name = model_index["_class_name"]
            for pipeline_cfg in _PIPELINE_REGISTRY.values():
                if pipeline_cfg.diffusers_class_name == class_name:
                    logger.info(
                        "Detected pipeline %r from model_index.json (_class_name=%r)",
                        pipeline_cfg.model_type,
                        class_name,
                    )
                    return pipeline_cfg.model_type, None
    except Exception as e:
        logger.debug("Failed to detect model type for diffusers-style models: %s", e)

    # Final fallback: some models (e.g. CosyVoice3) ship an empty
    # config.json and rely on naming conventions. Match the model path
    # basename against registered pipeline keys — longest match wins
    # so "cosyvoice3" (length 10) beats "cosyvoice" (length 9).
    model_lower = model.lower().replace("-", "").replace("_", "")
    best: str | None = None
    best_len = 0
    for registered_key in _PIPELINE_REGISTRY.keys():
        candidate = registered_key.lower().replace("-", "").replace("_", "")
        if candidate and candidate in model_lower and len(candidate) > best_len:
            best = registered_key
            best_len = len(candidate)
    if best is not None:
        return best, None

    return None, None
