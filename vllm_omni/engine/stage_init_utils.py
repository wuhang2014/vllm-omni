"""
Stage initialization helpers for vLLM-Omni multi-stage runtime.

Extracts orchestration-level init logic (config extraction, plugin loading,
multiprocessing setup, device mapping, device locking, engine args building)
out of StageEngineCoreClient into reusable functions.
"""

from __future__ import annotations

import fcntl
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.executor import Executor

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.stage_utils import _to_dict, set_stage_devices
from vllm_omni.entrypoints.utils import resolve_model_config_path
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.config.vllm_omni_config import StageResolvedConfig

logger = init_logger(__name__)


@dataclass
class ReplicaInitPlan:
    """One concrete replica startup unit within a logical stage."""

    replica_id: int
    num_replicas: int
    launch_mode: str
    stage_cfg: Any
    stage_connector_spec: dict[str, Any]
    omni_kv_connector: tuple[dict[str, Any] | None, str | None, str | None]
    stage_vllm_config: Any | None = None
    executor_class: type | None = None


@dataclass
class LogicalStageInitPlan:
    """Startup plan for one logical stage."""

    stage_idx: int
    configured_stage_id: int
    replicas: list[ReplicaInitPlan]


@dataclass
class StageRemoteFactoryContext:
    """Per-stage context cached by AsyncOmniEngine for dynamic replica attach.

    Populated once during ``_bootstrap_orchestrator`` from the per-stage
    init plans. ``_build_remote_replica`` consumes it to construct the
    right head-side stage client when a headless replica registers.
    """

    stage_id: int
    stage_type: str
    stage_cfg: Any
    # LLM-only fields:
    vllm_config: Any | None = None
    executor_class: type | None = None
    # Diffusion-only fields:
    diffusion_batch_size: int = 1


def capture_stage_factory_contexts(
    stage_plans: Sequence[LogicalStageInitPlan],
    diffusion_batch_size: int,
) -> dict[int, StageRemoteFactoryContext]:
    """Snapshot per-stage construction context for dynamic replica attach.

    Called once after ``_initialize_stages`` finishes. The captured
    context holds everything ``_build_remote_replica`` needs to build a
    fresh head-side client when a new headless replica registers
    (vllm_config / executor_class for LLM, batch_size for diffusion,
    plus the base stage metadata).

    Per-replica fields like ``replica_id`` are filled in at build time,
    not at capture time.
    """
    contexts: dict[int, StageRemoteFactoryContext] = {}
    for plan in stage_plans:
        if not plan.replicas:
            # Stage was declared but has zero replicas locally; we still
            # want to be able to attach incoming headless ones, so use
            # the stage_cfg-derived context if any replica plan exists.
            continue
        template = plan.replicas[0]
        stage_id = int(plan.configured_stage_id)
        stage_type = getattr(template.stage_cfg, "stage_type", "llm")
        contexts[stage_id] = StageRemoteFactoryContext(
            stage_id=stage_id,
            stage_type=stage_type,
            stage_cfg=template.stage_cfg,
            vllm_config=template.stage_vllm_config,
            executor_class=template.executor_class,
            diffusion_batch_size=diffusion_batch_size,
        )
    return contexts


def _resolve_model_to_local_path(model: str) -> str:
    """Resolve an HF Hub model ID to a local cache path."""
    if os.path.isdir(model):
        return model

    try:
        from huggingface_hub import snapshot_download

        # Keep init path resolution offline-friendly.
        return snapshot_download(model, local_files_only=True)
    except Exception:
        logger.warning(
            "[stage_init] Could not resolve %s to local snapshot; using as-is",
            model,
        )
        return model


def _resolve_model_tokenizer_paths(model: str, engine_args: dict[str, Any]) -> str:
    """Apply model_subdir/tokenizer_subdir indirections from stage engine args."""
    model_subdir = engine_args.pop("model_subdir", None)
    tokenizer_subdir = engine_args.pop("tokenizer_subdir", None)
    if model_subdir is None and tokenizer_subdir is None:
        return model

    resolved_base = _resolve_model_to_local_path(model)

    if model_subdir:
        model = os.path.join(resolved_base, model_subdir)
        logger.info("[stage_init] Using model subdirectory: %s", model)

    if tokenizer_subdir is not None:
        tokenizer_path = os.path.join(resolved_base, tokenizer_subdir) if tokenizer_subdir else resolved_base
        engine_args["tokenizer"] = tokenizer_path
        logger.info("[stage_init] Using tokenizer from: %s", tokenizer_path)
    elif model_subdir and "tokenizer" not in engine_args:
        # Keep legacy behavior: model in subdir, tokenizer defaults to base path.
        engine_args["tokenizer"] = resolved_base
        logger.info("[stage_init] Using tokenizer from base model path: %s", resolved_base)

    return model


def apply_cli_tokenizer(
    engine_args: dict[str, Any],
    *,
    cli_tokenizer: str | None,
    stage_defines_tokenizer: bool,
) -> None:
    """Forward CLI tokenizer unless the stage config defines its own."""
    if cli_tokenizer is None or stage_defines_tokenizer:
        return
    engine_args["tokenizer"] = cli_tokenizer


def terminate_alive_proc(proc, timeout=5):
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.kill()


def patch_generation_config_if_needed(model_config: Any) -> None:
    """Guard InputProcessor init for models whose config lacks model_type."""
    try:
        model_config.try_get_generation_config()
    except Exception:
        model_config.try_get_generation_config = lambda: {}


def resolve_worker_cls(engine_args: dict[str, Any]) -> None:
    """Resolve worker_cls from worker_type for non-diffusion stages."""
    worker_type = engine_args.get("worker_type", None)
    if not worker_type:
        return
    worker_cls = engine_args.get("worker_cls")
    if worker_cls is not None and worker_cls != "auto":
        return

    worker_type = str(worker_type).lower()
    if worker_type == "ar":
        engine_args["worker_cls"] = current_omni_platform.get_omni_ar_worker_cls()
    elif worker_type == "generation":
        engine_args["worker_cls"] = current_omni_platform.get_omni_generation_worker_cls()
    else:
        raise ValueError(f"Unknown worker_type: {worker_type}")


def _get_attr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from *obj* regardless of whether it's a dict or object."""
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tp_size_for_stage(stage_configs: Sequence[Any], stage_id: Any) -> int | None:
    """Resolve tensor_parallel_size for *stage_id* from the loaded stage configs."""
    id_strs = {str(stage_id)}
    try:
        id_strs.add(str(int(stage_id)))
    except (TypeError, ValueError):
        pass

    for stage_cfg in stage_configs:
        if str(getattr(stage_cfg, "stage_id", None)) not in id_strs:
            continue
        engine_args = getattr(stage_cfg, "engine_args", None)
        if engine_args is None:
            return 1
        parallel_config = _get_attr_or_item(engine_args, "parallel_config")
        if parallel_config is not None:
            tp = _get_attr_or_item(parallel_config, "tensor_parallel_size", 1)
        else:
            tp = _get_attr_or_item(engine_args, "tensor_parallel_size", 1)
        try:
            return max(1, int(tp))
        except (TypeError, ValueError):
            return 1
    return None


def _inject_inferred_kv_tp_topology(
    omni_kv: Any,
    stage_id: int,
    stage_configs: Sequence[Any],
    engine_input_source: Sequence[int] | None = None,
) -> None:
    """Infer adjacent-stage TP topology and inject it into omni_kv_config.

    This keeps heterogeneous TP working without requiring user-authored
    rank_mapping blocks in config files.
    """
    if omni_kv is None:
        return

    if hasattr(omni_kv, "get"):
        need_send = bool(omni_kv.get("need_send_cache", False))
        need_recv = bool(omni_kv.get("need_recv_cache", False))
        omni_from_stage = omni_kv.get("omni_from_stage")
        omni_to_stage = omni_kv.get("omni_to_stage")
        rank_mapping = omni_kv.get("rank_mapping")
    else:
        need_send = bool(getattr(omni_kv, "need_send_cache", False))
        need_recv = bool(getattr(omni_kv, "need_recv_cache", False))
        omni_from_stage = getattr(omni_kv, "omni_from_stage", None)
        omni_to_stage = getattr(omni_kv, "omni_to_stage", None)
        rank_mapping = getattr(omni_kv, "rank_mapping", None)

    if not need_send and not need_recv:
        return

    current_tp = _tp_size_for_stage(stage_configs, stage_id)
    if current_tp is None:
        return

    peer_stage_id = None
    from_tp = None
    to_tp = None
    if str(omni_from_stage) == str(stage_id):
        peer_stage_id = omni_to_stage
        from_tp = current_tp
        to_tp = _tp_size_for_stage(stage_configs, peer_stage_id)
    elif str(omni_to_stage) == str(stage_id):
        peer_stage_id = omni_from_stage
        from_tp = _tp_size_for_stage(stage_configs, peer_stage_id)
        to_tp = current_tp
    elif need_recv and engine_input_source:
        peer_stage_id = engine_input_source[0]
        from_tp = _tp_size_for_stage(stage_configs, peer_stage_id)
        to_tp = current_tp

    if from_tp is None or to_tp is None:
        return

    if not isinstance(rank_mapping, dict):
        rank_mapping = {}
    rank_mapping.setdefault("from_tp", int(from_tp))
    rank_mapping.setdefault("to_tp", int(to_tp))

    if hasattr(omni_kv, "__setitem__"):
        omni_kv["rank_mapping"] = rank_mapping
    else:
        setattr(omni_kv, "rank_mapping", rank_mapping)


def inject_kv_stage_info(stage_cfg: Any, stage_id: int, stage_configs: Sequence[Any] | None = None) -> None:
    """Inject stage_id, engine_input_source, and inferred TP topology into omni_kv_config.

    When *stage_configs* is provided, also infers from_tp/to_tp for
    heterogeneous TP topologies so the KV transfer manager can compute
    rank mappings automatically.
    """
    try:
        engine_args = stage_cfg.engine_args
        if hasattr(engine_args, "get"):
            omni_kv = engine_args.get("omni_kv_config", None)
        else:
            omni_kv = getattr(engine_args, "omni_kv_config", None)

        if omni_kv is None:
            return

        if hasattr(omni_kv, "setdefault"):
            omni_kv.setdefault("stage_id", stage_id)
        elif hasattr(omni_kv, "__setitem__"):
            if "stage_id" not in omni_kv:
                omni_kv["stage_id"] = stage_id

        engine_input_source = getattr(stage_cfg, "engine_input_source", None)
        if engine_input_source is not None:
            if hasattr(omni_kv, "setdefault"):
                omni_kv.setdefault("engine_input_source", list(engine_input_source))
            elif hasattr(omni_kv, "__setitem__") and "engine_input_source" not in omni_kv:
                omni_kv["engine_input_source"] = list(engine_input_source)

        if stage_configs:
            _inject_inferred_kv_tp_topology(
                omni_kv,
                stage_id=stage_id,
                stage_configs=stage_configs,
                engine_input_source=engine_input_source,
            )
    except Exception as e:
        logger.debug("Failed to inject stage info into omni_kv_config: %s", e)


def prepare_engine_environment() -> None:
    """One-time global setup: load plugins, set multiprocessing spawn method."""
    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("[stage_init] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def split_devices_for_replicas(
    devices_str: str | None,
    num_replicas: int,
    tp_size: int,
    stage_id: int,
) -> list[str]:
    """Split a devices string into per-replica subsets.

    When ``num_replicas`` is 1, returns ``[devices_str]`` unchanged.
    Otherwise, two YAML shapes are accepted:

    1. **Legacy / pool mode** — ``len(devices) == num_replicas * tp_size``:
       the string enumerates the full per-stage pool. Each replica gets
       ``tp_size`` consecutive entries. The values are logical indices
       into the launcher's ``CUDA_VISIBLE_DEVICES``.

       ``split_devices_for_replicas("1,2,3,4", 2, 2, 1) → ["1,2", "3,4"]``

    2. **Template mode** — ``len(devices) == tp_size``: the YAML declares
       a single per-replica template (the same shape one replica would
       use), and is **dp-independent**. Each replica r gets the offsets
       ``[r*tp_size + a for a in template]`` of the launcher's
       ``CUDA_VISIBLE_DEVICES``. The template's entries must lie in
       ``[0, tp_size)``.

       ``split_devices_for_replicas("0,1", 2, 2, 1) → ["0,1", "2,3"]``
       ``split_devices_for_replicas("0,1", 4, 2, 1) → ["0,1", "2,3", "4,5", "6,7"]``

       This lets the same ``devices: "0,1"`` YAML work for any
       ``--omni-dp-size-local``: the launcher's CVD scales, the YAML
       does not.

    Any other length raises ``ValueError`` (the two modes are
    length-disjoint for ``num_replicas > 1``).
    """
    if num_replicas <= 1 or devices_str is None:
        return [devices_str] if devices_str is not None else [devices_str]

    device_list = [d.strip() for d in devices_str.split(",") if d.strip()]

    if len(device_list) == num_replicas * tp_size:
        return [",".join(device_list[r * tp_size : (r + 1) * tp_size]) for r in range(num_replicas)]

    if len(device_list) == tp_size:
        try:
            offsets = [int(a) for a in device_list]
        except ValueError as e:
            raise ValueError(f"Stage {stage_id}: template-mode devices must be ints, got {devices_str!r}") from e
        bad = [a for a in offsets if not (0 <= a < tp_size)]
        if bad:
            raise ValueError(
                f"Stage {stage_id}: template-mode device offset(s) {bad} "
                f"out of range [0, {tp_size}); devices={devices_str!r}"
            )
        return [",".join(str(r * tp_size + a) for a in offsets) for r in range(num_replicas)]

    raise ValueError(
        f"Stage {stage_id}: devices={devices_str!r} has {len(device_list)} id(s); "
        f"need either {tp_size} (template, dp-independent) or "
        f"{num_replicas * tp_size} (pool / legacy). "
        f"num_replicas={num_replicas}, tensor_parallel_size={tp_size}."
    )


def get_stage_tp_size(stage_cfg: Any) -> int:
    """Extract tensor_parallel_size from a stage config object."""
    engine_args = getattr(stage_cfg, "engine_args", {})
    if hasattr(engine_args, "get"):
        return int(engine_args.get("tensor_parallel_size", 1) or 1)
    return int(getattr(engine_args, "tensor_parallel_size", 1) or 1)


def get_stage_devices_per_replica(stage_cfg: Any) -> int:
    """Return the number of devices consumed by one replica of *stage_cfg*."""
    if getattr(stage_cfg, "stage_type", "llm") != "diffusion":
        return get_stage_tp_size(stage_cfg)

    parallel_config = _get_attr_or_item(getattr(stage_cfg, "engine_args", {}), "parallel_config")
    if parallel_config is None:
        return 1

    world_size = _get_attr_or_item(parallel_config, "world_size")
    if world_size is not None:
        return max(1, int(world_size))

    try:
        from vllm_omni.diffusion.data import DiffusionParallelConfig

        return max(1, int(DiffusionParallelConfig.from_dict(_to_dict(parallel_config)).world_size))
    except Exception:
        return 1


def compute_replica_layout(
    stage_configs: Sequence[Any],
    *,
    allow_zero: bool = False,
) -> tuple[list[int], dict[int, list[str]]]:
    """Compute per-stage replica counts and device assignments.

    Args:
        stage_configs: per-stage config objects with a ``runtime`` sub-config
            exposing ``num_replicas`` and ``devices``.
        allow_zero: when True, ``num_replicas == 0`` is honored (used by
            single-stage / head-distributed mode for non-self stages that
            will be filled dynamically by remote registrations); when False
            (default), the count is clamped to at least 1.

    Returns:
        replicas_per_stage: num_replicas per logical stage.
        replica_devices_map: stage_idx -> per-replica device strings
            (only for stages with num_replicas > 1).
    """
    replicas_per_stage: list[int] = []
    for stage_cfg in stage_configs:
        runtime_cfg = getattr(stage_cfg, "runtime", {})
        num_replicas = int(
            runtime_cfg.get("num_replicas", 1)
            if hasattr(runtime_cfg, "get")
            else getattr(runtime_cfg, "num_replicas", 1)
        )
        if num_replicas < 0:
            raise ValueError(f"num_replicas must be >= 0, got {num_replicas}")
        replicas_per_stage.append(num_replicas if allow_zero else max(1, num_replicas))

    replica_devices_map: dict[int, list[str]] = {}
    for stage_id, stage_cfg in enumerate(stage_configs):
        num_replicas = replicas_per_stage[stage_id]
        if num_replicas <= 1:
            continue
        runtime_cfg = getattr(stage_cfg, "runtime", {})
        devices_str = (
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        )
        devices_per_replica = get_stage_devices_per_replica(stage_cfg)
        replica_devices_map[stage_id] = split_devices_for_replicas(
            devices_str,
            num_replicas,
            devices_per_replica,
            stage_id,
        )
        logger.info(
            "[stage_init] Stage %s: %d replicas, devices_per_replica=%d, devices split: %s",
            stage_id,
            num_replicas,
            devices_per_replica,
            replica_devices_map[stage_id],
        )

    return replicas_per_stage, replica_devices_map


def setup_stage_devices(stage_id: int, runtime_cfg: Any) -> None:
    """Device mapping via set_stage_devices for a single stage."""
    physical_devices = set_stage_devices(
        stage_id,
        runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else None,
    )
    # Only log if we actually set the env vars in the stage
    if physical_devices:
        logger.info(
            "[stage_init] Stage-%s set runtime devices: %s",
            stage_id,
            physical_devices,
        )


def build_engine_args_dict(
    stage_config: Any,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
    cli_tokenizer: str | None = None,
) -> dict[str, Any]:
    """Build the normalized engine args dict for one stage."""
    engine_args = stage_config.engine_args
    # HACK (Alex) Tensor parallel size should not be passed as None;
    # remove it if this is the case so that we fall back to default
    # creation from vLLM's engine args.
    # NOTE: This will be fixed more generically in ongoing work for engine arg filtering.
    if "tensor_parallel_size" in engine_args and engine_args["tensor_parallel_size"] is None:
        del engine_args["tensor_parallel_size"]

    stage_type = getattr(stage_config, "stage_type", "llm")
    stage_id = stage_config.stage_id

    engine_args_dict = _to_dict(engine_args)
    stage_defines_tokenizer = (
        engine_args_dict.get("tokenizer") is not None or engine_args_dict.get("tokenizer_subdir") is not None
    )
    model = _resolve_model_tokenizer_paths(model, engine_args_dict)
    apply_cli_tokenizer(
        engine_args_dict,
        cli_tokenizer=cli_tokenizer,
        stage_defines_tokenizer=stage_defines_tokenizer,
    )
    engine_args_dict["model"] = model
    # Stage id must come from stage config instead of inherited CLI kwargs
    # (e.g. `--stage-id` defaulting to None).
    engine_args_dict["stage_id"] = stage_id
    if engine_args_dict.get("async_chunk", False):
        engine_args_dict["stage_connector_spec"] = dict(stage_connector_spec or {})

    if stage_type == "diffusion":
        from vllm_omni.diffusion.data import parse_attention_config

        if engine_args_dict.get("diffusion_attention_config") is not None:
            engine_args_dict["diffusion_attention_config"] = parse_attention_config(
                engine_args_dict.get("diffusion_attention_config"),
            )

    if stage_type != "diffusion":
        resolve_worker_cls(engine_args_dict)

    if engine_args_dict.get("worker_type") == "generation":
        # Non-AR generation stages (e.g. code2wav) do not benefit from
        # prefix caching and can expose hybrid KV-cache layouts that vLLM's
        # prefix-cache coordinator does not handle.
        engine_args_dict.setdefault("disable_hybrid_kv_cache_manager", True)
        engine_args_dict.setdefault("enable_prefix_caching", False)

    # Check whether the stage's default_sampling_params defines extra_args.
    default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
    engine_args_dict["has_sampling_extra_args"] = bool(default_sp.get("extra_args"))

    return engine_args_dict


def build_vllm_config_from_engine_args(
    omni_engine_args: OmniEngineArgs,
    *,
    headless: bool = False,
) -> tuple[Any, type]:
    """Create VllmConfig and executor class from already-built OmniEngineArgs.

    The caller is responsible for building the per-stage ``OmniEngineArgs``
    (including model/tokenizer resolution, connector spec injection, etc.).
    """
    if omni_engine_args.max_model_len is not None and not os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        logger.debug(
            "Auto-set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for stage %s (max_model_len=%s).",
            omni_engine_args.stage_id,
            omni_engine_args.max_model_len,
        )

    # Resolve worker_cls (dc.replace skips __post_init__ so _prepare_hf_config
    # auto-detection never ran on per-stage engine args).
    if omni_engine_args.worker_cls is None:
        if omni_engine_args.worker_type == "generation":
            omni_engine_args.worker_cls = current_omni_platform.get_omni_generation_worker_cls()
        else:
            # Default to AR worker (covers worker_type=None or worker_type="ar").
            omni_engine_args.worker_cls = current_omni_platform.get_omni_ar_worker_cls()

    # Resolve hf_overrides["architectures"] (dc.replace skips __post_init__
    # so _prepare_hf_config never ran on per-stage engine args).
    if omni_engine_args.model_arch:
        if omni_engine_args.hf_overrides is None:
            omni_engine_args.hf_overrides = {}
        if isinstance(omni_engine_args.hf_overrides, dict):
            omni_engine_args.hf_overrides.setdefault("architectures", [omni_engine_args.model_arch])

    vllm_config = omni_engine_args.create_engine_config(
        usage_context=UsageContext.LLM_CLASS,
        headless=headless,
    )
    executor_class = Executor.get_class(vllm_config)

    from vllm_omni.quantization.inc_config import OmniINCConfig

    upgraded = OmniINCConfig.maybe_upgrade(vllm_config.quant_config)
    if upgraded is not vllm_config.quant_config:
        vllm_config = replace(vllm_config, quant_config=upgraded)

    return vllm_config, executor_class


def build_llm_stage_output_processor(plan: LogicalStageInitPlan, stage_vllm_config: Any) -> Any | None:
    """Build one output processor per logical LLM stage."""

    stage_cfg = plan.replicas[0].stage_cfg
    if stage_vllm_config.model_config.skip_tokenizer_init:
        tokenizer = None
    else:
        tokenizer = cached_tokenizer_from_config(
            model_config=stage_vllm_config.model_config,
        )
    return MultimodalOutputProcessor(
        tokenizer=tokenizer,
        log_stats=False,
        engine_core_output_type=stage_cfg.engine_output_type,
    )


def build_stage0_input_processor(stage_vllm_config: Any) -> InputProcessor:
    """Build the shared stage-0 input processor."""
    from vllm_omni.inputs.preprocess import OmniInputPreprocessor

    patch_generation_config_if_needed(stage_vllm_config.model_config)
    input_processor = InputProcessor(vllm_config=stage_vllm_config)
    input_processor.input_preprocessor = OmniInputPreprocessor(
        vllm_config=stage_vllm_config,
        renderer=input_processor.renderer,
    )
    return input_processor


def acquire_device_locks(
    stage_id: int,
    engine_args_dict_or_config: dict[str, Any] | Any,
    stage_init_timeout: int,
) -> list[int]:
    """Acquire exclusive file locks on devices needed by this stage.

    Returns list of lock file descriptors that must be released after init.
    """
    lock_fds: list[int] = []
    d = engine_args_dict_or_config
    try:
        # Accept VllmConfig (has parallel_config) or plain dict.
        if hasattr(d, "parallel_config"):
            pc = d.parallel_config
            tensor_parallel_size = getattr(pc, "tensor_parallel_size", 1)
            pipeline_parallel_size = getattr(pc, "pipeline_parallel_size", 1)
            data_parallel_size = getattr(pc, "data_parallel_size", 1)
            prefill_context_parallel_size = getattr(pc, "prefill_context_parallel_size", 1)
            sequence_parallel_size = getattr(pc, "sequence_parallel_size", 1)
            cfg_parallel_size = getattr(pc, "cfg_parallel_size", 1)
        elif isinstance(d, dict) and "parallel_config" in d:
            pc = d["parallel_config"]
            tensor_parallel_size = pc.get("tensor_parallel_size", 1)
            pipeline_parallel_size = pc.get("pipeline_parallel_size", 1)
            data_parallel_size = pc.get("data_parallel_size", 1)
            prefill_context_parallel_size = pc.get("prefill_context_parallel_size", 1)
            sequence_parallel_size = pc.get("sequence_parallel_size", 1)
            cfg_parallel_size = pc.get("cfg_parallel_size", 1)
        elif isinstance(d, dict):
            tensor_parallel_size = d.get("tensor_parallel_size", 1)
            pipeline_parallel_size = d.get("pipeline_parallel_size", 1)
            data_parallel_size = d.get("data_parallel_size", 1)
            prefill_context_parallel_size = d.get("prefill_context_parallel_size", 1)
            sequence_parallel_size = 1
            cfg_parallel_size = 1
        else:
            tensor_parallel_size = 1
            pipeline_parallel_size = 1
            data_parallel_size = 1
            prefill_context_parallel_size = 1
            sequence_parallel_size = 1
            cfg_parallel_size = 1

        num_devices_per_stage = (
            tensor_parallel_size
            * pipeline_parallel_size
            * data_parallel_size
            * prefill_context_parallel_size
            * sequence_parallel_size
            * cfg_parallel_size
        )

        # Get physical device IDs
        device_control_env = current_omni_platform.device_control_env_var
        visible_devices_str = os.environ.get(device_control_env)
        physical_devices: list[int] = []

        if visible_devices_str:
            try:
                physical_devices = [int(x.strip()) for x in visible_devices_str.split(",") if x.strip()]
            except (ValueError, IndexError):
                pass

        if not physical_devices:
            num_devices = current_omni_platform.get_device_count()
            physical_devices = list(range(num_devices))

        if len(physical_devices) < num_devices_per_stage:
            raise RuntimeError(
                f"Stage {stage_id} requires {num_devices_per_stage} device(s) based on parallel_config, "
                f"but only {len(physical_devices)} device(s) are available: {physical_devices}"
            )

        num_devices_to_lock = num_devices_per_stage
        devices_to_lock = sorted(physical_devices[:num_devices_to_lock])

        logger.debug(
            "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d, CFG=%d; will lock %d devices: %s",
            tensor_parallel_size,
            pipeline_parallel_size,
            data_parallel_size,
            prefill_context_parallel_size,
            sequence_parallel_size,
            cfg_parallel_size,
            num_devices_to_lock,
            devices_to_lock,
        )

        # Acquire locks
        wait_start = time.time()
        for device_id in devices_to_lock:
            lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
            lock_acquired = False

            while not lock_acquired:
                try:
                    lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        os.ftruncate(lock_fd, 0)
                        os.write(lock_fd, f"{os.getpid()}\n".encode())
                        os.fsync(lock_fd)
                        lock_acquired = True
                        lock_fds.append(lock_fd)
                        logger.debug("Acquired exclusive lock for device %s", device_id)
                    except BlockingIOError:
                        os.close(lock_fd)
                        if time.time() - wait_start > stage_init_timeout:
                            logger.warning(
                                "Timeout waiting for device %s initialization lock, proceeding anyway",
                                device_id,
                            )
                            break
                        time.sleep(0.01)
                except OSError as e:
                    logger.debug(
                        "Failed to acquire lock for device %s: %s, continuing anyway",
                        device_id,
                        e,
                    )
                    try:
                        os.close(lock_fd)
                    except (OSError, NameError):
                        pass
                    break

    except Exception as e:
        logger.debug(
            "[Stage-%s] Failed to set up sequential initialization lock: %s",
            stage_id,
            e,
        )

    return lock_fds


def release_device_locks(lock_fds: list[int]) -> None:
    """Release file locks acquired by acquire_device_locks."""
    for lock_fd in lock_fds:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            logger.debug("Released initialization lock (fd=%s)", lock_fd)
        except (OSError, ValueError):
            pass


def acquire_diffusion_device_locks(
    stage_id: int,
    od_config: Any,
    stage_init_timeout: int,
) -> list[int]:
    """Acquire init locks for the GPU set used by a diffusion stage.

    Diffusion stages express their device count via ``OmniDiffusionConfig``'s
    ``parallel_config.world_size`` rather than the LLM-style
    ``tensor_parallel_size`` knob, so adapt to the shape that
    ``acquire_device_locks`` understands.
    """
    parallel_config = getattr(od_config, "parallel_config", None)
    world_size = getattr(parallel_config, "world_size", 1)
    try:
        world_size = max(1, int(world_size))
    except (TypeError, ValueError):
        world_size = 1

    return acquire_device_locks(
        stage_id,
        {"tensor_parallel_size": world_size},
        stage_init_timeout,
    )


def load_omni_transfer_config_for_model(model: str, config_path: str | None) -> Any:
    """Load omni transfer config from an explicit path or resolved model config.

    Resolves ``base_config`` inheritance (CI overlay → base deploy YAML) so
    that connectors defined in the base config are visible to the transfer
    config parser.
    """
    from vllm_omni.distributed.omni_connectors import load_omni_transfer_config

    try:
        resolved_config_path = config_path or resolve_model_config_path(model)
        if resolved_config_path is None:
            return None
        from vllm_omni.config.stage_config import resolve_deploy_yaml

        resolved_dict = resolve_deploy_yaml(resolved_config_path)
        return load_omni_transfer_config(config_dict=resolved_dict)
    except Exception as e:
        logger.warning("[stage_init] Failed to load transfer config: %s", e)
        return None


def get_stage_connector_spec(
    omni_transfer_config: Any,
    stage_id: int,
    async_chunk: bool,
) -> dict[str, Any]:
    """Return the first connector spec for the stage when async chunking is enabled."""
    from vllm_omni.distributed.omni_connectors import get_stage_connector_config

    if not async_chunk:
        return {}

    stage_connectors_cfg = get_stage_connector_config(omni_transfer_config, stage_id)
    for cfg in stage_connectors_cfg.values():
        return dict(cfg.get("spec", {}))
    return {}


def build_diffusion_config_from_fields(
    fields: dict[str, Any],
    stage_id: int,
    model: str,
    *,
    cfg_kv_collect_func: Callable | None = None,
) -> Any:
    """Build ``OmniDiffusionConfig`` directly from a fields dict.

    Avoids ``_PerStageCfg`` and ``build_engine_args_dict`` — the caller
    has already assembled the required fields.
    """
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    # Resolve model/tokenizer paths before construction.
    model = _resolve_model_tokenizer_paths(model, fields)
    fields["model"] = model

    od_config = OmniDiffusionConfig.from_kwargs(**fields)

    num_devices_per_stage = od_config.parallel_config.world_size
    device_control_env = current_omni_platform.device_control_env_var
    visible_devices_str = os.environ.get(device_control_env) if device_control_env else None
    if visible_devices_str:
        physical_devices = [d.strip() for d in visible_devices_str.split(",") if d.strip()]
    else:
        physical_devices = list(range(current_omni_platform.get_device_count()))

    if len(physical_devices) < num_devices_per_stage:
        raise ValueError(
            f"Stage {stage_id} requires {num_devices_per_stage} device(s) "
            f"based on parallel_config, but {len(physical_devices)} device(s) "
            f"are available: {physical_devices}"
        )

    od_config.num_gpus = num_devices_per_stage
    if cfg_kv_collect_func is not None:
        od_config.cfg_kv_collect_func = cfg_kv_collect_func
    return od_config


def build_diffusion_config(
    model: str,
    stage_cfg: Any,
) -> Any:
    """Build diffusion config for a stage."""

    engine_args_dict = build_engine_args_dict(stage_cfg, model)
    od_config = OmniDiffusionConfig.from_kwargs(**engine_args_dict)

    num_devices_per_stage = od_config.parallel_config.world_size
    device_control_env = current_omni_platform.device_control_env_var
    visible_devices_str = os.environ.get(device_control_env) if device_control_env else None
    if visible_devices_str:
        physical_devices = [device.strip() for device in visible_devices_str.split(",") if device.strip()]
    else:
        physical_devices = list(range(current_omni_platform.get_device_count()))

    if len(physical_devices) < num_devices_per_stage:
        raise ValueError(
            f"Stage {stage_cfg.stage_id} requires {num_devices_per_stage} device(s) based on parallel_config, "
            f"but {len(physical_devices)} device(s) are available: {physical_devices}"
        )

    od_config.num_gpus = num_devices_per_stage
    cfg_func = getattr(stage_cfg, "cfg_kv_collect_func", None)
    if cfg_func is not None:
        od_config.cfg_kv_collect_func = cfg_func
    return od_config


def initialize_diffusion_stage(
    stage_id: int,
    model: str,
    stage_cfg: Any,
    stage_init_timeout: int,
    batch_size: int = 1,
    use_inline: bool = False,
    replica_id: int = 0,
) -> Any:
    """Build a diffusion stage client.

    If *stage_cfg* is a ``StageResolvedConfig`` (pre-built), uses the
    pre-built ``diffusion_config`` directly.  Otherwise falls back to
    ``build_diffusion_config``.
    """
    from vllm_omni.diffusion.stage_diffusion_client import create_diffusion_client

    od_config = getattr(stage_cfg, "diffusion_config", None)
    if od_config is None:
        od_config = build_diffusion_config(model, stage_cfg)
    return create_diffusion_client(
        model,
        od_config,
        stage_cfg,
        stage_init_timeout,
        batch_size,
        use_inline,
        replica_id=replica_id,
    )


# ---------------------------------------------------------------------------
# Headless helpers — shared between serve.py:run_headless and future head paths
# ---------------------------------------------------------------------------


def _get_devices_per_replica_from_resolved(stage: StageResolvedConfig) -> int:
    """Return the number of devices consumed by one replica of *stage*.

    Replaces ``get_stage_devices_per_replica()`` for the ``StageResolvedConfig``
    case — reads ``tensor_parallel_size`` from ``VllmConfig`` (LLM) or
    ``world_size`` from ``OmniDiffusionConfig.parallel_config`` (diffusion).
    """
    if stage.stage_type != "diffusion":
        if stage.vllm_config is not None:
            return stage.vllm_config.parallel_config.tensor_parallel_size
        return 1
    if stage.diffusion_config is not None:
        return max(1, stage.diffusion_config.parallel_config.world_size)
    return 1


def compute_per_replica_devices(
    stage: StageResolvedConfig,
    num_replicas: int,
    stage_id: int,
) -> list[str | None]:
    """Compute per-replica device strings for a stage.

    Reads ``runtime.devices`` from *stage*.  Returns a list of device strings
    (or ``None`` per replica when no split is needed).
    """
    runtime = stage.runtime or {}
    devices_str: str | None = runtime.get("devices") if isinstance(runtime, dict) else None
    devices_per_replica = _get_devices_per_replica_from_resolved(stage)
    if devices_str:
        return split_devices_for_replicas(devices_str, num_replicas, devices_per_replica, stage_id)
    return [None] * num_replicas


@contextmanager
def replica_device_context(stage_id: int, device_str: str | None) -> Iterator[None]:
    """Context manager that sets CUDA_VISIBLE_DEVICES per replica and restores after.

    Usage::

        with replica_device_context(stage_id, per_replica_devices[rep_idx]):
            # spawn / launch replica
    """
    device_control_env = current_omni_platform.device_control_env_var
    previous = os.environ.get(device_control_env)
    try:
        if device_str is not None:
            setup_stage_devices(stage_id, {"devices": device_str})
        yield
    finally:
        if previous is None:
            current_omni_platform.unset_device_control_env_var()
        else:
            current_omni_platform.set_device_control_env_var(previous)


def launch_headless_diffusion_replicas(
    *,
    model: str,
    stage: StageResolvedConfig,
    stage_id: int,
    omni_dp_size_local: int,
    omni_master_address: str,
    omni_master_port: int,
    omni_replica_address: str | None,
    stage_init_timeout: int,
    per_replica_devices: list[str | None],
) -> None:
    """Launch one or more diffusion replicas in headless mode.

    Blocks until a replica exits (sentinel wait).  Handles per-replica device
    assignment, master-server registration (auto-assigned replica ids), port
    settlement for multi-replica launches, process spawn + handshake, and
    cleanup on exit.
    """
    from multiprocessing import connection

    from vllm_omni.diffusion.stage_diffusion_proc import (
        complete_diffusion_handshake,
        spawn_diffusion_proc,
    )
    from vllm_omni.engine.stage_engine_startup import register_stage_with_omni_master

    od_config = stage.diffusion_config
    if od_config is None:
        raise ValueError(f"Diffusion stage {stage_id} has no diffusion_config")

    logger.info(
        "[Headless] Launching %d diffusion replica(s) for stage %d via OmniMasterServer at %s:%d",
        omni_dp_size_local,
        stage_id,
        omni_master_address,
        omni_master_port,
    )

    procs: list[Any] = []
    try:
        for rep_idx in range(omni_dp_size_local):
            # Auto-assign replica id via master server.
            response = register_stage_with_omni_master(
                omni_master_address=omni_master_address,
                omni_master_port=omni_master_port,
                omni_stage_id=stage_id,
                omni_stage_config=stage,
                replica_id=None,
                return_full_response=True,
                replica_bind_address=omni_replica_address,
            )

            with replica_device_context(stage_id, per_replica_devices[rep_idx]):
                # Multi-replica port settlement: avoid collisions when
                # torch.distributed master ports are allocated from the
                # same base.
                if omni_dp_size_local > 1:
                    od_config.master_port = od_config.settle_port(
                        61000 + rep_idx * 100,
                        port_inc=37,
                    )
                proc, _, _, _ = spawn_diffusion_proc(
                    model,
                    od_config,
                    handshake_address=response.handshake_address,
                    request_address=response.input_address,
                    response_address=response.output_address,
                    omni_coordinator_address=response.coordinator_router_address,
                    omni_stage_id=stage_id,
                    omni_replica_id=response.replica_id,
                )

            complete_diffusion_handshake(proc, response.handshake_address, stage_init_timeout)
            procs.append(proc)
            logger.info(
                "[Headless] Diffusion replica id=%d for stage %d is up (coord=%s)",
                response.replica_id,
                stage_id,
                response.coordinator_router_address,
            )

        # Block on the sentinel set so any replica crash is detected
        # immediately.  Any exit triggers fleet shutdown; non-zero exits
        # propagate.
        sentinel_to_proc = {p.sentinel: p for p in procs}
        died = connection.wait(list(sentinel_to_proc.keys()))
        first = sentinel_to_proc[died[0]]
        logger.info(
            "[Headless] Diffusion replica %s exited (code=%s); shutting down stage %d.",
            first.name,
            first.exitcode,
            stage_id,
        )
        if first.exitcode not in (None, 0):
            raise RuntimeError(f"Diffusion stage {stage_id} replica {first.name!r} exited with code {first.exitcode}")
    finally:
        logger.info(
            "[Headless] Shutting down %d diffusion replica(s) for stage %d.",
            len(procs),
            stage_id,
        )
        for p in procs:
            if p.is_alive():
                terminate_alive_proc(p)


def launch_headless_llm_replicas(
    *,
    stage: StageResolvedConfig,
    stage_id: int,
    omni_dp_size_local: int,
    omni_master_address: str,
    omni_master_port: int,
    omni_replica_address: str | None,
    log_stats: bool,
    disable_log_stats: bool,
    per_replica_devices: list[str | None],
) -> None:
    """Launch one or more LLM replicas in headless mode.

    Handles DP Coordinator creation, per-replica device assignment,
    master-server registration (auto-assigned replica ids),
    ``OmniCoreEngineProcManager`` spawn, multi-replica liveness monitoring,
    and cleanup on exit.
    """
    import signal
    import threading
    from types import FrameType

    from vllm.v1.engine.coordinator import DPCoordinator

    from vllm_omni.engine.omni_core_engine_proc_manager import OmniCoreEngineProcManager
    from vllm_omni.engine.stage_engine_startup import register_stage_with_omni_master

    vllm_config = stage.vllm_config
    executor_class = stage.executor_class
    if vllm_config is None:
        raise ValueError(f"Stage {stage_id} is LLM type but has no vllm_config")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local
    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    # Signal handling for graceful shutdown.
    shutdown_requested = False

    def _signal_handler(signum: int, frame: FrameType | None) -> None:
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # DP Coordinator (if needed).
    dp_rank = parallel_config.data_parallel_rank if parallel_config.data_parallel_rank is not None else 0
    coordinator: DPCoordinator | None = None
    if vllm_config.needs_dp_coordinator and dp_rank == 0:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )
        logger.info(
            "[Headless] Started DP Coordinator process for stage %d (PID: %d)",
            stage_id,
            coordinator.proc.pid,
        )

    if disable_log_stats:
        log_stats = False

    logger.info(
        "[Headless] Launching %d omni replica(s) (vLLM dp_size_local=%d each) "
        "for stage %d via OmniMasterServer at %s:%d",
        omni_dp_size_local,
        local_engine_count,
        stage_id,
        omni_master_address,
        omni_master_port,
    )

    engine_managers: list[Any] = []
    monitor_threads: list[threading.Thread] = []

    def _monitor_target(mgr: Any) -> None:
        try:
            mgr.monitor_engine_liveness()
        except Exception:
            logger.exception("[Headless] monitor_engine_liveness raised")

    try:
        for rep_idx in range(omni_dp_size_local):
            response = register_stage_with_omni_master(
                omni_master_address=omni_master_address,
                omni_master_port=omni_master_port,
                omni_stage_id=stage_id,
                omni_stage_config=stage,
                coordinator=coordinator,
                replica_id=None,
                return_full_response=True,
                replica_bind_address=omni_replica_address,
                replica_binds_sockets=False,
            )

            with replica_device_context(stage_id, per_replica_devices[rep_idx]):
                mgr = OmniCoreEngineProcManager(
                    local_engine_count=local_engine_count,
                    start_index=dp_rank,
                    local_start_index=0,
                    vllm_config=vllm_config,
                    local_client=False,
                    handshake_address=response.handshake_address,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    omni_stage_id=stage_id,
                    omni_coordinator_address=response.coordinator_router_address,
                    omni_replica_base_id=response.replica_id,
                )

            engine_managers.append(mgr)
            logger.info(
                "[Headless] Stage %d replica id=%d up (coord=%s)",
                stage_id,
                response.replica_id,
                response.coordinator_router_address,
            )

        # Liveness monitoring — blocks until all subprocesses exit.
        if len(engine_managers) == 1:
            engine_managers[0].monitor_engine_liveness()
        else:
            for mgr in engine_managers:
                t = threading.Thread(
                    target=_monitor_target,
                    args=(mgr,),
                    name=f"omni-replica-monitor-{id(mgr):x}",
                )
                t.start()
                monitor_threads.append(t)
            for t in monitor_threads:
                t.join()
    finally:
        logger.info(
            "[Headless] Shutting down stage %d (%d managers).",
            stage_id,
            len(engine_managers),
        )
        for mgr in engine_managers:
            try:
                mgr.shutdown()
            except Exception:
                logger.exception("[Headless] engine manager shutdown failed")
        if coordinator is not None:
            coordinator.shutdown()
