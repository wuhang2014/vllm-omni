from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    _PIPELINE_WIDE_ENGINE_FIELDS,
)
from vllm_omni.config.stage_config import (
    StageDeployConfig as _StageDeployConfig,
)
from vllm_omni.config.stage_config import (
    StageExecutionType as _StageExecutionType,
)
from vllm_omni.config.stage_config import (
    StagePipelineConfig as _StagePipelineConfig,
)
from vllm_omni.engine.output_modality import OutputModality
from vllm_omni.platforms import current_omni_platform
from vllm_omni.plugins import load_omni_general_plugins

logger = init_logger(__name__)

# Maps model architecture names to their HuggingFace model_type values.
# Used when auto-injecting hf_overrides for models with missing config.json.
_ARCH_TO_MODEL_TYPE: dict[str, str] = {
    "CosyVoice3Model": "cosyvoice3",
    "OmniVoiceModel": "omnivoice",
    "VoxCPM2TalkerForConditionalGeneration": "voxcpm2",
}

# Maps model architecture names to tokenizer subfolder paths within HF repos.
_TOKENIZER_SUBFOLDER_MAP: dict[str, str] = {
    "CosyVoice3Model": "CosyVoice-BlankEN",
}


def _register_omni_hf_configs() -> None:
    try:
        from transformers import AutoConfig

        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
    except Exception as exc:  # pragma: no cover - best-effort optional registration
        logger.warning("Skipping omni HF config registration due to import error: %s", exc)
        return

    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except ImportError:
        _CONFIG_REGISTRY = None

    for model_type, config_cls in [
        ("qwen3_tts", Qwen3TTSConfig),
    ]:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            pass
        if _CONFIG_REGISTRY is not None and model_type not in _CONFIG_REGISTRY:
            _CONFIG_REGISTRY[model_type] = config_cls


def register_omni_models_to_vllm():
    from vllm.model_executor.models import ModelRegistry

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    _register_omni_hf_configs()

    supported_archs = ModelRegistry.get_supported_archs()
    for arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items():
        if arch not in supported_archs:
            ModelRegistry.register_model(arch, f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}:{cls_name}")


# ============================================================================
# OmniEngineArgs — consolidated
# ============================================================================


@dataclass
class OmniEngineArgs(EngineArgs):
    """Unified engine arguments for vLLM-Omni.

    Absorbs all omni-specific fields from what were previously
    ``OrchestratorArgs``, ``SHARED_FIELDS``, and the old
    ``OmniEngineArgs`` into a single dataclass.

    Online:  ``OmniEngineArgs.from_cli_args(parsed_args)``
    Offline: ``OmniEngineArgs.from_kwargs(model, **kwargs)``

    The resolved ``VllmOmniConfig`` is produced by
    :meth:`create_omni_config`.
    """

    # ── Mode ─────────────────────────────────────────────────────

    omni: bool = False
    headless: bool = False

    # ── Orchestrator lifecycle ───────────────────────────────────

    stage_init_timeout: int = 300
    init_timeout: int = 600

    # ── Cross-stage communication ────────────────────────────────

    shm_threshold_bytes: int = 65536
    batch_timeout: int = 10
    async_chunk: bool | None = None

    # ── Backend ──────────────────────────────────────────────────

    worker_backend: str = "multi_process"
    ray_address: str | None = None

    # ── Config files ─────────────────────────────────────────────

    deploy_config: str | None = None
    stage_configs_path: str | None = None
    stage_overrides: str | None = None

    # ── Headless ─────────────────────────────────────────────────

    stage_id: int | None = None
    replica_id: int = 0
    omni_master_address: str | None = None
    omni_master_port: int | None = None
    omni_replica_address: str | None = None

    # ── OmniCoordinator ──────────────────────────────────────────

    omni_dp_size_local: int = 1
    omni_lb_policy: str = "random"
    omni_heartbeat_timeout: float = 30.0

    # ── Observability ────────────────────────────────────────────

    log_stats: bool = False
    log_file: str | None = None
    enable_diffusion_pipeline_profiler: bool = False
    enable_ar_profiler: bool = False

    # ── Output ───────────────────────────────────────────────────

    output_modalities: list[str] | None = None
    diffusion_batch_size: int = 1
    tts_batch_max_items: int = 32

    # ── Diffusion model ──────────────────────────────────────────

    num_gpus: int | None = None
    model_class_name: str | None = None
    diffusion_load_format: str | None = None
    diffusers_load_kwargs: dict[str, Any] | None = None
    diffusers_call_kwargs: dict[str, Any] | None = None

    # ── Diffusion parallelism ────────────────────────────────────

    ulysses_degree: int | None = None
    ulysses_mode: str = "strict"
    ring_degree: int | None = None
    cfg_parallel_size: int = 1
    vae_patch_parallel_size: int = 1
    use_hsdp: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = 1

    # ── Diffusion attention ──────────────────────────────────────

    diffusion_attention_backend: str | None = None
    diffusion_attention_config: dict[str, Any] | None = None

    # ── Diffusion cache ──────────────────────────────────────────

    cache_backend: str = "none"
    cache_config: str | None = None
    enable_cache_dit_summary: bool = False

    # ── Diffusion execution ──────────────────────────────────────

    step_execution: bool = False
    boundary_ratio: float | None = None
    flow_shift: float | None = None

    # ── VAE ──────────────────────────────────────────────────────

    vae_use_slicing: bool = False
    vae_use_tiling: bool = False

    # ── Weight loading ───────────────────────────────────────────

    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = 4

    # ── Offloading ───────────────────────────────────────────────

    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False

    # ── KV cache ─────────────────────────────────────────────────

    diffusion_kv_cache_dtype: str | None = None
    diffusion_kv_cache_skip_steps: str | None = None
    diffusion_kv_cache_skip_layers: str | None = None

    # ── Quantisation ─────────────────────────────────────────────

    quantization_config: Any | None = None
    force_cutlass_fp8: bool | None = None

    # ── LoRA ─────────────────────────────────────────────────────

    lora_path: str | None = None
    lora_scale: float | None = None

    # ── Sampling ─────────────────────────────────────────────────

    default_sampling_params: str | None = None
    max_generated_image_size: int | None = None
    tts_max_instructions_length: int | None = None

    # ── Per-stage forwarded ──────────────────────────────────────

    enable_sleep_mode: bool = False
    task_type: str | None = None
    model_stage: str = "thinker"
    model_arch: str | None = None
    engine_output_type: str | None = None
    hf_config_name: str | None = None
    custom_process_next_stage_input_func: str | None = None
    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    subtalker_sampling_params: dict[str, Any] | None = None
    omni_kv_config: dict[str, Any] | None = None
    worker_type: str | None = None
    worker_cls: str | None = None
    custom_pipeline_args: dict[str, Any] | None = None
    has_sampling_extra_args: bool = False

    parallel_config: Any = None
    tokenizer: str | None = None

    # ================================================================
    # CLI registration
    # ================================================================

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
        *,
        omni_args_only: bool = False,
    ) -> argparse.ArgumentParser:
        """Register all omni CLI flags in an ``OmniConfig`` argument group.

        Delegates to ``EngineArgs.add_cli_args`` first for all upstream
        vLLM flags, then adds omni-specific flags.  Pass
        ``omni_args_only=True`` when the parser already has upstream
        vLLM flags registered (e.g. via ``make_arg_parser``).
        """
        if not omni_args_only:
            parser = EngineArgs.add_cli_args(parser)

        group = parser.add_argument_group(
            title="OmniConfig",
            description="Configuration for vLLM-Omni multi-stage and diffusion models.",
        )

        # ── Mode switches ──
        try:
            group.add_argument("--omni", action="store_true", default=False, help="Enable vLLM-Omni mode.")
        except argparse.ArgumentError:
            pass
        try:
            group.add_argument(
                "--headless", action="store_true", default=False, help="Run a single stage in headless mode."
            )
        except argparse.ArgumentError:
            pass

        # ── Orchestrator lifecycle ──
        group.add_argument(
            "--stage-init-timeout", type=int, default=300, help="Timeout for single stage init (seconds)."
        )
        group.add_argument(
            "--init-timeout", type=int, default=600, help="Total orchestrator startup timeout (seconds)."
        )

        # ── Cross-stage communication ──
        group.add_argument("--shm-threshold-bytes", type=int, default=65536, help="Shared memory threshold.")
        group.add_argument("--batch-timeout", type=int, default=10, help="Batch collection timeout (seconds).")
        group.add_argument(
            "--async-chunk",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Override deploy YAML async_chunk setting.",
        )

        # ── Backend ──
        group.add_argument("--worker-backend", type=str, default="multi_process", choices=["multi_process", "ray"])
        group.add_argument("--ray-address", type=str, default=None)

        # ── Config files ──
        group.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="[Deprecated] Path to legacy stage_configs YAML. Use --deploy-config instead.",
        )
        group.add_argument(
            "--deploy-config",
            type=str,
            default=None,
            help="Path to deploy YAML (new format with stages).",
        )
        group.add_argument(
            "--stage-overrides",
            type=str,
            default=None,
            help='Per-stage JSON overrides. Example: \'{"0": {"gpu_memory_utilization": 0.8}}\'.',
        )

        # ── Headless / single-stage ──
        group.add_argument("--stage-id", type=int, default=None, help="Select a single stage by stage_id.")
        group.add_argument("--replica-id", type=int, default=0, help="Replica id for headless stage registration.")
        group.add_argument("--omni-master-address", type=str, default=None, help="Omni orchestrator hostname/IP.")
        group.add_argument("--omni-master-port", type=int, default=None, help="Omni orchestrator port.")
        group.add_argument("--omni-replica-address", type=str, default=None)

        # ── OmniCoordinator ──
        group.add_argument("--omni-dp-size-local", type=int, default=1)
        group.add_argument("--omni-lb-policy", type=str, default="random")
        group.add_argument("--omni-heartbeat-timeout", type=float, default=30.0)

        # ── Observability ──
        group.add_argument(
            "--log-stats", action="store_true", default=False, help="Enable per-request pipeline metrics logging."
        )
        group.add_argument("--log-file", type=str, default=None, help="Path to the log file.")
        group.add_argument("--enable-diffusion-pipeline-profiler", action="store_true", default=False)

        # ── Output ──
        group.add_argument("--output-modalities", type=str, nargs="*", default=None)
        group.add_argument("--diffusion-batch-size", type=int, default=1)
        group.add_argument("--tts-batch-max-items", type=int, default=32)

        # ── Diffusion model ──
        group.add_argument("--num-gpus", type=int, default=None)
        group.add_argument("--model-class-name", type=str, default=None)
        group.add_argument("--diffusion-load-format", type=str, default=None)
        group.add_argument("--diffusers-load-kwargs", type=str, default=None)
        group.add_argument("--diffusers-call-kwargs", type=str, default=None)

        # ── Diffusion parallelism ──
        group.add_argument("--ulysses-degree", type=int, default=None)
        group.add_argument("--ulysses-mode", type=str, default="strict")
        group.add_argument("--ring-degree", type=int, default=None)
        group.add_argument("--cfg-parallel-size", type=int, default=1)
        group.add_argument("--vae-patch-parallel-size", type=int, default=1)
        group.add_argument("--use-hsdp", action="store_true", default=False)
        group.add_argument("--hsdp-shard-size", type=int, default=-1)
        group.add_argument("--hsdp-replicate-size", type=int, default=1)

        # ── Diffusion attention ──
        group.add_argument("--diffusion-attention-backend", type=str, default=None)
        group.add_argument("--diffusion-attention-config", type=str, default=None)

        # ── Diffusion cache ──
        group.add_argument("--cache-backend", type=str, default="none")
        group.add_argument("--cache-config", type=str, default=None)
        group.add_argument("--enable-cache-dit-summary", action="store_true", default=False)

        # ── Diffusion execution ──
        group.add_argument("--step-execution", action="store_true", default=False)
        group.add_argument("--boundary-ratio", type=float, default=None)
        group.add_argument("--flow-shift", type=float, default=None)

        # ── VAE ──
        group.add_argument("--vae-use-slicing", action="store_true", default=False)
        group.add_argument("--vae-use-tiling", action="store_true", default=False)

        # ── Weight loading ──
        group.add_argument("--enable-multithread-weight-load", action="store_true", default=True)
        group.add_argument("--num-weight-load-threads", type=int, default=4)

        # ── Offloading ──
        group.add_argument("--enable-cpu-offload", action="store_true", default=False)
        group.add_argument("--enable-layerwise-offload", action="store_true", default=False)

        # ── KV cache ──
        group.add_argument("--diffusion-kv-cache-dtype", type=str, default=None)
        group.add_argument("--diffusion-kv-cache-skip-steps", type=str, default=None)
        group.add_argument("--diffusion-kv-cache-skip-layers", type=str, default=None)

        # ── Quantisation ──
        group.add_argument("--quantization-config", type=str, default=None)
        group.add_argument("--force-cutlass-fp8", type=bool, default=None)

        # ── LoRA ──
        group.add_argument("--lora-path", type=str, default=None)
        group.add_argument("--lora-scale", type=float, default=None)

        # ── Sampling ──
        group.add_argument("--default-sampling-params", type=str, default=None)
        group.add_argument("--max-generated-image-size", type=int, default=None)
        group.add_argument("--tts-max-instructions-length", type=int, default=None)

        # ── Per-stage / TTS ──
        try:
            group.add_argument("--enable-sleep-mode", action="store_true", default=False)
        except argparse.ArgumentError:
            pass
        group.add_argument("--task-type", type=str, default=None)
        group.add_argument("--enable-ar-profiler", action="store_true", default=False)

        return parser

    # ================================================================
    # Construction
    # ================================================================

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> OmniEngineArgs:
        """Build from a parsed argparse Namespace.

        Simple extraction — ``OmniArgumentParser`` already injected
        YAML defaults as ``action.default`` before parse, so no
        explicit-field tracking is needed.
        """
        attrs = [attr.name for attr in _dc.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})

    @classmethod
    def from_kwargs(cls, model: str, **kwargs: Any) -> OmniEngineArgs:
        """Build from raw keyword arguments (offline path).

        Calls ``_inject_deploy_defaults`` to fill YAML defaults via
        ``kwargs.setdefault()`` before construction.
        """
        # Defer import to avoid circular dependency.
        from vllm_omni.entrypoints.omni_base import _inject_deploy_defaults

        _inject_deploy_defaults(model, kwargs)
        # Build from kwargs, letting OmniEngineArgs defaults fill in the rest.
        engine_fields = {f.name for f in _dc.fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in engine_fields})

    def create_diffusion_config(self) -> Any:
        """Build ``OmniDiffusionConfig`` directly from ``OmniEngineArgs`` fields.

        Delegates to :meth:`OmniDiffusionConfig.from_engine_args` which
        auto-maps fields by name — no manual field lists to maintain.
        """
        from vllm_omni.diffusion.data import OmniDiffusionConfig

        return OmniDiffusionConfig.from_engine_args(self)

    def _build_single_diffusion_omni_config(self, model: str) -> VllmOmniConfig:
        """Build a single-stage diffusion ``VllmOmniConfig`` from this engine args.

        Called when no ``stage_overrides`` are present (no pipeline registered
        for the model).  Constructs ``OmniDiffusionConfig`` directly from the
        diffusion-specific fields on ``self``, then wraps it in a single-stage
        ``VllmOmniConfig``.
        """
        from vllm_omni.config.vllm_omni_config import StageResolvedConfig
        from vllm_omni.platforms import current_omni_platform

        worker_type = self.worker_type
        if worker_type in ("ar", "generation"):
            return VllmOmniConfig(model=model)

        # Build OmniDiffusionConfig directly from engine args fields.
        od_config = self.create_diffusion_config()
        od_config.model = model

        # Validate device count.
        num_devices_per_stage = od_config.parallel_config.world_size
        device_control_env = current_omni_platform.device_control_env_var
        visible_devices_str = os.environ.get(device_control_env) if device_control_env else None
        if visible_devices_str:
            physical_devices = [d.strip() for d in visible_devices_str.split(",") if d.strip()]
        else:
            physical_devices = list(range(current_omni_platform.get_device_count()))
        if len(physical_devices) < num_devices_per_stage:
            raise ValueError(
                f"Diffusion stage requires {num_devices_per_stage} device(s) "
                f"based on parallel_config, but {len(physical_devices)} device(s) "
                f"are available: {physical_devices}"
            )
        od_config.num_gpus = num_devices_per_stage

        return VllmOmniConfig(
            model=model,
            stages=(
                StageResolvedConfig(
                    stage_id=0,
                    stage_type="diffusion",
                    diffusion_config=od_config,
                    engine_output_type=None,
                    final_output=True,
                    num_replicas=1,
                ),
            ),
            diffusion_config=od_config,
        )

    def create_omni_config(
        self,
        model: str,
    ) -> VllmOmniConfig:
        """Build a resolved ``VllmOmniConfig`` from this engine args instance.

        All values are read from ``self`` — no further merge needed.
        ``OmniArgumentParser`` (online) or ``_inject_deploy_defaults`` (offline)
        already resolved all defaults before this method is called.
        """
        from vllm_omni.config.vllm_omni_config import (
            _detect_pd_config,
            _parse_stage_overrides,
            _resolve_stages,
        )
        from vllm_omni.engine.stage_init_utils import load_omni_transfer_config_for_model

        # 1. Parse stage_overrides (already merged by OmniArgumentParser / _inject_deploy_defaults).
        stage_overrides = _parse_stage_overrides(self.stage_overrides)

        # 2. Handle no-pipeline case → single diffusion stage.
        if not stage_overrides:
            return self._build_single_diffusion_omni_config(model)

        # 3. Resolve async_chunk: CLI > deploy YAML > default.
        async_chunk = bool(self.async_chunk)

        # 4. Load omni transfer config.
        omni_transfer_config = load_omni_transfer_config_for_model(model, self.deploy_config or self.stage_configs_path)

        # 5. Build per-stage resolved configs.
        resolved_stages, top_level_diffusion_config, prompt_expand_func = _resolve_stages(
            model=model,
            stage_overrides=stage_overrides,
            engine_args=self,
            async_chunk=async_chunk,
            omni_transfer_config=omni_transfer_config,
        )

        # 6. Detect PD disaggregation.
        pd_config = _detect_pd_config(resolved_stages)

        # 7. Assemble and return.
        return VllmOmniConfig(
            model=model,
            stages=tuple(resolved_stages),
            diffusion_config=top_level_diffusion_config,
            async_chunk=async_chunk,
            stage_init_timeout=self.stage_init_timeout,
            init_timeout=self.init_timeout,
            shm_threshold_bytes=self.shm_threshold_bytes,
            batch_timeout=self.batch_timeout,
            worker_backend=self.worker_backend,
            log_stats=self.log_stats,
            pd_config=pd_config,
            omni_transfer_config=omni_transfer_config,
            prompt_expand_func=prompt_expand_func,
        )

    def _ensure_omni_models_registered(self):
        if hasattr(self, "_omni_models_registered"):
            return True
        register_omni_models_to_vllm()
        self._omni_models_registered = True
        return True

    def _prepare_hf_config(self) -> None:
        """Prepare HF config overrides and auto-detect tokenizer.

        Runs in ``__post_init__`` so that models with empty config.json
        (e.g. CosyVoice3) are patched before vLLM tries to load them.
        """
        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])
                if "model_type" not in self.hf_overrides:
                    model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                    if model_type is not None:
                        self.hf_overrides.setdefault("model_type", model_type)
                if self.model_arch == "Qwen3TTSCode2Wav" and self.max_model_len is not None:
                    self.hf_overrides.setdefault("talker_config", {}).setdefault(
                        "max_position_embeddings", int(self.max_model_len)
                    )
            if not self.hf_config_path:
                model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                if model_type is not None:
                    self._patch_empty_hf_config(model_type)

        # Auto-detect tokenizer subdirectory.
        if not self.tokenizer and self.model:
            if os.path.isdir(self.model) and not os.path.isfile(os.path.join(self.model, "tokenizer_config.json")):
                for subfolder in sorted(os.listdir(self.model)):
                    candidate = os.path.join(self.model, subfolder)
                    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "tokenizer_config.json")):
                        self.tokenizer = candidate
                        logger.info("Auto-detected tokenizer at %s", candidate)
                        break
            elif not os.path.isdir(self.model):
                subfolder = _TOKENIZER_SUBFOLDER_MAP.get(self.model_arch)
                if subfolder:
                    try:
                        from huggingface_hub import snapshot_download

                        local_dir = snapshot_download(
                            self.model,
                            allow_patterns=[
                                f"{subfolder}/tokenizer*",
                                f"{subfolder}/special_tokens*",
                                f"{subfolder}/vocab*",
                                f"{subfolder}/merges*",
                                f"{subfolder}/added_tokens*",
                            ],
                        )
                        candidate = os.path.join(local_dir, subfolder)
                        if os.path.isdir(candidate):
                            self.tokenizer = candidate
                            logger.info("Downloaded tokenizer from %s/%s", self.model, subfolder)
                    except Exception as e:
                        logger.warning("Failed to download tokenizer subfolder: %s", e)

    def _patch_empty_hf_config(self, model_type: str) -> None:
        """For models with empty config.json (e.g. CosyVoice3), create a
        patched config in a temp directory with model_type set."""
        try:
            from transformers import PretrainedConfig

            config_dict, _ = PretrainedConfig.get_config_dict(self.model)
            if config_dict.get("model_type"):
                return
        except Exception:
            return

        temp_dir = tempfile.mkdtemp(prefix="omni_hf_config_")
        config_dict["model_type"] = model_type
        config_dict.setdefault("architectures", [self.model_arch])
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        self.hf_config_path = temp_dir
        self._temp_config_dir = temp_dir
        logger.info("Patched empty HF config with model_type=%s at %s", model_type, temp_dir)

    def __post_init__(self) -> None:
        if self.worker_cls is None:
            if self.worker_type == "ar":
                self.worker_cls = current_omni_platform.get_omni_ar_worker_cls()
            elif self.worker_type == "generation":
                self.worker_cls = current_omni_platform.get_omni_generation_worker_cls()
        load_omni_general_plugins()
        self._ensure_omni_models_registered()
        self._prepare_hf_config()
        super().__post_init__()


# ============================================================================
# OmniAsyncEngineArgs
# ============================================================================


@dataclass
class OmniAsyncEngineArgs(AsyncEngineArgs, OmniEngineArgs):
    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = AsyncEngineArgs.add_cli_args(parser)
        parser = OmniEngineArgs.add_cli_args(parser, omni_args_only=True)
        return parser

    @property
    def output_modality(self) -> OutputModality:
        return OutputModality.from_string(self.engine_output_type)


# ============================================================================
# OmniArgumentParser
# ============================================================================


class OmniArgumentParser(FlexibleArgumentParser):
    """FlexibleArgumentParser that injects model-specific defaults from
    deploy YAML as ``action.default`` before delegating to
    ``super().parse_args()``.

    Replaces the old nullify-then-refill pattern:

    BEFORE:
        nullify_stage_engine_defaults() → parse → YAML merge fills None

    AFTER:
        set YAML values as ``action.default`` → parse → argparse handles precedence
    """

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        if args is None:
            args = sys.argv[1:]

        args_list = list(args)

        # Clear stale state from a previous parse_args() call.
        self._omni_pipeline = None
        self._omni_model_type = None
        self._yaml_stage_overrides = None

        # Gate: only inject defaults for "serve" subcommand or flat-parser
        # mode (api_server direct launch). Avoids unnecessary HF config
        # fetches for unrelated subcommands like "bench".
        if self._is_serve_or_flat(args_list) and not self._is_help_or_version(args_list):
            if self._subparsers is None:
                model = self._peek_model_flat(args_list)
            else:
                model = self._peek_model(args_list)
            if model:
                stage_id = self._peek_stage_id(args_list)
                deploy_config_path = self._peek_deploy_config(args_list) or self._peek_stage_configs_path(args_list)
                self._inject_model_defaults(model, stage_id, deploy_config_path=deploy_config_path)

        result = super().parse_args(args, namespace)

        # Post-parse: deep-merge stashed YAML stage_overrides with user value.
        yaml_overrides = getattr(self, "_yaml_stage_overrides", None)
        if yaml_overrides is not None and hasattr(result, "stage_overrides"):
            user_raw = getattr(result, "stage_overrides", None)
            if user_raw:
                user_overrides = json.loads(user_raw) if isinstance(user_raw, str) else user_raw
                merged = _deep_merge(yaml_overrides, user_overrides)
            else:
                merged = yaml_overrides
            setattr(result, "stage_overrides", json.dumps(merged, sort_keys=True))

        return result

    # ── Peek helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_help_or_version(args: list[str]) -> bool:
        for a in args:
            if a in ("--help", "-h", "--version", "-v") or a.startswith("--help="):
                return True
        return False

    def _is_serve_or_flat(self, args: list[str]) -> bool:
        """True when injection should proceed: ``serve`` subcommand or flat-parser."""
        if self._subparsers is None:
            return True
        return bool(args) and args[0] == "serve"

    @classmethod
    def _peek_model(cls, args: list[str]) -> str | None:
        """Extract model from raw CLI args before full preprocessing.

        Handles:
        1. ``--config config.yaml`` / ``--config=config.yaml`` → reads yaml['model']
        2. ``--model foo/bar`` → promoted to positional by FlexibleArgumentParser
        3. ``vllm serve foo/bar`` → first positional after subcommand
        """
        import yaml

        current = list(args)

        # Case 1: --config YAML (both --config <path> and --config=<path>)
        for i, arg in enumerate(current):
            path: str | None = None
            if arg == "--config" and i + 1 < len(current):
                path = current[i + 1]
            elif arg.startswith("--config="):
                path = arg.split("=", 1)[1]
            if path:
                try:
                    with open(path) as fh:
                        cfg = yaml.safe_load(fh)
                    if isinstance(cfg, dict) and cfg.get("model"):
                        return cfg["model"]
                except Exception:
                    pass

        # Case 2: --model flag
        for i, arg in enumerate(current):
            if arg == "--model" and i + 1 < len(current):
                return current[i + 1]
            if arg.startswith("--model="):
                return arg.split("=", 1)[1]

        # Case 3: positional model for "serve" subcommand
        if current and current[0] == "serve" and len(current) > 1:
            if not current[1].startswith("-"):
                return current[1]

        return None

    def _peek_model_flat(self, args: list[str]) -> str | None:
        """Peek model for flat-parser mode (no subparsers — api_server)."""
        current = list(args)
        if current and not current[0].startswith("-"):
            return current[0]
        return None

    @staticmethod
    def _peek_stage_id(args: list[str]) -> int | None:
        for i, arg in enumerate(args):
            if arg == "--stage-id" and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except (ValueError, IndexError):
                    pass
            if arg.startswith("--stage-id="):
                try:
                    return int(arg.split("=", 1)[1])
                except ValueError:
                    pass
        return None

    @staticmethod
    def _peek_deploy_config(args: list[str]) -> str | None:
        """Peek --deploy-config from raw argv."""
        for i, arg in enumerate(args):
            if arg == "--deploy-config" and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith("--deploy-config="):
                return arg.split("=", 1)[1]
        return None

    @staticmethod
    def _peek_stage_configs_path(args: list[str]) -> str | None:
        """Peek --stage-configs-path from raw argv (deprecated alias for --deploy-config)."""
        for i, arg in enumerate(args):
            if arg == "--stage-configs-path" and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith("--stage-configs-path="):
                return arg.split("=", 1)[1]
        return None

    # ── Default injection ───────────────────────────────────────────────

    def _inject_model_defaults(
        self,
        model: str,
        stage_id: int | None,
        *,
        deploy_config_path: str | None = None,
    ) -> None:
        """Load deploy YAML + pipeline topology; inject as ``action.default``.

        Pipeline-wide DeployConfig fields → ``action.default``.
        Per-stage fields (headless) → ``action.default``.
        Per-stage fields (normal serve) → collapsed into JSON default on
        ``--stage-overrides``, stashed for post-parse deep-merge.

        Also loads ``PipelineConfig`` from registry and serialises its
        per-stage topology into the ``stage_overrides`` JSON alongside
        the deploy YAML per-stage values.
        """
        from dataclasses import fields as dc_fields
        from pathlib import Path

        from vllm_omni.config.stage_config import (
            _DEPLOY_DIR,
            _PIPELINE_REGISTRY,
            DeployConfig,
            StageDeployConfig,
            _auto_detect_model_type,
            load_deploy_config,
        )

        # Detect model_type + resolve deploy YAML path.
        if deploy_config_path:
            deploy_path = Path(deploy_config_path)
            if not deploy_path.exists():
                return
            # Detect model_type from the model (deploy file may have random suffix).
            model_type, _ = _auto_detect_model_type(model)
        else:
            model_type, _ = _auto_detect_model_type(model)
            if not model_type:
                return
            deploy_path = _DEPLOY_DIR / f"{model_type}.yaml"
            if not deploy_path.exists():
                return

        self._omni_model_type = model_type

        try:
            deploy = load_deploy_config(deploy_path)
        except Exception:
            logger.warning("Failed to load deploy config from %s", deploy_path)
            return

        serve_parser = self._get_serve_subparser()
        if serve_parser is None:
            if self._subparsers is None:
                serve_parser = self
            else:
                return

        # ── Pipeline-wide: DeployConfig fields → action.default ──
        for f in dc_fields(DeployConfig):
            val = getattr(deploy, f.name)
            if val is not None:
                self._set_action_default(serve_parser, f.name, val)

        # ── Load PipelineConfig from registry for topology ──
        pipeline_key = deploy.pipeline or (self._omni_model_type if self._omni_model_type else None)
        if pipeline_key and pipeline_key in _PIPELINE_REGISTRY:
            self._omni_pipeline = _PIPELINE_REGISTRY[pipeline_key]
        else:
            self._omni_pipeline = None

        # ── Build comprehensive stage_overrides from pipeline topology + deploy YAML ──
        if stage_id is not None:
            # Headless: set per-stage action.defaults directly from deploy YAML.
            stage_dep = next((s for s in deploy.stages if s.stage_id == stage_id), None)
            if stage_dep:
                for f in dc_fields(StageDeployConfig):
                    if f.name == "engine_extras":
                        continue
                    val = getattr(stage_dep, f.name)
                    if val is not None:
                        self._set_action_default(serve_parser, f.name, val)
                # Also inject pipeline topology for this stage.
                if self._omni_pipeline:
                    ps = self._omni_pipeline.get_stage(stage_id)
                    if ps:
                        _inject_stage_topology_defaults(self, serve_parser, ps, stage_dep)
            # Also build full stage_overrides so headless stages can resolve
            # their config via create_omni_config (run_headless needs all stages).
            stage_overrides: dict[str, dict[str, Any]] = _build_unified_stage_overrides(deploy, self._omni_pipeline)
            if stage_overrides:
                yaml_json = json.dumps(stage_overrides, sort_keys=True)
                self._set_action_default(serve_parser, "stage_overrides", yaml_json)
                self._yaml_stage_overrides = stage_overrides
        else:
            # Normal serve: build stage_overrides merging pipeline topology + deploy YAML.
            stage_overrides: dict[str, dict[str, Any]] = _build_unified_stage_overrides(deploy, self._omni_pipeline)
            if stage_overrides:
                yaml_json = json.dumps(stage_overrides, sort_keys=True)
                self._set_action_default(serve_parser, "stage_overrides", yaml_json)
                self._yaml_stage_overrides = stage_overrides

    @staticmethod
    def _set_action_default(parser: argparse.ArgumentParser, dest: str, value: Any) -> None:
        for action in parser._actions:
            if action.dest == dest:
                action.default = value
                break

    def _get_serve_subparser(self) -> argparse.ArgumentParser | None:
        if self._subparsers is None:
            return None
        subparsers_action = self._subparsers._group_actions[0]
        return subparsers_action.choices.get("serve")


# ============================================================================
# OmniArgumentParser helpers
# ============================================================================


def _inject_stage_topology_defaults(
    parser: OmniArgumentParser,
    serve_parser: argparse.ArgumentParser,
    ps: _StagePipelineConfig,
    ds: _StageDeployConfig | None,
) -> None:
    """Inject per-stage topology fields from ``StagePipelineConfig`` into the parser.

    Only used in headless mode for the selected stage.
    """
    mapping = {
        "model_arch": ps.model_arch,
        "engine_output_type": ps.engine_output_type,
        "hf_config_name": ps.hf_config_name,
        "custom_process_next_stage_input_func": ps.custom_process_next_stage_input_func,
        "model_subdir": ps.model_subdir,
        "tokenizer_subdir": ps.tokenizer_subdir,
        "model_stage": ps.model_stage,
    }
    for dest, val in mapping.items():
        if val is not None:
            parser._set_action_default(serve_parser, dest, val)


def _build_unified_stage_overrides(
    deploy: Any,  # DeployConfig
    pipeline: Any | None,  # PipelineConfig | None
) -> dict[str, dict[str, Any]]:
    """Build per-stage overrides dict merging pipeline topology + deploy YAML values."""
    deploy_by_id: dict[int, Any] = {s.stage_id: s for s in deploy.stages}

    result: dict[str, dict[str, Any]] = {}

    # Determine stages from pipeline (primary) or deploy YAML (fallback).
    if pipeline is not None:
        for ps in pipeline.stages:
            ds = deploy_by_id.get(ps.stage_id)
            entry = _build_one_stage_entry(ps, ds, deploy, _PIPELINE_WIDE_ENGINE_FIELDS)
            result[str(ps.stage_id)] = entry
    else:
        # No pipeline — stages defined purely by deploy YAML.
        for ds in deploy.stages:
            entry = _build_deploy_only_stage_entry(ds, deploy, _PIPELINE_WIDE_ENGINE_FIELDS)
            result[str(ds.stage_id)] = entry

    return result


def _build_one_stage_entry(
    ps: _StagePipelineConfig,
    ds: _StageDeployConfig | None,
    deploy: Any,  # DeployConfig
    pipeline_wide_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Build one stage override entry from pipeline topology + deploy YAML."""
    from dataclasses import fields as dc_fields

    # Map execution_type → stage_type + worker_type.
    stage_type_str = "llm"
    worker_type = None
    if ps.execution_type == _StageExecutionType.DIFFUSION:
        stage_type_str = "diffusion"
    elif ps.execution_type == _StageExecutionType.LLM_GENERATION:
        worker_type = "generation"

    engine_args: dict[str, Any] = {}

    # Pipeline topology fields.
    for name in ("engine_output_type", "hf_config_name", "model_subdir", "tokenizer_subdir"):
        val = getattr(ps, name, None)
        if val is not None:
            engine_args[name] = val
    engine_args["model_stage"] = ps.model_stage
    if worker_type:
        engine_args["worker_type"] = worker_type

    # Pipeline-wide deploy values.
    for name in pipeline_wide_fields:
        val = getattr(deploy, name, None)
        if val is not None:
            engine_args[name] = val

    # Deploy YAML async_chunk.
    engine_args["async_chunk"] = bool(deploy.async_chunk)

    # Per-stage deploy values.
    if ds is not None:
        for f in dc_fields(_StageDeployConfig):
            if f.name in ("engine_extras",):
                continue
            val = getattr(ds, f.name)
            if val is not None:
                engine_args[f.name] = val
        # Merge engine_extras (per-stage YAML fields that aren't
        # explicit StageDeployConfig fields, e.g. trust_remote_code,
        # enforce_eager, enable_prefix_caching).
        if ds.engine_extras:
            engine_args.update(ds.engine_extras)

    # Pipeline's omni_kv_config (need_send_cache, kv_transfer_criteria, etc.)
    # is consumed by the scheduler and KV transfer manager via
    # vllm_config.model_config.omni_kv_config.
    if ps.omni_kv_config:
        existing = engine_args.get("omni_kv_config") or {}
        if isinstance(existing, dict):
            existing.update(dict(ps.omni_kv_config))
        engine_args["omni_kv_config"] = existing

    entry: dict[str, Any] = {
        "stage_type": stage_type_str,
        "engine_args": engine_args,
    }

    # Stage-level overrides from pipeline topology.
    # is_comprehension maps from ps.owns_tokenizer (pipeline field name).
    if ps.owns_tokenizer:
        entry["is_comprehension"] = True
    for name in (
        "requires_multimodal_data",
        "custom_process_input_func",
        "prompt_expand_func",
        "cfg_kv_collect_func",
        "final_output",
        "final_output_type",
        "omni_kv_config",
        "sampling_constraints",
        "extras",
    ):
        val = getattr(ps, name, None)
        if val or (isinstance(val, bool) and val):
            entry[name] = val

    entry["input_sources"] = list(ps.input_sources)

    # Deploy YAML extras.
    if ds is not None:
        entry["default_sampling_params"] = dict(ds.default_sampling_params) if ds.default_sampling_params else {}
        entry.setdefault("num_replicas", ds.num_replicas)
        entry.setdefault("devices", ds.devices)

        runtime: dict[str, Any] = {}
        if ds.devices is not None:
            runtime["devices"] = ds.devices
        runtime["num_replicas"] = ds.num_replicas
        runtime["requires_multimodal_data"] = ps.requires_multimodal_data
        entry["runtime"] = runtime

        # PD flags from engine_extras.
        extras = ds.engine_extras if ds.engine_extras else {}
        if extras.get("is_prefill_only"):
            entry["is_prefill_only"] = True
        if extras.get("is_decode_only"):
            entry["is_decode_only"] = True

        # Input/output connectors.
        if ds.output_connectors:
            entry["output_connectors"] = ds.output_connectors
        if ds.input_connectors:
            entry["input_connectors"] = ds.input_connectors

    return entry


def _build_deploy_only_stage_entry(
    ds: _StageDeployConfig,
    deploy: Any,  # DeployConfig
    pipeline_wide_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Build a stage override entry from deploy YAML only (no pipeline)."""
    from dataclasses import fields as dc_fields

    stage_type_str = "llm"  # default; can be overridden in engine_extras
    engine_args: dict[str, Any] = {}

    # Pipeline-wide deploy values.
    for name in pipeline_wide_fields:
        val = getattr(deploy, name, None)
        if val is not None:
            engine_args[name] = val

    engine_args["async_chunk"] = bool(deploy.async_chunk)

    # Per-stage deploy values.
    for f in dc_fields(_StageDeployConfig):
        if f.name in ("engine_extras",):
            continue
        val = getattr(ds, f.name)
        if val is not None:
            engine_args[f.name] = val
    if ds.engine_extras:
        engine_args.update(ds.engine_extras)

    # engine_extras may contain stage_type override.
    extras = ds.engine_extras if ds.engine_extras else {}
    if extras.get("stage_type"):
        stage_type_str = extras["stage_type"]

    entry: dict[str, Any] = {
        "stage_type": stage_type_str,
        "engine_args": engine_args,
        "default_sampling_params": dict(ds.default_sampling_params) if ds.default_sampling_params else {},
        "num_replicas": ds.num_replicas,
    }

    runtime: dict[str, Any] = {}
    if ds.devices is not None:
        runtime["devices"] = ds.devices
    runtime["num_replicas"] = ds.num_replicas
    entry["runtime"] = runtime

    for key in (
        "is_prefill_only",
        "is_decode_only",
        "is_comprehension",
        "custom_process_input_func",
        "final_output",
        "final_output_type",
    ):
        if key in extras:
            entry[key] = extras[key]

    if extras.get("input_sources"):
        entry["input_sources"] = extras["input_sources"]

    return entry


# ============================================================================
# Post-parse deep merge
# ============================================================================


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*; override keys win.

    Stage IDs are normalised to strings to avoid duplicate keys when
    callers provide integer keys alongside YAML's string keys.
    """
    result = dict(base)
    for key, val in override.items():
        key_str = str(key)
        if key_str in result and isinstance(result[key_str], dict) and isinstance(val, dict):
            result[key_str] = _deep_merge(result[key_str], val)
        else:
            result[key_str] = val
    return result


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    "OmniEngineArgs",
    "OmniAsyncEngineArgs",
    "OmniArgumentParser",
    "register_omni_models_to_vllm",
]
