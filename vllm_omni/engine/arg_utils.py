from __future__ import annotations

import argparse
import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.logger import init_logger

from vllm_omni.config import OmniModelConfig, VllmOmniConfig
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
    "VoxCPMForConditionalGeneration": "voxcpm",
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

    # Register with both transformers AutoConfig and vLLM's config registry
    # so models with empty/missing config.json (e.g. CosyVoice3) can be
    # resolved when model_type is injected via hf_overrides.
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
            # Already registered elsewhere; ignore.
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


# ---------------------------------------------------------------------------
# OmniEngineArgs — unified engine arguments
# ---------------------------------------------------------------------------


@dataclass
class OmniEngineArgs(EngineArgs):
    """Unified engine arguments for vLLM-Omni, extending vLLM's EngineArgs.

    This is the **single user-facing config entrypoint** for both online
    (CLI) and offline (API) usage.  It absorbs what were previously
    ``OrchestratorArgs`` and per-stage ``OmniEngineArgs`` into one class.

    Pattern (matching vLLM's ``EngineArgs``):

        Online:  ``OmniEngineArgs.from_cli_args(parsed_args)``
        Offline: ``OmniEngineArgs(model="...", tensor_parallel_size=2, ...)``

    The resolved ``VllmOmniConfig`` is produced by
    :meth:`create_omni_config`.
    """

    # ── Omni mode switch ───────────────────────────────────────────

    omni: bool = False
    """Enable vLLM-Omni mode for multi-modal and diffusion models."""

    headless: bool = False
    """Run a single stage in headless mode (no orchestrator)."""

    # ── Orchestrator lifecycle ─────────────────────────────────────

    stage_init_timeout: int = 300
    """Timeout in seconds for a single stage to initialise."""

    init_timeout: int = 600
    """Total orchestrator startup timeout in seconds."""

    # ── Cross-stage communication ──────────────────────────────────

    shm_threshold_bytes: int = 65536
    """Byte threshold below which shared memory is used for transfer."""

    batch_timeout: int = 10
    """Batch collection timeout in seconds."""

    async_chunk: bool | None = None
    """Override the deploy YAML ``async_chunk`` setting.
    ``None`` leaves the YAML value in force."""

    # ── Cluster / backend ──────────────────────────────────────────

    worker_backend: str = "multi_process"
    """Backend for stage workers (``"multi_process"`` or ``"ray"``)."""

    ray_address: str | None = None
    """Address of the Ray cluster to connect to."""

    # ── Config files ───────────────────────────────────────────────

    stage_configs_path: str | None = None
    """[Deprecated] Path to legacy ``stage_args`` YAML.  Prefer ``deploy_config``."""

    deploy_config: str | None = None
    """Path to a deploy config YAML (new format with ``stages`` / ``engine_args``)."""

    stage_overrides: str | None = None
    """Per-stage JSON overrides.
    Example: ``'{"0": {"gpu_memory_utilization": 0.8}}'``."""

    # ── Single-stage / headless ────────────────────────────────────

    stage_id: int | None = None
    """Select and launch a single stage by stage_id."""

    replica_id: int = 0
    """Replica id to register when launching a single headless stage."""

    omni_master_address: str | None = None
    """Hostname or IP address of the Omni orchestrator (master)."""

    omni_master_port: int | None = None
    """Port of the Omni orchestrator (master)."""

    # ── Observability ──────────────────────────────────────────────

    log_stats: bool = False
    """Enable per-request pipeline metrics logging."""

    log_file: str | None = None
    """Path to the log file."""

    enable_diffusion_pipeline_profiler: bool = False
    """Enable diffusion pipeline profiler to display stage durations."""

    enable_ar_profiler: bool = False
    """Enable AR stage profiler to include AR stage timing."""

    # ── Output ─────────────────────────────────────────────────────

    output_modalities: list[str] | None = None
    """Optional list of output modality names (e.g. ``["text", "audio"]``)."""

    diffusion_batch_size: int = 1
    """Maximum number of requests to batch in the diffusion engine."""

    tts_batch_max_items: int = 32
    """Maximum items in a TTS batch."""

    # ── Diffusion model flags ──────────────────────────────────────

    num_gpus: int | None = None
    """Number of GPUs to use for diffusion model inference."""

    model_class_name: str | None = None
    """Override the diffusion pipeline class name."""

    diffusion_load_format: str | None = None
    """How to load the diffusion pipeline (``default``, ``custom_pipeline``, ``dummy``, ``diffusers``)."""

    diffusers_load_kwargs: dict[str, Any] | None = None
    """JSON object passed to ``DiffusionPipeline.from_pretrained()``."""

    diffusers_call_kwargs: dict[str, Any] | None = None
    """JSON object passed to ``pipeline.__call__()``."""

    # ── Diffusion parallelism ──────────────────────────────────────

    ulysses_degree: int | None = None
    """Ulysses Sequence Parallelism degree for diffusion models."""

    ulysses_mode: str = "strict"
    """Ulysses mode: ``"strict"`` or ``"advanced_uaa"``."""

    ring_degree: int | None = None
    """Ring Sequence Parallelism degree for diffusion models."""

    cfg_parallel_size: int = 1
    """Number of devices for CFG parallel computation (1 or 2)."""

    vae_patch_parallel_size: int = 1
    """VAE Patch Parallelism degree for diffusion models."""

    use_hsdp: bool = False
    """Enable HSDP (Hybrid Sharded Data Parallel) for diffusion models."""

    hsdp_shard_size: int = -1
    """Number of GPUs to shard weights across (-1 = auto)."""

    hsdp_replicate_size: int = 1
    """Number of replica groups for HSDP."""

    # ── Diffusion attention ────────────────────────────────────────

    diffusion_attention_backend: str | None = None
    """Diffusion attention backend shorthand (e.g. ``"FLASH_ATTN"``)."""

    diffusion_attention_config: dict[str, Any] | None = None
    """Diffusion attention config (per-role overrides)."""

    # ── Diffusion cache ────────────────────────────────────────────

    cache_backend: str = "none"
    """Cache backend for diffusion models (``"tea_cache"``, ``"cache_dit"``)."""

    cache_config: str | None = None
    """JSON string of cache configuration."""

    enable_cache_dit_summary: bool = False
    """Enable cache-dit summary logging after forward passes."""

    # ── Diffusion execution ────────────────────────────────────────

    step_execution: bool = False
    """Enable per-step diffusion execution for abort-ability."""

    boundary_ratio: float | None = None
    """Boundary split ratio for low/high DiT in video models."""

    flow_shift: float | None = None
    """Scheduler flow_shift for video models."""

    # ── VAE optimisation ───────────────────────────────────────────

    vae_use_slicing: bool = False
    """Enable VAE slicing for memory optimisation."""

    vae_use_tiling: bool = False
    """Enable VAE tiling for memory optimisation."""

    # ── Diffusion weight loading ───────────────────────────────────

    enable_multithread_weight_load: bool = True
    """Whether to use multi-threaded safetensors loading."""

    num_weight_load_threads: int = 4
    """Number of threads for parallel weight loading."""

    # ── Diffusion offloading ───────────────────────────────────────

    enable_cpu_offload: bool = False
    """Enable CPU offloading for diffusion models."""

    enable_layerwise_offload: bool = False
    """Enable layerwise (blockwise) offloading on DiT modules."""

    # ── Diffusion KV cache ─────────────────────────────────────────

    diffusion_kv_cache_dtype: str | None = None
    """Diffusion attention KV cache dtype (e.g. ``"fp8"``)."""

    diffusion_kv_cache_skip_steps: str | None = None
    """Diffusion KV-cache quantization skip-step selector."""

    diffusion_kv_cache_skip_layers: str | None = None
    """Diffusion KV-cache quantization skip-layer selector."""

    # ── Diffusion quantisation ─────────────────────────────────────

    quantization_config: Any | None = None
    """JSON string for diffusion quantization_config."""

    force_cutlass_fp8: bool | None = None
    """Force CUTLASS FP8 linear kernels on CUDA SM89+ devices."""

    # ── Diffusion LoRA ─────────────────────────────────────────────

    lora_path: str | None = None
    """Path to LoRA weights for diffusion models."""

    lora_scale: float | None = None
    """LoRA scale factor."""

    # ── Default sampling params ────────────────────────────────────

    default_sampling_params: str | None = None
    """JSON str for default sampling parameters per stage."""

    max_generated_image_size: int | None = None
    """Maximum size of generated images (height * width)."""

    # ── TTS ────────────────────────────────────────────────────────

    tts_max_instructions_length: int | None = None
    """Maximum length for TTS voice style instructions."""

    # ── Per-stage forwarded fields ─────────────────────────────────

    enable_sleep_mode: bool = False
    """Enable GPU memory pool for sleep mode."""

    task_type: str | None = None
    """Default task type for TTS models (``CustomVoice``, ``VoiceDesign``, ``Base``)."""

    model_stage: str = "thinker"
    """Stage type identifier (e.g. ``"thinker"`` or ``"talker"``)."""

    model_arch: str | None = None
    """Model architecture name."""

    engine_output_type: str | None = None
    """Optional output type specification for the engine."""

    hf_config_name: str | None = None
    """Optional key for HF config subkey (e.g. ``"talker_config"``)."""

    custom_process_next_stage_input_func: str | None = None
    """Optional path to a custom inter-stage input processing function."""

    stage_connector_spec: dict[str, Any] = field(default_factory=dict)
    """Extra configuration for stage connector."""

    subtalker_sampling_params: dict[str, Any] | None = None
    """Sub-talker sampling parameters for multi-speaker TTS."""

    omni_kv_config: dict[str, Any] | None = None
    """Omni KV cache transfer configuration."""

    worker_type: str | None = None
    """Worker type (``"ar"`` or ``"generation"``)."""

    worker_cls: str | None = None
    """Worker class (auto-resolved from ``worker_type`` if not set)."""

    custom_pipeline_args: dict[str, Any] | None = None
    """Arguments for custom pipeline initialisation."""

    has_sampling_extra_args: bool = False
    """Whether the stage default_sampling_params defines extra_args."""

    # ── Pre-built objects ──────────────────────────────────────────

    parallel_config: Any = None
    """Pre-built parallel config."""

    # ── Tokenizer ──────────────────────────────────────────────────
    # Captured by the orchestrator and forwarded to stages only when the
    # stage does not define tokenizer / tokenizer_subdir itself.

    tokenizer: str | None = None
    """Optional explicit tokenizer path or name."""

    # ==================================================================
    # CLI registration
    # ==================================================================

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

        This mirrors ``AsyncEngineArgs.add_cli_args`` and is intended to
        be called from ``OmniServeCommand.subparser_init``.
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
                "--headless",
                action="store_true",
                default=False,
                help="Run a single stage in headless mode.",
            )
        except argparse.ArgumentError:
            pass

        # ── Orchestrator lifecycle ──
        group.add_argument(
            "--stage-init-timeout",
            type=int,
            default=300,
            help="Timeout for single stage init (seconds).",
        )
        group.add_argument(
            "--init-timeout",
            type=int,
            default=600,
            help="Total orchestrator startup timeout (seconds).",
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

        # ── Cluster / backend ──
        group.add_argument("--worker-backend", type=str, default="multi_process", choices=["multi_process", "ray"])
        group.add_argument("--ray-address", type=str, default=None)

        # ── Config files ──
        group.add_argument(
            "--stage-configs-path",
            type=str,
            default=None,
            help="[Deprecated] Legacy stage_args YAML path.",
        )
        group.add_argument("--deploy-config", type=str, default=None, help="Path to deploy config YAML.")
        group.add_argument("--stage-overrides", type=str, default=None, help="Per-stage JSON overrides.")

        # ── Single-stage / headless ──
        group.add_argument("--stage-id", type=int, default=None, help="Select a single stage by stage_id.")
        group.add_argument("--replica-id", type=int, default=0, help="Replica id for headless mode.")
        group.add_argument("--omni-master-address", "-oma", type=str, default=None, help="Omni orchestrator address.")
        group.add_argument("--omni-master-port", "-omp", type=int, default=None, help="Omni orchestrator port.")

        # ── Observability ──
        group.add_argument("--log-stats", action="store_true", help="Enable per-request pipeline metrics.")
        group.add_argument("--log-file", type=str, default=None, help="Log file path.")
        group.add_argument("--enable-diffusion-pipeline-profiler", action="store_true")
        group.add_argument("--enable-ar-profiler", action="store_true")

        # ── Output ──
        group.add_argument("--output-modalities", type=str, nargs="*", default=None)
        group.add_argument("--diffusion-batch-size", type=int, default=1)
        group.add_argument("--tts-batch-max-items", type=int, default=32)

        # ── Diffusion model ──
        group.add_argument("--num-gpus", type=int, default=None)
        group.add_argument("--model-class-name", type=str, default=None)
        group.add_argument(
            "--diffusion-load-format",
            type=str,
            default=None,
            choices=["default", "custom_pipeline", "dummy", "diffusers"],
        )
        group.add_argument("--diffusers-load-kwargs", type=json.loads, default="{}")
        group.add_argument("--diffusers-call-kwargs", type=json.loads, default="{}")

        # ── Diffusion parallelism ──
        group.add_argument("--usp", "--ulysses-degree", dest="ulysses_degree", type=int, default=None)
        group.add_argument("--ulysses-mode", type=str, default="strict", choices=["strict", "advanced_uaa"])
        group.add_argument("--ring", "--ring-degree", dest="ring_degree", type=int, default=None)
        group.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2])
        group.add_argument("--vae-patch-parallel-size", type=int, default=1)
        group.add_argument("--use-hsdp", action="store_true")
        group.add_argument("--hsdp-shard-size", type=int, default=-1)
        group.add_argument("--hsdp-replicate-size", type=int, default=1)

        # ── Diffusion attention ──
        group.add_argument("--diffusion-attention-backend", type=str, default=None)
        group.add_argument("--diffusion-attention-config", "-dac", type=json.loads, default=None)

        # ── Diffusion cache ──
        group.add_argument("--cache-backend", type=str, default="none")
        group.add_argument("--cache-config", type=str, default=None)
        group.add_argument("--enable-cache-dit-summary", action="store_true")

        # ── Diffusion execution ──
        group.add_argument("--step-execution", action="store_true")
        group.add_argument("--boundary-ratio", type=float, default=None)
        group.add_argument("--flow-shift", type=float, default=None)

        # ── VAE ──
        group.add_argument("--vae-use-slicing", action="store_true")
        group.add_argument("--vae-use-tiling", action="store_true")

        # ── Weight loading ──
        group.add_argument(
            "--disable-multithread-weight-load",
            dest="enable_multithread_weight_load",
            action="store_false",
            default=True,
        )
        group.add_argument("--num-weight-load-threads", type=int, default=4)

        # ── Offloading ──
        group.add_argument("--enable-cpu-offload", action="store_true")
        group.add_argument("--enable-layerwise-offload", action="store_true")

        # ── Diffusion KV cache ──
        group.add_argument("--diffusion-kv-cache-dtype", type=str, default=None)
        group.add_argument("--diffusion-kv-cache-skip-steps", type=str, default=None)
        group.add_argument("--diffusion-kv-cache-skip-layers", type=str, default=None)

        # ── Quantisation ──
        group.add_argument("--quantization-config", type=json.loads, default=None)
        group.add_argument("--force-cutlass-fp8", action="store_true", default=None)

        # ── LoRA ──
        group.add_argument("--lora-path", type=str, default=None)
        group.add_argument("--lora-scale", type=float, default=None)

        # ── Sampling ──
        group.add_argument(
            "--default-sampling-params",
            type=str,
            default=None,
            help="JSON str for default sampling params.",
        )
        group.add_argument("--max-generated-image-size", type=int, default=None)

        # ── TTS ──
        group.add_argument("--tts-max-instructions-length", type=int, default=None)

        # ── Per-stage / sleep ──
        try:
            group.add_argument("--enable-sleep-mode", action="store_true", default=False)
        except argparse.ArgumentError:
            pass
        group.add_argument("--task-type", type=str, default=None, choices=["CustomVoice", "VoiceDesign", "Base"])

        return parser

    # ==================================================================
    # Construction helpers
    # ==================================================================

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> OmniEngineArgs:
        """Build ``OmniEngineArgs`` from a parsed CLI namespace (online path).

        Only copies attributes that exist on the namespace — missing fields
        fall back to the dataclass default.  Sets ``_explicit_fields`` so
        downstream code can distinguish user-typed flags from defaults.
        """
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)})
        engine_args._explicit_fields = frozenset(
            attr for attr in attrs if hasattr(args, attr) and getattr(args, attr) is not None
        )
        return engine_args

    @classmethod
    def create(cls, **explicit: Any) -> OmniEngineArgs:
        """Track caller-set fields for ``Omni(..., engine_args=ea)``."""
        ea = cls(**explicit)
        ea._explicit_fields = frozenset(explicit.keys())
        return ea

    def explicit_kwargs(self) -> dict[str, Any]:
        """Return only the fields explicitly set by the caller."""
        explicit = getattr(self, "_explicit_fields", None)
        if explicit is None:
            return {
                f.name: getattr(self, f.name) for f in dataclasses.fields(self) if getattr(self, f.name) is not None
            }
        return {k: getattr(self, k) for k in explicit}

    # ==================================================================
    # Post-init validation
    # ==================================================================

    def __post_init__(self) -> None:
        if self.worker_cls is None:
            if self.worker_type == "ar":
                self.worker_cls = current_omni_platform.get_omni_ar_worker_cls()
            elif self.worker_type == "generation":
                self.worker_cls = current_omni_platform.get_omni_generation_worker_cls()
        load_omni_general_plugins()
        super().__post_init__()

    # ==================================================================
    # HF config patching
    # ==================================================================

    def _ensure_omni_models_registered(self):
        if hasattr(self, "_omni_models_registered"):
            return True
        register_omni_models_to_vllm()
        self._omni_models_registered = True
        return True

    def _patch_empty_hf_config(self, model_type: str) -> None:
        """For models with empty config.json (e.g. CosyVoice3), create a
        patched config in a temp directory with model_type set so that
        transformers AutoConfig.from_pretrained can resolve the config class.
        Sets self.hf_config_path to point to the patched directory."""
        try:
            from transformers import PretrainedConfig

            config_dict, _ = PretrainedConfig.get_config_dict(self.model)
            if config_dict.get("model_type"):
                return  # config.json already has model_type, no patching needed
        except Exception:
            return  # can't load config, let vLLM handle the error

        # Create a temp dir with a patched config.json
        temp_dir = tempfile.mkdtemp(prefix="omni_hf_config_")
        config_dict["model_type"] = model_type
        config_dict.setdefault("architectures", [self.model_arch])
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        self.hf_config_path = temp_dir
        self._temp_config_dir = temp_dir
        logger.info("Patched empty HF config with model_type=%s at %s", model_type, temp_dir)

    def create_model_config(self) -> OmniModelConfig:
        """Create an OmniModelConfig from these engine arguments.
        Returns:
            OmniModelConfig instance with all configuration fields set
        """
        # register omni models to avoid model not found error
        self._ensure_omni_models_registered()

        # Build stage_connector_config from stage_connector_spec
        stage_connector_config = {
            "name": self.stage_connector_spec.get("name", "SharedMemoryConnector"),
            "extra": self.stage_connector_spec.get("extra", {}).copy(),
        }
        stage_connector_config["extra"]["stage_id"] = self.stage_id

        # If model_arch is specified, inject it into hf_overrides so vLLM can
        # resolve the architecture even when config.json lacks 'architectures'.
        # Also inject model_type so AutoConfig can resolve the correct config
        # class for models with empty or missing config.json (e.g. CosyVoice3).
        if self.model_arch:
            if self.hf_overrides is None:
                self.hf_overrides = {}
            if isinstance(self.hf_overrides, dict):
                self.hf_overrides.setdefault("architectures", [self.model_arch])
                if "model_type" not in self.hf_overrides:
                    model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                    if model_type is not None:
                        self.hf_overrides.setdefault("model_type", model_type)

                # Stage wrappers (e.g. Code2Wav) may need max_model_len larger
                # than the base checkpoint's text max_position_embeddings.
                if self.model_arch == "Qwen3TTSCode2Wav" and self.max_model_len is not None:
                    self.hf_overrides.setdefault("talker_config", {}).setdefault(
                        "max_position_embeddings", int(self.max_model_len)
                    )

            # For models whose HF config.json is empty or lacks model_type
            # (e.g. CosyVoice3), AutoConfig.from_pretrained fails because it
            # cannot determine which config class to use from the empty dict.
            # hf_overrides alone is not enough since transformers reads
            # model_type from config_dict before applying overrides.
            # Workaround: create a patched config.json in a temp directory
            # and point hf_config_path to it so vLLM reads model_type from it.
            if not self.hf_config_path:
                model_type = _ARCH_TO_MODEL_TYPE.get(self.model_arch)
                if model_type is not None:
                    self._patch_empty_hf_config(model_type)

        # Auto-detect tokenizer for models that store it in a subdirectory
        # rather than the root (e.g. CosyVoice3 uses CosyVoice-BlankEN/).
        if not self.tokenizer and self.model:
            model_path = self.model
            if os.path.isdir(model_path) and not os.path.isfile(os.path.join(model_path, "tokenizer_config.json")):
                for subfolder in sorted(os.listdir(model_path)):
                    candidate = os.path.join(model_path, subfolder)
                    if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "tokenizer_config.json")):
                        self.tokenizer = candidate
                        logger.info("Auto-detected tokenizer at %s", candidate)
                        break
            elif not os.path.isdir(model_path):
                subfolder = _TOKENIZER_SUBFOLDER_MAP.get(self.model_arch)
                if subfolder:
                    # Download just the tokenizer files from the subfolder
                    try:
                        from huggingface_hub import snapshot_download

                        local_dir = snapshot_download(
                            model_path,
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
                            logger.info("Downloaded tokenizer from %s/%s", model_path, subfolder)
                    except Exception as e:
                        logger.warning("Failed to download tokenizer subfolder: %s", e)

        # Build the vLLM config first, then use it to create the Omni config.
        try:
            model_config = super().create_model_config()
        finally:
            # Clean up temp config dir if we created one
            if hasattr(self, "_temp_config_dir"):
                import shutil

                shutil.rmtree(self._temp_config_dir, ignore_errors=True)
                del self._temp_config_dir

        omni_config = OmniModelConfig.from_vllm_model_config(
            model_config=model_config,
            # All kwargs below are Omni specific
            stage_id=self.stage_id,
            async_chunk=bool(self.async_chunk) if self.async_chunk is not None else False,
            model_stage=self.model_stage,
            model_arch=self.model_arch,
            worker_type=self.worker_type,
            engine_output_type=self.engine_output_type,
            hf_config_name=self.hf_config_name,
            custom_process_next_stage_input_func=self.custom_process_next_stage_input_func,
            stage_connector_config=stage_connector_config,
            subtalker_sampling_params=self.subtalker_sampling_params,
            omni_kv_config=self.omni_kv_config,
            task_type=self.task_type,
            has_sampling_extra_args=self.has_sampling_extra_args,
        )
        return omni_config

    # ==================================================================
    # create_omni_config — the main factory
    # ==================================================================

    def create_omni_config(self) -> VllmOmniConfig:
        """Build a resolved :class:`VllmOmniConfig` from these engine args.

        Delegates to :func:`~vllm_omni.config.vllm_omni_config.build_vllm_omni_config`
        with only explicitly-set args (not dataclass defaults).
        """
        import json as _json

        from vllm_omni.config.vllm_omni_config import build_vllm_omni_config

        # Use explicit_kwargs() so dataclass defaults don't leak into CLI overrides
        kwargs = self.explicit_kwargs()
        # Pop model since it's passed positionally
        model = getattr(self, "model", "")
        kwargs.pop("model", None)

        # Parse stage_overrides from JSON string if needed
        stage_overrides = kwargs.pop("stage_overrides", None)
        if isinstance(stage_overrides, str):
            try:
                stage_overrides = _json.loads(stage_overrides)
            except _json.JSONDecodeError:
                raise ValueError(f"--stage-overrides is not valid JSON: {stage_overrides!r}") from None

        return build_vllm_omni_config(
            model,
            engine_args=self,
            stage_overrides=stage_overrides,
            **kwargs,
        )


# ============================================================================
# OmniAsyncEngineArgs
# ============================================================================


@dataclass
class OmniAsyncEngineArgs(AsyncEngineArgs, OmniEngineArgs):
    """Async variant of OmniEngineArgs, combining AsyncEngineArgs + OmniEngineArgs."""

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = AsyncEngineArgs.add_cli_args(parser)
        parser = OmniEngineArgs.add_cli_args(parser, omni_args_only=True)
        return parser

    @property
    def output_modality(self) -> OutputModality:
        """Parse engine_output_type into a type-safe OutputModality flag."""
        return OutputModality.from_string(self.engine_output_type)


# ============================================================================
# stand-alone helpers
# ============================================================================


def nullify_stage_engine_defaults(parser: argparse.ArgumentParser) -> None:
    """Reset stage-level engine flag defaults to ``None``; preserve real
    default in help text. Only deploy-YAML override fields are touched.
    Idempotent."""
    from vllm_omni.config.stage_config import deploy_override_field_names

    override_dests = deploy_override_field_names()

    for action in parser._actions:
        if action.dest in ("help", "version") or not action.option_strings:
            continue
        if action.dest not in override_dests:
            continue
        if action.default is None or action.default is argparse.SUPPRESS:
            continue
        if action.help and "(default:" not in action.help and "%(default)" not in action.help:
            action.help = f"{action.help} (default: {action.default})"
        action.default = None

    parser._omni_nullified = True  # type: ignore[attr-defined]
