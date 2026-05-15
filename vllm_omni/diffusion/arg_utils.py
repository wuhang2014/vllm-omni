# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI argument dataclass for single-stage diffusion models.

``DiffusionEngineArgs`` is the authoritative flat representation of all
diffusion-specific CLI flags.  It mirrors the role that ``OmniEngineArgs``
plays for LLM stages: it owns the argparse registration, the
``from_cli_args`` factory, and the ``to_diffusion_config`` assembly method
that builds the nested ``OmniDiffusionConfig`` sub-configs in one place.

Usage
-----
    parser = argparse.ArgumentParser()
    DiffusionEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    od_config = DiffusionEngineArgs.from_cli_args(args).to_diffusion_config()

    # Or as a one-liner via the OmniDiffusionConfig façade:
    od_config = OmniDiffusionConfig.from_cli_args(args)

Multi-stage (YAML) path
-----------------------
When a stage_configs_path YAML is in use the old
``OmniDiffusionConfig.from_kwargs(**engine_args_dict)`` path is unchanged —
this module only adds a new CLI entry point; nothing is removed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Cache config helpers (moved from AsyncOmniEngine to the correct layer)
# ---------------------------------------------------------------------------


def get_default_cache_config(cache_backend: str | None) -> dict[str, Any] | None:
    """Return sensible default cache parameters for a given backend."""
    if cache_backend == "cache_dit":
        return {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }
    if cache_backend == "tea_cache":
        return {
            "rel_l1_thresh": 0.2,
        }
    return None


def normalize_cache_config(
    cache_backend: str | None,
    cache_config: Any | None,
) -> Any | None:
    """Coerce *cache_config* to a dict and fill defaults when it is absent."""
    if isinstance(cache_config, str):
        try:
            cache_config = json.loads(cache_config)
        except json.JSONDecodeError:
            logger.warning("Invalid cache_config JSON, using defaults.")
            cache_config = None
    if cache_config is None and cache_backend not in (None, "", "none"):
        cache_config = get_default_cache_config(cache_backend)
    return cache_config


# ---------------------------------------------------------------------------
# DiffusionEngineArgs
# ---------------------------------------------------------------------------


@dataclass
class DiffusionEngineArgs:
    """Flat CLI-facing dataclass for single-stage diffusion models.

    All fields map 1-to-1 onto ``OmniDiffusionConfig`` or onto the
    components assembled by ``to_diffusion_config()``.  Complex nested
    configs (``DiffusionParallelConfig``, ``DiffusionCacheConfig``,
    ``AttentionConfig``) are built inside ``to_diffusion_config()``.
    """

    # ------------------------------------------------------------------
    # Model & loading
    # ------------------------------------------------------------------
    model: str | None = None
    dtype: str = "auto"
    model_class_name: str | None = None
    trust_remote_code: bool = False
    revision: str | None = None
    diffusion_load_format: str = "default"
    distributed_executor_backend: str = "mp"
    enforce_eager: bool = False

    # ------------------------------------------------------------------
    # Parallelism (flat — assembled into DiffusionParallelConfig)
    # ------------------------------------------------------------------
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    ulysses_mode: str = "strict"
    sequence_parallel_size: int | None = None
    cfg_parallel_size: int = 1
    vae_patch_parallel_size: int = 1
    enable_expert_parallel: bool = False
    use_hsdp: bool = False
    hsdp_shard_size: int = -1
    hsdp_replicate_size: int = 1

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    cache_backend: str = "none"
    cache_config: dict[str, Any] | str | None = None

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------
    diffusion_attention_backend: str | None = None
    diffusion_attention_config: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------
    lora_path: str | None = None
    lora_scale: float = 1.0
    max_cpu_loras: int | None = None

    # ------------------------------------------------------------------
    # Quantization / KV-cache
    # ------------------------------------------------------------------
    quantization: str | None = None  # legacy alias, mapped to quantization_config
    quantization_config: Any | None = None
    force_cutlass_fp8: bool = False
    kv_cache_dtype: str | None = None
    kv_cache_skip_steps: str | None = None
    kv_cache_skip_layers: str | None = None

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False

    # ------------------------------------------------------------------
    # CPU offload
    # ------------------------------------------------------------------
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    pin_cpu_memory: bool = True

    # ------------------------------------------------------------------
    # Serving / scheduler
    # ------------------------------------------------------------------
    max_num_seqs: int = 1
    enable_sleep_mode: bool = False
    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = 4

    # ------------------------------------------------------------------
    # DiT-specific
    # ------------------------------------------------------------------
    boundary_ratio: float | None = None
    flow_shift: float | None = None
    skip_time_steps: int = 15
    VSA_sparsity: float = 0.0
    moba_config_path: str | None = None
    mask_strategy_file_path: str | None = None

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------
    enable_diffusion_pipeline_profiler: bool = False
    enable_cache_dit_summary: bool = False
    profiler_config: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Extra / passthrough
    # ------------------------------------------------------------------
    custom_pipeline_args: dict[str, Any] | None = None
    worker_extension_cls: str | None = None
    additional_config: dict[str, Any] | None = None
    diffusers_load_kwargs: dict[str, Any] | None = None
    diffusers_call_kwargs: dict[str, Any] | None = None
    # JSON string from CLI (parsed by to_diffusion_config)
    default_sampling_params: str | None = None
    step_execution: bool = False

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register all diffusion-specific CLI flags on *parser*.

        Flags are added inside a dedicated argument group so they appear
        together in ``--help`` output.  Each ``add_argument`` call is wrapped
        in a try/except to tolerate duplicate registration when this method is
        called alongside ``OmniAsyncEngineArgs.add_cli_args``.
        """
        grp = parser.add_argument_group("Diffusion model arguments")

        def _add(name: str, **kw: Any) -> None:
            try:
                grp.add_argument(name, **kw)
            except argparse.ArgumentError:
                pass  # already registered by a parent parser

        _add("--dtype", type=str, default=None, help="Model dtype (auto, bfloat16, float16, float32).")
        _add("--model-class-name", type=str, default=None, help="Diffusion model class name override.")
        _add("--revision", type=str, default=None, help="HuggingFace model revision.")
        _add(
            "--diffusion-load-format",
            type=str,
            default="default",
            help='Diffusion model loading format: "default", "diffusers", "custom_pipeline", "dummy".',
        )
        _add(
            "--distributed-executor-backend",
            type=str,
            default="mp",
            help="Distributed executor backend (mp, ray).",
        )
        _add("--enforce-eager", action="store_true", default=False, help="Disable CUDA graph capture.")

        # Parallelism
        _add("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size.")
        _add("--pipeline-parallel-size", type=int, default=1, help="Pipeline parallel size.")
        _add("--data-parallel-size", type=int, default=1, help="Data parallel size.")
        _add("--ulysses-degree", type=int, default=1, help="Ulysses sequence parallelism degree.")
        _add("--ring-degree", type=int, default=1, help="Ring sequence parallelism degree.")
        _add(
            "--ulysses-mode",
            type=str,
            default="strict",
            choices=["strict", "advanced_uaa"],
            help="Ulysses mode.",
        )
        _add("--sequence-parallel-size", type=int, default=None, help="Sequence parallel size (auto if omitted).")
        _add("--cfg-parallel-size", type=int, default=1, help="CFG (Classifier-Free Guidance) parallel size.")
        _add("--vae-patch-parallel-size", type=int, default=1, help="VAE patch/tile parallel size.")
        _add("--enable-expert-parallel", action="store_true", default=False, help="Enable expert parallelism.")
        _add("--use-hsdp", action="store_true", default=False, help="Enable Hybrid Sharded Data Parallel.")
        _add("--hsdp-shard-size", type=int, default=-1, help="HSDP shard size (-1 = auto).")
        _add("--hsdp-replicate-size", type=int, default=1, help="HSDP replicate size.")

        # Cache
        _add(
            "--cache-backend",
            type=str,
            default="none",
            help='Cache backend: "none", "cache_dit", "tea_cache".',
        )
        _add(
            "--cache-config",
            type=str,
            default=None,
            help="JSON string with cache backend configuration.",
        )

        # Attention
        _add(
            "--diffusion-attention-backend",
            type=str,
            default=None,
            help="Attention backend for diffusion transformer (e.g. flash_attn, xformers).",
        )
        _add(
            "--diffusion-attention-config",
            type=str,
            default=None,
            help="JSON string with full diffusion attention config.",
        )

        # LoRA
        _add("--lora-path", type=str, default=None, help="Path to LoRA weights.")
        _add("--lora-scale", type=float, default=1.0, help="LoRA scale factor.")
        _add("--max-cpu-loras", type=int, default=None, help="Maximum number of LoRA adapters in CPU memory.")

        # Quantization / KV-cache
        _add("--quantization-config", type=str, default=None, help="Quantization configuration (JSON or method name).")
        _add("--force-cutlass-fp8", action="store_true", default=False, help="Force CUTLASS FP8 kernels.")
        _add("--kv-cache-dtype", type=str, default=None, help='KV-cache dtype override (e.g. "fp8").')
        _add(
            "--kv-cache-skip-steps",
            type=str,
            default=None,
            help='Diffusion steps to skip KV-cache quantization (e.g. "0-9,20").',
        )
        _add(
            "--kv-cache-skip-layers",
            type=str,
            default=None,
            help='Transformer layers to skip KV-cache quantization (e.g. "0-3").',
        )

        # VAE
        _add("--vae-use-slicing", action="store_true", default=False, help="Enable VAE sliced encoding/decoding.")
        _add("--vae-use-tiling", action="store_true", default=False, help="Enable VAE tiled encoding/decoding.")

        # CPU offload
        _add("--enable-cpu-offload", action="store_true", default=False, help="Enable CPU offload for diffusion model.")
        _add(
            "--enable-layerwise-offload",
            action="store_true",
            default=False,
            help="Enable layer-wise CPU offloading.",
        )
        _add(
            "--pin-cpu-memory",
            action="store_true",
            default=True,
            help="Use pinned CPU memory for faster host-device transfers.",
        )

        # Serving
        _add("--max-num-seqs", type=int, default=1, help="Maximum batch size for diffusion engine.")
        _add(
            "--enable-multithread-weight-load",
            action="store_true",
            default=True,
            help="Enable parallel weight loading.",
        )
        _add("--num-weight-load-threads", type=int, default=4, help="Number of threads for weight loading.")

        # DiT-specific
        _add("--boundary-ratio", type=float, default=None, help="MoE boundary ratio (Wan2.2).")
        _add("--flow-shift", type=float, default=None, help="Scheduler flow_shift (Wan2.2).")
        _add("--skip-time-steps", type=int, default=15, help="STA skip time steps.")
        _add("--vsa-sparsity", type=float, default=0.0, dest="VSA_sparsity", help="VSA inference sparsity.")
        _add("--moba-config-path", type=str, default=None, help="Path to V-MoBA config file.")
        _add(
            "--mask-strategy-file-path",
            type=str,
            default=None,
            help="Path to STA mask strategy file.",
        )

        # Profiling
        _add(
            "--enable-diffusion-pipeline-profiler",
            action="store_true",
            default=False,
            help="Enable diffusion pipeline profiler.",
        )
        _add(
            "--enable-cache-dit-summary",
            action="store_true",
            default=False,
            help="Enable Cache-DiT summary logging.",
        )

        # Extra
        _add(
            "--custom-pipeline-args",
            type=str,
            default=None,
            help="JSON string with custom pipeline initialization arguments.",
        )
        _add("--worker-extension-cls", type=str, default=None, help="Dotted path to worker extension class.")
        _add(
            "--diffusers-load-kwargs",
            type=str,
            default=None,
            help="JSON string forwarded to DiffusionPipeline.from_pretrained().",
        )
        _add(
            "--diffusers-call-kwargs",
            type=str,
            default=None,
            help="JSON string forwarded to pipeline.__call__().",
        )
        _add(
            "--default-sampling-params",
            type=str,
            default=None,
            help='JSON dict of per-stage default sampling params, e.g. \'{"0": {"num_inference_steps": 50}}\'.',
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> DiffusionEngineArgs:
        """Construct from an ``argparse.Namespace`` returned by ``parse_args()``."""
        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {f: getattr(args, f) for f in valid if hasattr(args, f)}
        # JSON-string fields that argparse delivers as strings
        for key in ("custom_pipeline_args", "diffusers_load_kwargs", "diffusers_call_kwargs"):
            if isinstance(kwargs.get(key), str):
                try:
                    kwargs[key] = json.loads(kwargs[key])
                except json.JSONDecodeError:
                    logger.warning("Could not parse --%s as JSON, ignoring.", key.replace("_", "-"))
                    kwargs[key] = None
        if isinstance(kwargs.get("diffusion_attention_config"), str):
            try:
                kwargs["diffusion_attention_config"] = json.loads(kwargs["diffusion_attention_config"])
            except json.JSONDecodeError:
                logger.warning("Could not parse --diffusion-attention-config as JSON, ignoring.")
                kwargs["diffusion_attention_config"] = None
        return cls(**kwargs)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> DiffusionEngineArgs:
        """Construct from an arbitrary kwargs dict (e.g. from engine_args dicts).

        Unknown keys are silently dropped so callers do not need to pre-filter.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in kwargs.items() if k in valid})

    def to_diffusion_config(self) -> OmniDiffusionConfig:
        """Assemble and return an ``OmniDiffusionConfig``.

        This is the single authoritative place where:
        - flat parallel kwargs → ``DiffusionParallelConfig``
        - raw cache_config/cache_backend → normalised cache dict
        - diffusion_attention_backend shorthand → ``AttentionConfig``

        All other fields are forwarded verbatim to
        ``OmniDiffusionConfig.from_kwargs()``, which handles its own
        coercions (dtype str→torch.dtype, cache dict→DiffusionCacheConfig, …).
        """
        from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, parse_attention_config

        # 1. Build DiffusionParallelConfig from flat parallel fields
        sp_size = self.sequence_parallel_size
        if sp_size is None:
            sp_size = self.ulysses_degree * self.ring_degree
        parallel_config = DiffusionParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            data_parallel_size=self.data_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_expert_parallel=self.enable_expert_parallel,
            sequence_parallel_size=sp_size,
            ulysses_degree=self.ulysses_degree,
            ring_degree=self.ring_degree,
            ulysses_mode=self.ulysses_mode,
            cfg_parallel_size=self.cfg_parallel_size,
            vae_patch_parallel_size=self.vae_patch_parallel_size,
            use_hsdp=self.use_hsdp,
            hsdp_shard_size=self.hsdp_shard_size,
            hsdp_replicate_size=self.hsdp_replicate_size,
        )

        # 2. Normalize cache config
        cache_config = normalize_cache_config(self.cache_backend, self.cache_config)

        # 3. Build AttentionConfig when a backend shorthand or config dict is provided.
        #    Pass None when neither is set so OmniDiffusionConfig uses its default.
        attn_config: Any = None
        if self.diffusion_attention_config is not None or self.diffusion_attention_backend is not None:
            attn_config = parse_attention_config(
                self.diffusion_attention_config,
                attention_backend=self.diffusion_attention_backend,
            )

        # 4. Forward to OmniDiffusionConfig.from_kwargs for final assembly.
        #    Build the kwargs dict from all fields, replacing parallel/cache/attention
        #    with the assembled objects.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "dtype": self.dtype,
            "model_class_name": self.model_class_name,
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
            "diffusion_load_format": self.diffusion_load_format,
            "distributed_executor_backend": self.distributed_executor_backend,
            "enforce_eager": self.enforce_eager,
            "parallel_config": parallel_config,
            "cache_backend": self.cache_backend,
            "cache_config": cache_config,
            "lora_path": self.lora_path,
            "lora_scale": self.lora_scale,
            "max_cpu_loras": self.max_cpu_loras,
            "quantization_config": self.quantization_config,
            "quantization": self.quantization,
            "force_cutlass_fp8": self.force_cutlass_fp8,
            "kv_cache_dtype": self.kv_cache_dtype,
            "kv_cache_skip_steps": self.kv_cache_skip_steps,
            "kv_cache_skip_layers": self.kv_cache_skip_layers,
            "vae_use_slicing": self.vae_use_slicing,
            "vae_use_tiling": self.vae_use_tiling,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_layerwise_offload": self.enable_layerwise_offload,
            "pin_cpu_memory": self.pin_cpu_memory,
            "max_num_seqs": self.max_num_seqs,
            "enable_sleep_mode": self.enable_sleep_mode,
            "enable_multithread_weight_load": self.enable_multithread_weight_load,
            "num_weight_load_threads": self.num_weight_load_threads,
            "boundary_ratio": self.boundary_ratio,
            "flow_shift": self.flow_shift,
            "skip_time_steps": self.skip_time_steps,
            "VSA_sparsity": self.VSA_sparsity,
            "moba_config_path": self.moba_config_path,
            "mask_strategy_file_path": self.mask_strategy_file_path,
            "enable_diffusion_pipeline_profiler": self.enable_diffusion_pipeline_profiler,
            "enable_cache_dit_summary": self.enable_cache_dit_summary,
            "profiler_config": self.profiler_config,
            "custom_pipeline_args": self.custom_pipeline_args,
            "worker_extension_cls": self.worker_extension_cls,
            "additional_config": self.additional_config,
            "diffusers_load_kwargs": self.diffusers_load_kwargs,
            "diffusers_call_kwargs": self.diffusers_call_kwargs,
            "step_execution": self.step_execution,
        }
        if attn_config is not None:
            kwargs["diffusion_attention_config"] = attn_config

        return OmniDiffusionConfig.from_kwargs(**kwargs)
