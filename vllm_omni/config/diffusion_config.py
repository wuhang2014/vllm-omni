# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured configuration for diffusion stages.

Replaces the ~35 loose kwargs previously extracted in
``_create_default_diffusion_stage_cfg`` and the 12-field post-load
injection loop, providing a single typed container.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm_omni.diffusion.data import DiffusionParallelConfig

logger = init_logger(__name__)


@dataclass
class DiffusionConfig:
    """Canonical, fully-resolved config for a diffusion stage.

    Replaces the ~35 kwargs extracted in ``_create_default_diffusion_stage_cfg``
    and the 12-field post-load injection loop in ``_resolve_stage_configs``.

    Built once from the caller's kwargs dict via ``from_kwargs()``, then
    threaded through ``VllmOmniConfig`` to avoid repeated ``kwargs.get()``
    calls and error-prone ``hasattr``/``getattr`` guard chains.
    """

    # ── Parallelism ──
    parallel_config: DiffusionParallelConfig | None = None

    # ── Model identity ──
    model_class_name: str | None = None
    additional_config: dict[str, Any] | None = None
    custom_pipeline_args: dict[str, Any] | None = None
    worker_extension_cls: str | None = None

    # ── Execution ──
    max_num_seqs: int = 1
    enforce_eager: bool | None = None
    step_execution: bool = False

    # ── VAE ──
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False

    # ── Cache ──
    cache_backend: str = "none"
    cache_config: dict[str, Any] | None = None
    enable_cache_dit_summary: bool = False

    # ── Offload ──
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False

    # ── Sampling / schedule parameters ──
    boundary_ratio: float | None = None
    flow_shift: float | None = None

    # ── Weight loading ──
    diffusion_load_format: str = "default"
    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = 4

    # ── Quantization ──
    quantization: str | None = None
    quantization_config: Any = None
    force_cutlass_fp8: bool = False

    # ── KV Cache ──
    kv_cache_dtype: str | None = None
    kv_cache_skip_steps: int | None = None
    kv_cache_skip_layers: list[int] | None = None

    # ── LoRA ──
    lora_path: str | None = None
    lora_scale: float | None = None

    # ── Attention ──
    attention_config: Any = None

    # ── Observability ──
    enable_diffusion_pipeline_profiler: bool = False
    enable_ar_profiler: bool = False
    profiler_config: dict[str, Any] | None = None

    # ── Distributed infra ──
    trust_remote_code: bool = False
    distributed_executor_backend: str = "mp"

    # ── Sleep mode ──
    enable_sleep_mode: bool = False

    # ── Diffusers ──
    diffusers_load_kwargs: dict[str, Any] | None = None
    diffusers_call_kwargs: dict[str, Any] | None = None

    # ── Extras (structural, not user-facing) ──
    auxiliary_text_encoder: str | None = None
    default_llama_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # ── dtype (must be str for OmegaConf compat) ──
    dtype: str = "auto"

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> DiffusionConfig:
        """Build a ``DiffusionConfig`` from the current flat kwargs dict.

        This is the migration bridge: all existing callers keep working while
        internally producing a typed config.
        """
        parallel_config = kwargs.get("parallel_config")
        if isinstance(parallel_config, dict):
            parallel_config = DiffusionParallelConfig.from_dict(parallel_config)
        elif parallel_config is None:
            ulysses_degree = kwargs.get("ulysses_degree") or 1
            ring_degree = kwargs.get("ring_degree") or 1
            sp_size = kwargs.get("sequence_parallel_size")
            if sp_size is None:
                sp_size = ulysses_degree * ring_degree
            parallel_config = DiffusionParallelConfig(
                pipeline_parallel_size=kwargs.get("pipeline_parallel_size") or 1,
                data_parallel_size=kwargs.get("data_parallel_size") or 1,
                tensor_parallel_size=kwargs.get("tensor_parallel_size") or 1,
                enable_expert_parallel=kwargs.get("enable_expert_parallel", False),
                sequence_parallel_size=sp_size,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                ulysses_mode=kwargs.get("ulysses_mode", "strict"),
                cfg_parallel_size=kwargs.get("cfg_parallel_size") or 1,
                vae_patch_parallel_size=kwargs.get("vae_patch_parallel_size") or 1,
                use_hsdp=kwargs.get("use_hsdp", False),
                hsdp_shard_size=kwargs.get("hsdp_shard_size", -1),
                hsdp_replicate_size=kwargs.get("hsdp_replicate_size", 1),
            )

        attention_config = None
        if kwargs.get("diffusion_attention_config") or kwargs.get("diffusion_attention_backend"):
            from vllm_omni.diffusion.data import parse_attention_config

            attention_config = parse_attention_config(
                kwargs.get("diffusion_attention_config"),
                attention_backend=kwargs.get("diffusion_attention_backend"),
            )

        lora_scale = kwargs.get("lora_scale")
        if lora_scale is None:
            lora_scale = kwargs.get("static_lora_scale")

        dtype = kwargs.get("dtype", "auto")
        if not isinstance(dtype, str):
            dtype = str(dtype).removeprefix("torch.")

        profiler_config = kwargs.get("profiler_config")
        if profiler_config is not None and not isinstance(profiler_config, dict):
            if hasattr(profiler_config, "__dataclass_fields__"):
                profiler_config = asdict(profiler_config)

        return cls(
            parallel_config=parallel_config,
            model_class_name=kwargs.get("model_class_name"),
            additional_config=kwargs.get("additional_config"),
            custom_pipeline_args=kwargs.get("custom_pipeline_args"),
            worker_extension_cls=kwargs.get("worker_extension_cls"),
            max_num_seqs=kwargs.get("max_num_seqs") or 1,
            enforce_eager=False if kwargs.get("enforce_eager") is None else kwargs["enforce_eager"],
            step_execution=kwargs.get("step_execution", False),
            vae_use_slicing=kwargs.get("vae_use_slicing", False),
            vae_use_tiling=kwargs.get("vae_use_tiling", False),
            cache_backend=str(kwargs.get("cache_backend", "none")),
            cache_config=kwargs.get("cache_config"),
            enable_cache_dit_summary=kwargs.get("enable_cache_dit_summary", False),
            enable_cpu_offload=kwargs.get("enable_cpu_offload", False),
            enable_layerwise_offload=kwargs.get("enable_layerwise_offload", False),
            boundary_ratio=kwargs.get("boundary_ratio"),
            flow_shift=kwargs.get("flow_shift"),
            diffusion_load_format=kwargs.get("diffusion_load_format", "default"),
            enable_multithread_weight_load=kwargs.get("enable_multithread_weight_load", True),
            num_weight_load_threads=kwargs.get("num_weight_load_threads", 4),
            quantization=kwargs.get("quantization"),
            quantization_config=kwargs.get("quantization_config"),
            force_cutlass_fp8=bool(kwargs.get("force_cutlass_fp8", False)),
            kv_cache_dtype=kwargs.get("diffusion_kv_cache_dtype"),
            kv_cache_skip_steps=kwargs.get("diffusion_kv_cache_skip_steps"),
            kv_cache_skip_layers=kwargs.get("diffusion_kv_cache_skip_layers"),
            lora_path=kwargs.get("lora_path"),
            lora_scale=lora_scale,
            attention_config=attention_config,
            enable_diffusion_pipeline_profiler=kwargs.get("enable_diffusion_pipeline_profiler", False),
            enable_ar_profiler=kwargs.get("enable_ar_profiler", False),
            profiler_config=profiler_config,
            trust_remote_code=False if kwargs.get("trust_remote_code") is None else kwargs["trust_remote_code"],
            distributed_executor_backend=(
                "mp"
                if kwargs.get("distributed_executor_backend") is None
                else kwargs["distributed_executor_backend"]
            ),
            enable_sleep_mode=kwargs.get("enable_sleep_mode", False),
            diffusers_load_kwargs=kwargs.get("diffusers_load_kwargs"),
            diffusers_call_kwargs=kwargs.get("diffusers_call_kwargs"),
            auxiliary_text_encoder=kwargs.get("auxiliary_text_encoder"),
            default_llama_model_id=kwargs.get(
                "default_llama_model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct"
            ),
            dtype=dtype,
        )

    def to_stage_engine_args_dict(self, num_devices: int) -> dict[str, Any]:
        """Convert to the dict format expected by stage config consumers.

        Returns a dict compatible with both the old OmegaConf-based path
        (``create_default_diffusion``) and the new typed path.
        """
        result: dict[str, Any] = {
            "max_num_seqs": self.max_num_seqs,
            "model_class_name": self.model_class_name,
            "additional_config": self.additional_config,
            "step_execution": self.step_execution,
            "vae_use_slicing": self.vae_use_slicing,
            "vae_use_tiling": self.vae_use_tiling,
            "cache_backend": self.cache_backend,
            "cache_config": self.cache_config,
            "enable_cache_dit_summary": self.enable_cache_dit_summary,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_layerwise_offload": self.enable_layerwise_offload,
            "enforce_eager": self.enforce_eager,
            "boundary_ratio": self.boundary_ratio,
            "flow_shift": self.flow_shift,
            "diffusion_load_format": self.diffusion_load_format,
            "custom_pipeline_args": self.custom_pipeline_args,
            "worker_extension_cls": self.worker_extension_cls,
            "trust_remote_code": self.trust_remote_code,
            "distributed_executor_backend": self.distributed_executor_backend,
            "enable_sleep_mode": self.enable_sleep_mode,
            "enable_multithread_weight_load": self.enable_multithread_weight_load,
            "num_weight_load_threads": self.num_weight_load_threads,
            "quantization": self.quantization,
            "diffusion_kv_cache_dtype": self.kv_cache_dtype,
            "diffusion_kv_cache_skip_steps": self.kv_cache_skip_steps,
            "diffusion_kv_cache_skip_layers": self.kv_cache_skip_layers,
            "force_cutlass_fp8": self.force_cutlass_fp8,
            "enable_diffusion_pipeline_profiler": self.enable_diffusion_pipeline_profiler,
            "enable_ar_profiler": self.enable_ar_profiler,
            "extras": {
                "auxiliary_text_encoder": self.auxiliary_text_encoder,
                "default_llama_model_id": self.default_llama_model_id,
            },
        }

        if self.attention_config is not None:
            result["diffusion_attention_config"] = self.attention_config
        if self.parallel_config is not None:
            result["parallel_config"] = asdict(self.parallel_config)
        if self.quantization_config is not None:
            result["quantization_config"] = self.quantization_config
        if self.lora_path is not None:
            result["lora_path"] = self.lora_path
        if self.lora_scale is not None:
            result["lora_scale"] = self.lora_scale
        if self.diffusers_load_kwargs is not None:
            result["diffusers_load_kwargs"] = self.diffusers_load_kwargs
        if self.diffusers_call_kwargs is not None:
            result["diffusers_call_kwargs"] = self.diffusers_call_kwargs
        if self.profiler_config is not None:
            result["profiler_config"] = self.profiler_config
        if self.dtype != "auto":
            result["dtype"] = self.dtype
        result["model_stage"] = "diffusion"

        return result

    def to_default_stage_cfg(self, num_devices: int) -> dict[str, Any]:
        """Create a complete default diffusion stage config dict.

        Equivalent to what ``_create_default_diffusion_stage_cfg`` returned.
        """
        from vllm_omni.diffusion.diffusion_engine import supports_audio_output

        devices = ",".join(str(i) for i in range(max(1, int(num_devices))))
        final_output_type = "audio" if self.model_class_name and supports_audio_output(self.model_class_name) else "image"

        config_dict: dict[str, Any] = {
            "stage_id": 0,
            "stage_type": "diffusion",
            "runtime": {
                "process": True,
                "devices": devices,
            },
            "engine_args": self.to_stage_engine_args_dict(num_devices),
            "final_output": True,
            "final_output_type": final_output_type,
        }
        config_dict["engine_args"]["model_stage"] = "diffusion"
        return config_dict
