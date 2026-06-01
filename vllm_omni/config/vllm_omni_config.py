"""
Unified config classes for vLLM-Omni stage configuration.

Phase 2 of RFC #4021 — purely additive. These classes coexist with the
existing ``StageConfig`` / ``StageDeployConfig`` / ``merge_pipeline_deploy``
path during Phases 2-3. Consumers are cut over in Phases 3-4.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field
from vllm.config.utils import config

from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
)


@config(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """Per-stage model behavior and session defaults."""

    enable_sleep_mode: bool = False
    default_sampling_params: dict | None = None
    subtalker_sampling_params: dict | None = None
    has_sampling_extra_args: bool = False
    task_type: str | None = None
    codec_frame_rate_hz: float | None = None
    enforce_eager: bool = False
    enable_flashinfer_autotune: bool | None = None
    compilation_config: dict | None = None
    enable_multithread_weight_load: bool = True
    num_weight_load_threads: int = Field(default=4, ge=1)
    disable_autocast: bool = False


@config()
class LoadConfig:
    """Per-stage weight loading and tokenizer configuration."""

    load_format: str = "auto"
    tokenizer_mode: str = "auto"
    config_format: str | None = None
    skip_mm_profiling: bool | None = None


@config()
class CacheConfig:
    """Per-stage cache and memory behavior."""

    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    enable_prefix_caching: bool = False
    disable_hybrid_kv_cache_manager: bool = False
    mm_processor_cache_gb: float | None = Field(default=None, ge=0.0)


@config()
class SchedulerConfig:
    """Per-stage scheduling and batching behavior."""

    max_num_seqs: int = Field(default=128, ge=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    max_model_len: int | None = None
    enable_chunked_prefill: bool = False
    async_scheduling: bool = True


@config()
class ConnectorConfig:
    """Per-stage connector specification for inter-stage data transfer."""

    stage_connector: dict = Field(default_factory=lambda: {"name": "SharedMemoryConnector", "extra": {}})
    output_connectors: dict | None = None
    input_connectors: dict | None = None


@config()
class RuntimeConfig:
    """Per-stage process placement and runtime configuration."""

    devices: str | None = None
    num_replicas: int = Field(default=1, ge=1)
    env: dict | None = None
    requires_multimodal_data: bool = False
    process: bool = True
    log_level: str = "info"
    log_stats: bool = False
    profiler_config: dict | None = None


@config()
class ParallelConfig:
    """Per-stage distributed parallelism configuration."""

    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    sequence_parallel_size: int | None = Field(default=None, ge=1)
    ulysses_degree: int = Field(default=1, ge=1)
    ring_degree: int = Field(default=1, ge=1)
    ulysses_mode: Literal["strict", "advanced_uaa"] = "strict"
    cfg_parallel_size: int = Field(default=1, ge=1, le=3)
    vae_patch_parallel_size: int = Field(default=1, ge=1)
    use_hsdp: bool = False
    hsdp_shard_size: int = -1  # -1 = auto, cannot constrain with Field
    hsdp_replicate_size: int = Field(default=1, ge=1)
    enable_expert_parallel: bool = False
    world_size: int = Field(init=False)

    def __post_init__(self) -> None:
        """Resolve world_size and hsdp defaults (mirrors DiffusionParallelConfig)."""
        if self.sequence_parallel_size is None:
            self.sequence_parallel_size = self.ulysses_degree * self.ring_degree

        other_world = (
            self.pipeline_parallel_size
            * self.data_parallel_size
            * self.tensor_parallel_size
            * self.ulysses_degree
            * self.ring_degree
            * self.cfg_parallel_size
        )

        if self.use_hsdp:
            if self.tensor_parallel_size > 1 or self.data_parallel_size > 1:
                raise ValueError(
                    "HSDP cannot be combined with TP or DP "
                    f"(tp={self.tensor_parallel_size}, dp={self.data_parallel_size})"
                )
            if self.hsdp_shard_size == -1:
                if other_world == 1:
                    raise ValueError("Cannot auto-calculate hsdp_shard_size when other parallelism is all 1")
                if other_world % self.hsdp_replicate_size != 0:
                    raise ValueError(
                        f"hsdp_replicate_size ({self.hsdp_replicate_size}) "
                        f"must evenly divide world_size ({other_world}) "
                        "when shard_size is -1."
                    )
                self.hsdp_shard_size = other_world // self.hsdp_replicate_size
                self.world_size = other_world
            else:
                hsdp_world = self.hsdp_replicate_size * self.hsdp_shard_size
                if other_world == 1:
                    self.world_size = hsdp_world
                elif hsdp_world != other_world:
                    raise ValueError(f"HSDP world ({hsdp_world}) must match other parallelism world ({other_world})")
                else:
                    self.world_size = other_world
        else:
            self.world_size = other_world


@config(config=ConfigDict(arbitrary_types_allowed=True))
class DiffusionConfig:
    """Per-stage diffusion-specific configuration. None for AR stages."""

    model_class_name: str | None = None
    tf_model_config: dict = Field(default_factory=dict)
    diffusion_attention_config: dict = Field(default_factory=dict)
    cache_strategy: str = "none"
    cache_backend: str = "none"
    cache_config: dict = Field(default_factory=dict)
    enable_cache_dit_summary: bool = False
    diffusion_load_format: str = "default"
    diffusers_load_kwargs: dict = Field(default_factory=dict)
    diffusers_call_kwargs: dict = Field(default_factory=dict)
    diffusers_pipeline_cls: str | None = None
    lora_path: str | None = None
    lora_scale: float = 1.0
    max_cpu_loras: int | None = None
    output_type: str = "pil"
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    pin_cpu_memory: bool = True
    vae_use_slicing: bool = False
    vae_use_tiling: bool = False
    mask_strategy_file_path: str | None = None
    skip_time_steps: int = 15
    VSA_sparsity: float = 0.0
    moba_config_path: str | None = None
    boundary_ratio: float | None = None
    flow_shift: float | None = None
    diffusion_kv_cache_dtype: str | None = None
    diffusion_kv_cache_skip_steps: str | None = None
    diffusion_kv_cache_skip_layers: str | None = None
    force_cutlass_fp8: bool = False
    enable_diffusion_pipeline_profiler: bool = False
    step_execution: bool = False
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    model_paths: dict = Field(default_factory=dict)
    model_loaded: dict = Field(default_factory=lambda: {"transformer": True, "vae": True})
    override_transformer_cls_name: str | None = None
    worker_extension_cls: str | None = None
    custom_pipeline_args: dict | None = None
    additional_config: dict = Field(default_factory=dict)
    enable_stage_verification: bool = True
    host: str | None = None
    port: int | None = None
    scheduler_port: int = 5555
    master_port: int | None = None
    nccl_port: int | None = None
    dist_timeout: int | None = None
    prompt_file_path: str | None = None
    cfg_kv_collect_func: str | None = None
    quantization_config: str | dict | None = None
    extras: dict = Field(default_factory=dict)


@config()
class OrchestratorConfig:
    """Consumed only by the orchestrator process — not forwarded to stage engines."""

    stage_init_timeout: int = Field(default=300, ge=0)
    init_timeout: int = Field(default=600, ge=0)
    worker_backend: str = "multi_process"
    ray_address: str | None = None
    omni_master_address: str | None = None
    omni_master_port: int | None = None
    omni_dp_size_local: int = Field(default=1, ge=1)
    omni_lb_policy: str = "random"
    omni_heartbeat_timeout: float = Field(default=30.0, gt=0.0)
    shm_threshold_bytes: int = Field(default=65536, ge=0)
    batch_timeout: int = Field(default=10, ge=0)
    deploy_config_path: str | None = None


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniStageConfig:
    """All configuration for one pipeline stage, ready for engine initialization."""

    # Required fields must come before default fields (dataclass ordering rule).
    stage_pipeline_config: StagePipelineConfig  # required — no default
    input_processor: str | None = None
    model_config: ModelConfig = Field(default_factory=ModelConfig)
    load_config: LoadConfig = Field(default_factory=LoadConfig)
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = Field(default_factory=SchedulerConfig)
    connector_config: ConnectorConfig = Field(default_factory=ConnectorConfig)
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)
    diffusion_config: DiffusionConfig | None = None


@config(config=ConfigDict(arbitrary_types_allowed=True))
class VllmOmniConfig:
    """Constructed once at startup, passed to every stage engine during initialization."""

    pipeline: PipelineConfig  # required — no default
    stage_configs: tuple[VllmOmniStageConfig, ...]  # required — no default
    orchestrator_config: OrchestratorConfig  # required — no default

    def stage_by_id(self, stage_id: int) -> VllmOmniStageConfig:
        """Return the stage config for the given stage_id."""
        for s in self.stage_configs:
            if s.stage_pipeline_config.stage_id == stage_id:
                return s
        raise KeyError(f"No stage {stage_id}")

    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        pipeline: PipelineConfig,
        deploy: DeployConfig,
        explicit_kwargs: dict[str, Any] | None = None,
        platform: str | None = None,
    ) -> VllmOmniConfig:
        """Create from pipeline topology + deploy YAML + CLI overrides.

        Mirrors ``merge_pipeline_deploy()`` but produces strongly-typed
        sub-configs instead of a flat ``StageConfig`` with dict-based args.

        Args:
            pipeline: Frozen pipeline topology from the registry.
            deploy: Deploy config loaded from YAML (platform overrides already
                applied, or will be applied internally).
            explicit_kwargs: Dict of explicit keyword overrides.
            platform: Auto-detected if None.

        Returns:
            Fully resolved ``VllmOmniConfig`` ready for engine initialization.
        """
        if explicit_kwargs is None:
            explicit_kwargs = {}

        # Strip None values — the old flow filters these before applying
        # overrides (see ``_create_from_registry`` at stage_config.py:1258
        # and ``build_stage_runtime_overrides`` at stage_config.py:68).
        explicit_kwargs = {k: v for k, v in explicit_kwargs.items() if v is not None}

        # Apply async_chunk CLI override (mirrors stage_config.py:1243).
        cli_async_chunk = explicit_kwargs.get("async_chunk")
        if cli_async_chunk is not None:
            deploy.async_chunk = bool(cli_async_chunk)

        # Apply platform-specific overrides (reuse existing function).
        from vllm_omni.config.stage_config import _apply_platform_overrides

        deploy = _apply_platform_overrides(deploy, platform)

        # Validate async_chunk — pipeline must declare a processor.
        if deploy.async_chunk and not any(
            ps.async_chunk_process_next_stage_input_func or ps.custom_process_next_stage_input_func
            for ps in pipeline.stages
        ):
            raise ValueError(
                f"Pipeline {pipeline.model_type!r} has async_chunk=True in "
                "deploy but no stage declares a next-stage input processor."
            )

        deploy_stages_by_id = {s.stage_id: s for s in deploy.stages}

        stage_configs: list[VllmOmniStageConfig] = []
        for ps in pipeline.stages:
            ds = deploy_stages_by_id.get(ps.stage_id)
            # Resolve per-stage overrides from both global keys and
            # stage_N_* keys (mirrors stage_config.py:1261).
            from vllm_omni.config.stage_config import build_stage_runtime_overrides

            stage_kwargs = build_stage_runtime_overrides(ps.stage_id, explicit_kwargs)
            stage_cfg = cls._build_stage_config(ps, ds, pipeline, deploy, explicit_kwargs, stage_kwargs)
            stage_configs.append(stage_cfg)

        orchestrator = cls._build_orchestrator_config(explicit_kwargs)

        return cls(
            pipeline=pipeline,
            stage_configs=tuple(stage_configs),
            orchestrator_config=orchestrator,
        )

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_stage_config(
        ps: StagePipelineConfig,
        ds: StageDeployConfig | None,
        pipeline: PipelineConfig,
        deploy: DeployConfig,
        explicit_kwargs: dict[str, Any],
        stage_overrides: dict[str, Any] | None = None,
    ) -> VllmOmniStageConfig:
        """Build one ``VllmOmniStageConfig`` from pipeline + deploy + CLI.

        Reuses the existing ``_build_engine_args`` and ``_build_extras``
        logic from ``stage_config.py`` for guaranteed field parity, then
        extracts typed sub-configs from the resulting dicts.

        ``stage_overrides`` is the resolved per-stage override dict computed
        by ``build_stage_runtime_overrides`` — overrides from global keys
        and ``stage_N_*`` keys that apply to this particular stage.
        """
        if stage_overrides is None:
            stage_overrides = {}

        # Resolve execution mode and processor functions.
        # ``input_proc`` is the resolved dotted path (stored on the stage
        # config so async_chunk-dependent switching works at runtime).
        # ``next_stage_proc`` is threaded into engine_args via _build_engine_args.
        from vllm_omni.config.stage_config import (
            OmniARAsyncScheduler,
            _build_engine_args,
            _build_extras,
            _resolve_execution_mode,
            _select_processor_funcs,
        )

        stage_type, worker_type = _resolve_execution_mode(ps.execution_type)
        input_proc, next_stage_proc = _select_processor_funcs(ps, deploy.async_chunk)

        # Build the same flat dicts as ``merge_pipeline_deploy``.
        engine_args: dict[str, Any] = _build_engine_args(ps, ds, pipeline, deploy, next_stage_proc)
        extras: dict[str, Any] = _build_extras(ps, ds)

        # LLM_AR scheduler wiring (mirrors merge_pipeline_deploy).
        if ps.execution_type == StageExecutionType.LLM_AR:
            from vllm_omni.config.stage_config import _resolve_scheduler

            sched_cls = _resolve_scheduler(ps.execution_type, engine_args.get("async_scheduling", True))
            engine_args["async_scheduling"] = sched_cls is OmniARAsyncScheduler

        # Runtime (mirrors merge_pipeline_deploy).
        runtime: dict[str, Any] = {"process": True}
        if ds is not None:
            if ds.devices is not None:
                runtime["devices"] = ds.devices
            runtime["num_replicas"] = ds.num_replicas
            if ds.env is not None:
                runtime["env"] = ds.env
        runtime["requires_multimodal_data"] = ps.requires_multimodal_data

        # --- Populate typed sub-configs from the flat dicts ---
        # ``_coerce`` returns ``value`` when not None, else ``default``.
        # This is used instead of ``or`` so valid falsey values (False, 0)
        # are preserved by both stage_overrides and engine_args.
        def _coerce(value: Any, default: Any) -> Any:
            return value if value is not None else default

        # ModelConfig
        sampling = extras.get("default_sampling_params") or {}
        model_config = ModelConfig(
            enable_sleep_mode=bool(stage_overrides.get("enable_sleep_mode", False)),
            default_sampling_params=sampling if sampling else None,
            subtalker_sampling_params=_coerce(
                stage_overrides.get("subtalker_sampling_params"),
                engine_args.get("subtalker_sampling_params"),
            ),
            has_sampling_extra_args=bool(sampling.get("extra_args")),
            task_type=explicit_kwargs.get("task_type"),
            codec_frame_rate_hz=None,
            enforce_eager=bool(
                _coerce(
                    stage_overrides.get("enforce_eager"),
                    _coerce(engine_args.get("enforce_eager"), False),
                )
            ),
            enable_flashinfer_autotune=_coerce(
                stage_overrides.get("enable_flashinfer_autotune"),
                engine_args.get("enable_flashinfer_autotune"),
            ),
            compilation_config=_coerce(
                stage_overrides.get("compilation_config"),
                engine_args.get("compilation_config"),
            ),
            enable_multithread_weight_load=_coerce(engine_args.get("enable_multithread_weight_load"), True),
            num_weight_load_threads=_coerce(
                _coerce(
                    stage_overrides.get("num_weight_load_threads"),
                    engine_args.get("num_weight_load_threads"),
                ),
                4,
            ),
            disable_autocast=bool(_coerce(engine_args.get("disable_autocast"), False)),
        )

        # LoadConfig
        load_config = LoadConfig(
            load_format=engine_args.get("load_format") or "auto",
            tokenizer_mode=engine_args.get("tokenizer_mode") or "auto",
            config_format=engine_args.get("config_format"),
            skip_mm_profiling=engine_args.get("skip_mm_profiling"),
        )

        # CacheConfig
        cache_config = CacheConfig(
            gpu_memory_utilization=float(
                _coerce(
                    stage_overrides.get("gpu_memory_utilization"),
                    _coerce(engine_args.get("gpu_memory_utilization"), 0.90),
                )
            ),
            enable_prefix_caching=bool(
                _coerce(
                    stage_overrides.get("enable_prefix_caching"),
                    _coerce(engine_args.get("enable_prefix_caching"), False),
                )
            ),
            disable_hybrid_kv_cache_manager=bool(
                _coerce(
                    stage_overrides.get("disable_hybrid_kv_cache_manager"),
                    _coerce(engine_args.get("disable_hybrid_kv_cache_manager"), False),
                )
            ),
            mm_processor_cache_gb=_coerce(
                stage_overrides.get("mm_processor_cache_gb"),
                engine_args.get("mm_processor_cache_gb"),
            ),
        )

        # SchedulerConfig
        scheduler_config = SchedulerConfig(
            max_num_seqs=int(
                _coerce(
                    stage_overrides.get("max_num_seqs"),
                    _coerce(engine_args.get("max_num_seqs"), 128),
                )
            ),
            max_num_batched_tokens=_coerce(
                stage_overrides.get("max_num_batched_tokens"),
                engine_args.get("max_num_batched_tokens"),
            ),
            max_model_len=_coerce(
                stage_overrides.get("max_model_len"),
                engine_args.get("max_model_len"),
            ),
            enable_chunked_prefill=bool(
                _coerce(
                    stage_overrides.get("enable_chunked_prefill"),
                    _coerce(engine_args.get("enable_chunked_prefill"), False),
                )
            ),
            async_scheduling=bool(
                _coerce(
                    stage_overrides.get("async_scheduling"),
                    _coerce(engine_args.get("async_scheduling"), True),
                )
            ),
        )

        # ConnectorConfig
        connector_config = ConnectorConfig(
            stage_connector=extras.get("stage_connector")
            or {
                "name": "SharedMemoryConnector",
                "extra": {},
            },
            output_connectors=extras.get("output_connectors"),
            input_connectors=extras.get("input_connectors"),
        )

        # RuntimeConfig
        runtime_config = RuntimeConfig(
            devices=stage_overrides.get("devices", runtime.get("devices")),
            num_replicas=int(
                _coerce(
                    stage_overrides.get("num_replicas"),
                    _coerce(runtime.get("num_replicas"), 1),
                )
            ),
            env=runtime.get("env"),
            requires_multimodal_data=runtime.get("requires_multimodal_data", False),
            process=runtime.get("process", True),
            log_level=engine_args.get("log_level") or "info",
            log_stats=bool(explicit_kwargs.get("log_stats", False)),
            profiler_config=engine_args.get("profiler_config"),
        )

        # ParallelConfig — per-stage overrides win over deploy YAML values.
        parallel_config = ParallelConfig(
            tensor_parallel_size=int(
                _coerce(
                    stage_overrides.get("tensor_parallel_size"),
                    _coerce(engine_args.get("tensor_parallel_size"), 1),
                )
            ),
            pipeline_parallel_size=int(
                _coerce(
                    stage_overrides.get("pipeline_parallel_size"),
                    _coerce(engine_args.get("pipeline_parallel_size"), 1),
                )
            ),
            data_parallel_size=int(
                _coerce(
                    stage_overrides.get("data_parallel_size"),
                    _coerce(engine_args.get("data_parallel_size"), 1),
                )
            ),
            sequence_parallel_size=_coerce(
                stage_overrides.get("sequence_parallel_size"),
                engine_args.get("sequence_parallel_size"),
            ),
            ulysses_degree=int(
                _coerce(
                    stage_overrides.get("ulysses_degree"),
                    _coerce(engine_args.get("ulysses_degree"), 1),
                )
            ),
            ring_degree=int(
                _coerce(
                    stage_overrides.get("ring_degree"),
                    _coerce(engine_args.get("ring_degree"), 1),
                )
            ),
            ulysses_mode=(stage_overrides.get("ulysses_mode") or engine_args.get("ulysses_mode") or "strict"),
            cfg_parallel_size=int(
                _coerce(
                    stage_overrides.get("cfg_parallel_size"),
                    _coerce(engine_args.get("cfg_parallel_size"), 1),
                )
            ),
            vae_patch_parallel_size=int(
                _coerce(
                    stage_overrides.get("vae_patch_parallel_size"),
                    _coerce(engine_args.get("vae_patch_parallel_size"), 1),
                )
            ),
            use_hsdp=bool(
                _coerce(
                    stage_overrides.get("use_hsdp"),
                    _coerce(engine_args.get("use_hsdp"), False),
                )
            ),
            hsdp_shard_size=(
                int(stage_overrides.get("hsdp_shard_size"))
                if stage_overrides.get("hsdp_shard_size") is not None
                else engine_args.get("hsdp_shard_size")
                if engine_args.get("hsdp_shard_size") is not None
                else -1
            ),
            hsdp_replicate_size=int(
                _coerce(
                    stage_overrides.get("hsdp_replicate_size"),
                    _coerce(engine_args.get("hsdp_replicate_size"), 1),
                )
            ),
            enable_expert_parallel=bool(
                _coerce(
                    stage_overrides.get("enable_expert_parallel"),
                    _coerce(engine_args.get("enable_expert_parallel"), False),
                )
            ),
        )

        # DiffusionConfig
        diffusion_config: DiffusionConfig | None = None
        if ps.execution_type == StageExecutionType.DIFFUSION:
            diffusion_config = DiffusionConfig(
                model_class_name=engine_args.get("model_class_name"),
                tf_model_config=engine_args.get("tf_model_config") or {},
                diffusion_attention_config=engine_args.get("diffusion_attention_config") or {},
                cache_strategy=engine_args.get("cache_strategy") or "none",
                cache_backend=engine_args.get("cache_backend") or "none",
                cache_config=engine_args.get("cache_config") or {},
                enable_cache_dit_summary=bool(engine_args.get("enable_cache_dit_summary") or False),
                diffusion_load_format=engine_args.get("diffusion_load_format") or "default",
                diffusers_load_kwargs=engine_args.get("diffusers_load_kwargs") or {},
                diffusers_call_kwargs=engine_args.get("diffusers_call_kwargs") or {},
                diffusers_pipeline_cls=engine_args.get("diffusers_pipeline_cls"),
                lora_path=engine_args.get("lora_path"),
                lora_scale=float(_coerce(engine_args.get("lora_scale"), 1.0)),
                max_cpu_loras=engine_args.get("max_cpu_loras"),
                output_type=engine_args.get("output_type") or "pil",
                enable_cpu_offload=bool(engine_args.get("enable_cpu_offload") or False),
                enable_layerwise_offload=bool(engine_args.get("enable_layerwise_offload") or False),
                pin_cpu_memory=_coerce(engine_args.get("pin_cpu_memory"), True),
                vae_use_slicing=bool(engine_args.get("vae_use_slicing") or False),
                vae_use_tiling=bool(engine_args.get("vae_use_tiling") or False),
                mask_strategy_file_path=engine_args.get("mask_strategy_file_path"),
                skip_time_steps=int(_coerce(engine_args.get("skip_time_steps"), 15)),
                VSA_sparsity=float(_coerce(engine_args.get("VSA_sparsity"), 0.0)),
                moba_config_path=engine_args.get("moba_config_path"),
                boundary_ratio=engine_args.get("boundary_ratio"),
                flow_shift=engine_args.get("flow_shift"),
                diffusion_kv_cache_dtype=engine_args.get("diffusion_kv_cache_dtype"),
                diffusion_kv_cache_skip_steps=engine_args.get("diffusion_kv_cache_skip_steps"),
                diffusion_kv_cache_skip_layers=engine_args.get("diffusion_kv_cache_skip_layers"),
                force_cutlass_fp8=bool(engine_args.get("force_cutlass_fp8") or False),
                enable_diffusion_pipeline_profiler=bool(engine_args.get("enable_diffusion_pipeline_profiler") or False),
                step_execution=bool(engine_args.get("step_execution") or False),
                supports_multimodal_inputs=bool(engine_args.get("supports_multimodal_inputs") or False),
                max_multimodal_image_inputs=engine_args.get("max_multimodal_image_inputs"),
                model_paths=engine_args.get("model_paths") or {},
                model_loaded=engine_args.get("model_loaded") or {"transformer": True, "vae": True},
                override_transformer_cls_name=engine_args.get("override_transformer_cls_name"),
                worker_extension_cls=engine_args.get("worker_extension_cls"),
                custom_pipeline_args=engine_args.get("custom_pipeline_args"),
                additional_config=engine_args.get("additional_config") or {},
                enable_stage_verification=_coerce(engine_args.get("enable_stage_verification"), True),
                host=engine_args.get("host"),
                port=engine_args.get("port"),
                scheduler_port=int(_coerce(engine_args.get("scheduler_port"), 5555)),
                master_port=engine_args.get("master_port"),
                nccl_port=engine_args.get("nccl_port"),
                dist_timeout=engine_args.get("dist_timeout"),
                prompt_file_path=engine_args.get("prompt_file_path"),
                cfg_kv_collect_func=ps.cfg_kv_collect_func,
                quantization_config=engine_args.get("quantization_config") or engine_args.get("quantization"),
                extras=engine_args.get("extras") or {},
            )

        return VllmOmniStageConfig(
            stage_pipeline_config=ps,
            input_processor=input_proc,
            model_config=model_config,
            load_config=load_config,
            cache_config=cache_config,
            scheduler_config=scheduler_config,
            connector_config=connector_config,
            runtime_config=runtime_config,
            parallel_config=parallel_config,
            diffusion_config=diffusion_config,
        )

    @staticmethod
    def _build_orchestrator_config(
        explicit_kwargs: dict[str, Any],
    ) -> OrchestratorConfig:
        """Build orchestrator config from CLI overrides.

        Uses ``.get(key, default)`` (not ``or``) because valid values include
        ``0`` (timeouts, thresholds) that must not be replaced by defaults.
        """
        return OrchestratorConfig(
            stage_init_timeout=explicit_kwargs.get("stage_init_timeout", 300),
            init_timeout=explicit_kwargs.get("init_timeout", 600),
            worker_backend=explicit_kwargs.get("worker_backend", "multi_process"),
            ray_address=explicit_kwargs.get("ray_address"),
            omni_master_address=explicit_kwargs.get("omni_master_address"),
            omni_master_port=explicit_kwargs.get("omni_master_port"),
            omni_dp_size_local=explicit_kwargs.get("omni_dp_size_local", 1),
            omni_lb_policy=explicit_kwargs.get("omni_lb_policy", "random"),
            omni_heartbeat_timeout=explicit_kwargs.get("omni_heartbeat_timeout", 30.0),
            shm_threshold_bytes=explicit_kwargs.get("shm_threshold_bytes", 65536),
            batch_timeout=explicit_kwargs.get("batch_timeout", 10),
            deploy_config_path=explicit_kwargs.get("deploy_config_path") or explicit_kwargs.get("deploy_config"),
        )
