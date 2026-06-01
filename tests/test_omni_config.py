"""Tests for vllm_omni.config.vllm_omni_config — RFC #4021 Phase 2.

Verifies that ``VllmOmniConfig.from_registry()`` produces field-identical
results to the existing ``merge_pipeline_deploy()`` for all registered models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    _PIPELINE_REGISTRY,
    DeployConfig,
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.config.vllm_omni_config import (
    CacheConfig,
    ConnectorConfig,
    DiffusionConfig,
    ModelConfig,
    OrchestratorConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmOmniConfig,
    VllmOmniStageConfig,
)

# ---------------------------------------------------------------------------
# Basic instantiation and validation tests
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig()
        assert mc.enable_sleep_mode is False
        assert mc.enforce_eager is False
        assert mc.default_sampling_params is None

    def test_overrides(self):
        mc = ModelConfig(enforce_eager=True, compilation_config={"mode": "max-autotune"})
        assert mc.enforce_eager is True
        assert mc.compilation_config == {"mode": "max-autotune"}


class TestCacheConfig:
    def test_valid(self):
        CacheConfig()
        CacheConfig(gpu_memory_utilization=0.95, mm_processor_cache_gb=4.0)

    def test_invalid_gpu_memory_utilization(self):
        with pytest.raises(ValidationError):
            CacheConfig(gpu_memory_utilization=0.0)

    def test_invalid_mm_cache(self):
        with pytest.raises(ValidationError):
            CacheConfig(mm_processor_cache_gb=-1.0)


class TestSchedulerConfig:
    def test_defaults(self):
        sc = SchedulerConfig()
        assert sc.max_num_seqs == 128
        assert sc.max_num_batched_tokens is None
        assert sc.async_scheduling is True


class TestParallelConfig:
    def test_world_size_simple(self):
        pc = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
        assert pc.world_size == 4

    def test_world_size_ulysses_ring(self):
        pc = ParallelConfig(ulysses_degree=2, ring_degree=2)
        assert pc.sequence_parallel_size == 4
        assert pc.world_size == 4

    def test_ulysses_mode_validation(self):
        with pytest.raises(ValidationError):
            ParallelConfig(ulysses_mode="invalid")

    def test_cfg_parallel_validation(self):
        with pytest.raises(ValidationError):
            ParallelConfig(cfg_parallel_size=5)


class TestConnectorConfig:
    def test_defaults(self):
        cc = ConnectorConfig()
        assert cc.stage_connector["name"] == "SharedMemoryConnector"

    def test_isolation(self):
        a = ConnectorConfig()
        b = ConnectorConfig()
        a.stage_connector["extra"] = {"changed": True}
        assert b.stage_connector["extra"] == {}


class TestDiffusionConfig:
    def test_defaults(self):
        dc = DiffusionConfig()
        assert dc.model_loaded["transformer"] is True
        assert dc.cache_backend == "none"

    def test_isolation(self):
        a = DiffusionConfig()
        b = DiffusionConfig()
        a.model_loaded["transformer"] = False
        assert b.model_loaded["transformer"] is True


class TestVllmOmniStageConfig:
    def test_requires_stage_pipeline_config(self):
        spc = StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
        )
        stage = VllmOmniStageConfig(stage_pipeline_config=spc)
        assert stage.model_config.enable_sleep_mode is False
        assert stage.diffusion_config is None
        assert stage.stage_pipeline_config.stage_id == 0

    def test_diffusion_stage(self):
        spc = StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
        )
        stage = VllmOmniStageConfig(
            stage_pipeline_config=spc,
            diffusion_config=DiffusionConfig(cache_backend="tea_cache"),
        )
        assert stage.diffusion_config is not None
        assert stage.diffusion_config.cache_backend == "tea_cache"


class TestVllmOmniConfig:
    def test_create_and_stage_by_id(self):
        spc = StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
        )
        pipeline = PipelineConfig(model_type="test", stages=(spc,))
        stage = VllmOmniStageConfig(stage_pipeline_config=spc)
        cfg = VllmOmniConfig(
            pipeline=pipeline,
            stage_configs=(stage,),
            orchestrator_config=OrchestratorConfig(),
        )

        assert cfg.stage_by_id(0) is stage
        with pytest.raises(KeyError):
            cfg.stage_by_id(99)


# ---------------------------------------------------------------------------
# Field-parity tests — ensure from_registry() == merge_pipeline_deploy()
# ---------------------------------------------------------------------------


def _model_ids_with_deploy() -> list[str]:
    """Return registered model types that have a deploy YAML."""
    return [mt for mt in sorted(_PIPELINE_REGISTRY.keys()) if (_DEPLOY_DIR / f"{mt}.yaml").exists()]


@pytest.mark.parametrize("model_type", _model_ids_with_deploy())
def test_from_registry_field_parity(model_type: str) -> None:
    """``from_registry()`` must produce identical values to ``merge_pipeline_deploy()``."""
    pipeline = _PIPELINE_REGISTRY[model_type]
    deploy = load_deploy_config(str(_DEPLOY_DIR / f"{model_type}.yaml"))

    old_stages = merge_pipeline_deploy(pipeline, deploy)
    new_config = VllmOmniConfig.from_registry(pipeline, deploy)

    assert len(old_stages) == len(new_config.stage_configs), (
        f"{model_type}: old stages={len(old_stages)} new stages={len(new_config.stage_configs)}"
    )

    for old, new in zip(old_stages, new_config.stage_configs):
        o = old.yaml_engine_args
        r = old.yaml_runtime
        e = old.yaml_extras
        sp = new.stage_pipeline_config

        # Structural identity
        assert old.stage_id == sp.stage_id, f"stage_id: {old.stage_id} vs {sp.stage_id}"
        assert old.model_stage == sp.model_stage
        assert old.final_output == sp.final_output
        assert old.final_output_type == sp.final_output_type
        assert old.is_comprehension == sp.owns_tokenizer
        assert old.hf_config_name == sp.hf_config_name

        # Resolved input processor (async_chunk-dependent switch).
        assert new.input_processor == old.custom_process_input_func

        # ModelConfig
        mc = new.model_config
        assert mc.enforce_eager == bool(o.get("enforce_eager", False)), (
            f"enforce_eager: {mc.enforce_eager} vs {o.get('enforce_eager')}"
        )
        assert mc.enable_multithread_weight_load == bool(o.get("enable_multithread_weight_load", True))
        assert mc.num_weight_load_threads == int(o.get("num_weight_load_threads") or 4)

        # LoadConfig
        lc = new.load_config
        assert lc.load_format == (o.get("load_format") or "auto")
        assert lc.tokenizer_mode == (o.get("tokenizer_mode") or "auto")
        assert lc.config_format == o.get("config_format")
        assert lc.skip_mm_profiling == o.get("skip_mm_profiling")

        # CacheConfig
        cc = new.cache_config
        assert abs(cc.gpu_memory_utilization - float(o.get("gpu_memory_utilization") or 0.90)) < 0.001
        assert cc.enable_prefix_caching == bool(o.get("enable_prefix_caching", False))
        assert cc.disable_hybrid_kv_cache_manager == bool(o.get("disable_hybrid_kv_cache_manager", False))
        assert cc.mm_processor_cache_gb == o.get("mm_processor_cache_gb")

        # SchedulerConfig
        sc = new.scheduler_config
        assert sc.max_num_seqs == int(o.get("max_num_seqs") or 128)
        assert sc.max_num_batched_tokens == o.get("max_num_batched_tokens")
        assert sc.max_model_len == o.get("max_model_len")
        assert sc.enable_chunked_prefill == bool(o.get("enable_chunked_prefill", False))
        assert sc.async_scheduling == bool(o.get("async_scheduling", True))

        # RuntimeConfig
        rc = new.runtime_config
        assert rc.devices == r.get("devices")
        assert rc.num_replicas == r.get("num_replicas", 1)
        assert rc.env == r.get("env")
        assert rc.requires_multimodal_data == r.get("requires_multimodal_data", False)
        assert rc.process == r.get("process", True)

        # ConnectorConfig
        conn = new.connector_config
        assert conn.output_connectors == e.get("output_connectors")
        assert conn.input_connectors == e.get("input_connectors")

        # ParallelConfig
        pc = new.parallel_config
        assert pc.tensor_parallel_size == int(o.get("tensor_parallel_size") or 1)
        assert pc.ulysses_degree == int(o.get("ulysses_degree") or 1)
        assert pc.ring_degree == int(o.get("ring_degree") or 1)
        assert pc.enable_expert_parallel == bool(o.get("enable_expert_parallel", False))

        # DiffusionConfig (only if stage is diffusion)
        if sp.execution_type == StageExecutionType.DIFFUSION:
            assert new.diffusion_config is not None, "Diffusion stage missing diffusion_config"
            dc = new.diffusion_config
            assert dc.model_class_name == o.get("model_class_name")
            assert dc.cache_backend == (o.get("cache_backend") or "none")
            assert dc.diffusers_load_kwargs == (o.get("diffusers_load_kwargs") or {})
            assert dc.pin_cpu_memory == (o.get("pin_cpu_memory") if o.get("pin_cpu_memory") is not None else True)
            assert dc.custom_pipeline_args == o.get("custom_pipeline_args")
            assert dc.additional_config == (o.get("additional_config") or {})
        else:
            assert new.diffusion_config is None, f"Non-diffusion stage {sp.model_stage} has diffusion_config set"


def test_from_registry_async_chunk_validation() -> None:
    """Pipeline without async processor should raise if async_chunk=True."""
    spc = StagePipelineConfig(
        stage_id=0,
        model_stage="thinker",
        execution_type=StageExecutionType.LLM_AR,
    )
    pipeline = PipelineConfig(model_type="test_no_async", stages=(spc,))
    deploy = DeployConfig(async_chunk=True)

    with pytest.raises(ValueError, match="async_chunk"):
        VllmOmniConfig.from_registry(pipeline, deploy)


def test_from_registry_with_platform_overrides() -> None:
    """Platform overrides should propagate correctly."""
    if "qwen3_omni_moe" not in _PIPELINE_REGISTRY:
        pytest.skip("qwen3_omni_moe not in registry")

    pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
    deploy_path = _DEPLOY_DIR / "qwen3_omni_moe.yaml"
    if not deploy_path.exists():
        pytest.skip("qwen3_omni_moe deploy YAML not found")

    deploy = load_deploy_config(str(deploy_path))

    # Auto-detected platform
    cfg = VllmOmniConfig.from_registry(pipeline, deploy)
    tp0 = cfg.stage_configs[0].parallel_config.tensor_parallel_size
    assert tp0 >= 1, f"Expected tp >= 1, got {tp0}"

    # Explicit platform=cuda override
    cfg_cuda = VllmOmniConfig.from_registry(pipeline, deploy, platform="cuda")
    assert cfg_cuda.stage_configs[0].parallel_config.tensor_parallel_size >= 1


def test_from_registry_with_stage_overrides() -> None:
    """stage_N_* overrides should apply to the correct stage, falsey values preserved."""
    if "bagel_single_stage" not in _PIPELINE_REGISTRY:
        pytest.skip("bagel_single_stage not in registry")

    pipeline = _PIPELINE_REGISTRY["bagel_single_stage"]
    deploy_path = _DEPLOY_DIR / "bagel_single_stage.yaml"
    if not deploy_path.exists():
        pytest.skip("bagel_single_stage deploy YAML not found")

    deploy = load_deploy_config(str(deploy_path))

    # Override tensor_parallel_size for stage 0 via stage_0_* key.
    cfg = VllmOmniConfig.from_registry(pipeline, deploy, explicit_kwargs={"stage_0_tensor_parallel_size": 2})
    assert cfg.stage_configs[0].parallel_config.tensor_parallel_size == 2

    # Override via a global key that becomes a stage override.
    cfg_global = VllmOmniConfig.from_registry(pipeline, deploy, explicit_kwargs={"gpu_memory_utilization": 0.50})
    assert abs(cfg_global.stage_configs[0].cache_config.gpu_memory_utilization - 0.50) < 0.001

    # Falsey stage override should be preserved (not fall through to default).
    cfg_falsey = VllmOmniConfig.from_registry(
        pipeline,
        deploy,
        explicit_kwargs={"stage_0_enable_prefix_caching": False},
    )
    assert cfg_falsey.stage_configs[0].cache_config.enable_prefix_caching is False


def test_hsdp_divisibility_check() -> None:
    """Invalid HSDP replicate_size should raise."""
    from vllm_omni.config.vllm_omni_config import ParallelConfig

    with pytest.raises(ValueError, match="evenly divide"):
        ParallelConfig(ulysses_degree=2, ring_degree=2, use_hsdp=True, hsdp_replicate_size=3)


def test_falsey_values_preserved() -> None:
    """batch_timeout=0 and enable_multithread_weight_load=False should be preserved."""
    from vllm_omni.config.vllm_omni_config import OrchestratorConfig

    oc = OrchestratorConfig(batch_timeout=0)
    assert oc.batch_timeout == 0

    from vllm_omni.config.vllm_omni_config import ModelConfig

    mc = ModelConfig(enable_multithread_weight_load=False)
    assert mc.enable_multithread_weight_load is False
