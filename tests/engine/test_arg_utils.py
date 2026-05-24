"""
Tests for OmniEngineArgs — the consolidated config entrypoint.
"""

from __future__ import annotations

import argparse
from dataclasses import fields as dc_fields
from types import SimpleNamespace

import pytest

from vllm_omni.config.vllm_omni_config import (
    StageResolvedConfig,
    VllmOmniConfig,
    _detect_pd_config,
    _resolve_dotted_func,
)
from vllm_omni.engine.arg_utils import OmniEngineArgs, OmniArgumentParser
from vllm_omni.engine.stage_init_utils import (
    _get_devices_per_replica_from_resolved,
    build_vllm_config_from_engine_args,
    compute_per_replica_devices,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ─────────────────────────────────────────────────────────
# OmniEngineArgs — basic construction
# ─────────────────────────────────────────────────────────


def test_omni_engine_args_defaults():
    """All fields have sensible defaults."""
    ea = OmniEngineArgs()
    assert ea.stage_init_timeout == 300
    assert ea.init_timeout == 600
    assert ea.worker_backend == "multi_process"
    assert ea.log_stats is False
    assert ea.stage_id is None


def test_omni_engine_args_from_cli_args():
    """from_cli_args extracts all fields from a Namespace."""
    ns = argparse.Namespace(
        model="test-model",
        stage_init_timeout=120,
        async_chunk=True,
        log_stats=True,
        worker_backend="ray",
    )
    ea = OmniEngineArgs.from_cli_args(ns)
    assert ea.model == "test-model"
    assert ea.stage_init_timeout == 120
    assert ea.async_chunk is True
    assert ea.log_stats is True
    assert ea.worker_backend == "ray"


# ─────────────────────────────────────────────────────────
# create_diffusion_config — explicit field construction
# ─────────────────────────────────────────────────────────


def test_create_diffusion_config_builds_parallel_config():
    """Parallelism fields are mapped to DiffusionParallelConfig."""
    ea = OmniEngineArgs(
        ulysses_degree=4,
        ring_degree=2,
        cfg_parallel_size=2,
        vae_patch_parallel_size=2,
        ulysses_mode="advanced_uaa",
    )
    config = ea.create_diffusion_config()
    pc = config.parallel_config
    assert pc.ulysses_degree == 4
    assert pc.ring_degree == 2
    assert pc.cfg_parallel_size == 2
    assert pc.vae_patch_parallel_size == 2
    assert pc.ulysses_mode == "advanced_uaa"


def test_create_diffusion_config_default_parallel_config():
    """Default parallelism produces a valid DiffusionParallelConfig."""
    ea = OmniEngineArgs()
    config = ea.create_diffusion_config()
    pc = config.parallel_config
    assert pc.ulysses_degree == 1
    assert pc.ring_degree == 1
    assert pc.cfg_parallel_size == 1


def test_create_diffusion_config_tp_dp_pp_propagation():
    """tensor_parallel_size, data_parallel_size, pipeline_parallel_size
    are passed from OmniEngineArgs to DiffusionParallelConfig and
    correctly compute world_size."""
    ea = OmniEngineArgs(tensor_parallel_size=4, data_parallel_size=2, pipeline_parallel_size=1)
    config = ea.create_diffusion_config()
    pc = config.parallel_config
    assert pc.tensor_parallel_size == 4
    assert pc.data_parallel_size == 2
    assert pc.pipeline_parallel_size == 1
    # world_size is computed in DiffusionParallelConfig.__post_init__
    assert pc.world_size == 8  # 4*2*1


def test_create_diffusion_config_tp_dp_pp_defaults():
    """When not set, tensor_parallel_size defaults to 1."""
    ea = OmniEngineArgs(ulysses_degree=2)
    config = ea.create_diffusion_config()
    pc = config.parallel_config
    assert pc.tensor_parallel_size == 1
    assert pc.data_parallel_size == 1
    assert pc.pipeline_parallel_size == 1
    # world_size: ulysses_degree=2 * (1*1*1*2*1*1) = 2
    assert pc.world_size == 2


def test_create_diffusion_config_explicit_field_mapping():
    """All explicit fields are mapped correctly."""
    ea = OmniEngineArgs(
        model_class_name="TestPipeline",
        cache_backend="tea_cache",
        enable_cache_dit_summary=True,
        lora_path="/path/to/lora",
        lora_scale=0.5,
        enable_cpu_offload=True,
        enable_layerwise_offload=True,
        vae_use_slicing=True,
        vae_use_tiling=True,
        enforce_eager=True,
        enable_multithread_weight_load=False,
        num_weight_load_threads=8,
        enable_sleep_mode=True,
        worker_extension_cls="my.Worker",
        custom_pipeline_args={"key": "val"},
        diffusion_load_format="diffusers",
        diffusers_load_kwargs={"load": "kwarg"},
        diffusers_call_kwargs={"call": "kwarg"},
        trust_remote_code=True,
    )
    config = ea.create_diffusion_config()
    assert config.model_class_name == "TestPipeline"
    assert config.cache_backend == "tea_cache"
    assert config.enable_cache_dit_summary is True
    assert config.lora_path == "/path/to/lora"
    assert config.lora_scale == 0.5
    assert config.enable_cpu_offload is True
    assert config.enable_layerwise_offload is True
    assert config.vae_use_slicing is True
    assert config.vae_use_tiling is True
    assert config.enforce_eager is True
    assert config.enable_multithread_weight_load is False
    assert config.num_weight_load_threads == 8
    assert config.enable_sleep_mode is True
    assert config.worker_extension_cls == "my.Worker"
    assert config.custom_pipeline_args == {"key": "val"}
    assert config.diffusion_load_format == "diffusers"
    assert config.diffusers_load_kwargs == {"load": "kwarg"}
    assert config.diffusers_call_kwargs == {"call": "kwarg"}
    assert config.trust_remote_code is True


# ─────────────────────────────────────────────────────────
# StageResolvedConfig
# ─────────────────────────────────────────────────────────


def test_stage_resolved_config_defaults():
    """StageResolvedConfig has correct defaults."""
    cfg = StageResolvedConfig(stage_id=0, stage_type="llm")
    assert cfg.vllm_config is None
    assert cfg.diffusion_config is None
    assert cfg.executor_class is None
    assert cfg.engine_output_type is None
    assert cfg.is_comprehension is False
    assert cfg.requires_multimodal_data is False
    assert cfg.engine_input_source == []
    assert cfg.final_output is False
    assert cfg.final_output_type is None
    assert cfg.default_sampling_params is None
    assert cfg.custom_process_input_func is None
    assert cfg.model_stage is None
    assert cfg.model_arch is None
    assert cfg.cfg_kv_collect_func is None
    assert cfg.num_replicas == 1
    assert cfg.prompt_expand_func is None
    assert cfg.is_prefill_only is False
    assert cfg.is_decode_only is False


# ─────────────────────────────────────────────────────────
# VllmOmniConfig
# ─────────────────────────────────────────────────────────


def test_vllm_omni_config_validation():
    """VllmOmniConfig validates stage invariants."""
    cfg = VllmOmniConfig(model="test")
    assert cfg.num_stages == 0
    assert cfg.is_single_stage() is False

    with pytest.raises(ValueError, match="must be a non-empty string"):
        VllmOmniConfig(model="")

    # LLM stage without vllm_config should raise.
    with pytest.raises(ValueError, match="LLM stage must have vllm_config"):
        VllmOmniConfig(
            model="test",
            stages=(StageResolvedConfig(stage_id=0, stage_type="llm"),),
        )

    # Diffusion stage without diffusion_config should raise.
    with pytest.raises(ValueError, match="diffusion stage must have diffusion_config"):
        VllmOmniConfig(
            model="test",
            stages=(StageResolvedConfig(stage_id=0, stage_type="diffusion"),),
        )


# ─────────────────────────────────────────────────────────
# _detect_pd_config
# ─────────────────────────────────────────────────────────


def test_detect_pd_config_no_pd():
    """Returns None when no PD separation is found."""
    stages = [
        StageResolvedConfig(stage_id=0, stage_type="llm"),
    ]
    assert _detect_pd_config(stages) is None


def test_detect_pd_config_finds_pair():
    """Detects a prefill/decode pair from is_prefill_only / is_decode_only."""
    stages = [
        StageResolvedConfig(
            stage_id=0,
            stage_type="llm",
            is_prefill_only=True,
            engine_input_source=[],
        ),
        StageResolvedConfig(
            stage_id=1,
            stage_type="llm",
            is_decode_only=True,
            engine_input_source=[0],
        ),
    ]
    result = _detect_pd_config(stages)
    assert result is not None
    assert result["pd_pair"] == (0, 1)
    assert "bootstrap_addr" in result


# ─────────────────────────────────────────────────────────
# _resolve_dotted_func
# ─────────────────────────────────────────────────────────


def test_resolve_dotted_func_none():
    assert _resolve_dotted_func(None) is None
    assert _resolve_dotted_func("") is None


def test_resolve_dotted_func_callable_passthrough():
    def foo():
        return 42

    assert _resolve_dotted_func(foo) is foo


def test_resolve_dotted_func_invalid_path():
    assert _resolve_dotted_func("nonexistent.module.func") is None


# ─────────────────────────────────────────────────────────
# _get_devices_per_replica_from_resolved
# ─────────────────────────────────────────────────────────


def test_devices_per_replica_llm():
    """LLM reads tensor_parallel_size from VllmConfig.parallel_config."""
    from unittest.mock import Mock

    mock_vllm = Mock()
    mock_vllm.parallel_config.tensor_parallel_size = 4

    stage = StageResolvedConfig(
        stage_id=0,
        stage_type="llm",
        vllm_config=mock_vllm,
        executor_class=Mock,
    )

    assert _get_devices_per_replica_from_resolved(stage) == 4


def test_devices_per_replica_diffusion():
    """Diffusion reads world_size from OmniDiffusionConfig.parallel_config."""
    from unittest.mock import Mock

    mock_diff = Mock()
    mock_diff.parallel_config.world_size = 8

    stage = StageResolvedConfig(
        stage_id=0,
        stage_type="diffusion",
        diffusion_config=mock_diff,
    )

    assert _get_devices_per_replica_from_resolved(stage) == 8


def test_devices_per_replica_defaults():
    """Defaults to 1 when no config is set."""
    stage = StageResolvedConfig(stage_id=0, stage_type="llm")
    assert _get_devices_per_replica_from_resolved(stage) == 1

    stage = StageResolvedConfig(stage_id=0, stage_type="diffusion")
    assert _get_devices_per_replica_from_resolved(stage) == 1


# ─────────────────────────────────────────────────────────
# compute_per_replica_devices
# ─────────────────────────────────────────────────────────


def test_compute_per_replica_devices_no_split():
    """Returns None per replica when no device split is needed."""
    stage = StageResolvedConfig(stage_id=0, stage_type="llm")
    result = compute_per_replica_devices(stage, num_replicas=2, stage_id=0)
    assert result == [None, None]


def test_compute_per_replica_devices_with_runtime():
    """Splits devices from runtime when set — template mode."""
    stage = StageResolvedConfig(
        stage_id=0,
        stage_type="llm",
        runtime={"devices": "0,1"},
    )
    result = compute_per_replica_devices(stage, num_replicas=2, stage_id=0)
    assert len(result) == 2
    assert result[0] is not None
    assert result[1] is not None


# ─────────────────────────────────────────────────────────
# OmniEngineArgs.add_cli_args
# ─────────────────────────────────────────────────────────


def test_add_cli_args_registers_flags():
    """add_cli_args registers omni-specific flags on an ArgumentParser with omni_args_only=True."""
    parser = argparse.ArgumentParser()
    parser = OmniEngineArgs.add_cli_args(parser, omni_args_only=True)

    # Check a few representative flags.
    args, _ = parser.parse_known_args(["--omni", "--stage-init-timeout", "120", "--async-chunk"])
    assert args.omni is True
    assert args.stage_init_timeout == 120
    assert args.async_chunk is True


# ─────────────────────────────────────────────────────────
# OmniArgumentParser
# ─────────────────────────────────────────────────────────


def test_omni_argument_parser_skip_help():
    """Parser skips injection for --help — exits cleanly."""
    parser = OmniArgumentParser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_is_help_or_version():
    assert OmniArgumentParser._is_help_or_version(["--help"])
    assert OmniArgumentParser._is_help_or_version(["-h"])
    assert OmniArgumentParser._is_help_or_version(["--version"])
    assert not OmniArgumentParser._is_help_or_version(["--model", "foo"])


def test_peek_stage_id():
    assert OmniArgumentParser._peek_stage_id(["--stage-id", "3"]) == 3
    assert OmniArgumentParser._peek_stage_id(["--stage-id=5"]) == 5
    assert OmniArgumentParser._peek_stage_id(["--other", "flag"]) is None


def test_peek_deploy_config():
    assert OmniArgumentParser._peek_deploy_config(["--deploy-config", "foo.yaml"]) == "foo.yaml"
    assert OmniArgumentParser._peek_deploy_config(["--deploy-config=bar.yaml"]) == "bar.yaml"
    assert OmniArgumentParser._peek_deploy_config(["--other", "flag"]) is None
