"""Unit tests for the Omni serve CLI helpers."""

from __future__ import annotations

import argparse
from unittest.mock import Mock, patch

import pytest

from vllm_omni.entrypoints.cli.serve import run_headless

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_headless_args() -> argparse.Namespace:
    return argparse.Namespace(
        model="fake-model",
        stage_id=3,
        omni_master_address="127.0.0.1",
        omni_master_port=26000,
        api_server_count=0,
        worker_backend="multi_process",
        stage_configs_path=None,
        disable_log_stats=False,
    )


def test_run_headless_registers_stage_once_and_launches_all_local_engines() -> None:
    args = _make_headless_args()
    stage_cfg = Mock(stage_id=3)
    stage_cfgs = [stage_cfg]
    parallel_config = Mock(
        data_parallel_size_local=2,
        data_parallel_rank=4,
        data_parallel_rank_local=1,
        node_rank_within_dp=0,
    )
    vllm_config = Mock(parallel_config=parallel_config)
    executor_class = Mock()
    engine_manager = Mock()

    with (
        patch(
            "vllm_omni.entrypoints.utils.load_and_resolve_stage_configs",
            return_value=("/fake/stages.yaml", stage_cfgs),
        ),
        patch("vllm_omni.engine.stage_init_utils.prepare_engine_environment"),
        patch("vllm_omni.engine.stage_init_utils.load_omni_transfer_config_for_model", return_value=Mock()),
        patch("vllm_omni.engine.stage_init_utils.get_stage_connector_spec", return_value={}),
        patch("vllm_omni.engine.stage_init_utils.build_engine_args_dict", return_value={}),
        patch(
            "vllm_omni.distributed.omni_connectors.utils.initialization.resolve_omni_kv_config_for_stage",
            return_value=(None, None, None),
        ),
        patch(
            "vllm_omni.engine.stage_init_utils.build_vllm_config",
            return_value=(vllm_config, executor_class),
        ) as mock_build_vllm_config,
        patch(
            "vllm_omni.engine.omni_core_engine.register_stage_with_omni_master",
            return_value="tcp://127.0.0.1:26001",
        ) as mock_register,
        patch("vllm.v1.engine.utils.CoreEngineProcManager", return_value=engine_manager) as mock_manager_cls,
        patch("signal.signal"),
    ):
        run_headless(args)

    mock_build_vllm_config.assert_called_once_with(
        stage_cfg,
        "fake-model",
        stage_connector_spec={},
        engine_args_dict={},
        headless=True,
    )
    mock_register.assert_called_once_with(
        omni_master_address="127.0.0.1",
        omni_master_port=26000,
        omni_stage_id=3,
        omni_stage_config=stage_cfg,
    )
    mock_manager_cls.assert_called_once()
    manager_kwargs = mock_manager_cls.call_args.kwargs
    assert manager_kwargs["local_engine_count"] == 2
    assert manager_kwargs["start_index"] == 4
    assert manager_kwargs["local_start_index"] == 0
    assert manager_kwargs["handshake_address"] == "tcp://127.0.0.1:26001"
    engine_manager.join_first.assert_called_once_with()
    engine_manager.close.assert_called_once_with()
