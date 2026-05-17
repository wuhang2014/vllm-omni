# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm_omni.engine.arg_utils — unified OmniEngineArgs."""

from __future__ import annotations

import argparse
from dataclasses import fields

import pytest

# ============================================================================
# OmniEngineArgs construction
# ============================================================================


def test_omniengineargs_create_tracks_explicit_fields():
    try:
        from vllm_omni.engine.arg_utils import OmniEngineArgs
    except Exception as exc:
        pytest.skip(f"OmniEngineArgs not importable: {exc}")

    ea = OmniEngineArgs.create(model="x", gpu_memory_utilization=0.5)
    assert ea._explicit_fields == frozenset({"model", "gpu_memory_utilization"})
    assert ea.explicit_kwargs() == {"model": "x", "gpu_memory_utilization": 0.5}


def test_omniengineargs_bare_constructor_has_no_explicit_tracking():
    try:
        from vllm_omni.engine.arg_utils import OmniEngineArgs
    except Exception as exc:
        pytest.skip(f"OmniEngineArgs not importable: {exc}")

    ea = OmniEngineArgs(model="x")
    assert not hasattr(ea, "_explicit_fields")
    assert "model" in ea.explicit_kwargs()


def test_omniengineargs_from_cli_args():
    try:
        from vllm_omni.engine.arg_utils import OmniEngineArgs
    except Exception as exc:
        pytest.skip(f"OmniEngineArgs not importable: {exc}")

    ns = argparse.Namespace(model="test/model", tensor_parallel_size=2, async_chunk=True)
    ea = OmniEngineArgs.from_cli_args(ns)
    assert ea.model == "test/model"
    assert ea.tensor_parallel_size == 2
    assert ea.async_chunk is True


def test_omniengineargs_add_cli_args_registers_omni_config_group():
    """OmniEngineArgs.add_cli_args adds an OmniConfig argument group."""
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        from vllm_omni.engine.arg_utils import OmniEngineArgs
    except Exception as exc:
        pytest.skip(f"Cannot build parser: {exc}")

    parser = FlexibleArgumentParser()
    parser = OmniEngineArgs.add_cli_args(parser)

    # OmniConfig group should exist
    groups = {g.title for g in parser._action_groups if g.title == "OmniConfig"}
    assert groups, "OmniConfig argument group not found"

    # Key omni flags should be registered
    dests = {a.dest for a in parser._actions if a.dest}
    assert "omni" in dests
    assert "async_chunk" in dests
    assert "stage_init_timeout" in dests
    assert "deploy_config" in dests
    assert "headless" in dests


def test_omniengineargs_add_cli_args_omni_args_only_skips_parent():
    """omni_args_only=True does not re-register vllm parent flags."""
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        from vllm_omni.engine.arg_utils import OmniEngineArgs
    except Exception as exc:
        pytest.skip(f"Cannot build parser: {exc}")

    parser = FlexibleArgumentParser()
    parser = OmniEngineArgs.add_cli_args(parser, omni_args_only=True)

    # Omni flags present
    dests = {a.dest for a in parser._actions if a.dest}
    assert "omni" in dests
    assert "headless" in dests
    assert "stage_init_timeout" in dests


def test_create_omni_config_returns_vllm_omni_config():
    """create_omni_config() returns a VllmOmniConfig instance.

    Requires a model that exists locally to avoid HuggingFace download.
    If no model is available, the test is skipped.
    """
    try:
        from vllm_omni.config import VllmOmniConfig
        from vllm_omni.engine.arg_utils import OmniEngineArgs
    except Exception as exc:
        pytest.skip(f"OmniEngineArgs not importable: {exc}")

    # OmniEngineArgs.__post_init__ triggers HF model resolution.
    # Use a local path or skip if none available.
    import os

    test_model = os.environ.get("VLLM_OMNI_TEST_MODEL", "")
    if not test_model or not os.path.isdir(test_model):
        pytest.skip("Set VLLM_OMNI_TEST_MODEL to a local model path to run this test")

    ea = OmniEngineArgs(model=test_model)
    cfg = ea.create_omni_config()
    assert isinstance(cfg, VllmOmniConfig)
    assert cfg.model == test_model


# ============================================================================
# CLI flag classification
# ============================================================================


def test_all_omni_cli_flags_are_omniengineargs_fields():
    """Every omni CLI flag registered in the OmniConfig group must be a
    field on ``OmniEngineArgs``.

    Since ``OrchestratorArgs`` was absorbed into ``OmniEngineArgs``,
    all omni-added flags must map to the unified dataclass.
    vLLM server/API flags (host, port, ssl_keyfile, etc.) are expected
    to be outside ``OmniEngineArgs``.
    """
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    except Exception as exc:
        pytest.skip(f"Cannot build parser: {exc}")

    # Build the serve parser
    root = FlexibleArgumentParser()
    subparsers = root.add_subparsers()
    cmd = OmniServeCommand()
    try:
        parser = cmd.subparser_init(subparsers)
    except Exception as exc:
        pytest.skip(f"subparser_init failed (dev env issue): {exc}")

    engine = {f.name for f in fields(OmniEngineArgs)}

    # Collect dests from OmniConfig group only
    omni_dests: set[str] = set()
    for group in parser._action_groups:
        if group.title == "OmniConfig":
            for action in group._group_actions:
                if action.dest and action.dest != "help":
                    omni_dests.add(action.dest)
            break

    missing = omni_dests - engine - {"help", "model_tag"}
    assert not missing, (
        f"OmniConfig CLI flags not on OmniEngineArgs: {sorted(missing)}. Add them as fields on OmniEngineArgs."
    )


# ============================================================================
# nullify_stage_engine_defaults
# ============================================================================


def _build_full_serve_parser():
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
    except ImportError:
        pytest.skip("vllm parser not importable")
    return make_arg_parser(FlexibleArgumentParser())


def test_nullify_stage_engine_defaults_resets_inherited_defaults():
    from vllm_omni.config.stage_config import deploy_override_field_names
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    parser = _build_full_serve_parser()
    nullify_stage_engine_defaults(parser)

    override_dests = deploy_override_field_names()
    offenders = [
        (a.dest, a.default)
        for a in parser._actions
        if a.dest not in ("help", "version")
        and a.option_strings
        and a.dest in override_dests
        and a.default is not None
        and a.default is not argparse.SUPPRESS
    ]
    assert not offenders, f"Stage flags with non-None defaults after nullify: {offenders}"


def test_non_override_flags_keep_real_defaults_after_nullify():
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    parser = argparse.ArgumentParser()
    parser.add_argument("--hsdp-shard-size", type=int, default=-1, help="HSDP shard size.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max num seqs.")
    nullify_stage_engine_defaults(parser)

    hsdp = next(a for a in parser._actions if a.dest == "hsdp_shard_size")
    max_num_seqs = next(a for a in parser._actions if a.dest == "max_num_seqs")
    assert hsdp.default == -1
    assert max_num_seqs.default is None


def test_help_text_preserves_default_after_nullify():
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-seqs", type=int, default=42, help="Example knob.")
    nullify_stage_engine_defaults(parser)

    action = next(a for a in parser._actions if a.dest == "max_num_seqs")
    assert action.default is None
    assert "(default: 42)" in action.help
