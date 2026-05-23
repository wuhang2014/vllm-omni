# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OmniEngineArgs and OmniArgumentParser."""

from __future__ import annotations

import argparse
from dataclasses import fields as dc_fields

import pytest

from vllm_omni.engine.arg_utils import OmniEngineArgs, OmniArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ─────────────────────────────────────────────────────────
# OmniEngineArgs field coverage
# ─────────────────────────────────────────────────────────


def test_all_cli_friendly_fields_have_defaults():
    """Every field on OmniEngineArgs has a default (no required fields)."""
    fields_without_defaults = []
    for f in dc_fields(OmniEngineArgs):
        if f.default is dc_fields.MISSING and f.default_factory is dc_fields.MISSING:
            fields_without_defaults.append(f.name)
    assert not fields_without_defaults, (
        f"Fields without defaults: {fields_without_defaults}"
    )


def test_omni_engine_args_from_cli_args_partial():
    """from_cli_args works with a partial Namespace."""
    ns = argparse.Namespace(model="test", async_chunk=True)
    ea = OmniEngineArgs.from_cli_args(ns)
    assert ea.model == "test"
    assert ea.async_chunk is True
    assert ea.stage_init_timeout == 300  # default


def test_add_cli_args_omni_only():
    """omni_args_only=True only adds omni flags."""
    parser = argparse.ArgumentParser()
    parser = OmniEngineArgs.add_cli_args(parser, omni_args_only=True)
    args = parser.parse_args([])
    assert args.stage_init_timeout == 300
    assert args.async_chunk is None  # BooleanOptionalAction default


# ─────────────────────────────────────────────────────────
# OmniArgumentParser
# ─────────────────────────────────────────────────────────


class TestOmniArgumentParser:
    def test_skip_on_help(self):
        parser = OmniArgumentParser()
        result = parser.parse_args(["--help"])
        assert result is not None

    def test_skip_on_version(self):
        parser = OmniArgumentParser()
        result = parser.parse_args(["--version"])
        assert result is not None

    def test_peek_model_from_positional(self):
        assert OmniArgumentParser._peek_model(["serve", "Qwen/Qwen-Image"]) == "Qwen/Qwen-Image"

    def test_peek_model_from_model_flag(self):
        assert OmniArgumentParser._peek_model(["--model", "foo/bar"]) == "foo/bar"
        assert OmniArgumentParser._peek_model(["--model=baz/qux"]) == "baz/qux"

    def test_peek_model_none(self):
        assert OmniArgumentParser._peek_model(["--port", "8000"]) is None
        assert OmniArgumentParser._peek_model([]) is None

    def test_peek_stage_id(self):
        assert OmniArgumentParser._peek_stage_id(["--stage-id", "2"]) == 2
        assert OmniArgumentParser._peek_stage_id(["--stage-id=3"]) == 3
        assert OmniArgumentParser._peek_stage_id([]) is None

    def test_peek_deploy_config(self):
        assert OmniArgumentParser._peek_deploy_config(["--deploy-config", "x.yaml"]) == "x.yaml"
        assert OmniArgumentParser._peek_deploy_config(["--deploy-config=y.yaml"]) == "y.yaml"
        assert OmniArgumentParser._peek_deploy_config([]) is None
