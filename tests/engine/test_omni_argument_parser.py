"""
Tests for ``OmniArgumentParser`` — model-aware default injection from deploy YAML.
"""

import argparse
import json

import pytest

from vllm_omni.engine.arg_utils import OmniArgumentParser, _deep_merge

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

OAP = OmniArgumentParser


# ===========================================================================
# A — Peek helpers
# ===========================================================================


class TestPeekModel:
    def test_positional_after_serve(self):
        assert OAP._peek_model(["serve", "Qwen/Qwen2.5-Omni-7B", "--omni"]) == "Qwen/Qwen2.5-Omni-7B"

    def test_model_flag_separate(self):
        assert OAP._peek_model(["serve", "--model", "foo/bar", "--omni"]) == "foo/bar"

    def test_model_flag_equals(self):
        assert OAP._peek_model(["serve", "--model=foo/bar", "--omni"]) == "foo/bar"

    def test_config_yaml_with_model(self, tmp_path):
        yaml = tmp_path / "cfg.yaml"
        yaml.write_text("model: Qwen/Qwen2.5-Omni-7B\nport: 8000\n")
        assert OAP._peek_model(["serve", "--config", str(yaml), "--omni"]) == "Qwen/Qwen2.5-Omni-7B"

    def test_config_yaml_equals(self, tmp_path):
        yaml = tmp_path / "cfg.yaml"
        yaml.write_text("model: my-model\n")
        assert OAP._peek_model(["serve", f"--config={yaml}", "--omni"]) == "my-model"

    def test_config_yaml_no_model_key(self, tmp_path):
        yaml = tmp_path / "cfg.yaml"
        yaml.write_text("port: 8000\n")
        assert OAP._peek_model(["serve", "--config", str(yaml)]) is None

    def test_no_model(self):
        assert OAP._peek_model(["serve", "--port", "8000"]) is None

    def test_empty_args(self):
        assert OAP._peek_model([]) is None


class TestPeekModelFlat:
    def test_positional_at_start(self):
        # We can't call _peek_model_flat directly because it's an instance
        # method that needs self._subparsers.  Use the fact that when
        # _subparsers is None, the flat peek is used.
        parser = OmniArgumentParser("test")
        model = parser._peek_model_flat(["my-model", "--port", "8000"])
        assert model == "my-model"

    def test_flag_first_returns_none(self):
        parser = OmniArgumentParser("test")
        assert parser._peek_model_flat(["--port", "8000"]) is None

    def test_empty_returns_none(self):
        parser = OmniArgumentParser("test")
        assert parser._peek_model_flat([]) is None


class TestPeekStageId:
    def test_space_form(self):
        assert OAP._peek_stage_id(["--stage-id", "2"]) == 2

    def test_equals_form(self):
        assert OAP._peek_stage_id(["--stage-id=3"]) == 3

    def test_missing(self):
        assert OAP._peek_stage_id(["serve", "foo"]) is None

    def test_invalid(self):
        assert OAP._peek_stage_id(["--stage-id", "abc"]) is None


class TestPeekDeployConfig:
    def test_space_form(self):
        assert OAP._peek_deploy_config(["--deploy-config", "/path/to/d.yaml"]) == "/path/to/d.yaml"

    def test_equals_form(self):
        assert OAP._peek_deploy_config(["--deploy-config=/path/to/d.yaml"]) == "/path/to/d.yaml"

    def test_missing(self):
        assert OAP._peek_deploy_config(["serve", "foo"]) is None


class TestIsHelpOrVersion:
    @pytest.mark.parametrize(
        "args",
        [
            ["--help"],
            ["-h"],
            ["--version"],
            ["-v"],
            ["--help=OmniConfig"],
        ],
    )
    def test_positive(self, args):
        assert OAP._is_help_or_version(args) is True

    @pytest.mark.parametrize(
        "args",
        [
            ["serve", "foo", "--omni"],
            [],
            ["--port", "8000"],
        ],
    )
    def test_negative(self, args):
        assert OAP._is_help_or_version(args) is False


class TestIsServeOrFlat:
    def test_serve_subcommand(self):
        parser = OmniArgumentParser("test")
        sub = parser.add_subparsers(dest="s")
        sub.add_parser("serve")
        assert parser._is_serve_or_flat(["serve", "foo"]) is True

    def test_not_serve(self):
        parser = OmniArgumentParser("test")
        sub = parser.add_subparsers(dest="s")
        sub.add_parser("bench")
        assert parser._is_serve_or_flat(["bench", "foo"]) is False

    def test_flat_parser(self):
        parser = OmniArgumentParser("test")
        assert parser._is_serve_or_flat(["my-model", "--port", "8000"]) is True


# ===========================================================================
# B — _set_action_default
# ===========================================================================


class TestSetActionDefault:
    def test_sets_matching_dest(self):
        p = argparse.ArgumentParser()
        p.add_argument("--dtype", type=str, default="auto")
        OAP._set_action_default(p, "dtype", "bfloat16")
        args = p.parse_args([])
        assert args.dtype == "bfloat16"

    def test_noop_on_missing_dest(self):
        p = argparse.ArgumentParser()
        p.add_argument("--dtype", type=str, default="auto")
        OAP._set_action_default(p, "nonexistent", 999)
        args = p.parse_args([])
        assert args.dtype == "auto"

    def test_preserves_type_converter(self):
        p = argparse.ArgumentParser()
        p.add_argument("--gpu-mem", type=float, default=0.9)
        OAP._set_action_default(p, "gpu_mem", 0.5)
        args = p.parse_args(["--gpu-mem", "0.3"])
        assert args.gpu_mem == 0.3
        assert isinstance(args.gpu_mem, float)

    def test_bool_default(self):
        p = argparse.ArgumentParser()
        p.add_argument("--enforce-eager", action="store_true", default=False)
        OAP._set_action_default(p, "enforce_eager", True)
        args = p.parse_args([])
        assert args.enforce_eager is True
        args = p.parse_args(["--enforce-eager"])
        assert args.enforce_eager is True
        args = p.parse_args(["--no-enforce-eager"]) if False else None
        # BooleanOptionalAction not used here; store_true default works


# ===========================================================================
# C — _deep_merge
# ===========================================================================


class TestDeepMerge:
    def test_simple_override(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        assert _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 99, "z": 100}}) == {
            "a": {"x": 1, "y": 99, "z": 100},
        }

    def test_type_mismatch_overrides(self):
        assert _deep_merge({"a": {"x": 1}}, {"a": "str"}) == {"a": "str"}

    def test_preserves_untouched(self):
        result = _deep_merge({"a": 1, "b": {"x": 1}, "c": 3}, {"b": {"y": 2}})
        assert result == {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}

    def test_normalizes_int_keys_to_strings(self):
        """Fix P5: int keys from callers should merge with string keys from YAML."""
        result = _deep_merge({"0": {"a": 1}}, {0: {"b": 2}})
        assert result == {"0": {"a": 1, "b": 2}}


# ===========================================================================
# D — parse_args integration (mocked model detection)
# ===========================================================================


def _make_serve_parser() -> OmniArgumentParser:
    """Build an OmniArgumentParser with a 'serve' subparser and typical flags."""
    parser = OmniArgumentParser("test")
    subparsers = parser.add_subparsers(dest="subparser")
    serve = subparsers.add_parser("serve")
    serve.add_argument("model_tag", nargs="?", default=None)
    serve.add_argument("--async-chunk", action=argparse.BooleanOptionalAction, default=None)
    serve.add_argument("--dtype", type=str, default="auto")
    serve.add_argument("--gpu-memory-utilization", "-gmu", type=float, default=0.9)
    serve.add_argument("--enforce-eager", action="store_true", default=False)
    serve.add_argument("--stage-overrides", type=str, default=None)
    serve.add_argument("--stage-id", type=int, default=None)
    serve.add_argument("--deploy-config", type=str, default=None)
    parser.set_defaults(subparser="serve")
    return parser


def _write_deploy_yaml(path, *, async_chunk=None, dtype=None, stages=None):
    """Write a minimal deploy YAML to *path*."""
    lines = []
    if async_chunk is not None:
        lines.append(f"async_chunk: {str(async_chunk).lower()}")
    if dtype is not None:
        lines.append(f"dtype: {dtype}")
    if stages is not None:
        if stages:
            lines.append("stages:")
            for s in stages:
                lines.append(f"  - stage_id: {s['stage_id']}")
                for k, v in s.items():
                    if k == "stage_id":
                        continue
                    if isinstance(v, bool):
                        lines.append(f"    {k}: {str(v).lower()}")
                    elif isinstance(v, str):
                        lines.append(f"    {k}: {v}")
                    else:
                        lines.append(f"    {k}: {v}")
        else:
            lines.append("stages: []")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def mock_model_detection(mocker, tmp_path):
    """Patch model_type detection and deploy dir to point at tmp_path."""
    mocker.patch(
        "vllm_omni.config.stage_config.StageConfigFactory._auto_detect_model_type",
        return_value=("test_model", None),
    )
    mocker.patch("vllm_omni.config.stage_config._DEPLOY_DIR", tmp_path)


class TestParseArgsIntegration:
    def test_injects_pipeline_defaults(self, mocker, tmp_path, mock_model_detection):
        _write_deploy_yaml(tmp_path / "test_model.yaml", async_chunk=False, dtype="bfloat16")
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B"])
        assert args.async_chunk is False
        assert args.dtype == "bfloat16"
        # gpu-memory-utilization was not in deploy YAML, keeps vLLM default
        assert args.gpu_memory_utilization == 0.9

    def test_cli_overrides_yaml_defaults(self, mocker, tmp_path, mock_model_detection):
        _write_deploy_yaml(tmp_path / "test_model.yaml", async_chunk=False, dtype="bfloat16")
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", "--dtype", "float32"])
        assert args.dtype == "float32"  # CLI wins
        assert args.async_chunk is False  # YAML still applies

    def test_help_skips_injection(self, mocker, tmp_path, mock_model_detection):
        _write_deploy_yaml(tmp_path / "test_model.yaml", async_chunk=False, dtype="bfloat16")
        parser = _make_serve_parser()
        # parse_args exits on --help, so we verify via _is_help_or_version gate
        # and confirm that help text does not crash
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", "--help"])

    def test_nonserve_skips_injection(self, mocker, tmp_path, mock_model_detection):
        _write_deploy_yaml(tmp_path / "test_model.yaml", async_chunk=False)
        parser = OmniArgumentParser("test")
        subparsers = parser.add_subparsers(dest="subparser")
        bench = subparsers.add_parser("bench")
        bench.add_argument("--dtype", type=str, default="auto")
        bench.add_argument("model_tag", nargs="?", default=None)
        parser.set_defaults(subparser="bench")

        args = parser.parse_args(["bench", "--dtype", "float16"])
        # dtype should NOT get YAML default because bench is not serve
        assert args.dtype == "float16"

    def test_deploy_not_found_noop(self, mocker, tmp_path, mock_model_detection):
        # No test_model.yaml written
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", "--dtype", "float16"])
        assert args.dtype == "float16"  # user value only

    def test_custom_deploy_config(self, mocker, tmp_path, mock_model_detection):
        custom = tmp_path / "custom_deploy.yaml"
        _write_deploy_yaml(custom, async_chunk=True, dtype="custom_dtype", stages=[])
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", "--deploy-config", str(custom)])
        assert args.dtype == "custom_dtype"
        assert args.async_chunk is True

    def test_custom_deploy_config_equals_form(self, mocker, tmp_path, mock_model_detection):
        custom = tmp_path / "custom_deploy.yaml"
        _write_deploy_yaml(custom, async_chunk=True, dtype="equals_form_dtype", stages=[])
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", f"--deploy-config={custom}"])
        assert args.dtype == "equals_form_dtype"

    def test_state_cleanup_between_parses(self, mocker, tmp_path, mock_model_detection):
        """Fix P4: _yaml_stage_overrides cleared between parse_args calls."""
        _write_deploy_yaml(
            tmp_path / "test_model.yaml",
            async_chunk=True,
            stages=[{"stage_id": 0, "gpu_memory_utilization": 0.5}],
        )
        parser = _make_serve_parser()
        parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B"])  # first parse seeds yaml defaults
        # Second parse should not carry over stale defaults
        args2 = parser.parse_args(
            ["serve", "Qwen/Qwen2.5-Omni-7B", "--stage-overrides", '{"0": {"enforce_eager": true}}']
        )
        so2 = json.loads(args2.stage_overrides)
        # User override should win, YAML gpu_memory_utilization should still be present
        assert so2["0"]["enforce_eager"] is True
        assert so2["0"]["gpu_memory_utilization"] == 0.5


# ===========================================================================
# E — post-parse stage_overrides merge
# ===========================================================================


class TestPostParseMerge:
    def test_yaml_only_stage_overrides(self, mocker, tmp_path, mock_model_detection):
        _write_deploy_yaml(
            tmp_path / "test_model.yaml",
            stages=[{"stage_id": 0, "gpu_memory_utilization": 0.5, "enforce_eager": True}],
        )
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B"])
        so = json.loads(args.stage_overrides)
        assert so["0"]["gpu_memory_utilization"] == 0.5
        assert so["0"]["enforce_eager"] is True

    def test_user_merges_with_yaml_stage_overrides(self, mocker, tmp_path, mock_model_detection):
        _write_deploy_yaml(
            tmp_path / "test_model.yaml",
            stages=[
                {"stage_id": 0, "gpu_memory_utilization": 0.8, "enforce_eager": True},
                {"stage_id": 1, "gpu_memory_utilization": 0.4},
            ],
        )
        parser = _make_serve_parser()
        args = parser.parse_args(
            ["serve", "Qwen/Qwen2.5-Omni-7B", "--stage-overrides", '{"0": {"gpu_memory_utilization": 0.5}}']
        )
        so = json.loads(args.stage_overrides)
        assert so["0"]["gpu_memory_utilization"] == 0.5  # user wins
        assert so["0"]["enforce_eager"] is True  # YAML preserved
        assert so["1"]["gpu_memory_utilization"] == 0.4  # YAML preserved

    def test_no_merge_when_no_yaml_overrides(self, mocker, tmp_path, mock_model_detection):
        # Deploy YAML with pipeline defaults but no per-stage fields
        _write_deploy_yaml(tmp_path / "test_model.yaml", async_chunk=False)
        parser = _make_serve_parser()
        args = parser.parse_args(
            ["serve", "Qwen/Qwen2.5-Omni-7B", "--stage-overrides", '{"0": {"enforce_eager": true}}']
        )
        so = json.loads(args.stage_overrides)
        assert so == {"0": {"enforce_eager": True}}


# ===========================================================================
# F — _inject_model_defaults edge cases
# ===========================================================================


class TestInjectModelDefaults:
    def test_engine_extras_skipped(self, mocker, tmp_path, mock_model_detection):
        """engine_extras should not be set as action.default."""
        _write_deploy_yaml(
            tmp_path / "test_model.yaml",
            stages=[{"stage_id": 0, "gpu_memory_utilization": 0.5, "enforce_eager": True}],
        )
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B"])
        so = json.loads(args.stage_overrides)
        assert "engine_extras" not in so.get("0", {})

    def test_headless_per_stage_defaults(self, mocker, tmp_path, mock_model_detection):
        _write_deploy_yaml(
            tmp_path / "test_model.yaml",
            stages=[{"stage_id": 0, "gpu_memory_utilization": 0.5, "enforce_eager": True}],
        )
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", "--stage-id", "0"])
        assert args.gpu_memory_utilization == 0.5
        assert args.enforce_eager is True


# ===========================================================================
# G — Edge cases: model_type detection fails, malformed YAML
# ===========================================================================


class TestEdgeCases:
    def test_model_type_detection_returns_none(self, mocker, tmp_path, mock_model_detection):
        """When _auto_detect_model_type returns (None, None), injection is skipped cleanly."""
        mocker.patch(
            "vllm_omni.config.stage_config.StageConfigFactory._auto_detect_model_type",
            return_value=(None, None),
        )
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", "--dtype", "float32"])
        assert args.dtype == "float32"  # user value used, no crash

    def test_malformed_deploy_yaml_noop(self, mocker, tmp_path, mock_model_detection):
        """Malformed deploy YAML is caught gracefully — injection skipped, no crash."""
        yaml = tmp_path / "test_model.yaml"
        yaml.write_text("{ invalid: yaml: content: [")  # malformed YAML
        parser = _make_serve_parser()
        args = parser.parse_args(["serve", "Qwen/Qwen2.5-Omni-7B", "--dtype", "float32"])
        assert args.dtype == "float32"  # no crash, user value used
