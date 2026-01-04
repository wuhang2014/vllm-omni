"""Unit tests for vllm_omni/entrypoints/utils.py"""
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from omegaconf import OmegaConf

from vllm_omni.entrypoints.utils import (
    _convert_dataclasses_to_dict,
    _try_get_class_name_from_diffusers_config,
    get_final_stage_id_for_e2e,
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)


class TestConvertDataclassesToDict:
    """Tests for _convert_dataclasses_to_dict function"""

    def test_convert_counter_object(self):
        """Test converting Counter objects to dict"""
        counter = Counter(["a", "a", "b"])
        result = _convert_dataclasses_to_dict(counter)
        assert isinstance(result, dict)
        assert result == {"a": 2, "b": 1}

    def test_convert_set_object(self):
        """Test converting set objects to list"""
        test_set = {1, 2, 3}
        result = _convert_dataclasses_to_dict(test_set)
        assert isinstance(result, list)
        assert set(result) == test_set

    def test_convert_dataclass(self):
        """Test converting dataclass objects"""

        @dataclass
        class TestDataClass:
            value: int
            name: str

        obj = TestDataClass(value=42, name="test")
        result = _convert_dataclasses_to_dict(obj)
        assert isinstance(result, dict)
        assert result == {"value": 42, "name": "test"}

    def test_convert_nested_dict(self):
        """Test converting nested dictionaries with Counter"""
        nested = {"outer": {"counter": Counter(["x", "x", "y"]), "value": 1}}
        result = _convert_dataclasses_to_dict(nested)
        assert result == {"outer": {"counter": {"x": 2, "y": 1}, "value": 1}}

    def test_convert_list_with_mixed_types(self):
        """Test converting lists with mixed types"""
        mixed_list = [Counter(["a"]), {1, 2}, "string", 42]
        result = _convert_dataclasses_to_dict(mixed_list)
        assert isinstance(result, list)
        assert result[0] == {"a": 1}
        assert set(result[1]) == {1, 2}
        assert result[2] == "string"
        assert result[3] == 42

    def test_convert_tuple_preserves_type(self):
        """Test that tuples remain tuples"""
        test_tuple = (1, 2, Counter(["a"]))
        result = _convert_dataclasses_to_dict(test_tuple)
        assert isinstance(result, tuple)
        assert result[2] == {"a": 1}

    def test_primitive_types_unchanged(self):
        """Test that primitive types pass through unchanged"""
        assert _convert_dataclasses_to_dict(42) == 42
        assert _convert_dataclasses_to_dict("string") == "string"
        assert _convert_dataclasses_to_dict(3.14) == 3.14
        assert _convert_dataclasses_to_dict(True) is True
        assert _convert_dataclasses_to_dict(None) is None

    def test_empty_containers(self):
        """Test empty containers"""
        assert _convert_dataclasses_to_dict({}) == {}
        assert _convert_dataclasses_to_dict([]) == []
        assert _convert_dataclasses_to_dict(set()) == []
        assert _convert_dataclasses_to_dict(Counter()) == {}


class TestTryGetClassNameFromDiffusersConfig:
    """Tests for _try_get_class_name_from_diffusers_config function"""

    @patch("vllm_omni.entrypoints.utils.get_hf_file_to_dict")
    def test_returns_class_name_when_present(self, mock_get_file):
        """Test successful retrieval of _class_name"""
        mock_get_file.return_value = {"_class_name": "StableDiffusionPipeline"}
        result = _try_get_class_name_from_diffusers_config("test_model")
        assert result == "StableDiffusionPipeline"
        mock_get_file.assert_called_once_with("model_index.json", "test_model", revision=None)

    @patch("vllm_omni.entrypoints.utils.get_hf_file_to_dict")
    def test_returns_none_when_class_name_missing(self, mock_get_file):
        """Test returns None when _class_name is missing"""
        mock_get_file.return_value = {"other_field": "value"}
        result = _try_get_class_name_from_diffusers_config("test_model")
        assert result is None

    @patch("vllm_omni.entrypoints.utils.get_hf_file_to_dict")
    def test_returns_none_when_file_not_dict(self, mock_get_file):
        """Test returns None when file content is not a dict"""
        mock_get_file.return_value = "not a dict"
        result = _try_get_class_name_from_diffusers_config("test_model")
        assert result is None

    @patch("vllm_omni.entrypoints.utils.get_hf_file_to_dict")
    def test_returns_none_when_file_none(self, mock_get_file):
        """Test returns None when file is None"""
        mock_get_file.return_value = None
        result = _try_get_class_name_from_diffusers_config("test_model")
        assert result is None


class TestResolveModelConfigPath:
    """Tests for resolve_model_config_path function"""

    @patch("vllm_omni.entrypoints.utils.get_config")
    @patch("vllm_omni.entrypoints.utils.detect_device_type")
    @patch("os.path.exists")
    def test_returns_cuda_config_for_standard_model(self, mock_exists, mock_device, mock_get_config):
        """Test resolving config path for standard transformers model"""
        mock_config = MagicMock()
        mock_config.model_type = "qwen2_5_omni"
        mock_get_config.return_value = mock_config
        mock_device.return_value = "cuda"
        mock_exists.return_value = True

        result = resolve_model_config_path("test_model")
        assert result is not None
        assert "qwen2_5_omni.yaml" in result

    @patch("vllm_omni.entrypoints.utils.get_config")
    @patch("vllm_omni.entrypoints.utils.file_or_path_exists")
    @patch("vllm_omni.entrypoints.utils._try_get_class_name_from_diffusers_config")
    @patch("vllm_omni.entrypoints.utils.detect_device_type")
    @patch("os.path.exists")
    def test_returns_config_for_diffusers_model(
        self, mock_exists, mock_device, mock_class_name, mock_file_exists, mock_get_config
    ):
        """Test resolving config path for diffusers model"""
        mock_get_config.side_effect = ValueError("Not a transformers model")
        mock_file_exists.return_value = True
        mock_class_name.return_value = "StableDiffusionPipeline"
        mock_device.return_value = "cuda"
        mock_exists.return_value = True

        result = resolve_model_config_path("test_model")
        assert result is not None
        assert "StableDiffusionPipeline.yaml" in result

    @patch("vllm_omni.entrypoints.utils.get_config")
    @patch("vllm_omni.entrypoints.utils.file_or_path_exists")
    def test_raises_value_error_when_model_type_not_found(self, mock_file_exists, mock_get_config):
        """Test raises ValueError when model type cannot be determined"""
        mock_get_config.side_effect = ValueError("Config error")
        mock_file_exists.return_value = False

        with pytest.raises(ValueError, match="Could not determine model_type"):
            resolve_model_config_path("test_model")

    @patch("vllm_omni.entrypoints.utils.get_config")
    @patch("vllm_omni.entrypoints.utils.file_or_path_exists")
    @patch("vllm_omni.entrypoints.utils._try_get_class_name_from_diffusers_config")
    def test_raises_value_error_for_diffusers_without_class_name(
        self, mock_class_name, mock_file_exists, mock_get_config
    ):
        """Test raises ValueError for diffusers model without _class_name"""
        mock_get_config.side_effect = ValueError("Not transformers")
        mock_file_exists.return_value = True
        mock_class_name.return_value = None

        with pytest.raises(ValueError, match="Could not determine model_type for diffusers model"):
            resolve_model_config_path("test_model")

    @patch("vllm_omni.entrypoints.utils.get_config")
    @patch("vllm_omni.entrypoints.utils.detect_device_type")
    @patch("vllm_omni.entrypoints.utils.is_rocm")
    @patch("os.path.exists")
    def test_returns_device_specific_config_for_npu(self, mock_exists, mock_is_rocm, mock_device, mock_get_config):
        """Test returns device-specific config for NPU"""
        mock_config = MagicMock()
        mock_config.model_type = "qwen2_5_omni"
        mock_get_config.return_value = mock_config
        mock_device.return_value = "npu"
        mock_is_rocm.return_value = False
        # First call (device-specific) returns True, second call (fallback) returns True
        mock_exists.side_effect = [True, True]

        result = resolve_model_config_path("test_model")
        assert result is not None
        assert "npu" in result


class TestLoadStageConfigs:
    """Tests for load_stage_configs_from_yaml and load_stage_configs_from_model"""

    def test_load_stage_configs_from_yaml_basic(self, tmp_path):
        """Test loading stage configs from YAML file"""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
stage_args:
  - stage_id: 0
    engine_args:
      model: test_model
      tensor_parallel_size: 1
"""
        config_file.write_text(config_content)

        result = load_stage_configs_from_yaml(str(config_file))
        assert len(result) == 1
        assert result[0].stage_id == 0

    def test_load_stage_configs_with_base_engine_args(self, tmp_path):
        """Test merging base_engine_args with stage-specific args"""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
stage_args:
  - stage_id: 0
    engine_args:
      tensor_parallel_size: 2
"""
        config_file.write_text(config_content)

        base_args = {"max_model_len": 1024, "gpu_memory_utilization": 0.9}
        result = load_stage_configs_from_yaml(str(config_file), base_engine_args=base_args)

        assert len(result) == 1
        # Base args should be merged with stage-specific args
        assert result[0].engine_args.tensor_parallel_size == 2
        assert result[0].engine_args.max_model_len == 1024

    @patch("vllm_omni.entrypoints.utils.resolve_model_config_path")
    @patch("vllm_omni.entrypoints.utils.load_stage_configs_from_yaml")
    def test_load_stage_configs_from_model(self, mock_load_yaml, mock_resolve_path):
        """Test loading stage configs from model name"""
        mock_resolve_path.return_value = "/path/to/config.yaml"
        mock_stage_config = MagicMock()
        mock_load_yaml.return_value = [mock_stage_config]

        result = load_stage_configs_from_model("test_model")

        assert len(result) == 1
        mock_resolve_path.assert_called_once_with("test_model")
        mock_load_yaml.assert_called_once()

    @patch("vllm_omni.entrypoints.utils.resolve_model_config_path")
    def test_load_stage_configs_from_model_returns_empty_when_no_config(self, mock_resolve_path):
        """Test returns empty list when no config path found"""
        mock_resolve_path.return_value = None

        result = load_stage_configs_from_model("test_model")

        assert result == []


class TestGetFinalStageIdForE2e:
    """Tests for get_final_stage_id_for_e2e function"""

    def test_returns_last_stage_when_output_modalities_none(self):
        """Test returns last stage ID when output_modalities is None"""
        stage_list = [MagicMock(final_output=True, final_output_type="text") for _ in range(3)]
        result = get_final_stage_id_for_e2e(None, ["text"], stage_list)
        # Should use default modalities and find the last stage with matching type
        assert result == 2

    def test_returns_matching_stage_from_end(self):
        """Test finds first matching stage from end"""
        stage_list = [
            MagicMock(final_output=True, final_output_type="text"),
            MagicMock(final_output=True, final_output_type="image"),
            MagicMock(final_output=True, final_output_type="text"),
        ]
        result = get_final_stage_id_for_e2e(["text"], ["text", "image"], stage_list)
        assert result == 2  # Last text stage

    def test_filters_invalid_modalities(self):
        """Test filters out invalid output modalities"""
        stage_list = [
            MagicMock(final_output=True, final_output_type="text"),
            MagicMock(final_output=True, final_output_type="image"),
        ]
        # "invalid" should be filtered out
        result = get_final_stage_id_for_e2e(["invalid", "text"], ["text", "image"], stage_list)
        assert result == 0  # Should find text stage

    def test_returns_last_stage_when_no_match(self):
        """Test returns last stage when no matching modality found"""
        stage_list = [
            MagicMock(final_output=False, final_output_type="text"),
            MagicMock(final_output=False, final_output_type="image"),
            MagicMock(final_output=True, final_output_type="audio"),
        ]
        result = get_final_stage_id_for_e2e(["text"], ["text"], stage_list)
        assert result == 2  # Falls back to last stage

    def test_handles_exception_gracefully(self):
        """Test handles exceptions and falls back to last stage"""
        stage_list = [
            MagicMock(final_output=True, final_output_type="text"),
            MagicMock(final_output=True),  # Missing final_output_type
        ]
        # Should handle missing attribute
        result = get_final_stage_id_for_e2e(["text"], ["text"], stage_list)
        assert result >= 0
        assert result < len(stage_list)
