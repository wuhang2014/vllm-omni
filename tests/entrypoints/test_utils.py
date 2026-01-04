from collections import Counter
from dataclasses import dataclass

import pytest

from vllm_omni.entrypoints import utils as entry_utils


@dataclass
class _Inner:
    value: int


def test_convert_dataclasses_to_dict_handles_collections():
    payload = {
        "counts": Counter({"a": 2, "b": 1}),
        "tags": {"x", "y"},
        "inner": _Inner(5),
    }

    converted = entry_utils._convert_dataclasses_to_dict(payload)

    assert converted["counts"] == {"a": 2, "b": 1}
    assert sorted(converted["tags"]) == ["x", "y"]
    assert converted["inner"] == {"value": 5}


def test_load_stage_configs_from_yaml_merges_base_args(tmp_path):
    config_path = tmp_path / "stage_config.yaml"
    config_path.write_text(
        """
stage_args:
  - engine_args:
      stage_specific: 2
  - engine_args: null
        """,
        encoding="utf-8",
    )

    stage_args = entry_utils.load_stage_configs_from_yaml(
        str(config_path), base_engine_args={"common": 1}
    )

    assert len(stage_args) == 2
    first_args = stage_args[0].engine_args
    second_args = stage_args[1].engine_args

    assert first_args.common == 1
    assert first_args.stage_specific == 2
    assert second_args.common == 1


def test_get_final_stage_id_for_e2e_prefers_requested_modalities():
    class _Stage:
        def __init__(self, final_output, final_output_type):
            self.final_output = final_output
            self.final_output_type = final_output_type

    stages = [
        _Stage(final_output=True, final_output_type="text"),
        _Stage(final_output=False, final_output_type=None),
        _Stage(final_output=True, final_output_type="image"),
    ]

    final_stage = entry_utils.get_final_stage_id_for_e2e(
        output_modalities=["image"],
        default_modalities=["text", "image"],
        stage_list=stages,
    )
    assert final_stage == 2

    final_stage_text = entry_utils.get_final_stage_id_for_e2e(
        output_modalities=["audio", "text"],
        default_modalities=["text", "image"],
        stage_list=stages,
    )
    assert final_stage_text == 0
