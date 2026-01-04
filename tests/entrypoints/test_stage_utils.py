import json
import os
import sys

import pytest

from vllm_omni.entrypoints.stage_utils import (
    append_jsonl,
    encode_for_ipc,
    maybe_load_from_ipc_with_metrics,
    set_stage_devices,
)


def _make_dummy_torch(call_log):
    class _Props:
        def __init__(self, total):
            self.total_memory = total

    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def set_device(idx):
            call_log.append(idx)

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_properties(idx):
            return _Props(total=16000)

        @staticmethod
        def mem_get_info(idx):
            return (8000, 16000)

        @staticmethod
        def get_device_name(idx):
            return f"gpu-{idx}"

    class _Torch:
        cuda = _Cuda

    return _Torch


@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
def test_set_stage_devices_respects_logical_ids(monkeypatch):
    # Preserve an existing logical mapping and ensure devices "0,1" map through it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    call_log: list[int] = []
    dummy_torch = _make_dummy_torch(call_log)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    monkeypatch.setattr("vllm_omni.utils.detect_device_type", lambda: "cuda")
    monkeypatch.setattr("vllm_omni.utils.get_device_control_env_var", lambda: "CUDA_VISIBLE_DEVICES")

    set_stage_devices(stage_id=0, devices="0,1")

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert call_log and call_log[0] == 0  # current device set after remap


class _DummySerializer:
    @staticmethod
    def serialize(obj):
        return json.dumps(obj).encode()

    @staticmethod
    def deserialize(buf):
        return json.loads(buf.decode())


def _patch_serializer(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.distributed.omni_connectors.utils.serialization.OmniSerializer",
        _DummySerializer,
        raising=False,
    )


def test_encode_for_ipc_prefers_inline_payload(monkeypatch):
    _patch_serializer(monkeypatch)
    obj = {"message": "small"}

    payload = encode_for_ipc(obj, threshold=10_000, obj_key="obj", shm_key="shm")

    assert payload == {"obj": obj}
    restored, metrics = maybe_load_from_ipc_with_metrics(payload, "obj", "shm")
    assert restored == obj
    assert metrics["rx_transfer_bytes"] > 0
    assert metrics["rx_decode_time_ms"] >= 0.0


def test_encode_for_ipc_uses_shared_memory_when_large(monkeypatch):
    _patch_serializer(monkeypatch)
    obj = {"payload": "x" * 256}

    payload = encode_for_ipc(obj, threshold=1, obj_key="obj", shm_key="shm")

    assert "shm" in payload
    restored, metrics = maybe_load_from_ipc_with_metrics(payload, "obj", "shm")
    assert restored == obj
    assert metrics["rx_transfer_bytes"] >= len(_DummySerializer.serialize(obj))


def test_append_jsonl_appends_records(tmp_path):
    log_path = tmp_path / "nested" / "records.jsonl"
    record_one = {"id": 1, "text": "hello"}
    record_two = {"id": 2, "text": "world"}

    append_jsonl(str(log_path), record_one)
    append_jsonl(str(log_path), record_two)

    contents = log_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in contents] == [record_one, record_two]
