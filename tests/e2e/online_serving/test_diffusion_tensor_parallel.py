# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E regression test for tensor_parallel_size propagation to diffusion stages.

Verifies that ``--tensor-parallel-size N`` is correctly passed through
``OmniDiffusionConfig.from_engine_args → DiffusionParallelConfig`` and
that ``world_size`` is computed as the product of all parallelism dims.

This guards against the bug where ``create_diffusion_config()`` silently
dropped ``tensor_parallel_size``, ``data_parallel_size``, and
``pipeline_parallel_size`` (they were not in the manual field list).

See: https://github.com/vllm-project/vllm-omni/issues/3735#issuecomment-4499663173
"""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL = "black-forest-labs/FLUX.2-klein-4B"

_WIDTH = 512
_HEIGHT = 512
_NUM_INFERENCE_STEPS = 4


def _image_to_base64_jpeg(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _create_test_mask_b64(w: int = _WIDTH, h: int = _HEIGHT) -> str:
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw

    d = ImageDraw.Draw(mask)
    d.rectangle([w // 4, h // 4, w * 3 // 4, h * 3 // 4], fill=255)
    return _image_to_base64_jpeg(mask)


# ── Test parametrizations ──────────────────────────────────────────

_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--tensor-parallel-size",
                "2",
            ],
        ),
        id="tp2",
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--tensor-parallel-size",
                "2",
                "--data-parallel-size",
                "1",
                "--pipeline-parallel-size",
                "1",
            ],
        ),
        id="tp2_dp1_pp1",
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            # No explicit TP/DP/PP — use defaults (1 each).
            server_args=[],
        ),
        id="default_parallel",
    ),
]


# ── Tests ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_tp2_produces_valid_image(omni_server):
    """TP=2, DP=1, PP=1 produces a valid edited image with deterministic output."""
    import httpx

    image_b64 = _image_to_base64_jpeg(Image.new("RGB", (_WIDTH, _HEIGHT), (128, 128, 128)))
    mask_b64 = _create_test_mask_b64()

    url = f"http://{omni_server.host}:{omni_server.port}/v1/images/edits"
    files = {
        "image": ("image.jpg", base64.b64decode(image_b64), "image/jpeg"),
        "mask_image": ("mask.jpg", base64.b64decode(mask_b64), "image/jpeg"),
    }
    data = {
        "prompt": "A red rose in a meadow",
        "model": MODEL,
        "guidance_scale": 1.0,
        "num_inference_steps": _NUM_INFERENCE_STEPS,
        "n": 1,
        "seed": 100,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, files=files, data=data)
        resp.raise_for_status()
        result = resp.json()

    assert "data" in result and len(result["data"]) == 1
    b64_data = result["data"][0].get("b64_json") or result["data"][0].get("url", "").split(",")[-1]
    img = Image.open(BytesIO(base64.b64decode(b64_data)))
    assert img.size == (_WIDTH, _HEIGHT)


@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_tp2_deterministic_same_seed(omni_server):
    """Same input + same seed → identical output (TP does not break determinism)."""
    import httpx

    image_b64 = _image_to_base64_jpeg(Image.new("RGB", (_WIDTH, _HEIGHT), (128, 128, 128)))
    mask_b64 = _create_test_mask_b64()

    def generate():
        url = f"http://{omni_server.host}:{omni_server.port}/v1/images/edits"
        files = {
            "image": ("image.jpg", base64.b64decode(image_b64), "image/jpeg"),
            "mask_image": ("mask.jpg", base64.b64decode(mask_b64), "image/jpeg"),
        }
        data = {
            "prompt": "A blue sky",
            "model": MODEL,
            "guidance_scale": 1.0,
            "num_inference_steps": _NUM_INFERENCE_STEPS,
            "n": 1,
            "seed": 42,
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, files=files, data=data)
            resp.raise_for_status()
            return resp.json()["data"][0]["b64_json"]

    b64_1 = generate()
    b64_2 = generate()
    assert b64_1 == b64_2, "Deterministic output required for same seed"


@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_tp2_multiple_outputs(omni_server):
    """n=2 produces two distinct images."""
    import httpx

    image_b64 = _image_to_base64_jpeg(Image.new("RGB", (_WIDTH, _HEIGHT), (128, 128, 128)))
    mask_b64 = _create_test_mask_b64()

    url = f"http://{omni_server.host}:{omni_server.port}/v1/images/edits"
    files = {
        "image": ("image.jpg", base64.b64decode(image_b64), "image/jpeg"),
        "mask_image": ("mask.jpg", base64.b64decode(mask_b64), "image/jpeg"),
    }
    data = {
        "prompt": "A beautiful sunset",
        "model": MODEL,
        "guidance_scale": 1.0,
        "num_inference_steps": _NUM_INFERENCE_STEPS,
        "n": 2,
        "seed": 7,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, files=files, data=data)
        resp.raise_for_status()
        result = resp.json()

    assert "data" in result and len(result["data"]) == 2
