# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E regression test for tensor_parallel_size propagation to
a headless diffusion stage (stage_id=1).

Launches a multi-stage model (BAGEL: thinker AR + DiT diffusion) with
``use_stage_cli=True`` so the diffusion stage runs as a separate
headless process.  ``tensor_parallel_size`` is set in the per-stage
deploy YAML and must be picked up by
:meth:`DiffusionParallelConfig.from_engine_args`.

Guards against the bug where ``create_diffusion_config()`` silently
dropped ``tensor_parallel_size`` — the stage would launch with
``world_size=1`` and OOM or misbehave.

See: https://github.com/vllm-project/vllm-omni/issues/3735#issuecomment-4499663173
"""

from __future__ import annotations

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
_CI_DEPLOY = get_deploy_config_path("ci/bagel.yaml")

_NUM_INFERENCE_STEPS = 2


def get_tp2_deploy() -> str:
    """Produce a deploy YAML with TP=2 for the diffusion stage (stage 1)."""
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"devices": "0"},
                1: {"tensor_parallel_size": 2, "devices": "1,3"},
            },
        },
    )


# ── Test parametrizations ──────────────────────────────────────────

test_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_tp2_deploy(),
            server_args=["--disable-log-stats"],
            use_stage_cli=True,
        ),
        id="tp2_diffusion_stage_1",
    ),
]


# ── Tests ───────────────────────────────────────────────────────────


@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_diffusion_stage1_with_tp2_generates_image(omni_server, openai_client) -> None:
    """Stage-1 diffusion with ``tensor_parallel_size=2`` produces a valid image."""
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            content_text="<|im_start|>A small red cube on a white table.<|im_end|>"
        ),
        "modalities": ["image"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": _NUM_INFERENCE_STEPS,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    responses = openai_client.send_diffusion_request(request_config, request_num=1)
    assert len(responses) == 1
    response = responses[0]
    assert response.success, f"Diffusion request failed: {response.error}"
    assert response.images is not None and len(response.images) == 1
    img = response.images[0]
    assert img.size == (512, 512)


@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_diffusion_stage1_with_tp2_deterministic(omni_server, openai_client) -> None:
    """Same seed → same image (TP does not break determinism)."""
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            content_text="<|im_start|>A blue flower.<|im_end|>"
        ),
        "modalities": ["image"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": _NUM_INFERENCE_STEPS,
            "guidance_scale": 0.0,
            "seed": 12345,
        },
    }

    r1 = openai_client.send_diffusion_request(request_config, request_num=1)
    r2 = openai_client.send_diffusion_request(request_config, request_num=1)

    assert r1[0].success and r2[0].success
    assert r1[0].images and r2[0].images
    assert r1[0].images[0].size == r2[0].images[0].size == (512, 512)

    img1 = r1[0].images[0]
    img2 = r2[0].images[0]
    assert list(img1.getdata()) == list(img2.getdata()), (
        "Same seed must produce identical output"
    )
