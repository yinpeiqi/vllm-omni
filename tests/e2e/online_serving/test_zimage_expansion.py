"""
Comprehensive online serving tests for Z-Image diffusion features.

Coverage:
- TeaCache
- Tensor Parallel (TP=2)
- VAE Patch Parallel (with TP=2)

Validation is delegated to assert_diffusion_response in tests.conftest,
which checks successful generation and expected output dimensions.
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
)
from tests.utils import hardware_marks

MODEL = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A high-detail studio photo of an orange tabby cat sitting on a laptop keyboard."

SINGLE_CARD_FEATURE_MARKS = hardware_marks(
    res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"},
)
PARALLEL_FEATURE_MARKS = hardware_marks(
    res={"cuda": "L4", "rocm": "MI325"},
    num_cards=2,
)


def _get_diffusion_feature_cases():
    """Return L4 diffusion feature cases for Z-Image."""
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                ],
            ),
            id="single_card_teacache",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                ],
            ),
            id="parallel_tp2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_tp2_vae_pp2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases(),
    indirect=True,
)
def test_zimage(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Exercise supported Z-Image online diffusion feature combinations."""
    request_config = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
