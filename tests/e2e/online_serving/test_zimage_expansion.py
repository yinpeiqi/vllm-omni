"""
Targeted online serving tests for Z-Image diffusion.

Coverage kept intentionally small because TeaCache / Cache-DiT are already
covered elsewhere in CI:
- Ulysses-SP
- Ring-Attention
- multi-image count guard
- returned image correctness (decodable + expected dimensions)
"""

import pytest

from tests.conftest import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    assert_image_valid,
    decode_b64_image,
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


BASE_EXTRA_BODY = {
    "height": 512,
    "width": 512,
    "num_inference_steps": 2,
    "guidance_scale": 0.0,
    "seed": 42,
}

PARALLEL_FEATURE_CASES = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--ulysses-degree",
                "2",
            ],
        ),
        id="parallel_ulysses_sp2",
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--ring",
                "2",
            ],
        ),
        id="parallel_ring2",
        marks=PARALLEL_FEATURE_MARKS,
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
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

SINGLE_CARD_CASES = [
    pytest.param(
        OmniServerParams(model=MODEL),
        id="single_card_base",
        marks=SINGLE_CARD_FEATURE_MARKS,
    ),
]


def _build_request_config(*, num_outputs_per_prompt: int = 1) -> dict:
    extra_body = dict(BASE_EXTRA_BODY)
    extra_body["num_outputs_per_prompt"] = num_outputs_per_prompt
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "extra_body": extra_body,
    }


def _collect_image_url_items(openai_client: OpenAIClientHandler, request_config: dict):
    chat_completion = openai_client.client.chat.completions.create(
        model=request_config["model"],
        messages=request_config["messages"],
        extra_body=request_config.get("extra_body"),
    )
    image_items = []
    for choice in chat_completion.choices:
        content = getattr(choice.message, "content", None)
        assert content is not None, "API response content is None"
        assert not isinstance(content, str), "API response content should be image items, not plain text"

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image_url" and item.get("image_url") is not None:
                    image_items.append(item)
            elif hasattr(item, "image_url") and item.image_url is not None:
                image_items.append(item)
    return image_items


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    PARALLEL_FEATURE_CASES,
    indirect=True,
)
def test_zimage_parallel_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Exercise the sequence-parallel modes supported by Z-Image."""
    request_config = _build_request_config()
    request_config["model"] = omni_server.model

    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", SINGLE_CARD_CASES, indirect=True)
def test_zimage_num_outputs_per_prompt(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """The online API should return exactly the requested number of images."""
    request_config = _build_request_config(num_outputs_per_prompt=2)
    request_config["model"] = omni_server.model

    openai_client.send_diffusion_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", SINGLE_CARD_CASES, indirect=True)
def test_zimage_output_image_count(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Guard that the API does not collapse a multi-image response."""
    request_config = _build_request_config(num_outputs_per_prompt=2)
    request_config["model"] = omni_server.model

    image_items = _collect_image_url_items(openai_client, request_config)

    assert len(image_items) == 2, f"Expected 2 image_url items in response, got {len(image_items)}"


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", SINGLE_CARD_CASES, indirect=True)
def test_zimage_output_image_correctness(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """Verify returned image payloads are data URIs that decode to 512x512 images."""
    request_config = _build_request_config()
    request_config["model"] = omni_server.model

    image_items = _collect_image_url_items(openai_client, request_config)

    assert len(image_items) == 1, f"Expected 1 image_url item in response, got {len(image_items)}"

    for item in image_items:
        if isinstance(item, dict):
            url = item["image_url"]["url"]
        else:
            url = item.image_url.url
        assert url.startswith("data:image"), f"image_url item is not a data URI: {url[:80]}"
        image = decode_b64_image(url.split(",", 1)[1])
        assert_image_valid(image, width=BASE_EXTRA_BODY["width"], height=BASE_EXTRA_BODY["height"])
