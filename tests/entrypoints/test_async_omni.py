from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_get_supported_tasks_returns_engine_supported_tasks():
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(supported_tasks=("generate", "speech"))

    supported_tasks = await omni.get_supported_tasks()

    assert supported_tasks == ("generate", "speech")
