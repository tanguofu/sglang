"""Shared fixtures for gfx942 kernel unit tests."""

from __future__ import annotations

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs a HIP/CUDA device")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    mark = pytest.mark.gpu
    for item in items:
        item.add_marker(mark)


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    return bool(torch.cuda.is_available())


@pytest.fixture(autouse=True)
def _require_gpu(request: pytest.FixtureRequest, gpu_available: bool) -> None:
    if request.node.get_closest_marker("gpu") and not gpu_available:
        pytest.skip("GPU required")
