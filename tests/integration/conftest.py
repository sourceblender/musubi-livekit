"""Connect integration tests to an operator-provided disposable Musubi stack."""

from __future__ import annotations

import os
from typing import Any

import pytest


@pytest.fixture
def api_client() -> Any:
    url = os.getenv("MUSUBI_TEST_API_URL")
    token = os.getenv("MUSUBI_TEST_TOKEN")
    if not url or not token:
        pytest.skip("set MUSUBI_TEST_API_URL and MUSUBI_TEST_TOKEN for a disposable test stack")

    from musubi_sdk import AsyncMusubiClient

    return AsyncMusubiClient(base_url=url, token=token)
