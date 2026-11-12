"""Shared test fixtures: each test's evidence directory, and the four-can round."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOUR_CANS = ["--seed", "4", "--cans", "4"]
"""The four-can round ``test_four_cans`` and ``test_planning_latency`` both read."""


@pytest.fixture
def artifacts(request: pytest.FixtureRequest) -> Path:
    """Where a test writes its evidence: ``$GEODUDE_ARTIFACTS`` (default ``artifacts/``)
    / <test name>."""
    base = Path(os.environ.get("GEODUDE_ARTIFACTS", ROOT / "artifacts"))
    out = base / str(request.node.name)
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def four_cans(tmp_path_factory: pytest.TempPathFactory) -> tuple[int, Path]:
    """``geodude-recycling`` on ``FOUR_CANS`` with ``--record``, run once per session: its
    exit code and the directory it recorded into."""
    from geodude import recycling

    out = tmp_path_factory.mktemp("four_cans")
    return recycling.main([*FOUR_CANS, "--record", str(out)]), out
