"""One test per journey id so a regression names the conversation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fx1.journeys.engine import run_journey
from tests.fx1.journeys.schema import load_journeys

_JOURNEYS = load_journeys()


@pytest.mark.parametrize("journey_id", [journey.id for journey in _JOURNEYS], ids=lambda item: item)
def test_journey(journey_id: str, tmp_path: Path) -> None:
    journey = next(item for item in _JOURNEYS if item.id == journey_id)
    result = run_journey(journey, root=tmp_path / journey_id)
    assert result.passed, "\n".join(result.failures)
