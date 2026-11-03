"""Empty-module guard for ``tests/formal``. The root conftest is not modified."""

from __future__ import annotations

import pytest
from tests.support.collection_guard import finish_collection, note_collect_report, reset_collection


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    reset_collection()


@pytest.hookimpl(tryfirst=True)
def pytest_collectreport(report: pytest.CollectReport) -> None:
    note_collect_report(report)


def pytest_collection_finish(session: pytest.Session) -> None:
    finish_collection(session)
