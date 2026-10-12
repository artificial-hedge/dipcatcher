"""``BackendNotConfiguredError`` stays importable from the harness client.

The explicit ``as`` alias is the re-export. ``__all__`` does not list it.
"""

from __future__ import annotations

import fx1.serve.client as client
from fx1.serve.backends import BackendNotConfiguredError as Canonical
from fx1.serve.client import BackendNotConfiguredError


def test_backend_not_configured_error_reexport() -> None:
    assert BackendNotConfiguredError is Canonical
    assert client.BackendNotConfiguredError is Canonical
    assert issubclass(BackendNotConfiguredError, RuntimeError)
    assert "BackendNotConfiguredError" not in client.__all__
    raised = BackendNotConfiguredError("unconfigured", code="x_probe")
    assert str(raised) == "unconfigured"
    assert raised.code == "x_probe"
