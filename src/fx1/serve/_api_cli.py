"""CLI entry for ``python -m fx1.serve.api`` (kept out of api.py's line budget)."""

from __future__ import annotations

import os

from fx1.serve.api import _API_KEY_ENV, create_app
from fx1.serve.release_gate import ALLOW_TEST_FIXTURE_ENV


def main() -> int:
    """Serve the fx1 API on a loopback host unless an API key is configured."""
    import uvicorn

    host = os.environ.get("FX1_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FX1_API_PORT", "8011"))
    if host not in {"127.0.0.1", "::1", "localhost"} and not os.environ.get(_API_KEY_ENV):
        raise SystemExit(
            "non-loopback binding requires FX1_API_KEY; refusing unauthenticated exposure"
        )
    public_release = host not in {"127.0.0.1", "::1", "localhost"}
    if public_release and os.environ.get(ALLOW_TEST_FIXTURE_ENV) == "1":
        raise SystemExit(f"non-loopback binding refuses {ALLOW_TEST_FIXTURE_ENV}=1")
    app = create_app(public_release=public_release)
    uvicorn.run(app, host=host, port=port, reload=False, server_header=False)
    return 0
