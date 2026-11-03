"""Liveness probe for the container HEALTHCHECK. Stdlib only.

Reads the bind port from ``DIPCATCHER_BIND_PORT``, then
``RESEARCH_API_PORT``, then ``PORT``, and requests
``http://127.0.0.1:<port>/health``. Exit 0 only on HTTP 200.
"""

from __future__ import annotations

import http.client
import os
import sys

_PORT_ENV = ("DIPCATCHER_BIND_PORT", "RESEARCH_API_PORT", "PORT")


def probe_port() -> str | None:
    for name in _PORT_ENV:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        if raw.isdigit() and 1 <= int(raw) <= 65535:
            return raw
        return None
    return "8000"


def main() -> int:
    port = probe_port()
    if port is None:
        return 1
    try:
        connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=3)
        try:
            connection.request("GET", "/health")
            status = connection.getresponse().status
        finally:
            connection.close()
    except (OSError, TimeoutError, ValueError, http.client.HTTPException):
        return 1
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
