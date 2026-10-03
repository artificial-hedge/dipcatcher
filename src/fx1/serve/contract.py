"""Wire-contract version shared by the harness server and its clients.

``API_VERSION`` is bumped on breaking changes to the pinned OpenAPI
surface. The server stamps it on every response as
``X-Fx1-Api-Version`` and reports it via ``GET /harness/version``;
``HarnessClient.check_compat`` compares it against the constant it was
built with, so a client can negotiate before sending work. It lives in
this module — not ``fx1.serve.api`` — so the client and CLI resolve it
without importing FastAPI.
"""

from __future__ import annotations

API_VERSION = "1"

__all__ = ["API_VERSION"]
