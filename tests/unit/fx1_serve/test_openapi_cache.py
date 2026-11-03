"""The shared OpenAPI cache must not change the document an app publishes."""

from __future__ import annotations

import fastapi.applications as fa
from fastapi.testclient import TestClient

import fx1.serve.api as api_mod
from fx1.harness import Harness
from tests.unit.fx1_serve.openapi_cache import REAL_GET_OPENAPI


def _app(*, rate_limit_rps: float | None = None):
    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    return api_mod.create_app(
        harness=Harness(runner=fake_runner),
        rate_limit_rps=rate_limit_rps,
    )


def _health_headers(spec: dict) -> dict:
    op = spec["paths"]["/health"]["get"]
    return op["responses"]["200"].get("headers", {})


def test_cached_schema_matches_a_direct_build_except_declared_headers() -> None:
    app = _app()
    published = app.openapi()
    fresh = REAL_GET_OPENAPI(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        summary=app.summary,
        description=app.description,
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
        routes=app.routes,
        webhooks=app.webhooks.routes,
        tags=app.openapi_tags,
        servers=app.servers,
        separate_input_output_schemas=app.separate_input_output_schemas,
        external_docs=app.openapi_external_docs,
    )
    assert set(published["paths"]) == set(fresh["paths"])
    assert published["components"]["schemas"] == fresh["components"]["schemas"]
    # The cache returns a copy: stamping headers on this app must not
    # leak into the document a second identical app publishes, and a
    # direct rebuild still sees the unstamped operation.
    assert "X-Request-ID" in _health_headers(published)
    assert "X-Request-ID" not in _health_headers(fresh)
    again = _app().openapi()
    assert again == published


def test_rate_limit_header_stamp_stays_on_the_app_that_asked_for_it() -> None:
    plain = _health_headers(_app().openapi())
    limited = _health_headers(_app(rate_limit_rps=5).openapi())
    assert "X-RateLimit-Limit" not in plain
    assert "X-RateLimit-Limit" in limited


def test_cache_is_what_create_app_calls() -> None:
    assert fa.get_openapi is not REAL_GET_OPENAPI
    client = TestClient(_app())
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/health" in resp.json()["paths"]
