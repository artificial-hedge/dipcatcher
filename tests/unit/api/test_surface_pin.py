"""Tests for api/surface_pin.py — the API contract ratchet."""

from __future__ import annotations

from fastapi import FastAPI

from quant_fund.api.app import app as real_app
from quant_fund.api.surface_pin import check, diff, surface


def _app() -> FastAPI:
    a = FastAPI()

    @a.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @a.post("/run")
    def run(mode: str, strict: bool = False) -> dict[str, str]:
        return {"mode": mode, "strict": str(strict)}

    return a


def test_surface_shape():
    s = surface(_app())
    assert set(s) == {"/health", "/run"}
    assert s["/run"]["POST"]["params"] == [
        {"name": "mode", "required": True},
        {"name": "strict", "required": False},
    ]


def test_diff_no_drift():
    s = surface(_app())
    d = diff(s, s)
    assert d == {"issues": [], "breaking": False, "n_issues": 0}


def test_route_removed_is_breaking():
    p, live = surface(_app()), surface(_app())
    del live["/health"]
    d = diff(p, live)
    assert d["breaking"] and "route_removed:/health" in d["issues"]


def test_route_added_is_not_breaking():
    p, live = surface(_app()), surface(_app())
    live["/new"] = {"GET": {"params": []}}
    d = diff(p, live)
    assert not d["breaking"] and "route_added:/new" in d["issues"]


def test_required_param_flip_breaks():
    p, live = surface(_app()), surface(_app())
    live["/run"]["POST"]["params"][1]["required"] = True
    d = diff(p, live)
    assert d["breaking"]
    assert any(i.startswith("param_changed") for i in d["issues"])


def test_real_app_surface_nonempty():
    s = surface(real_app)
    assert "/health" in s and "/forecast/{symbol}" in s


def test_check_against_pin(tmp_path):
    import json

    pin = tmp_path / "api_surface.json"
    pin.write_text(json.dumps(surface(_app())))
    out = check(_app(), pin)
    assert out["ok"]
