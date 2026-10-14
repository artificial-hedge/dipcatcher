"""Interactive surface seal — registry/trigger digest anti-fork guard."""

from __future__ import annotations

import json

from fx1.harness import HARNESS_REGISTRY
from fx1.interactive.superpower import (
    _TRIGGERS,
    _WEB_TRIGGERS,
    CAPABILITIES,
    interactive_surface_digest,
)
from fx1.interactive.surface_seal import (
    build_surface_seal,
    surface_digest,
    surface_payload,
    surface_weave,
)


def test_surface_payload_covers_every_harness_command() -> None:
    payload = surface_payload(
        capabilities=CAPABILITIES,
        triggers=_TRIGGERS,
        web_triggers=_WEB_TRIGGERS,
    )
    names = {row["name"] for row in payload["harness"]}
    assert names == {c.name for c in HARNESS_REGISTRY}


def test_surface_digest_is_stable_hex() -> None:
    seal = build_surface_seal(
        capabilities=CAPABILITIES,
        triggers=_TRIGGERS,
        web_triggers=_WEB_TRIGGERS,
    )
    digest = seal["digest"]
    assert len(digest) == 64
    assert digest == interactive_surface_digest()
    raw = json.dumps(
        surface_payload(
            capabilities=CAPABILITIES,
            triggers=_TRIGGERS,
            web_triggers=_WEB_TRIGGERS,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert surface_digest(raw=raw) == digest


def test_surface_weave_is_stable_secondary_fingerprint() -> None:
    payload = surface_payload(
        capabilities=CAPABILITIES,
        triggers=_TRIGGERS,
        web_triggers=_WEB_TRIGGERS,
    )
    weave = surface_weave(payload)
    assert len(weave) == 64
    seal = build_surface_seal(
        capabilities=CAPABILITIES,
        triggers=_TRIGGERS,
        web_triggers=_WEB_TRIGGERS,
    )
    assert seal["weave"] == weave
    assert weave != seal["digest"]
