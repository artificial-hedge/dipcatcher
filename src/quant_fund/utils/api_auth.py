"""Shared API-key check for readiness and ``/metrics``.

Accepts ``X-API-Key`` or ``Authorization: Bearer``. A caller that sends both
must send the same value. The comparison does not echo the credential.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping


def _header(headers: Mapping[str, str], name: str) -> str | None:
    direct = headers.get(name)
    if direct is not None:
        return str(direct)
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value)
    return None


def _bearer(authorization: str | None) -> str | None:
    """Return the token, ``None`` when the header is absent, ``""`` when malformed."""
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token or " " in token:
        return ""
    return token


def presented_api_key(headers: Mapping[str, str]) -> str | None:
    """The single presented credential, or ``None`` when it cannot be used."""
    header_key = _header(headers, "X-API-Key")
    bearer = _bearer(_header(headers, "Authorization"))
    if bearer == "":
        return None
    if header_key is not None and bearer is not None:
        if not hmac.compare_digest(header_key.encode("utf-8"), bearer.encode("utf-8")):
            return None
        return header_key or None
    token = header_key if header_key is not None else bearer
    if not token:
        return None
    return token


def credential_matches(presented: str | None, expected: str) -> bool:
    if presented is None or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
