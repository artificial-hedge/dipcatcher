"""Pinned HTTP for the public vendor fetch scripts.

Redirects are followed only when every hop passes the outbound guard.
A refusal becomes ``URLError`` so the existing retry loops stay intact.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from quant_fund.utils.outbound import OutboundRefused, read_http_redirects


def read_vendor(request: urllib.request.Request, *, timeout: float) -> bytes:
    """Read ``request`` through the outbound guard, including bounded redirects."""
    try:
        return read_http_redirects(request, timeout=float(timeout))
    except OutboundRefused as exc:
        raise urllib.error.URLError(str(exc)) from exc
