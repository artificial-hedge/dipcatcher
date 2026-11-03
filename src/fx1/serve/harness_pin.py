"""One pinned HTTP exchange for the harness client.

``fx1.serve.client`` re-exports ``_pinned_harness_exchange``. Callers in
that module look the name up there, so a patch on the client module is the
function that runs.
"""

from __future__ import annotations

import urllib.request

from quant_fund.utils.outbound import OutboundRefused, open_pinned


def _pinned_harness_exchange(
    req: urllib.request.Request, timeout_s: float
) -> tuple[int, dict[str, str], bytes]:
    """Resolve once and dial that numeric address. Redirects stay as statuses.

    Loopback and RFC1918 literals stay allowed. Other hostnames must resolve
    entirely to public addresses; ``localhost`` may resolve to loopback.
    """
    from fx1.serve.client import HarnessTransportError

    try:
        with open_pinned(req, timeout=timeout_s, allow_loopback=True, allow_private=True) as handle:
            return int(handle.status), dict(handle.headers), handle.read()
    except OutboundRefused as exc:
        raise HarnessTransportError(f"harness destination refused: {exc}") from exc
