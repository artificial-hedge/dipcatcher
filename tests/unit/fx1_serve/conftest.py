"""fx1 serve tests share one OpenAPI document across identical apps.

Job-callback delivery retries through ``deliver_signed`` with the
production backoff (0.5s, then 1s). The probes assert attempt counts,
not the pause, so the suite passes a 10ms backoff into the api module's
binding. Production ``create_app`` is unchanged.
"""

from __future__ import annotations

import fx1.serve.api as api_mod
from fx1.serve.webhooks import deliver_signed
from tests.unit.fx1_serve.openapi_cache import install

install()

_REAL_DELIVER = deliver_signed


def _fast_deliver(url: str, secret: str | None, body: bytes, **kwargs: object):
    kwargs.setdefault("backoff_s", 0.01)
    return _REAL_DELIVER(url, secret, body, **kwargs)


api_mod.deliver_signed = _fast_deliver
