"""Watch the operator kill file and latch the harness drain event.

``src/fx1/serve/api.py`` is hash-pinned, so this wrapper is applied by
``fx1 harness serve``. The pinned ``python -m fx1.serve.api`` entry does not
call it. The drain event is one-way, matching ``/harness/drain``.
"""

from __future__ import annotations

import threading
from typing import Any

from quant_fund.utils.operator_kill import latch_drain


def serve_until_signal(app: Any, *, host: str, port: int) -> None:
    """Serve ``app`` and latch ``app.state.metrics.draining`` from the kill file."""
    import uvicorn

    event = app.state.metrics.draining
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.25):
            latch_drain(event)

    latch_drain(event)
    thread = threading.Thread(target=watch, name="fx1-operator-kill", daemon=True)
    thread.start()
    try:
        uvicorn.run(app, host=host, port=port, reload=False, server_header=False)
    finally:
        stop.set()
        thread.join(timeout=1)
