"""Watch the operator kill file and latch the harness drain event.

``src/fx1/serve/api.py`` is hash-pinned. ``fx1 harness serve`` calls this
wrapper directly. ``python -m fx1.serve.api`` reaches it through the loader
hook in ``fx1.serve.module_entry``, which passes the original ``uvicorn.run``
as ``runner``. ``POST /harness/drain`` stays one-way. This watcher clears only
a latch it set, after the kill file has stayed ``ENABLED`` through the debounce.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from quant_fund.utils.operator_kill import DrainLatch


def serve_until_signal(
    app: Any,
    *,
    host: str,
    port: int,
    runner: Callable[..., object] | None = None,
) -> None:
    """Serve ``app`` and latch ``app.state.metrics.draining`` from the kill file.

    ``runner``, when set, is the unwrapped server call. The module-entry hook
    passes it so this function does not call itself.
    """
    import uvicorn

    run = uvicorn.run if runner is None else runner
    event = app.state.metrics.draining
    latch = DrainLatch.from_env()
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.25):
            latch.poll(event)

    latch.poll(event)
    thread = threading.Thread(target=watch, name="fx1-operator-kill", daemon=True)
    thread.start()
    try:
        run(app, host=host, port=port, reload=False, server_header=False)
    finally:
        stop.set()
        thread.join(timeout=1)
