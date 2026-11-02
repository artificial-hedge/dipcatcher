"""SIGTERM unwinds like Ctrl-C, then the previous handler is restored.

``ThreadingHTTPServer.shutdown`` deadlocks when the serving thread calls it.
Raising ``KeyboardInterrupt`` from the signal handler lets the caller’s
existing ``finally`` close the socket or join a worker. Tests invoke the
installed handler directly: Windows can register SIGTERM, but the OS does
not deliver it the way POSIX does.
"""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager


def _raise_keyboard_interrupt(signum: int, frame: object | None) -> None:
    del signum, frame
    raise KeyboardInterrupt


@contextmanager
def sigterm_as_keyboard_interrupt(
    on_interrupt: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Install a main-thread SIGTERM handler for the duration of the block.

    ``on_interrupt`` runs when ``KeyboardInterrupt`` leaves the block (SIGTERM,
    or Ctrl-C that the block did not already handle). The previous handler is
    restored on every exit.
    """
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        try:
            yield
        except KeyboardInterrupt:
            if on_interrupt is None:
                raise
            on_interrupt()
    finally:
        signal.signal(signal.SIGTERM, previous)
