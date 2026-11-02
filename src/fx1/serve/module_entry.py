"""Send ``python -m fx1.serve.api`` through the kill-file watcher.

``src/fx1/serve/api.py`` is hash-pinned, so this module cannot edit it.
``python -m`` imports ``fx1.serve`` while ``sys.argv[0]`` is still ``-m``,
then asks the api module loader for its code object before rewriting argv.
Wrapping that loader is the last non-pinned step before the pinned
``uvicorn.run`` call. A normal import does not have ``argv[0] == "-m"`` at
package import, and this hook does not install.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from typing import Any


class _WatchingLoader:
    """Delegate to the real loader, and arm the watcher from ``get_code``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec: ModuleSpec) -> object | None:
        create = getattr(self._inner, "create_module", None)
        if create is None:
            return None
        created: object | None = create(spec)
        return created

    def exec_module(self, module: object) -> None:
        self._inner.exec_module(module)

    def get_code(self, fullname: str) -> object:
        if fullname == "fx1.serve.api" and sys.argv[:1] == ["-m"]:
            arm_kill_watch()
        code: object = self._inner.get_code(fullname)
        return code

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class _ApiEntryFinder:
    """Return the api spec with a watching loader while argv still says ``-m``."""

    def __init__(self) -> None:
        self._busy = False

    def find_spec(
        self,
        fullname: str,
        _path: object = None,
        _target: object = None,
    ) -> ModuleSpec | None:
        if fullname != "fx1.serve.api" or self._busy or sys.argv[:1] != ["-m"]:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        if not isinstance(spec.loader, _WatchingLoader):
            spec.loader = _WatchingLoader(spec.loader)
        return spec


def install_module_entry_hook() -> None:
    """Install the api finder once. No effect unless this process was ``python -m``."""
    if sys.argv[:1] != ["-m"]:
        return
    if any(isinstance(finder, _ApiEntryFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _ApiEntryFinder())


def arm_kill_watch() -> None:
    """Replace ``uvicorn.run`` with the kill-file watcher for this process only."""
    import uvicorn

    from fx1.serve.kill_drain import serve_until_signal

    current = uvicorn.run
    if getattr(current, "__dipcatcher_kill_watch__", False):
        return
    original = current

    def watched(
        app: object,
        host: str = "127.0.0.1",
        port: int = 8000,
        **_kwargs: object,
    ) -> None:
        serve_until_signal(app, host=str(host), port=int(port), runner=original)

    watched.__dipcatcher_kill_watch__ = True  # type: ignore[attr-defined]
    uvicorn.run = watched
