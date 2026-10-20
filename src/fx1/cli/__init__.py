"""``fx1`` command-line interface — the model project's front door.

dipcatcher remains the harness CLI (``dipcatcher``/``quant`` aliases).
This CLI drives corpus, eval, training manifests, and inspection.
"""

import fx1.cli.data_cmds as _data_cmds  # noqa: F401
import fx1.cli.eval_cmds as _eval_cmds  # noqa: F401
import fx1.cli.forecast_cmds as _forecast_cmds  # noqa: F401
from fx1.cli._app import app
from fx1.cli.eval_cmds import _resolve_judge


def main() -> None:
    app()


__all__ = ["app", "main", "_resolve_judge"]
