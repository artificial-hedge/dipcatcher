"""``python -m quant_fund.cli`` — one execution of the Typer app.

Importing the package loads ``cli.main`` (the ``app`` re-export). Running
``python -m quant_fund.cli.main`` therefore finds that module already in
``sys.modules`` and runpy warns that execution may be unpredictable.
This module is not imported by the package, so runpy executes it once.
"""

from __future__ import annotations

from quant_fund.cli.main import app

if __name__ == "__main__":
    app()
