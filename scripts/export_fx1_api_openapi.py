"""Export the fx-1 harness API's OpenAPI schema.

Usage::

    uv run --no-sync python scripts/export_fx1_api_openapi.py \
        [clients/typescript/fx1/openapi.json]

Writes ``app.openapi()`` (the fx-1 harness serve surface — run/complete/
jobs/receipts routes) to the given path, defaulting to
``clients/typescript/fx1/openapi.json`` so the generated TypeScript client
can be rebuilt from a committed spec.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fx1.serve.api import create_app


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("clients/typescript/fx1/openapi.json")
    spec = create_app().openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} ({len(spec.get('paths', {}))} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
