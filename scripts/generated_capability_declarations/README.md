# Generated capability declarations

This directory contains executable registration source split by the real
fx-1 extension owner:

- `skills/` — one file for each registered harness command.
- `plugins/` — one file for each registered datasource adapter.
- `features/` — one file for each reviewed feature output.

Every `_register(seed_id)` statement maps to the generated runtime wrapper named
at the top of its file. The runtime uses the compact seed and the package under
`src/fx1/extensions/`; these declaration files are the reproducible
million-line source ledger. They do not implement independent capabilities or
satisfy the requested one-million-capability, one-file-per-capability target.

Regenerate from the repository root:

```bash
uv run python scripts/seed_fx1_capabilities.py
```
