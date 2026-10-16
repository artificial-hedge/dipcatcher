# Getting started

Install the locked environment, confirm the lab is research-only, and verify a receipt before treating any score as evidence.

## Requirements

- Python 3.12 or newer (`requires-python` is `>=3.12`)
- [uv](https://docs.astral.sh/uv/)

`MOONSHOT_API_KEY` is used only by `make fx1-eval`, which grades Moonshot's hosted Kimi K3 base. The harness commands below run without it.

## Install

From the repository root:

```bash
make sync
```

That runs `uv sync --frozen --all-groups --all-extras`. `uv.lock` is authoritative. The `docs` dependency group (MkDocs) is included in `--all-groups` and is omitted from the runtime image, which syncs with `--no-dev`.

## Check the lab

```bash
uv run dipcatcher doctor --config configs/research.yaml
```

A fresh checkout has no `data/metadata/data_manifest.json` and no `data/metadata/research/latest.json`. `doctor` then exits 1 with `data_manifest=missing` and `research_receipt=missing`. That is the empty tree, not a broken install. `live_allowed` stays false on the research profile. A receipt that fails verification is `invalid`; fix the inputs, and do not weaken the verifier. The operations detail is in the [runbook](OPERATIONS_RUNBOOK.md).

## SYNTHETIC research smoke

```bash
uv run python examples/01_receipt_round_trip.py
```

Unset, `dipcatcher research` runs the full wired optional battery, which is thousands of benches. The example sets `DIPCATCHER_RESEARCH_REQUIRED_ONLY=1`, seals a **SYNTHETIC** required-family receipt, verifies it, and rejects a one-byte edit. It is a correctness test. It is not market evidence and it does not authorize promotion.

Real-file and public-source workflows are separate. Read [Data source labels](DATA_SOURCE_LABELS.md) before publishing a score, and [Receipt verification](RECEIPT_VERIFICATION.md) before citing one.

## This site

```bash
make docs
make docs-serve
```

`make docs` runs `mkdocs build --strict`. `make docs-serve` serves the site at `http://127.0.0.1:8000`.

## Next

- [Architecture](ARCHITECTURE.md)
- [Data contracts](DATA_CONTRACTS.md)
- [fx-1 overview](FX1.md)
- [API reference](api/index.md)
