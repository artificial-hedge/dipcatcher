# Proposed evidence chain: `data/manifests` and `packaging/`

**Status:** proposal for QA-0K / key-holder stamping (ZR-9V-5). Not yet in `quality/epoch_heads.json`.

## Problem

`dipcatcher verify-repo` closed-world coverage does not yet attest:

- `data/manifests/replay/*.json` (16 replay carriers plus membership sidecars)
- `data/manifests/*_tape*.json` and related tape manifests
- `packaging/` (Homebrew formula, release wheel metadata)

These paths are load-bearing for replay sweeps and install surfaces but can drift without an epoch receipt.

## Proposed chain: `manifests_packaging`

| Field | Value |
|---|---|
| Corpus id | `manifests_packaging` |
| Members glob | `data/manifests/**`, `packaging/**` |
| Mutable | yes (`--allow-member-updates` when carriers or formulas change) |
| Admission | same honesty bar as other mutable corpora; replay carrier edits require migration record in `quality/evidence_migrations/` |

## Stamping steps (QA-0K only, after owner signs)

1. Add chain head to `quality/epoch_heads.json` and jewel pin if needed.
2. Run per-chain stamp (not full `make stamp-epochs` unless all chains drifted):

   ```bash
   uv run dipcatcher corpus-epoch \
     --corpus-dir data/manifests \
     --glob '**/*' \
     --out-dir data/manifests \
     --heads-pin quality/epoch_heads.json
   ```

   Repeat for `packaging/` with its corpus dir / out-dir.

3. `make sign-pins` (human H1), then `uv run dipcatcher verify-repo`.

## Test hook

Once stamped, extend `tests/unit/research/test_corpus_epoch.py` coverage-closure tests so an unchained file under `data/manifests/` fails verify-repo closure (red until the chain lands).

## Out of scope for agents

Do not re-pin carriers or manifests solely to green a check; every drift needs a migration label in `quality/evidence_migrations/`.
