# Proposed evidence chain: `data/manifests` and `packaging/`

Proposal only. Do **not** add these corpora to `quality/epoch_heads.json`
until the key holder re-signs pins (QA-0K stamps; `gate_pins.sig` is stale
since 2026-09-30). This document does not chain, pin, or re-seal anything.

## Closed-world miss (reproduced)

`verify_repo`'s `CORPORA` table is the closed world: every tracked file
under a listed directory is a chain member, an exemption, or an error.
Two sibling trees sit next to chained dirs and are not listed:

- `data/manifests/` — replay carriers, corpus-member lists, tape
  manifests. `data/metadata` *is* a corpus; `data/manifests` is not.
- `packaging/` — Homebrew formula (`packaging/homebrew/fxi.rb`).
  `tests/unit/research/test_repo_integrity.py` currently files this under
  `NON_CORPUS_TOP_DIRS`, which documents the hole rather than covering it.

Tracked members at the time of this proposal (counts move; the gate
fails on any tracked file under those prefixes):

- 19 files under `data/manifests/` (replay carriers, `yahoo_eod*.json`,
  `replay_corpus_members.json`)
- 1 file under `packaging/` (`homebrew/fxi.rb`)

`declared_tree_coverage_gate` reports each as `unchained_member:<path>`.
That gate is fail-closed in `verify_repo` so the miss cannot stay silent.
It does **not** invent an epoch chain or edit signed pins.

## Proposed chain (QA-0K, after re-sign)

Add two `CORPORA` rows — same shape as the existing policy table — and
stamp them with `make stamp-epochs`, then `make sign-pins` (human key):

| corpus_dir | glob | require_stamped | allow_member_updates | notes |
|---|---|---|---|---|
| `data/manifests` | `*` | True | True | Replay carriers and tape manifests are claim-bearing inputs. Mutable between restamps; arrivals must be stamped. Exempt nothing that GitHub/CI would execute. Do not fold into `data/metadata`. |
| `packaging` | `*` | True | True | Distribution formulas are supply-chain inputs (Homebrew tap). |

Also:

1. Drop `packaging` from `NON_CORPUS_TOP_DIRS` once the chain exists.
2. Extend `make stamp-epochs` / `make epoch-consistency` / evidence-audit
   `corpus-epoch:` loops with the two dirs (same `require-stamped` /
   `allow-member-updates` flags as the table).
3. Keep `data/manifests/replay/corpus.json` and
   `data/manifests/replay_corpus_members.json` as chain members — do not
   specially untrack them; in-flight generators remain owned elsewhere.
4. After stamp, the key holder signs pins. Until then this proposal
   stays a doc plus the fail-closed coverage-closure gate.
