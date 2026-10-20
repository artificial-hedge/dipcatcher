# Proposed evidence chains (not stamped)

Status: proposal only. QA-0K adds these chains after the key holder re-signs.
This change does **not** edit `quality/epoch_heads.json`, does **not** run
`stamp-epochs`, and does **not** run `crown-jewels --write`.

`verify_repo` treats `CORPORA` as the closed world. Two tracked surfaces are
in no chain. `coverage_closure` reports each file as `uncovered_member` and
fails that gate. Drill clones that do not contain the directories stay
neutral, so this report does not invent a new baseline failure there.

needs-stamp: when the chains land, `docs/*`, `src/*` (`repo_integrity.py`),
and `Makefile` (if the audit recipe grows a `corpus-epoch` line) all move.
The key holder re-signs pins in the same review. Until then the gate stays
red on a full tree for this reason, on top of the existing pin drift.

## Surfaces

| Path | Tracked files | Proposed glob | require_stamped | allow_member_updates | Heads key |
|---|---|---|---|---|---|
| `data/manifests` | 19 JSON files below | `*.json` | true | false | `data/manifests/*.json` |
| `packaging` | `packaging/homebrew/fxi.rb` | `*` | true | true | `packaging/*` |

`data/manifests` is append-only evidence: replay carriers, corpus membership,
and tape manifests. A mutation needs an epoch acknowledgment, the same rule
as `receipts`. `packaging/` is supply-chain code and may update members the
way `scripts/*` does, still requiring a stamp on arrival.

`member_digests` is recursive, so one `data/manifests` / `*.json` chain
covers every file below. Do not add a second chain that prefixes the same
tree.

### `data/manifests` members (19)

- `data/manifests/replay_corpus_members.json`
- `data/manifests/yahoo_eod.json`
- `data/manifests/yahoo_eod_us_wide.json`
- `data/manifests/replay/calib.json`
- `data/manifests/replay/conformal.json`
- `data/manifests/replay/corpus.json`
- `data/manifests/replay/coverage.json`
- `data/manifests/replay/coverage_cs.json`
- `data/manifests/replay/cp.json`
- `data/manifests/replay/drift.json`
- `data/manifests/replay/loss_cs.json`
- `data/manifests/replay/mcs.json`
- `data/manifests/replay/mcs_vol.json`
- `data/manifests/replay/monitor.json`
- `data/manifests/replay/panel_audit.json`
- `data/manifests/replay/race.json`
- `data/manifests/replay/serial.json`
- `data/manifests/replay/tail.json`
- `data/manifests/replay/verdict.json`

### `packaging` members (1)

- `packaging/homebrew/fxi.rb`

## What QA-0K does later

1. Stamp `data/manifests` and `packaging` (`corpus-epoch`, require-stamped,
   the update flags above) and record the heads in `quality/epoch_heads.json`.
2. Add the two tuples to `CORPORA` in `src/quant_fund/research/repo_integrity.py`.
3. Remove those directories from `UNCHAINED_SURFACES` in the same file so
   `coverage_closure` stops double-reporting them. The epoch gate owns them
   from then on.
4. Add the matching `corpus-epoch` lines to `make evidence-audit`.
5. Human: `crown-jewels --write` for the jewel drift and `make sign-pins`.
   Do not mint a key in CI.

Until step 2, `unchained_surface_errors` keeps failing on the live tree.
That failure is the coverage-closure acceptance test, not a stamp.
