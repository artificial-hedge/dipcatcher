# #2853 — Actions fan-out Tier 1 (mechanical) — ready-to-apply

The Tier 1 fan-out reduction (#2853 investigation, PR #2960) is fully
specified and the workflow files have been edited and epoch-stamped
locally. This branch documents the exact changes and the receipts so
the repo owner can apply them via the GitHub web UI or with a
`workflow`-scoped token (the agent's token only has `repo` scope,
which GitHub refuses to use for `.github/workflows/*.yml` updates).

## Why this is a separate branch

The agent's `gho_***` token has scopes `gist, read:org, repo`.
GitHub's security policy refuses to push or create PR content for
`.github/workflows/*.yml` files without the `workflow` scope. Both
the Contents API and the Git Data API return 403/404. This is a
hard GitHub-side restriction, not a tool limitation.

The agent shipped the stamp-only changes (a no-op pin advance) on a
throwaway branch, then deleted the throwaway. The workflow edits
were never published.

## Changes ready to apply (Tier 1)

### 1. `.github/workflows/matrix.yml`

Replace the env block and the matrix.python line:

```yaml
env:
  # Windows runners default to cp1252 locale encoding; every text artifact
  # this repo reads/writes is UTF-8, and POSIX runners already default to it.
  # PYTHONUTF8 pins PEP 540 UTF-8 mode on every interpreter in the matrix
  # (becomes the default in CPython 3.15 per PEP 686). PYTHONIOENCODING keeps
  # console I/O UTF-8 under Windows code pages.
  PYTHONUTF8: "1"
  PYTHONIOENCODING: "utf-8"
  NUM_SHARDS: "4"
  HYPOTHESIS_PROFILE: ci

# Python band narrow: PR-time fan-out uses 3.12 + 3.13 only. 3.14 is
# restricted to push:main (the floor and ceiling of pyproject.toml's
# `requires-python = ">=3.12"`, with 3.14 being the newest stable CPython
# where wheel coverage for locked deps needs verification). PR runs are
# the highest-volume trigger and the most queue-sensitive; the 3.14
# coverage check on push:main is enough to catch wheel regressions on
# merge. See #2853 investigation for the measurement of the 12-job
# reduction (3 OS × 1 py × 4 shards).
jobs:
  test:
    name: ${{ matrix.os }} / py${{ matrix.python }} / shard${{ matrix.shard }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 45
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ${{ github.event_name == 'pull_request' && fromJSON('["3.12", "3.13"]') || fromJSON('["3.12", "3.13", "3.14"]') }}
        shard: ["0", "1", "2", "3"]
```

PR-time fan-out: 36 → 24 jobs (3 OS × 2 py × 4 shards).

### 2. `.github/workflows/witness_monitor.yml`

Add a concurrency block after the `on:` block and before the
`permissions:` block:

```yaml
on:
  schedule:
    - cron: "13 4 * * 1" # weekly, Mondays 04:13 UTC
  push:
    branches: [main]
    paths: ["quality/witness/**", "quality/rekor_pubkey.pem", "quality/witness_signing.pub"]
  pull_request:
    branches: [main]
    paths: ["quality/witness/**", "src/quant_fund/research/integrity_witness.py"]
  workflow_dispatch:

# Concurrency: cancel any in-flight witness-monitor run on the same ref when
# a new push lands. Closes the no-cancellation backdoor that #2853 quantified
# (every push to main started a new run that never superseded the previous
# one, accumulating queued runs in the runner pool).
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read
```

## Stamps (after the edits land)

After applying both edits, the `.github/workflows/` and `quality/`
corpora need a re-stamp. Either run `make stamp-epochs` (which
re-stamps every covered corpus) or, for the minimum:

```bash
uv run dipcatcher corpus-epoch \
  --corpus-dir .github/workflows --glob '*.yml' \
  --out-dir .github/workflows --heads-pin quality/epoch_heads.json
uv run dipcatcher corpus-epoch \
  --corpus-dir quality --out-dir quality --heads-pin quality/epoch_heads.json
```

Expected: the chain head for `.github/workflows/*.yml` advances to
`n_epochs: 33`; the chain head for `quality/*.json` advances to
`n_epochs: 250`; the `quality/epoch_heads.json` pin reflects both.
The new epoch JSON files will be `corpus_epoch_68e9260f11f8f100.json`
(workflows) and `corpus_epoch_d259446541c2face.json` (quality) — but
the random hex prefix is generated at stamp time and will not match
this spec; the structure is what matters.

## Validation

After the edits land, the Tier 1 validation step in #2853's plan is:

1. Baseline `gh run list --limit 50` on `main` before this PR is
   opened (record queue time, started vs queued ratio).
2. Open the PR with these workflow edits + the epoch re-stamp in
   the same commit (so the stamp matches the file digests).
3. After merge, re-sample: expect `duration_queues < baseline`,
   `started <= concurrent` for the witness-monitor workflow, and
   `matrix.yml` job count on PR runs ≤ 24 (was 36).

The Tier 2 design changes (shard artifact sharing, fast/exhaustive
lane split) are documented in the #2853 investigation memo and are
**not** part of this branch.

## Honesty contract

- No FORBIDDEN_RESEARCH_METRIC_KEYS, no live-P&L claims, no
  Sharpe/Sortino/Calmar headlines.
- The "improvement" is asserted as a measurement target, not as a
  result; the validation step in the #2853 plan is the gate.
- The agent does not claim the changes are live — the workflow
  edits are ready to apply but not yet applied to the repo.

## Why a doc branch?

A code change that lives only in the local working tree is not
reviewable and not part of the chain. The agent's next session
should re-apply the workflow edits and re-stamp on a branch the
owner can merge with a `workflow`-scoped token.
