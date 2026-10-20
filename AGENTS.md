# AGENTS.md — working in this repo

## What this is

**fx-1** is the product program: a quant research LLM planned as a fine-tune
of Kimi K3 open weights, with chat ids **fx1** and **fx1-lite**
([artificialhedge.co](https://artificialhedge.co)). No fine-tuned weights
are published, and `fx1.__version__` is the harness API, not a model
release. **dipcatcher** is the
state-of-the-art harness that orchestrates them from a chat-first,
Codex-style CLI: plain text is a conversation with the active model, and
`/superpower` plans and runs the research families that fit the goal (design
and current status: `docs/DIP_CONCIERGE.md`). Bare `dipcatcher` in a terminal
opens the chat via the outer launcher `src/dipcatcher_cli`; `quant` is the lab
CLI without it, and `fxi` stays compatible.
Underneath, `src/quant_fund` is the data engine, evaluation, and verification
that builds and gates fx-1 (`src/fx1`).

## Setup

```bash
make sync          # uv sync --frozen --all-groups --all-extras (uv.lock is
                   # authoritative; --frozen must pass — regenerate with
                   # `uv lock` after touching [project]). Includes the torch
                   # `nn` extra, matching the CI test job.
```

`MOONSHOT_API_KEY` is needed only for hosted eval (`make fx1-eval`); see
`.env.example` for all env vars.

## Gates (run before committing)

| Gate | Command |
|---|---|
| Lint | `make lint` (ruff check + format --check, mypy-strict allowlist, McCabe ratchet on `src`/`tests`) |
| Types | `make typecheck` (mypy `src/quant_fund` + public API `--strict`); fx1: `uv run mypy src/fx1` |
| Security | `make security` (bandit; the `ci.yml` `lint` job runs this too) |
| Lab tests | `make test` (PR gate: not network, not slow, xdist). `make test-full` includes slow. CI shards `test` on 3.12/3.13. |
| Local CI mirror | `make ci` — `lint` + `typecheck` + `security` + `coverage` (the `ci.yml` `lint` job + PR `test`/`coverage` gate) |
| Other `ci.yml` jobs | `make audit`, `make mc-engine-smoke`, `make native-lint`, `make market-sim-test`, `make parity-smoke`, `make examples`, `make audit-obs`, `make diffbacktest`, `make robustness-smoke`, `make formal`, `make evidence-audit`, `make pretrade-bench`, `make stress-smoke`, `make test-durations`; GitHub-only: `package`, `container`, `test-inventory`, `perf`, `smoke`, `rust-accel`, `required-gate` (`ci-required` merge-queue check) |
| fx1 suite | `make fx1-test` — separate lane (`fx1.yml`), ~300+ tests |
| fx1 full gate | `make fx1-gate` — lint + types + tests + honesty + corpus smoke |
| CI parity | `.github/workflows/ci.yml` + `fx1.yml` use `uv sync --frozen`. A `v*.*.*` tag does not sign unless the latest completed `CI` run on that SHA is green and its `evidence-audit` job succeeded (`scripts/require_release_gates.py`, `release.yml` preflight). |

## Honesty contract (hard rules — do not weaken)

1. Research results are **proper scores** (pinball, CRPS, PIT, QLIKE, Brier,
   ECE, Kupiec, HMM likelihood). Never headline Sharpe/Sortino/Calmar/P&L/NAV.
   `FORBIDDEN_RESEARCH_METRIC_KEYS` in `quant_fund.research.catalog` is
   mirrored by `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` — change them together;
   `tests/fx1/test_honesty_inheritance.py` blocks drift.
2. SYNTHETIC results are correctness tests, always labeled, never market
   evidence.
3. Receipts are immutable evidence (`receipts/`, `verify-research`). Every
   research claim should be reproducible from a receipt hash.

## Conventions

- **Commits land directly on `main`** for fx-1 lanes; cross-cutting repo
  changes go through PRs. Match the existing terse conventional-commit style.
- `data/fx1/`, `data/*` derived dirs, `mlflow.db`, coverage files are
  gitignored — regenerate, never commit.
- `tests/fx1` is excluded from the default pytest testpaths on purpose; run it
  via `make fx1-test`. (The `tests/tests` mirror was dropped; if a stray
  reappears, keep it out of default collection.)
- Verifier acceptance history lives in `verifier/vN/`; run receipts in
  `verifier/runs/`. `INFLIGHT` / `day_grind_progress.md` / `docs/DATA_CONTRACTS.md`
  are the lab's work-tracking files — read them before grabbing work.
- `fx1_seed_corpus.jsonl` and the root docs (`APPLY.md`, `MATH_SPEC.md`,
  `RESEARCH_REFERENCES.md`) are tracked inputs/notes — don't delete blindly.

## Remote fleet (Windows box `D:\dipcatcher`)

The SOTA/eval fleet runs on a dedicated Windows host; conventions below are
load-bearing — jobs break silently otherwise.

- **PowerShell-only.** Remote ops scripts are `.ps1`; bash heredocs do not
  exist there. Quote `cmd /c` inner strings with backtick-escaped quotes.
- **WMI spawn, not `Start-Process`.** Jobs are created via
  `Invoke-CimMethod -ClassName Win32_Process -MethodName Create` so they are
  owned by WMI and survive ssh session teardown. Always wrap the payload in
  `cmd /c "... 1> stdout.log 2> stderr.log"` — `Win32_Process` has no built-in
  redirection.
- **Parametrized launcher.** New fleet work goes through
  `scripts/fleet_spawn.ps1 -Manifest jobs.json` (one JSON job list, one
  spawn path) plus `scripts/fleet_watchdog.ps1` (heartbeat file at
  `.dsh-24x7\fleet_heartbeat.json`, auto-respawn bounded by `-MaxRespawns`).
  `scripts/fleet_manifest_sota.ps1` regenerates the canonical SOTA manifest.
  The legacy `spawn_*.ps1` one-offs stay for reference; prefer the launcher.
- **Durable paths.** Long-running artifacts live under `.dsh-24x7\` (gitignored,
  durable remotely); model weights under `data\models\`; bars under
  `data\raw\sources\`. Do not write fleet state into temp dirs. **Caveat: 912
  files under `.dsh-24x7/` are still tracked despite the ignore rule.** Some
  are pinned evidence: the SOTA reproduction workflow reads committed `native/`
  and `mega-arena/` inputs, and the integrity gate chains this directory as a
  corpus. `.gitignore` only prevents new scratch from being added. Do not run a
  blanket `git rm --cached`; first preserve the evidence bytes and references
  through an explicit signed migration (`docs/ULTRA_PROD_READINESS.md`, Phase 2).
- **Thread pinning.** Jobs set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `TOKENIZERS_PARALLELISM=false` in their `env` block — the manifest schema
  carries this; keep it when authoring new jobs.
- **Defender exclusions.** The remote host needs `D:\dipcatcher` excluded
  from real-time scanning or parquet/bar reads get throttled mid-fleet.

## Key docs

`docs/FX1.md` (model), `docs/FX1_TRAINING.md` (compute ladder + gates),
`docs/FX1_API_STABILITY.md` (versioning — `fx1.__version__` is canonical
semver; pyproject reads it via hatch dynamic version),
`docs/FX1_DATA.md`, `docs/FX1_DATASOURCES.md`, `docs/FX1_ARCHITECTURE.md`,
`docs/OPERATIONS_RUNBOOK.md`.
