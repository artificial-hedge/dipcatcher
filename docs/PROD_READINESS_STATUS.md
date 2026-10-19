# Production readiness status (QA-0K integrator)

**Measured at:** `origin/main` `f6f061f79` (2026-10-19 UTC)  
**Work-order base:** `cb4f2c52f` is an ancestor of `main` (+16 commits through `f6f061f79`).  
**Owner:** QA-0K maintains this file; other agents do not edit unless coordinating a criterion flip.

Do not call the repo production-ready until every exit criterion below is **green** on a completed CI run of `main`, or explicitly **parked** in [Human-only parked items](#human-only-parked-items).

---

## Exit criteria

| # | Status | Command / check | Last CI run (placeholder) |
|---|--------|-----------------|---------------------------|
| 1 | **RED** — no completed all-jobs-green CI on current `main` tip | Full push workflow set: lint, typecheck, test shards, coverage, evidence-audit, proofcore, fx1, arch-guards, platform matrix | In flight: https://github.com/artificial-hedge/dipcatcher/actions/runs/39201259280 (CI, `f6f061f79`, pending). Prior pushes cancelled mid-queue (`39201252179` `cfa8d1961`, `39201184002` `0febe4d69`, `39200925380` `ce6cfe32c`). Last **completed** CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510 (`5c474b6d3`, **failure**). |
| 2 | **RED** (local `make lint` on `cf6b3613b`, 4 vCPU) | `make lint && make typecheck && uv run mypy src/fx1` | CI lint job **green** on https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510 (`5c474b6d3`). Local `make lint` failed `check_mypy_strict_allowlist.py` (regressions in `hedge_lab/resources.py`, `models/deep_rl.py`, `models/nbeats.py`, `utils/seeds.py`). Log: `/opt/cursor/artifacts/qa0k_make_lint.log`. |
| 3 | **RED** (expected until human pin re-sign) | `uv run dipcatcher crown-jewels --check` and `uv run dipcatcher verify-repo` | evidence-audit on https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510: `pin_signatures: FAIL pin_drift:quality/crown_jewels.json,pin_drift:quality/epoch_heads.json`. Local `crown-jewels --check`: **intact**; `verify-repo` still reports `pin_drift` on those pins. |
| 4 | **RED** | `make evidence-audit` (all 7 gates reported; drills calibrated) | Same run as row 3 — job failed at `verify-repo` / pin drift; downstream drills not a reliable green signal until item H1. |
| 5 | **PENDING CI** | `make replay-sweep` (executed vs skipped carriers separated) | **VT-2M** landed `0febe4d69`: lane TMPDIR under run cwd, `evidence-replay-sweep` + bundle artifacts on `evidence-audit`. Subagent log: local `make replay-sweep` pass (~91–109s); confirm on next green `evidence-audit` (run 39201259280+). `mcs_vol` budget documented in `docs/OPERATIONS_RUNBOOK.md` (no manifest re-pin). |
| 6 | **UNKNOWN** | `make admission-gate ADMISSION_BASE=<last green>` and `make epoch-consistency EPOCH_BASE=<last green>` | No **last green** CI SHA recorded yet; use post-green run URL once criterion 1 flips. |
| 7 | **PASS** (spot-check) | `git diff cb4f2c52f..origin/main -- quality/ pyproject.toml tests \| grep -E 'skip\|xfail\|baseline\|deselect\|fail_under'` — review every hit | No integrator-driven baseline loosening in this audit; parallel agents must keep it that way. |
| 8 | **IN PROGRESS** | This file matches reality line by line | Updated by QA-0K integrator pass 2026-10-19. |

### Recent CI failures (last completed CI `39157190510`, SHA `5c474b6d3`)

| Job | Result | Notes |
|-----|--------|--------|
| `lint` | success | |
| `test (3.12\|3.13, *)` | failure / cancelled | e.g. `pin_drift` in `test_auditor_bundle.py`, `test_bundle_differential.py`, canon qualification drift |
| `coverage` | failure | (log not fully mined this pass) |
| `evidence-audit` | failure | `verify-repo pin_signatures: pin_drift` on crown jewels + epoch heads |
| `pretrade-risk` | failure | |
| `atlas` (`0350d4346`) | failure | stale `docs/ARCHITECTURE_ATLAS.md` / `docs/architecture/*` — regenerate with `scripts/gen_arch_diagrams.py` |

---

## Human-only parked items

Parked items are **not** fixable in-agent. Each line is the exact action required.

| ID | Status | Action |
|----|--------|--------|
| **H1 — Re-sign pins** (D1-03/D2-02/D3-04) | **BLOCKING** | `gate_pins.sig` stale since 2026-09-30. Holder of `GATE_SIGNING_KEY` (Ed25519, `key_id` `bc3bcff676056adf`) runs `make sign-pins`, then `make checkpoint`, `make witness-checkpoint`, and anchor steps. Until then: `verify-repo`, `make evidence-audit`, tamper/fuzz drills, and several unit tests stay red. Do not mint a new key or weaken the signature check. |
| **H2 — Branch protection** (D1-01/D2-10) | **BLOCKING** for “production” governance | Repo admin defines required checks / merge queue / revert bot for `main` (currently advisory CI). |
| **H3 — Epoch ack remediation** (D1-02) | **OWNER DECISION** | 13 self-excused epoch acks (12 in-place receipt mutations) plus 39 ungated acks from `84d3d31b7`. Agents list only; owner chooses remediate vs grandfather. |
| **H4 — Private tapes & Windows fleet** (D1-06, D5-11) | **BLOCKING** for full replay | 14/16 replay carriers need private tapes; NTFS/ACL behaviour needs `D:\dipcatcher`. |
| **H5 — fx1 model endpoint** (D5-01/D5-02) | **BLOCKING** for live chat | No production `fx1` / `fx1-lite` weights endpoint; do not substitute another vendor as “fx1”. |
| **H6 — Release tag & versioning** (D4-06/D5-06) | **OWNER DECISION** | `fx1.__version__` frozen at 0.4.0; install/Homebrew assets may not exist until owner tags. |
| **H7 — Admission honesty for stamped history** (D1-08) | **OWNER DECISION** | 176/346 stamped non-legacy receipts lack stamp admission would require today; 6 registered. Owner: register, re-admit, or document grandfathering. |

---

## Agent queue hints (for parallel workers)

| Agent | Start when | Current signal on `main` |
|-------|------------|---------------------------|
| **WP-3J** | in progress | `6172922c8` bounded benches; `be9f8639a` pytest-timeout 900s; shard balance / `.test_durations` still open. |
| **KX-7Q** | done (`e6222f455`) | `fx1/cli.py` → `cli_harness` + `cli_common`; `test_modules_stay_under_max_lines` green. |
| **ZR-9V** | parallel | evidence-audit masks; base-ref gating; anchor staleness reporting. |
| **VT-2M** | done (`0febe4d69`) | Lane TMPDIR isolation, CI replay/bundle artifacts, runbook budget note; await evidence-audit on 39201259280+. |
| **HX-5D** | done (this pass) | `test_stateful.py` TZ/LC_ALL + repo-root config path on `main`. |
| **FX-1A** | done (`f6f061f79`) | corpus quarantine refuse, honesty lexical holes, ship fixture CLI. |
| **PD-8S** | done (`f8ec9e649`) | Key scrub, repo-root paths, sim-live honesty, superpower partial exit, lab doctor, env/README parity (`test_pd8s_frontdoor.py`). |
| **RL-4C** | **landed** (`cf6b3613b`) | release preflight, checkout creds, `make ci` parity (`test_workflow_security.py` guards `ci.yml` / `release.yml`). |
| **QA-0K** | continuous | **No stamping** this pass unless `needs-stamp:` commits land and pin holder signs (H1). Reconcile this doc after each green CI candidate. |

### Parallel landings on `main` (inventory)

| SHA | Agent / theme | Summary |
|-----|----------------|---------|
| `6172922c8` | WP-3J | Bound research oracle/CLI to required benches |
| `802d170c0` | honesty | Tighten honesty corpus ship gate and doctor jewels |
| `cf6b3613b` | RL-4C Shipwright | Release preflight, checkout `persist-credentials: false`, `make ci` parity |
| `0febe4d69` | VT-2M | Replay lane TMPDIR isolation + CI evidence artifacts |
| `f6f061f79` | FX-1A | Corpus quarantine refuse, honesty lexical holes, ship fixture CLI |
| `ef717fc99` | HX-5D | Formal Hypothesis collection, env isolation tests, webhook injected clock |
| `b42a31864` | HX-5D | Formal repo-root `paper.yaml`; TZ/LC_ALL (superseded by formal `conftest` fixture) |
| `7fcf09a10` | HX-5D | Formal locale via `tests/formal/conftest.py` (env baseline guard) |
| `f8ec9e649` | PD-8S | Receipt census, `.env` parity, fx1 doctor repo root |
| `be9f8639a` | WP-3J | `pytest-timeout` 900s default (shard hang visibility) |
| `7e920fd25` | KX-7Q | Split `fx1 harness` CLI into `cli_harness` / `cli_common` |
| `e6222f455` | KX-7Q | Remove duplicate help constants post-split (ruff F811) |
| `ea351a1b1` | PD-8S | `/use` env scrub, repo-root paths, sim-live honesty, superpower `--sync` |
| `779306f27` | PD-8S | Interactive tests + lab doctor `dir_*_created` reporting |
| `ce6cfe32c` | QA-0K | Initial `PROD_READINESS_STATUS.md` ledger |

Open PRs (2026-10-19): #3008 release hardening (`sesori/goal-shipping-dipcatcher`), #3006/#3004 harness drafts — unrelated to KX-7Q/PD-8S (landed on `main`).

---

## Integrator log

| When (UTC) | SHA | Note |
|------------|-----|------|
| 2026-10-19 | `cf6b3613b` | Initial status file; `cb4f2c52f` verified ancestor; local `make lint` mypy-allowlist red; `crown-jewels --check` intact; `verify-repo` pin_drift expected pending H1. |
| 2026-10-19 | `ce6cfe32c` | QA-0K status ledger on `main`. |
| 2026-10-19 | `0febe4d69` | **VT-2M** replay TMPDIR + CI evidence artifacts; criterion 5 pending CI replay-sweep. |
| 2026-10-19 | `f6f061f79` | **FX-1A** honesty/corpus fixes; CI run 39201259280 queued. |
| 2026-10-19 | `b42a31864` | QA-0K refresh + HX-5D formal locale/repo-root follow-up on `main`. |
| 2026-10-19 | `f8ec9e649` | **PD-8S** front-door lane landed; `test_pd8s_frontdoor.py` 10/10 locally. |
| 2026-10-19 | `e6222f455` | **KX-7Q** CLI line-budget split + ruff cleanup on `main`. |
