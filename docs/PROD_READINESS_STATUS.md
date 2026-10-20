# Production readiness status (QA-0K integrator)

**Measured at:** `origin/main` `8a59e26d8` (2026-10-20 UTC). Prior doc: `9af260100`. Harness/superpower fix: `46a3fb91e`.  
**Work-order base:** `cb4f2c52f` is an ancestor of `main`.  
**Owner:** QA-0K. **Not production-ready** until every exit line is **green** on a completed CI run of `main`, or **parked** under human-only items below.

**Landed sibling work (on `main`):**

| Lane | Tip SHA | Note |
|---|---|---|
| RL-4C | `de448576d` | `release.yml` preflight requires green CI + evidence-audit; `persist-credentials: false`; `AGENTS.md` / `Makefile` ci aligned with `ci.yml`. |
| VT-2M | `b5f1f6d11`, `0febe4d69`, `ead2bd4f9` | Replay lane env isolation; `mcs_vol` carrier shrink (`lane_evolution`); CI replay / integrity drill uploads. Migration: `evidence_migration_20261019.json` (`mcs_vol_smaller_inputs`). |
| ZR-9V | `b34527e55` | Fail-closed `evidence-audit` (`_ea`); `evidence-base-ref` admission; stale `timestamp_anchors` with age; replay executed/skipped headlines; `coverage_closure` red on unchained `data/manifests/*` (proposal only — **do not stamp** here). |
| PD-8S | `7e5fbc655` | FrontDoor credentials, paths, honesty, superpower opt-in; `tests/unit/frontdoor/test_pd8s_frontdoor.py`. |
| WP-3J | `34818a153` | PR required-only tests; slow full catalog `timeout(5400)`; `pytest-timeout` 900s; `check-test-durations` CI job; path inventory shards. |
| KX-7Q | `f4f817d11`, `59b9c4533` | `fx1/cli/*` split; **KX-7Q-1** line-budget **green** (`api_request_ingress.py`; shrink-only pins + `evidence_migration_20261020.json`). |
| FX-1A | `9a76eb4af` | Corpus quarantine refuse (9 SFT pins); lexical P&L/NAV guards; doctor crown-jewels; ship gate / `--fixture` SYNTHETIC; fx1 coverage floor **90** (91.19% − 1pp documented) — **`make fx1-coverage` not re-run** at land. **130** targeted honesty tests passed. |
| HX-5D | `e7ebe13d9` | **HX-5D-1** stateful Hypothesis collects/passes `-n 4`. **HX-5D-2** Numba / TZ / `LC_ALL` isolation + guarded env test. **HX-5D-3** `OPENBLAS_CORETYPE` pin + cross-kernel threshold test. **HX-5D-4** webhook audit injected clock; wall-clock timing gates removed from probes — **park:** occasional `receipt_sha256` mismatch on executor races. **HX-5D-5** `slow` marker registered; `pytest --strict-markers --collect-only` exit 0. |
| Leftovers | `48c79ce44` | Interactive import/planning repair: `ruff check src/fx1/interactive` **clean**; `test_research_banner` collects/passes (`_echo_synthetic_marker` export restored). |
| Leftovers | `46a3fb91e` | `popen_cancellable` restored on `fx1.harness`; `test_harness_cancel` collects/passes. Family selector summary includes **Evidence (harness-verified)**; superpower test **passed**. |
| Leftovers | `8a59e26d8` | `quant_fund/research/` McCabe **clean** (refactor splits); `quality/mccabe_baseline.txt` entries **lowered/removed only** — not raised. |

**mcs_vol epoch stamp:** satisfied in `d0b9d45cd` (`scripts/*` through `b5f1f6d11`). **No restamp** this pass. `needs-stamp:` on `b34527e55` for `docs/UNGATED_EPOCH_ACKS.md` / `docs/PROPOSED_EVIDENCE_CHAINS.md` — **not stamped** (do-not-stamp guidance; manifests chain remains proposal).

Supporting lists: [`D1_02_EPOCH_ACK_INVENTORY.md`](D1_02_EPOCH_ACK_INVENTORY.md), [`UNGATED_EPOCH_ACKS_84d3d31b7.md`](UNGATED_EPOCH_ACKS_84d3d31b7.md), [`UNGATED_EPOCH_ACKS.md`](UNGATED_EPOCH_ACKS.md), [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md).

## Remaining code follow-ups (agent-owned, not H1–H7)

These are real on clean `main` at `8a59e26d8`; fixing them is engineering work, not owner key ceremony.

| Area | Status | Command / note |
|---|---|---|
| **Lint (McCabe)** | **RED** | `quant_fund/research/` **clean** at `8a59e26d8`. Repo-wide `make lint` still **RED**: `superpower._stage_web_research` **+10** new complexity; `webresearch.Researcher.run` **15>14**; `ops_cmds` `sim_live` **13>10**; `pipeline.doctor` **16>15**; plus canonical-form nag / stale pins in `fx1/flash`, `fx1/serve`. **Do not raise baseline** — refactor or lower pins only. |
| **Pytest** | **PARTIAL** | `test_harness_cancel` + superpower Evidence summary **green** (`46a3fb91e`). **Pre-existing fail:** `tests/unit/research/test_replay_binding.py::test_contract_still_accepts_legacy_bodies`. |
| **Interactive ruff** | **GREEN** | `ruff check src/fx1/interactive` clean (since `48c79ce44`). |
| **Crown jewels** | **RED** | `uv run quant crown-jewels --check` → `jewel_mutated` (e.g. `AGENTS.md`, `Makefile`, `pyproject.toml`, `corpus_epoch.py`, `repo_integrity.py`, `tests/conftest.py`, …) + `jewel_unpinned` ed25519 helpers. |
| **Epoch chains** | **RED** | `verify-repo` epoch forks/multiple heads on `src`/`tests`/`docs`; digest drift on `scripts`, `quality`, `.github/workflows`; unstamped scripts/docs members. Needs integrator stamp pass **after** H1 or coordinated re-pin — **not done here**. |
| **Coverage closure** | **RED (honest)** | ZR-9V-5: all `data/manifests/**` + `packaging/homebrew/fxi.rb` `uncovered_member` until manifests chain lands. |
| **HX-5D webhook load proof** | **PARK (flake)** | Occasional `receipt_sha256` mismatch under executor races — watch CI / harden, not a gate weaken. |
| **FX-1A coverage** | **NOT VERIFIED** | Floor raised to 90 in ratchet; full `make fx1-coverage` blocked by collection errors above. |

## Exit criteria

1. **RED** — No completed all-green CI on `main`. Command: `gh run list --repo artificial-hedge/dipcatcher --branch main --workflow CI --limit 10`. Latest at measure: **pending** https://github.com/artificial-hedge/dipcatcher/actions/runs/39407250669 (`3cab21659`). Prior pushes mostly **cancelled** mid-queue; last old completed run still **failure** https://github.com/artificial-hedge/dipcatcher/actions/runs/39257948317.
2. **RED** — `make lint` at `8a59e26d8` **fails** (repo-wide McCabe drift listed above; `quant_fund/research/` fixed). Interactive ruff **green**. `test_modules_stay_under_max_lines` **passes**. `test_replay_binding` legacy-body contract **fails** (pre-existing).
3. **RED** — `uv run quant verify-repo` at `48c79ce44` (not re-run this micro-pass; prior at `3cab21659`): `crown_jewels` **FAIL**; `pin_signatures` **FAIL** `pin_drift` (expected until **H1**); `coverage_closure` **FAIL**; `timestamp_anchors` **stale age=20d** (ZR-9V-3 honest); epoch **FAIL** on multiple chains; `lattice` **ok**; `checkpoint`/`witness`/`spine` **ok**.
4. **RED until H1** — `make evidence-audit` (seven trailing gates via `_ea`) blocked by jewel/pin/epoch redness. CI evidence-audit not green on latest runs.
5. **PARTIAL** — Replay sweep policy (executed vs skipped headlines) landed (ZR-9V-4); tape-backed carriers skip without private bars; `mcs_vol` synthetic lane documented. No fresh all-green CI replay artifact at measure.
6. **POLICY GREEN, no green CI SHA** — `evidence-base-ref` / `ADMISSION_BASE` policy (ZR-9V-2). **No** all-jobs-green `main` commit for production admission. `ADMISSION_BASE=b5f1f6d11 make admission-gate` not re-run this pass.
7. **GREEN vs `cb4f2c52f`** (FX-1A caveat) — No pytest **skip/xfail** retreat; WP-3J time bounds only. FX-1A-5 **tightened** fx1 `fail_under` to 90 — not a loosening. Full fx1 coverage % not re-measured end-to-end.
8. **THIS FILE** — Exit 1–5 red or partial; 6–7 qualified green; H1–H7 parked. **Not production-ready.**

## Human-only parked items

Agents do not generate `GATE_SIGNING_KEY`, do not run `make sign-pins`, do not grandfather receipts, and do not point another vendor at the fx1 chat id.

1. **H1 Re-sign pins (D1-03/D2-02/D3-04).** Ed25519 `key_id` `bc3bcff676056adf`. Only the holder of `GATE_SIGNING_KEY` runs `make sign-pins`, then checkpoint / witness / anchor steps. Until then `pin_signatures` and dependent gates stay red.
2. **H2 Branch protection and required checks (D1-01/D2-10).** Repo admin must require CI status checks / merge queue. `gh api` branch protection may return 403 for agents.
3. **H3 Epoch acks (D1-02).** **13** self-excused acks at `84d3d31b7` plus **39** ungated entries ([`UNGATED_EPOCH_ACKS_84d3d31b7.md`](UNGATED_EPOCH_ACKS_84d3d31b7.md), [`UNGATED_EPOCH_ACKS.md`](UNGATED_EPOCH_ACKS.md) — not reverted). Owner decides remediate vs grandfather.
4. **H4 Private tapes and Windows fleet (D1-06, D5-11).** Tape-backed replay carriers and `D:\dipcatcher` fleet host not verified in cloud integrator passes.
5. **H5 fx1 model endpoint (D5-01/D5-02).** `fx1.__version__` `0.4.0`; no production weights endpoint; do not mislabel third-party chat as fx1.
6. **H6 Release tag and versioning (D4-06/D5-06).** Release asset `v0.4.0` / Homebrew `sha256` `FILL_AT_RELEASE` — owner tags and publishes (`gh api …/releases/tags/v0.4.0` → 404).
7. **H7 Admission honesty for stamped history (D1-08).** Work-order bar **176 / 346** stamped receipts failing honesty admission (owner triage). Detail: [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md). Agents do not grandfather.

## Integrator log

| When (UTC) | SHA | Note |
|---|---|---|
| 2026-10-20 | `a1e266414` | **Close-out:** all lanes recorded; HX-5D; no stamp. |
| 2026-10-20 | `48c79ce44` | Leftover interactive lint + `test_research_banner`. |
| 2026-10-20 | `46a3fb91e` / `8a59e26d8` | Harness cancel + superpower Evidence; research McCabe clean; repo-wide McCabe still red. |
