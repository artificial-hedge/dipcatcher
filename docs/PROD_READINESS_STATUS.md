# Production readiness status (QA-0K integrator)

**Measured at:** `origin/main` `59b9c4533` (2026-10-20 UTC). Prior doc: `5e90d1a5f`.  
**Work-order base:** `cb4f2c52f` is an ancestor of `main`.  
**Owner:** QA-0K. Do not call the repo production-ready until every exit line is **green** on a completed CI run of `main`, or **parked** below with an owner action.

**Landed sibling work (on `main`):**

| Lane | Tip SHA | Note |
|---|---|---|
| RL-4C | `de448576d` | `release.yml` preflight requires green CI + evidence-audit; `persist-credentials: false`; `AGENTS.md` / `Makefile` ci targets aligned with `ci.yml`. |
| VT-2M | `b5f1f6d11`, `0febe4d69`, `ead2bd4f9` | Replay lane `HOME`/`TMP`/`GIT_CEILING_DIRECTORIES` isolation; `mcs_vol` producer shrink + carrier re-pin (`claims_equal: false`, `lane_evolution`; `receipts/mcs_vol_drill.json` unchanged); CI uploads replay / verify-repo / tamper / fuzz artifacts. Migration: `quality/evidence_migrations/evidence_migration_20261019.json` (`mcs_vol_smaller_inputs`). |
| ZR-9V | `b34527e55` | **ZR-9V-1** `make evidence-audit` runs all seven trailing gates via Makefile `_ea` (fail-closed: every gate executes before exit). **ZR-9V-2** CI `ADMISSION_BASE` / `EPOCH_BASE` resolve from last green **evidence-audit** (`evidence-base-ref`); `tests/unit/research/test_evidence_base.py`; owner list [`UNGATED_EPOCH_ACKS.md`](UNGATED_EPOCH_ACKS.md) (**39** acks, not reverted). **ZR-9V-3** `verify-repo` `timestamp_anchors` reports **stale with age** (no fake-green attestation). **ZR-9V-4** replay coverage headlines **executed vs skipped** separately. **ZR-9V-5** `coverage_closure` **FAIL** on unchained `data/manifests/*` (+ `packaging/`); proposal [`PROPOSED_EVIDENCE_CHAINS.md`](PROPOSED_EVIDENCE_CHAINS.md) — agents **do not** stamp that chain or `epoch_heads.json` here. |
| PD-8S | `7e5fbc655` | **P0** `quant` lab CLI import restored (`capabilities_app` `NameError` fixed in `6f2bbfedb`). **PD-8S-1** stored credentials do not leak into harness/lab subprocess env (`/use`, `Harness.run` scrub). **PD-8S-2** endpoint normalization + profile/credential imports repaired. **PD-8S-3** repo-root path resolution and `quant doctor` from non-repo cwd (`/tmp`). **PD-8S-4** sim-live default stdout passes honesty validator; `--diagnostic` may headline sharpe. **PD-8S-5** superpower `--sync` partial run exits non-zero. **PD-8S-6** README receipt census + `.env.example` vars documented on operator surfaces (incl. CI/OS parity test). **PD-8S-7** superpower web research requires explicit opt-in; capability catalog works off-repo cwd. Acceptance: `tests/unit/frontdoor/test_pd8s_frontdoor.py`. Chat launcher `dipcatcher` ≠ lab CLI `quant`. |
| WP-3J | `34818a153` | **Shardbreaker** CI test latency. **WP-3J-1** PR path runs required-only tests (~25s / ~47–63s observed); full catalog on `slow` with `timeout(5400)`. **WP-3J-2** `pytest-timeout` 900s default + `process_map` start method forkserver/spawn/fork. **WP-3J-3** `check-test-durations` CI job guards `.test_durations` — do **not** regen `.test_durations` locally in integrator passes. **WP-3J-4** path-based inventory shards in `ci.yml`. Tip on `main` also includes `f4f817d11` (KX-7Q). One late local qualification-audit hash mismatch blamed on sibling churn — **watch next CI**. |
| KX-7Q | `f4f817d11`, `59b9c4533` | **Ratchet:** `src/fx1/cli.py` → `src/fx1/cli/*`; `tiny_lm` extract (`f4f817d11`). **KX-7Q-1** `test_modules_stay_under_max_lines` **GREEN** at `59b9c4533`: `fx1/serve/api.py` split to `api_request_ingress.py` + shrink-only pins in `quality/module_line_budgets.txt` (migration `quality/evidence_migrations/evidence_migration_20261020.json`). CLI help unchanged except Typer sign/attestation ordering. |

**Parked follow-ups (not gate weakens):** `pytest -k` may still hit **unrelated collection errors** (e.g. `superpower`, `test_research_banner`) — environment/import debt, **not** a loosened line ratchet. PD-8S `concierge.py` ruff **F821** on unhit paths may remain until owner cleanup.

**mcs_vol epoch stamp:** `needs-stamp: mcs_vol` on `b5f1f6d11` satisfied in `d0b9d45cd` (`scripts/*` through `b5f1f6d11`). **No restamp.** `data/manifests/replay/mcs_vol.json` remains uncovered until manifests chain lands (ZR-9V-5).

Supporting lists: [`D1_02_EPOCH_ACK_INVENTORY.md`](D1_02_EPOCH_ACK_INVENTORY.md), [`UNGATED_EPOCH_ACKS_84d3d31b7.md`](UNGATED_EPOCH_ACKS_84d3d31b7.md), [`UNGATED_EPOCH_ACKS.md`](UNGATED_EPOCH_ACKS.md), [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md).

## Exit criteria

1. **RED (in progress / cancelled)** — Full CI on `main` with no cancelled/failed jobs. Command: `gh run list --repo artificial-hedge/dipcatcher --branch main --workflow CI --limit 20`. Last **partial** run for RL-4C tip: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964 (`de448576d`; lint/audit/evidence-audit failed — not all-green). Newer pushes cancel mid-queue. Last **completed** CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39257948317 (**failure**).
2. **PARTIAL** — `test_modules_stay_under_max_lines` **green** at `59b9c4533` (KX-7Q-1). Local full gate not re-run this pass. Command: `make lint && make typecheck && uv run mypy src/fx1`; PR CI uses WP-3J shards + `check-test-durations`. Stale full-CI ref: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964 — watch CI on `59b9c4533`.
3. **RED** — `uv run quant verify-repo` at `ce36d0464`: `crown_jewels` **FAIL**; `pin_signatures` **FAIL** `pin_drift` (expected until H1); `coverage_closure` **FAIL** `uncovered_member` for all `data/manifests/**` and `packaging/homebrew/fxi.rb` (ZR-9V-5 honest red); `timestamp_anchors` **stale** with age (~20d at measure — ZR-9V-3, not treated as green); epoch forks/drift on several chains. Command: `uv run quant verify-repo`.
4. **RED until H1** — `make evidence-audit` must run and report all seven trailing gates (`crown-jewels`, `verify-witness`, `verify-repo`, `checkpoint-chain`, `verify-rotations`, `tamper-drill`, `fuzz-drill`); ZR-9V-1 `_ea` wrapper runs each gate fail-closed. Still blocked by pin/jewel drift and CI redness. Last observed: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964.
5. **PARTIAL** — `make replay-sweep` / `replay_coverage.v1` now separates **executed** vs **skipped** headline counts (ZR-9V-4). 16 carriers; 14 tape-skipped locally; `mcs_vol` synthetic executes. CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964 (not full green).
6. **POLICY GREEN (ZR-9V-2), no green CI base** — CI sets `ADMISSION_BASE` / `EPOCH_BASE` from last completed green **evidence-audit** commit (`dipcatcher evidence-base-ref`), not a cancelled push tip. Unit contract: `tests/unit/research/test_evidence_base.py`. **No all-jobs-green CI SHA** for production. Manual `ADMISSION_BASE=b5f1f6d11 make admission-gate` still exits 0 (`no corpus receipt changes`).
7. **GREEN vs `cb4f2c52f`** — No integrator-driven pytest skip / coverage floor loosening since work-order base. WP-3J bounds wall time (shards, `pytest-timeout`, slow catalog) without lowering `fail_under` or adding skips.
8. **THIS FILE** — Lines 1–5 red or partial; line 6 policy improved, still no green CI base; line 7 green; H1–H7 parked. **Not production-ready.**

## Human-only parked items

Agents do not generate `GATE_SIGNING_KEY`, do not run `make sign-pins`, do not grandfather receipts, and do not point another vendor at the fx1 chat id.

1. **H1 Re-sign pins (D1-03/D2-02/D3-04).** Ed25519 `key_id` `bc3bcff676056adf`. Only the holder of `GATE_SIGNING_KEY` runs `make sign-pins`, then checkpoint / witness / anchor steps. Until then `pin_signatures` and dependent gates stay red.
2. **H2 Branch protection and required checks (D1-01/D2-10).** Repo admin must require CI status checks / merge queue. `gh api` branch protection may return 403 for agents.
3. **H3 Epoch acks (D1-02).** **13** self-excused acks at `84d3d31b7` plus **39** ungated entries ([`UNGATED_EPOCH_ACKS_84d3d31b7.md`](UNGATED_EPOCH_ACKS_84d3d31b7.md), owner inventory [`UNGATED_EPOCH_ACKS.md`](UNGATED_EPOCH_ACKS.md) — not reverted). Owner decides remediate vs grandfather. Agents only list; `needs-stamp:` on `b34527e55` for those docs — **not stamped** this pass.
4. **H4 Private tapes and Windows fleet (D1-06, D5-11).** Tape-backed replay carriers and `D:\dipcatcher` fleet host not verified in cloud integrator passes.
5. **H5 fx1 model endpoint (D5-01/D5-02).** `fx1.__version__` `0.4.0`; no production weights endpoint; do not mislabel third-party chat as fx1.
6. **H6 Release tag and versioning (D4-06/D5-06).** Release asset `v0.4.0` / Homebrew `sha256` `FILL_AT_RELEASE` — owner tags and publishes (`gh api …/releases/tags/v0.4.0` → 404).
7. **H7 Admission honesty for stamped history (D1-08).** Work-order bar **176 / 346** stamped receipts failing honesty admission (owner triage). Detail: [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md). Agents do not grandfather.

## Integrator log

| When (UTC) | SHA | Note |
|---|---|---|
| 2026-10-19 | `d0b9d45cd` | Stamp `scripts`/`src`/`tests`/`docs`/`workflows`/`quality` through `b5f1f6d11` (includes **mcs_vol** producer). |
| 2026-10-20 | `b34527e55` | **ZR-9V** honest evidence gates (landed; docs `needs-stamp` not applied). |
| 2026-10-20 | `465d8cd23` | QA-0K: ZR-9V recorded in readiness doc. |
| 2026-10-20 | `7e5fbc655` | **PD-8S** FrontDoor landed; readiness doc updated (concierge F821 follow-up parked). |
| 2026-10-20 | `34818a153` / `f4f817d11` | **WP-3J** Shardbreaker + **KX-7Q** ratchet/cli split recorded (no stamp). |
| 2026-10-20 | `59b9c4533` | **KX-7Q-1** line-budget green; readiness doc updated. |
