# Production readiness status (QA-0K integrator)

**Measured at:** `origin/main` `6f2bbfedb` (2026-10-20 UTC). Prior integrator measure: `ba9537459`.  
**Work-order base:** `cb4f2c52f` is an ancestor of `main`.  
**Owner:** QA-0K. Do not call the repo production-ready until every exit line is **green** on a completed CI run of `main`, or **parked** below with an owner action.

**Landed sibling work (on `main`):**

| Lane | Tip SHA | Note |
|---|---|---|
| RL-4C | `de448576d` | `release.yml` preflight requires green CI + evidence-audit; `persist-credentials: false`; `AGENTS.md` / `Makefile` ci targets aligned with `ci.yml`. |
| VT-2M | `b5f1f6d11`, `0febe4d69`, `ead2bd4f9` | Replay lane `HOME`/`TMP`/`GIT_CEILING_DIRECTORIES` isolation; `mcs_vol` producer shrink + carrier re-pin (`claims_equal: false`, `lane_evolution`; `receipts/mcs_vol_drill.json` unchanged); CI uploads replay / verify-repo / tamper / fuzz artifacts. Migration: `quality/evidence_migrations/evidence_migration_20261019.json` (`mcs_vol_smaller_inputs`). |

**mcs_vol epoch stamp:** `needs-stamp: mcs_vol` on `b5f1f6d11` was satisfied by integrator commit `d0b9d45cd` (`scripts/*` corpus-epoch through `b5f1f6d11`, including `scripts/mcs_vol_drill.py`). This pass re-checked at `6f2bbfedb`: `mcs_vol_drill.py` is not in `head_member_digest_drift` for `scripts/*`. The replay carrier `data/manifests/replay/mcs_vol.json` is not yet in `quality/epoch_heads.json` (see `docs/PROPOSED_EVIDENCE_CHAINS.md` / [`EVIDENCE_MANIFESTS_PACKAGING_CHAIN.md`](EVIDENCE_MANIFESTS_PACKAGING_CHAIN.md)). **No restamp.**

Supporting lists: [`D1_02_EPOCH_ACK_INVENTORY.md`](D1_02_EPOCH_ACK_INVENTORY.md), [`UNGATED_EPOCH_ACKS_84d3d31b7.md`](UNGATED_EPOCH_ACKS_84d3d31b7.md), [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md).

## Exit criteria

1. **RED (in progress / cancelled)** — Full CI on `main` with no cancelled/failed jobs. Command: `gh run list --repo artificial-hedge/dipcatcher --branch main --workflow CI --limit 20`. Last **partial** run for RL-4C tip: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964 (`de448576d`; **lint** and **audit** failed; **evidence-audit** failed; replay upload path exercised — not all-green). Newer pushes cancel mid-queue (`39400490233`, `39401015545`). Last **completed** CI at earlier measure: https://github.com/artificial-hedge/dipcatcher/actions/runs/39257948317 (**failure**).
2. **NOT RUN this pass** — Local full gate (sibling dirty trees). Command: `make lint && make typecheck && uv run mypy src/fx1`. `uv run python scripts/check_mypy_strict_allowlist.py` reported passing on parallel `cursor/prod-ready-0343` work; `main` lint still carries McCabe ratchet debt. Last CI **lint** failure: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964.
3. **RED** — `verify_repo` at `6f2bbfedb` (Python API; lab CLI: `uv run quant verify-repo` when importable): `crown_jewels` **FAIL** (26 `jewel_mutated` / `jewel_unpinned`); `pin_signatures` **FAIL** `pin_drift:quality/crown_jewels.json,pin_drift:quality/epoch_heads.json` (expected until H1). Multiple `epoch:*` chains **FAIL** (forks/multiple heads on `src`/`tests`/`docs`; digest drift on `scripts`, `quality`, workflows). `epoch:scripts` drift does **not** include `mcs_vol_drill.py`. CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964.
4. **RED until H1** — `make evidence-audit` (seven trailing gates). Last observed **evidence-audit** failure: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964.
5. **PARTIAL / inventory** — `make replay-sweep`. 16 carriers under `data/manifests/replay/`; 14 need gitignored tape. `mcs_vol` executes on CI (synthetic). CI replay sweep + uploads: https://github.com/artificial-hedge/dipcatcher/actions/runs/39398901964 (not paired with full green CI).
6. **GREEN vs `b5f1f6d11` base (limited)** — `EPOCH_BASE=b5f1f6d11 make epoch-consistency` → exit 0. `ADMISSION_BASE=b5f1f6d11 make admission-gate` → exit 0 (`no corpus receipt changes vs b5f1f6d11`). **No all-jobs-green CI SHA** for production admission. Last completed full CI failure-heavy: https://github.com/artificial-hedge/dipcatcher/actions/runs/39247868051.
7. **GREEN vs `cb4f2c52f`** — No integrator-driven pytest skip / coverage floor loosening since work-order base.
8. **THIS FILE** — Exit lines 1–5 red or partial; line 6 narrowly green; line 7 green; H1–H7 parked. **Not production-ready.**

## Human-only parked items

Agents do not generate `GATE_SIGNING_KEY`, do not run `make sign-pins`, do not grandfather receipts, and do not point another vendor at the fx1 chat id.

1. **H1 Re-sign pins (D1-03/D2-02/D3-04).** Ed25519 `key_id` `bc3bcff676056adf`. Only the holder of `GATE_SIGNING_KEY` runs `make sign-pins`, then checkpoint / witness / anchor steps. Until then `pin_signatures` and dependent gates stay red.
2. **H2 Branch protection and required checks (D1-01/D2-10).** Repo admin must require CI status checks / merge queue. `gh api` branch protection may return 403 for agents.
3. **H3 Epoch acks (D1-02).** **13** self-excused acks at `84d3d31b7` plus **39** ungated entries in [`UNGATED_EPOCH_ACKS_84d3d31b7.md`](UNGATED_EPOCH_ACKS_84d3d31b7.md). Owner decides remediate vs grandfather. Agents only list.
4. **H4 Private tapes and Windows fleet (D1-06, D5-11).** Tape-backed replay carriers and `D:\dipcatcher` fleet host not verified in cloud integrator passes.
5. **H5 fx1 model endpoint (D5-01/D5-02).** `fx1.__version__` `0.4.0`; no production weights endpoint; do not mislabel third-party chat as fx1.
6. **H6 Release tag and versioning (D4-06/D5-06).** Release asset `v0.4.0` / Homebrew `sha256` `FILL_AT_RELEASE` — owner tags and publishes (`gh api …/releases/tags/v0.4.0` → 404).
7. **H7 Admission honesty for stamped history (D1-08).** Work-order bar **176 / 346** stamped receipts failing honesty admission (owner triage). Detail: [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md). Agents do not grandfather.

## Integrator log

| When (UTC) | SHA | Note |
|---|---|---|
| 2026-10-19 | `d0b9d45cd` | Stamp `scripts`/`src`/`tests`/`docs`/`workflows`/`quality` through `b5f1f6d11` (includes **mcs_vol** producer). |
| 2026-10-19 | `fb801e6c4` | Concurrent stamp: `src`/`tests`/`docs`/`quality` (epoch fork pressure). |
| 2026-10-20 | `cursor/prod-ready-0343` | Parallel agent: torch stubs, ruff UP017, manifests chain proposal doc. |
| 2026-10-20 | `6f2bbfedb` | QA-0K integrator: **mcs_vol already stamped**; readiness refresh; RL-4C + VT-2M recorded. Open `needs-stamp:` on `b34527e55` for ungated/proposed docs — **not stamped** per do-not-stamp guidance. |
