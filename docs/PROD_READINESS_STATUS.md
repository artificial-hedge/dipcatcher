# Production readiness status (QA-0K integrator)

**Measured at:** `origin/main` `b5f1f6d11` (2026-10-19 UTC), then this integrator stamp of drifted chains plus the D1 inventories.  
**Work-order base:** `cb4f2c52f` is an ancestor of `main`.  
**Owner:** QA-0K. Do not call the repo production-ready until every exit line is **green** on a completed CI run of `main`, or **parked** below with an owner action.

Supporting lists: [`D1_02_EPOCH_ACK_INVENTORY.md`](D1_02_EPOCH_ACK_INVENTORY.md), [`UNGATED_EPOCH_ACKS_84d3d31b7.md`](UNGATED_EPOCH_ACKS_84d3d31b7.md), [`D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`](D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md).

## Exit criteria

1. **RED** — complete CI on `main` with no cancelled/failed jobs (lint, typecheck, all test shards, coverage, evidence-audit, proofcore, fx1, arch-guards, platform matrix). Command: `gh run list --branch main --workflow CI --limit 50`. Last **completed** CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510 (`5c474b6d3`, **failure**; jobs: lint success; evidence-audit failure; coverage failure; pretrade-risk failure; test (3.12, 3)/(3.13, 3)/(3.13, 4) failure; five test shards cancelled; 17 other jobs success). Pushes since then cancel mid-queue. Newest listing at measure time: pending CI https://github.com/artificial-hedge/dipcatcher/actions/runs/39203555298 (`b5f1f6d11`).
2. **NOT RUN this pass** (4-vCPU box, sibling agents live; do not run full pytest). Command: `make lint && make typecheck && uv run mypy src/fx1`. Last CI lint job **success** on https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510. Earlier local `make lint` on `cf6b3613b` failed `check_mypy_strict_allowlist.py` (`hedge_lab/resources.py`, `models/deep_rl.py`, `models/nbeats.py`, `utils/seeds.py`).
3. **RED until H1** — `uv run dipcatcher crown-jewels --check` at `b5f1f6d11` exited 1 (`jewel_mutated:AGENTS.md`, `jewel_mutated:Makefile`); this stamp re-pins those jewels. `uv run dipcatcher verify-repo` is allowed to keep `pin_signatures: FAIL pin_drift:quality/crown_jewels.json,pin_drift:quality/epoch_heads.json` until H1. Other verify-repo gates must print. Last evidence-audit: https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510.
4. **RED until H1** — `make evidence-audit` must show all 7 trailing gates (`crown-jewels`, `verify-witness`, `verify-repo`, `checkpoint-chain`, `verify-rotations`, `tamper-drill`, `fuzz-drill`). Last completed evidence-audit **failed** on pin_drift in https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510. Tamper/fuzz 0-escape calibration is not a green signal until pins are re-signed.
5. **INVENTORY, sweep not re-run this pass** — `make replay-sweep` (`dipcatcher replay-all --strict --timeout 300`). 16 carriers under `data/manifests/replay/`. `data/file_us_wide/bronze/bars.parquet` is missing (gitignored). 14 carriers bind that tape (`calib`, `conformal`, `coverage`, `coverage_cs`, `cp`, `drift`, `loss_cs`, `mcs`, `monitor`, `panel_audit`, `race`, `serial`, `tail`, `verdict`). `corpus.json` pins in-repo receipts. `mcs_vol.json` has no `input_tapes`. Skips allowed only for private tapes; executed vs skipped must be reported separately on the next completed evidence-audit replay artifact. CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510.
6. **NO LAST GREEN BASE** — `make admission-gate ADMISSION_BASE=<pre-push tip>` and `make epoch-consistency EPOCH_BASE=<pre-push tip>` run against this stamp's parent (`b5f1f6d11`), not against a green CI SHA. There is no completed all-jobs-green CI SHA. Last completed CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510.
7. **GREEN vs `cb4f2c52f`** — `git grep -E 'pytest.mark.(skip|xfail)|pytest\.skip\(|pytest\.xfail\(' cb4f2c52f -- tests | wc -l` → 173; same grep at `b5f1f6d11` → 173. `git diff cb4f2c52f..b5f1f6d11 -- tests pyproject.toml Makefile | rg 'pytest.mark.(skip|xfail)|fail_under|deselect'` → empty. `pyproject.toml` adds `pytest-timeout` / `timeout = 900` (not a skip). No integrator-driven baseline loosening.
8. **THIS FILE** — lines 1–6 are red, not-run, or parked; line 7 is green vs `cb4f2c52f`; human items H1–H7 are parked with owner actions. Not production-ready.

## Human-only parked items

Agents do not generate `GATE_SIGNING_KEY`, do not run `make sign-pins`, do not grandfather receipts, and do not point another vendor at the fx1 chat id.

1. **H1 Re-sign pins (D1-03/D2-02/D3-04).** `gate_pins.sig` last changed in `93c257586` on 2026-09-30. Ed25519 `key_id` `bc3bcff676056adf`. Only the holder of `GATE_SIGNING_KEY` runs `make sign-pins`, then `make checkpoint`, `make witness-checkpoint`, then anchor steps. Until then `verify-repo` / `evidence-audit` / tamper-drill / fuzz-drill and the pin-drift unit tests stay red. Never mint a new key.
2. **H2 Branch protection and required checks (D1-01/D2-10).** `gh api repos/artificial-hedge/dipcatcher/branches/main/protection` → HTTP 403. Rulesets: `24646693` creation (excludes `main`/`isolated`/`others`), `24648163` deletion (includes those refs). Neither requires status checks or a merge queue. Repo admin must set required checks / merge queue / revert bot. Until then CI is advisory.
3. **H3 Epoch acks (D1-02).** Live ledger 46 acks (13 mutated, 33 removed). Genesis 5 at `02fbf8612`. `84d3d31b7` added 13 (12 in-place lane mutations + 1 quality-fork removal). 39 entries after the genesis 5 (44−5 at that inventory; two later `tests/*` removals in `43602dd9d` make 41 post-genesis). Owner decides remediate vs grandfather. Lists: `docs/D1_02_EPOCH_ACK_INVENTORY.md`, `docs/UNGATED_EPOCH_ACKS_84d3d31b7.md`. Agents only list.
4. **H4 Private tapes and Windows fleet (D1-06, D5-11).** 14/16 replay carriers need `data/file_us_wide/bronze/bars.parquet`. NTFS/ACL needs `D:\dipcatcher`. Not verified here.
5. **H5 fx1 model endpoint (D5-01/D5-02).** `fx1.__version__` is `0.4.0`. Default hosted URL is `FX1_BASE_URL` or Moonshot. Do not point another vendor at chat ids `fx1` / `fx1-lite` and call it fx1. No production weights endpoint.
6. **H6 Release tag and versioning (D4-06/D5-06).** `fx1.__version__` `0.4.0`. `install.sh` and `packaging/homebrew/fxi.rb` use `https://github.com/artificial-hedge/dipcatcher/releases/download/v0.4.0/fx_1-0.4.0-py3-none-any.whl`. `gh api repos/artificial-hedge/dipcatcher/releases/tags/v0.4.0` → HTTP 404. Homebrew `sha256` is `FILL_AT_RELEASE`. Owner tags and publishes the asset.
7. **H7 Admission honesty for stamped history (D1-08).** Work-order 176/346/6 is stale. Live: **170 of 338** stamped non-legacy claim receipts fail `honesty_stamp_errors`; 0 reject-class; 1 registry `admission_honesty_gap` (`p0_evidence_inventory_v1.json`). Owner decides register / re-admit / grandfather. This pass does not grandfather. List: `docs/D1_08_STAMPED_RECEIPTS_FAILING_ADMISSION.md`.

## Integrator log

| When (UTC) | SHA | Note |
|---|---|---|
| 2026-10-19 | `219f6c4bb` | Integrator stamp through `6d15a409c` (workflows/configs/docs/scripts/src/tests/typings/quality). |
| 2026-10-19 | `65080a71e` | ZR-9V stamp left some pin receipts uncommitted; repaired later. |
| 2026-10-19 | `43602dd9d` / `8249426e0` | Epoch-head repair + omitted workflows/configs receipts committed. |
| 2026-10-19 | `b5f1f6d11` | Pre-stamp tip for this pass (replay isolation). Jewels `AGENTS.md`+`Makefile` mutated; src/tests/scripts/docs/workflows/quality live-drifted. |
