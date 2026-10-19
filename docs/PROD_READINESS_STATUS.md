# Production readiness status

Commands below were run in `/tmp/wt-qa0k` at `a66ea98952eef0e1667ef04390345e486d79590e`. `cb4f2c52f` is an ancestor of that commit. `origin/main` was still moving while this file was written; a later tip was not re-measured.

## Exit criteria

1. red — A complete CI run on `main` with no cancelled or failed jobs, covering lint, typecheck, all test shards, coverage, evidence-audit, proofcore, fx1, arch-guards, and the platform matrix. `gh run list --commit a66ea98952eef0e1667ef04390345e486d79590e` shows that push cancelled: CI https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153973, proofcore https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153966, fx1 https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153976, arch-guards https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153972, Platform Matrix https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153952. The newest CI row at listing time was still pending on a later push: https://github.com/artificial-hedge/dipcatcher/actions/runs/39202224076. The last completed CI run in `gh run list --workflow CI --limit 25` that was not cancelled is failure https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510 (`5c474b6d30af63292273413d72120c38ace20d5f`); failed jobs were pretrade-risk, test (3.13, 4), test (3.13, 3), test (3.12, 3), evidence-audit, and coverage, with other test shards cancelled.
2. not-run — `make lint && make typecheck && uv run mypy src/fx1` was not executed this pass. Latest CI URL from the list above: https://github.com/artificial-hedge/dipcatcher/actions/runs/39202224076. Main's ledger, not re-run here, records `make lint` failing on `cf6b3613b` in `check_mypy_strict_allowlist.py` (`hedge_lab/resources.py`, `models/deep_rl.py`, `models/nbeats.py`, `utils/seeds.py`) and records KX-7Q strict mypy for `tiny_lm` without the `nn` extra at `6d15a409c` (`3e603bd03`, after this pass's measurement SHA).
3. red at `a66ea9895`, before later stamps — `uv run --offline --frozen dipcatcher crown-jewels --check` at `989705c44` exited 1 before the checker: uv could not download `pytest-timeout==2.4.0` (network disabled, wheel not in the cache). The checker was then run as `PYTHONPATH=/tmp/wt-qa0k/src /workspace/.venv/bin/dipcatcher crown-jewels --check` at `a66ea98952eef0e1667ef04390345e486d79590e` and exited 1: `jewel_mutated` on `.env.example`, `.gitignore`, `AGENTS.md`, `Makefile`, `README.md`, `pyproject.toml`, `src/fx1/harness.py`, `src/fx1/interactive/concierge.py`, `src/fx1/interactive/superpower.py`, `tests/conftest.py`, `uv.lock`. `PYTHONPATH=/tmp/wt-qa0k/src /workspace/.venv/bin/dipcatcher verify-repo` at the same SHA exited 1. `pin_signatures` is `FAIL pin_drift:quality/crown_jewels.json,pin_drift:quality/epoch_heads.json`. Also red: `crown_jewels` (the same eleven files); `epoch:quality` (`ratchet_migrations.json`); `epoch:.github/workflows` (`ci.yml`, `proofcore.yml`, `replay_viz.yml`, `simtest.yml`, `web_explorer.yml`, `witness_monitor.yml`); `epoch:configs` (`arch_boundaries.toml`); `epoch:src` (digest drift plus unstamped `dipcatcher_cli/paths.py`, `dipcatcher_cli/truthfulness.py`, `fx1/cli/data_cmds.py`, `fx1/cli/eval_cmds.py`, `fx1/cli_common.py`, `fx1/cli_harness.py`, `quant_fund/utils/forbidden_research_keys.py`); `epoch:tests` (digest drift, `head_member_missing_live:formal/_test_sm_probe.py`, and unstamped `formal/conftest.py`, `fx1/test_honesty_forbidden_keys_mirror.py`, `unit/frontdoor/test_pd8s_frontdoor.py`, `unit/fx1_serve/conftest.py`, `unit/test_test_env_isolation.py`); `epoch:scripts` (`fx1_goldenpath.py`); `epoch:docs` (`OPERATIONS_RUNBOOK.md` drift, unstamped `PROD_READINESS_STATUS.md`). Printed ok: timestamp_anchors, checkpoint, witness, spine, key_rotation, quorum_rotations, lattice, and the receipts, verifier, artifacts, `.dsh-24x7`, and `data/metadata` epochs. Log: `/opt/cursor/artifacts/qa-0k/verify-repo.txt`. CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153973. Newer result already on `main`, not re-run here: `43602dd9` records `crown-jewels --check` intact after restoring `epoch_heads` from `219f6c4bb`, re-stamping drift through `3e603bd03`, and acknowledging removals of `formal/_test_sm_probe.py` and `test_hf_rv_scaffolding.py`. That note still has `pin_signatures` `FAIL pin_drift` until human item 1.
4. not-run — `make evidence-audit` was not run. The target walks every corpus-epoch chain before the seven trailing gates (`crown-jewels`, `verify-witness`, `verify-repo`, `checkpoint-chain`, `verify-rotations`, `tamper-drill`, `fuzz-drill`), so it does not finish in a few minutes. This pass did not see whether all seven gates print. The last completed evidence-audit job failed in https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510. Newer note already on `main`, not re-run here: `43602dd9` records ZR-9V unmasked the seven gates (`ea_fail` at the end) and that evidence-audit stays red on inherited `pin_drift` until human item 1. The same ledger also records, at `06cf40e73`, anchor staleness as `stale` plus `age_s` in `verify-repo` and `repo_integrity.json`, and a local `test_replay_sweep` plus `test_timestamp_anchor` run of 16/16. This pass did not re-run those.
5. not-run — `make replay-sweep` was not run this pass. CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153973. Main's ledger records a local `make replay-sweep` pass of about 91–109s after `0febe4d69` (VT-2M). This pass did not re-run that command.
6. not-run — `make admission-gate ADMISSION_BASE=<last green>` and `make epoch-consistency EPOCH_BASE=<last green>` were not run. No completed green CI SHA was available to use as the base. CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39157190510.
7. green — `git diff cb4f2c52f..a66ea98952eef0e1667ef04390345e486d79590e -- quality/ pyproject.toml tests | grep -E 'skip|xfail|baseline|deselect|fail_under'` returned 41 lines. Reviewed: migration prose in `quality/evidence_migrations/evidence_migration_20261018.json` mentions skipped replay carriers; repeated epoch receipts name existing files `perf/baselines/baseline.json`, `unit/core/test_perf_baseline.py`, and `unit/research/test_skip_intervals_flag.py`; `tests/fx1/test_audit_fixes.py` and `tests/fx1/test_corpus.py` still assert corpus-load counters that include a `skipped` field; `tests/unit/research/test_replay_proof.py` comments describe an honest skip; `tests/unit/test_test_env_isolation.py` adds `test_guarded_env_matches_session_baseline_after_collection`. No new pytest `skip`, `xfail`, `deselect`, or `fail_under`. `pyproject.toml` in this range adds `pytest-timeout` and `timeout = 900`, which the grep does not match and which is not a deselection. `quality/mccabe_baseline.txt` moves `harness_message` 14 and `harness_respond` 17 onto `src/fx1/cli_harness.py` without raising those numbers. CI: https://github.com/artificial-hedge/dipcatcher/actions/runs/39202153973. Newer note already on `main`, not re-run here: `08164a895` records WP-3J done at `e0cd82717` (required-only oracle/CLI plus slow twins, `pytest-timeout` 900s, `.test_durations` retime and a `refresh-test-durations` CI job; `process_map` spawn/fork not changed). That note says `.test_durations` still needs a stamp.
8. green — `docs/PROD_READINESS_STATUS.md` records the commands and outputs of this pass at `a66ea98952eef0e1667ef04390345e486d79590e`. Lines 1 and 3 are red. Lines 2, 4, 5, and 6 are not-run. Line 7 is green.

## Human-only items

Parked. No signing key was generated, no epoch ack was reverted, and no receipt was rewritten.

1. Re-sign the pins. `gate_pins.sig` was last changed in `93c257586bb3e8c4384344e4fb07eb661ab0f26f` on 2026-09-30. The file is Ed25519 `gate_signatures.v2` and the signature `key_id` is `bc3bcff676056adf`. Only the holder of `GATE_SIGNING_KEY` can run `make sign-pins`, then `make checkpoint`, `make witness-checkpoint`, and the anchor steps. Until then `pin_signatures` stays red. A re-sign does not by itself clear the `jewel_mutated` and unstamped epoch drift in exit criterion 3. Do not generate a new key.
2. Branch protection and required checks. `gh api repos/artificial-hedge/dipcatcher/branches/main/protection` returned HTTP 403, so classic branch protection was not read. `gh api repos/artificial-hedge/dipcatcher/rulesets` returned two active rulesets and neither requires status checks or a merge queue: `Only main isolated others` (id `24646693`, rule `creation`, excludes `refs/heads/main`, `refs/heads/isolated`, `refs/heads/others`) and `Retain main isolated others` (id `24648163`, rule `deletion`, includes those three refs). A repo admin must set a ruleset with required checks plus a merge queue, or a post-push revert bot. Until then CI is advisory.
3. Epoch acks. Do not revert. `quality/epoch_allowed_removals.json` has 44 entries at this SHA. The ledger contains 13 `receipts/*.json` `mutated` acks. Twelve of them are six lane receipts recorded on two epochs, all introduced in `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae`. The thirteenth is `corpus_real_drill.json`, introduced in `02fbf86122b58aaa9d4cc566e42d0460e84a00a1`. A parent-blob check shows only the six `corpus_epoch_6f6d75cd3751c48d.json` rows have `from_sha256` equal to the parent blob; the six `corpus_epoch_05bbdbfc2bab9f19.json` rows and `corpus_real_drill.json` do not.
- `corpus_real_drill.json` epoch `corpus_epoch_2bd794275af2b3d8.json` introducing `02fbf86122b58aaa9d4cc566e42d0460e84a00a1` from `082f16234bae90433c9f699d575a57b2ef5f1bf826fdb61ccba838e936f9f119` to `25230923d07164342facbe10ec0520975ee04c0ef690b3661f5daaf9d69f250e`
- `capacity_eval_3622d0c059d42009.json` epoch `corpus_epoch_05bbdbfc2bab9f19.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `57ad3394513fe58e48127141858a990580fade56ae4755ad0a3b868aa2568f2b` to `4f9f7af188507dc602ae782b2743ee62a992f5d99eede4b4478e8101493df128`
- `fleet_eval_5e907be710811a44.json` epoch `corpus_epoch_05bbdbfc2bab9f19.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `c95ce361609bdb00912147e0f87548fdbd2c0221ad27b8c1a4d0ba558b9330e1` to `b5444f12bf0e774c547dc6720517453bce98aa6c2ac2dbb84731bc0f6ad664af`
- `honest_verdict_b133e8b3992af893.json` epoch `corpus_epoch_05bbdbfc2bab9f19.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `1a003a670945452a5c86d033900bc33429f2981422c8938b1b8e2469fc764f3a` to `70c391cb69796d05353b9c03c53bf8aa05c14465f919be9d790b08aa0c265ce7`
- `monitor_run_13d20aef00774027.json` epoch `corpus_epoch_05bbdbfc2bab9f19.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `3ce1cd7f59c301be6b4710326ab905313725317bee3669ae8e755f6d422bc12d` to `2d804ee35deee30e8e14a35d3c2361bbfe6e0482967fdf8e88a2a322132f2f03`
- `rankic_eval_8d740c0ddf6d5c54.json` epoch `corpus_epoch_05bbdbfc2bab9f19.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `e48fce1ae4cebf94e417773bdef20c9ed7b0822397505c6a51e8843a8da06c81` to `bb6859f2b8f4d18dcf90fd388bc2f804c911210fdefafe9d58466154a67636a6`
- `vol_bench_f3bf2afc2f159ace.json` epoch `corpus_epoch_05bbdbfc2bab9f19.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `b9b70f7186d75054e91ee734b87b8aeeba22d66af713806a8d5410a9bc603ee3` to `6bbb8b5ec1745ef5dcd186c5037211374c29dcd3078214e5b86a4a4682740241`
- `capacity_eval_3622d0c059d42009.json` epoch `corpus_epoch_6f6d75cd3751c48d.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `4f9f7af188507dc602ae782b2743ee62a992f5d99eede4b4478e8101493df128` to `b867f73565dd828e6ae2c2f7091b04915126622b11e5ae7854e7675cce9b048f`
- `fleet_eval_5e907be710811a44.json` epoch `corpus_epoch_6f6d75cd3751c48d.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `b5444f12bf0e774c547dc6720517453bce98aa6c2ac2dbb84731bc0f6ad664af` to `adbe15a247e23fd26cc874a509d3448881536aef586be335bbad7b44ba97812f`
- `honest_verdict_b133e8b3992af893.json` epoch `corpus_epoch_6f6d75cd3751c48d.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `70c391cb69796d05353b9c03c53bf8aa05c14465f919be9d790b08aa0c265ce7` to `ed1736ed83c33d0c44775292c3e4979aaf11bb5bd2d18cfeb7df3baa78575a84`
- `monitor_run_13d20aef00774027.json` epoch `corpus_epoch_6f6d75cd3751c48d.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `2d804ee35deee30e8e14a35d3c2361bbfe6e0482967fdf8e88a2a322132f2f03` to `82c474e25984e9418b6709f81dd34e55457c512c6eef7d3b3275c249e2e82b38`
- `rankic_eval_8d740c0ddf6d5c54.json` epoch `corpus_epoch_6f6d75cd3751c48d.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `bb6859f2b8f4d18dcf90fd388bc2f804c911210fdefafe9d58466154a67636a6` to `3f1aba6b04d9b7757852334f213059ef971bffe5100e2bbda807eeba28014f06`
- `vol_bench_f3bf2afc2f159ace.json` epoch `corpus_epoch_6f6d75cd3751c48d.json` introducing `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` from `6bbb8b5ec1745ef5dcd186c5037211374c29dcd3078214e5b86a4a4682740241` to `dfca0a6263a360b3b89cb8588bbb3e03224f37e069cf2764f4caa24b6a2cd453`
A separate filter, "named epoch file is absent from the parent of the commit that introduced the ack", returns 23 acks, not 13:
- `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` `receipts/*.json` `mutated` `capacity_eval_3622d0c059d42009.json` @ `corpus_epoch_6f6d75cd3751c48d.json`
- `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` `receipts/*.json` `mutated` `fleet_eval_5e907be710811a44.json` @ `corpus_epoch_6f6d75cd3751c48d.json`
- `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` `receipts/*.json` `mutated` `honest_verdict_b133e8b3992af893.json` @ `corpus_epoch_6f6d75cd3751c48d.json`
- `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` `receipts/*.json` `mutated` `monitor_run_13d20aef00774027.json` @ `corpus_epoch_6f6d75cd3751c48d.json`
- `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` `receipts/*.json` `mutated` `rankic_eval_8d740c0ddf6d5c54.json` @ `corpus_epoch_6f6d75cd3751c48d.json`
- `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` `receipts/*.json` `mutated` `vol_bench_f3bf2afc2f159ace.json` @ `corpus_epoch_6f6d75cd3751c48d.json`
- `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` `quality/*.json` `removed` `corpus_epoch_cd4b7ee4ad32d328.json` @ `corpus_epoch_792426f656af6eb2.json`
- `f327796b039d15a8b3415ca42449dff4cdd90dc3` `tests/*` `removed` `corpus_epoch_05724f90cfec84f6.json` @ `corpus_epoch_446ddadac2efb9d5.json`
- `adf8ba95234d73877a2f2a37e85cd09b9f30a13e` `data/metadata/*` `removed` `resolved_config.json` @ `corpus_epoch_e73c101a7ab5d51b.json`
- `1457ee3a5c1564d624eddc827739cb33ab121beb` `docs/*` `removed` `corpus_epoch_bf34205674030d20.json` @ `corpus_epoch_56ab5b59290569ce.json`
- `1457ee3a5c1564d624eddc827739cb33ab121beb` `docs/*` `removed` `corpus_epoch_6b3d85be9a477735.json` @ `corpus_epoch_56ab5b59290569ce.json`
- `2b72e0501897571117175c41916052b8aa078b95` `receipts/*.json` `removed` `capacity_eval_3622d0c059d42009.json` @ `corpus_epoch_1d1a6cfdadf37c0c.json`
- `2b72e0501897571117175c41916052b8aa078b95` `receipts/*.json` `removed` `fleet_eval_5e907be710811a44.json` @ `corpus_epoch_1d1a6cfdadf37c0c.json`
- `2b72e0501897571117175c41916052b8aa078b95` `receipts/*.json` `removed` `honest_verdict_b133e8b3992af893.json` @ `corpus_epoch_1d1a6cfdadf37c0c.json`
- `2b72e0501897571117175c41916052b8aa078b95` `receipts/*.json` `removed` `monitor_run_13d20aef00774027.json` @ `corpus_epoch_1d1a6cfdadf37c0c.json`
- `2b72e0501897571117175c41916052b8aa078b95` `receipts/*.json` `removed` `rankic_eval_8d740c0ddf6d5c54.json` @ `corpus_epoch_1d1a6cfdadf37c0c.json`
- `2b72e0501897571117175c41916052b8aa078b95` `receipts/*.json` `removed` `vol_bench_f3bf2afc2f159ace.json` @ `corpus_epoch_1d1a6cfdadf37c0c.json`
- `021b79fec5624774a5debc833c35fc59da18d5e0` `web/*` `removed` `public/fixtures/receipts/capacity_eval_3622d0c059d42009.json` @ `corpus_epoch_12bce57f7c47989b.json`
- `021b79fec5624774a5debc833c35fc59da18d5e0` `web/*` `removed` `public/fixtures/receipts/fleet_eval_5e907be710811a44.json` @ `corpus_epoch_12bce57f7c47989b.json`
- `021b79fec5624774a5debc833c35fc59da18d5e0` `web/*` `removed` `public/fixtures/receipts/honest_verdict_b133e8b3992af893.json` @ `corpus_epoch_12bce57f7c47989b.json`
- `021b79fec5624774a5debc833c35fc59da18d5e0` `web/*` `removed` `public/fixtures/receipts/monitor_run_13d20aef00774027.json` @ `corpus_epoch_12bce57f7c47989b.json`
- `021b79fec5624774a5debc833c35fc59da18d5e0` `web/*` `removed` `public/fixtures/receipts/rankic_eval_8d740c0ddf6d5c54.json` @ `corpus_epoch_12bce57f7c47989b.json`
- `021b79fec5624774a5debc833c35fc59da18d5e0` `web/*` `removed` `public/fixtures/receipts/vol_bench_f3bf2afc2f159ace.json` @ `corpus_epoch_12bce57f7c47989b.json`
refuted: the count of 39 acks added in `84d3d31b7`. `git` history of `quality/epoch_allowed_removals.json` shows that commit introduced 13 ledger entries, not 39. `docs/UNGATED_EPOCH_ACKS_84d3d31b7.md` is not on this commit. ZR-9V's list belongs in that forthcoming doc. The 13 entries this pass identified in `84d3d31b7cc33bfda58e739f5da6291f3afcc2ae` are:
- `receipts/*.json` `mutated` `capacity_eval_3622d0c059d42009.json` @ `corpus_epoch_05bbdbfc2bab9f19.json` from `57ad3394513fe58e48127141858a990580fade56ae4755ad0a3b868aa2568f2b` to `4f9f7af188507dc602ae782b2743ee62a992f5d99eede4b4478e8101493df128`
- `receipts/*.json` `mutated` `fleet_eval_5e907be710811a44.json` @ `corpus_epoch_05bbdbfc2bab9f19.json` from `c95ce361609bdb00912147e0f87548fdbd2c0221ad27b8c1a4d0ba558b9330e1` to `b5444f12bf0e774c547dc6720517453bce98aa6c2ac2dbb84731bc0f6ad664af`
- `receipts/*.json` `mutated` `honest_verdict_b133e8b3992af893.json` @ `corpus_epoch_05bbdbfc2bab9f19.json` from `1a003a670945452a5c86d033900bc33429f2981422c8938b1b8e2469fc764f3a` to `70c391cb69796d05353b9c03c53bf8aa05c14465f919be9d790b08aa0c265ce7`
- `receipts/*.json` `mutated` `monitor_run_13d20aef00774027.json` @ `corpus_epoch_05bbdbfc2bab9f19.json` from `3ce1cd7f59c301be6b4710326ab905313725317bee3669ae8e755f6d422bc12d` to `2d804ee35deee30e8e14a35d3c2361bbfe6e0482967fdf8e88a2a322132f2f03`
- `receipts/*.json` `mutated` `rankic_eval_8d740c0ddf6d5c54.json` @ `corpus_epoch_05bbdbfc2bab9f19.json` from `e48fce1ae4cebf94e417773bdef20c9ed7b0822397505c6a51e8843a8da06c81` to `bb6859f2b8f4d18dcf90fd388bc2f804c911210fdefafe9d58466154a67636a6`
- `receipts/*.json` `mutated` `vol_bench_f3bf2afc2f159ace.json` @ `corpus_epoch_05bbdbfc2bab9f19.json` from `b9b70f7186d75054e91ee734b87b8aeeba22d66af713806a8d5410a9bc603ee3` to `6bbb8b5ec1745ef5dcd186c5037211374c29dcd3078214e5b86a4a4682740241`
- `receipts/*.json` `mutated` `capacity_eval_3622d0c059d42009.json` @ `corpus_epoch_6f6d75cd3751c48d.json` from `4f9f7af188507dc602ae782b2743ee62a992f5d99eede4b4478e8101493df128` to `b867f73565dd828e6ae2c2f7091b04915126622b11e5ae7854e7675cce9b048f`
- `receipts/*.json` `mutated` `fleet_eval_5e907be710811a44.json` @ `corpus_epoch_6f6d75cd3751c48d.json` from `b5444f12bf0e774c547dc6720517453bce98aa6c2ac2dbb84731bc0f6ad664af` to `adbe15a247e23fd26cc874a509d3448881536aef586be335bbad7b44ba97812f`
- `receipts/*.json` `mutated` `honest_verdict_b133e8b3992af893.json` @ `corpus_epoch_6f6d75cd3751c48d.json` from `70c391cb69796d05353b9c03c53bf8aa05c14465f919be9d790b08aa0c265ce7` to `ed1736ed83c33d0c44775292c3e4979aaf11bb5bd2d18cfeb7df3baa78575a84`
- `receipts/*.json` `mutated` `monitor_run_13d20aef00774027.json` @ `corpus_epoch_6f6d75cd3751c48d.json` from `2d804ee35deee30e8e14a35d3c2361bbfe6e0482967fdf8e88a2a322132f2f03` to `82c474e25984e9418b6709f81dd34e55457c512c6eef7d3b3275c249e2e82b38`
- `receipts/*.json` `mutated` `rankic_eval_8d740c0ddf6d5c54.json` @ `corpus_epoch_6f6d75cd3751c48d.json` from `bb6859f2b8f4d18dcf90fd388bc2f804c911210fdefafe9d58466154a67636a6` to `3f1aba6b04d9b7757852334f213059ef971bffe5100e2bbda807eeba28014f06`
- `receipts/*.json` `mutated` `vol_bench_f3bf2afc2f159ace.json` @ `corpus_epoch_6f6d75cd3751c48d.json` from `6bbb8b5ec1745ef5dcd186c5037211374c29dcd3078214e5b86a4a4682740241` to `dfca0a6263a360b3b89cb8588bbb3e03224f37e069cf2764f4caa24b6a2cd453`
- `quality/*.json` `removed` `corpus_epoch_cd4b7ee4ad32d328.json` @ `corpus_epoch_792426f656af6eb2.json` from `4a428c8f6834cbbd495e55115f50a398bfec7b3b6a1f04ba4061951288a60a7e` to `None`
The owner decides how to remediate the acks.
4. Private tapes and the Windows fleet host. A read of `data/manifests/replay/*.json` at this SHA finds 16 carriers. Fourteen resolve `replay.input_tapes` through `data/manifests/yahoo_eod_us_wide.json` to `data/file_us_wide/bronze/bars.parquet`, which is missing and gitignored: `calib.json`, `conformal.json`, `coverage.json`, `coverage_cs.json`, `cp.json`, `drift.json`, `loss_cs.json`, `mcs.json`, `monitor.json`, `panel_audit.json`, `race.json`, `serial.json`, `tail.json`, `verdict.json`. `corpus.json` pins receipt paths that are present. `mcs_vol.json` declares no `input_tapes`. This was not `make replay-sweep`. NTFS and ACL behaviour on `D:\dipcatcher` was not verified. Those carriers can only be verified on that host.
5. fx1 model endpoint. At this SHA `fx1.interactive.profiles.default_base_url` returns `FX1_BASE_URL` or `https://api.moonshot.ai/v1/chat/completions` (`MOONSHOT_API_URL`). `fx1.interactive.actions.chat` builds `HostedK3Backend` with `model` set to the chat id (`fx1` or `fx1-lite`). `fx1.__version__` is `0.4.0`. No endpoint was called. A valid endpoint and credentials are needed. Do not point another vendor at the fx1 chat id and call it fx1.
6. Release tag and versioning. `fx1.__version__` in `src/fx1/__init__.py` is `0.4.0` (also printed by the venv import). `install.sh` defaults `FXI_VERSION` to `0.4.0` and the wheel URL `https://github.com/artificial-hedge/dipcatcher/releases/download/v0.4.0/fx_1-0.4.0-py3-none-any.whl`. `packaging/homebrew/fxi.rb` uses that URL and `sha256 "FILL_AT_RELEASE"`. `gh release list --limit 5` printed no releases. `gh api repos/artificial-hedge/dipcatcher/releases/tags/v0.4.0` returned HTTP 404. The owner decides the version and the tag.
7. Admission honesty bar. refuted: `PYTHONPATH=/tmp/wt-qa0k/src /workspace/.venv/bin/python /opt/cursor/artifacts/qa-0k/investigate.py` (library calls `load_heads_pin`, `pinned_head_members`, `honesty_stamp_errors`, `load_quarantine`; no receipt writes). Re-run at `989705c44802ba0a1fb9f69ce751fa970845ff8f`; the commits through `a66ea98952eef0e1667ef04390345e486d79590e` do not touch `receipts/`, `quality/epoch_allowed_removals.json`, or `quality/legacy_quarantine.json`. It does not reproduce 176 of 346 lacking a stamp with 6 registered. Measured: receipts head `corpus_epoch_1d1a6cfdadf37c0c.json`; 403 stamped members; 7 under `legacy-unsealed/`; 396 non-legacy; 348 non-legacy whose names do not start with `corpus_epoch_`; 178 of those lack a quarantine-class honesty stamp (`live_pnl_claim_missing` 143, `research_only_missing` 10, both of those 8, `data_label_missing` 6, `research_only_missing` plus `data_label_missing` 6, all three 5); registry `admission_honesty_gap` count is 1 (`p0_evidence_inventory_v1.json`). Other registry categories: `tape_manifest_unpinned` 2, `detached_prereg_seal` 4, `contract_postdates_receipt` 2, `pre_seal_era` 7, `claim_absent` 1. The 178 members are:
- `abc_calibrate_amzn.json`
- `aftermath_flow_amzn.json`
- `anchor_scan_amzn.json`
- `api_fuzz.json`
- `band_occupancy.json`
- `band_shape.json`
- `cancel_cluster_amzn.json`
- `cancel_gradient_amzn.json`
- `cancel_gradient_bench.json`
- `cancel_lead_amzn.json`
- `capacity_eval_5cf0042e91c7953c.json`
- `capacity_eval_cd0854242ed8a9ec.json`
- `churn_reseed.json`
- `churn_stability.json`
- `closure_fit_amzn.json`
- `closure_stack_amzn.json`
- `coherence_2dd641ab766a536a.json`
- `concordance_df424fa2f6b1c4e9.json`
- `continuation_attr_amzn.json`
- `cost_aware_rerun_20261007.json`
- `cost_aware_rerun_20261007.json.seal.json`
- `coverage_cs_real_drill.json`
- `coverage_real_drill.json`
- `cp_real_drill_gaussian_minus_conf_t_pinball.json`
- `cp_real_drill_gaussian_pit.json`
- `crown_behind.json`
- `crown_density_amzn.json`
- `crown_join.json`
- `crown_size.json`
- `cxl_shield_amzn.json`
- `decay_watch_synth.json`
- `deep_book_bench.json`
- `deep_microprice_amzn.json`
- `depth_consumption_amzn.json`
- `depth_tilt_amzn.json`
- `empirical_flow_bench.json`
- `event_burst_amzn.json`
- `event_granger_amzn.json`
- `event_matrix_amzn.json`
- `evidence_audit_3464d8f8197bf737.json`
- `evidence_audit_5c9f07c823727701.json`
- `evidence_audit_d449e1ca0cc119a6.json`
- `evidence_audit_f45ae8d6bc7cd677.json`
- `exec_cost_real_amzn.json`
- `exec_cost_split.json`
- `fast_replay_p42_conformance_20260928.json`
- `fifo_priority_amzn.json`
- `flee_wide.json`
- `fleet_eval_5ddf15b0dc7d3ca1.json`
- `fleet_eval_ab613c423b36627d.json`
- `floor_compose.json`
- `floor_pins.json`
- `floor_rate.json`
- `floor_reseed.json`
- `floor_stability.json`
- `flow_couple.json`
- `forward_record_preregistration_v1.json`
- `forward_record_preregistration_v1.json.seal.json`
- `full_impact.json`
- `full_stack.json`
- `gap_close.json`
- `glft_bench_synth.json`
- `grammar_map.json`
- `hawkes_clock_bench.json`
- `hawkes_real_amzn.json`
- `hidden_depth_amzn.json`
- `hidden_depth_bench.json`
- `hit_flee_amzn.json`
- `hit_starve_amzn.json`
- `hmm_verify.json`
- `ice_budget.json`
- `ice_crown.json`
- `iceberg_bench.json`
- `iid_floor.json`
- `imbalance_predict_amzn.json`
- `impact_instant_amzn.json`
- `impact_persist_amzn.json`
- `improve_flow_bench.json`
- `initiative_fade_bench.json`
- `instant_decomp_amzn.json`
- `intraday_exec_amzn.json`
- `intraday_shape_amzn.json`
- `joint_fit_amzn.json`
- `joint_stability.json`
- `joint_tune.json`
- `lane_power_drill.json`
- `lattice_drill_vol_bench_a.json`
- `lattice_drill_vol_bench_b.json`
- `level_gap_amzn.json`
- `lo_response_amzn.json`
- `lob_exec_amzn.json`
- `lob_invariants_amzn.json`
- `lob_resilience_amzn.json`
- `lobster_replay_amzn_2012-06-21.json`
- `loss_cs_real_drill_conf_t_vs_empirical.json`
- `loss_cs_real_drill_gaussian_vs_conf_t.json`
- `maker_age_bench.json`
- `map_parity.json`
- `marketable_limit_amzn.json`
- `markout_amzn.json`
- `meta_model_synth.json`
- `metaorder_detect_amzn.json`
- `mid_dark_amzn.json`
- `mid_jump_amzn.json`
- `mortal_repost.json`
- `multih_fleet_eval_5db1cab214e291d7.json`
- `native_conformance.json`
- `nautilus_conformance_7bf19a08c147547b.json`
- `order_lifetime_amzn.json`
- `order_revision_amzn.json`
- `p0_evidence_inventory_v1.json`
- `pin_stability.json`
- `place_law_amzn.json`
- `place_mix_amzn.json`
- `post_trade_drift_amzn.json`
- `price_improvement_amzn.json`
- `promotion_gate_conf_t.json`
- `propagator_real_amzn.json`
- `queue_class_bench.json`
- `queue_fate_amzn.json`
- `queue_jump_amzn.json`
- `queue_occupancy_amzn.json`
- `quote_floor.json`
- `quote_place_amzn.json`
- `rankic_eval_9ebdad7da83e7348.json`
- `rankic_eval_addd924f107eedd8.json`
- `real_benchmark_us_wide_manifest.json`
- `real_benchmark_us_wide_test.json`
- `real_benchmark_us_wide_validation.json`
- `refill_hazard_amzn.json`
- `regime_clock_bench.json`
- `release_chase_amzn.json`
- `reload_gate.json`
- `repost_frontier.json`
- `repost_latency.json`
- `reseed_hazard.json`
- `round_lot_amzn.json`
- `seed_sweep_demo_amzn.json`
- `shield_decay_amzn.json`
- `side_imbalance_amzn.json`
- `sign_autocorr_real_amzn.json`
- `sign_predict_amzn.json`
- `sim_real_ledger_amzn.json`
- `sim_sensitivity.json`
- `split_flow_bench.json`
- `spread_dynamics_amzn.json`
- `spread_floor_bench.json`
- `spread_reopen.json`
- `spread_response_amzn.json`
- `spread_response_bench.json`
- `stack_watch_synth.json`
- `stale_quote_amzn.json`
- `streak_calibrate_bench.json`
- `streak_stats_amzn.json`
- `sweep_crown.json`
- `sweep_width_amzn.json`
- `sweep_width_bench.json`
- `tail_real_drill.json`
- `tape_digest_amzn.json`
- `tape_surgery_amzn.json`
- `tick_rule_amzn.json`
- `touch_empty.json`
- `touch_follow_amzn.json`
- `trade_decomp_amzn.json`
- `unhit_chase_amzn.json`
- `vac_chase_amzn.json`
- `vol_bench_dc16af46c3f2975b.json`
- `vol_signature_amzn.json`
- `vpin_amzn.json`
- `warmup_spec.json`
- `wave23_map_amzn.json`
- `wave24_map.json`
- `zone_card.json`
- `zone_churn.json`
- `zone_embargo.json`
- `zone_map.json`
- `zone_stability.json`
- `zone_ttl.json`
The owner decides register vs re-admit vs grandfather.
