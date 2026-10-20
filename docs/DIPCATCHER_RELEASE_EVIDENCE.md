# dipcatcher release evidence

**Date:** 2026-10-12. This is the evidence table named in
[the product contract](DIPCATCHER_PRODUCT_CONTRACT.md). Every gate below
stays **OPEN**. A passing local number is not the evidence that gate's
text requires.

The JSON files live under `/opt/cursor/artifacts/` and are not committed.
Hashes are `sha256sum` of those files. `reference_hardware` is false in
every artifact that has the field. Nothing here is a latency, price, or
context-window claim for a hosted model.

## What was measured, and at which tree

`git diff --name-only c2e624f36 ef35e0e04` is only:

- `.gitignore`
- `Makefile`
- `docs/DIPCATCHER_JOURNEYS_PTY.md`
- `tests/fx1/journeys_pty/pty_run.py`

`src/fx1` and `src/dipcatcher_cli` do not differ between those commits.
`c2e624f36` is 2026-10-12 12:38:59 +0000. `ef35e0e04` is 2026-10-12
13:55:48 +0000. The journeys JSON records `tree.head`
`c2e624f3695bde98a2672c476f308bcbd6713913` and a dirty tree (`Makefile`,
`tests/fx1/journeys_pty/pty_run.py`). The latency, soak, smoke, and
relevance JSON files do not store a git SHA. Their logs sit in that same
window, before `ef35e0e04`. The fx1 lane log at 13:01 UTC prints
`HEAD c2e624f36` and the same two dirty paths.

`os.cpu_count()` in the journey and latency records is 4. That is the
number those processes reported. It is not a statement that this machine
is the contract's reference hardware.

## Table

Owner is **unassigned**. No artifact names a release owner.

| Gate | SHA | Command / scenario | Platform | Artifact (sha256) | Timestamp | Result | Remaining blocker |
|---|---|---|---|---|---|---|---|
| P5 latency | `c2e624f36` (not stored in the JSON) | `scripts/dipcatcher_latency.py`. Log `/tmp/r3_latency.log`. Cold prompt n=30. Local ack n=200, cycle `/help` `/status` `/model` `/session`. Cancel is `procs.run_owned` on the script's `--sleeper`, not chat `/cancel`. | Linux 6.12.94+, CPython 3.12.3, `os.cpu_count()` 4, load `[0.0293, 0.4116, 0.6841]`, `reference_hardware` false | `/opt/cursor/artifacts/dipcatcher_latency_final.json` `7996d5675a5991891efca279122dd2d3c48d1519358156ae0d7cfbf9805da4e2` | `measured_at` `2026-10-12T13:20:25.088365+00:00`. Log EXIT 0, elapsed 13 s | `pass` true. `p5_gate_closed` false. Cold p95 `0.244281` s (min `0.184711`, median `0.201407`, max `0.247129`, 30/30 ok). Input ack p95 `0.03819` s (p50 `0.020015`, min `0.005707`, max `0.04624`, 200/200 ok). Owned stop `0.100916` s, exit 130, `owned_stop_pass` true. Numeric targets in the file pass (cold limit 2 s, ack limit 0.1 s, cancel limit 1 s). Percentile method `nearest_rank` | P5 stays open. Not reference hardware. The file's own note: crash/restart coverage and separate model-latency measurement are out of scope. Chat `/cancel` was not timed |
| P6 soak | `c2e624f36` (not stored in the JSON) | `scripts/dipcatcher_soak.py`. Log `/tmp/r3_soak.log` elapsed 121 s, EXIT 0. `duration_s` in the file is `120.0`. `make dipcatcher-evidence` passes `--duration-s 60`, so that target does not reproduce this file | same Linux, CPython 3.12.3, `reference_24h` false, stub model | `/opt/cursor/artifacts/dipcatcher_soak_final.json` `fa11ce2a2b887a4e763605c7d0ed78c86d8c537590c602d8a90dc6f6f4ecb242` | log `2026-10-12T13:22:52+00:00` to `2026-10-12T13:24:53+00:00`. `elapsed_s` `120.616435`, `warmup_s` `0.086937` | `ok` true. `failures` `[]`. 200 cycles, every `recovered` true. `orphan_children` `[]`, `secret_hits` `[]`, `session_problems` `[]`, `registry_live` `[]`. 43 jobs: cancelled 15, interrupted 14, succeeded 14. fd 5 → 6. traced bytes 302057 → 67292802 (delta 66990745), under the file's 100663296 growth bound. Bench is SYNTHETIC `lane-power` at `--n-steps 16`, `--n-seeds 1`, `--lanes serial_watch`, `--defects 0.0`. The file says catalog defaults are larger and this is not a full bench | P6 stays open. Scope string in the file: this is not the 24-hour mixed-workload soak. Crash-restart cycles here are a bounded sample |
| P7 wheel smoke | `c2e624f36` (not stored in the JSON) | `scripts/dipcatcher_release_smoke.py`. Log `/tmp/r3_smoke.log` | Linux 6.12.94+, x86_64, CPython 3.12.3, uv 0.13.0 | `/opt/cursor/artifacts/dipcatcher_release_smoke_final.json` `a0defb0857bd00f816472fc0e0ff0419e13999c2a9fad19e3e5b570dc7ce0df4` | `started_at` `2026-10-12T13:28:18Z`, `finished_at` `2026-10-12T13:28:27Z`. Log EXIT 0, elapsed 9 s | `passed` true. Wheel `fx_1-0.4.0-py3-none-any.whl` sha256 `5753d67d61d315f4282c71e3910de5b6a2da2a5a8410376455f038364d121d6b`. Build `uv-build-offline-isolated` ok. `uv pip install --offline` of the wheel alone exited 1 (cache missing `exchange-calendars`; network disabled). The recorded install is `uv sync --frozen --offline --no-dev --no-install-project` then `uv pip install --offline --no-deps`. `modules_from_install` true for `fx1`, `quant_fund`, `dipcatcher_cli`. 15 checks passed, including bare non-tty `dipcatcher` and `exec` for both model names. Temporary venvs removed. `not_covered`: signing, rollback drills, platform matrix (macOS/Windows) | P7 stays open. Those three items, plus the production-audit blockers, are the contract's remaining evidence |
| P3 relevance | `c2e624f36` (not stored in the JSON) | `scripts/dipcatcher_relevance.py`. Log `/tmp/r3_rel.log` EXIT 0, elapsed 3 s. Does not execute benches | Linux, this tree's catalog | `/opt/cursor/artifacts/dipcatcher_relevance_final.json` `ecc901d7c0a268f30b0a15f58f48e09fa59dbc0b997a2177084308f8af00f604` | log `2026-10-12T13:32:12+00:00` to `2026-10-12T13:32:15+00:00`. The JSON has no `generated_at`. File mtime `2026-10-12 13:33:58 +0000` | `ok` true. 55 cases. `supported_case_pass_rate` 1.0. `paraphrase_consistency` 1.0. `unsafe_refused_rate` 1.0. `invariant_failures` none. `model_refinement_ok` true (six attacks). `family_exclusion_code_retention` 1.0. Catalog digest `1cc0274867d6a886309bbaf87ed23f968989521d8772da73a9f0751aa274042d`. Suite sha256 `c37d0ef6d135561a1e154765278ee4844f1ae0d8fb459a16934cc6d88341365f`. Intent on 32 positive and 32 negative phrases: precision 1.0 (tp 20, fp 0), recall 0.625 (fn 12, tn 32). False negatives `P021`–`P032` | P3 stays open. The file's scope string: the expert rubric (at least 90% of supported cases) needs a human rating. This run is not a state-of-the-art claim. Plain-text recall 0.625 means `/superpower` is still required for the misses |
| P1 journeys, pty/in-proc | `c2e624f36` dirty as above | `scripts/dipcatcher_journeys_pty.py`. Log `/tmp/r3_jp.log` EXIT 0, elapsed 21 s. Stub model. 24 pty / 76 in-proc (see `docs/DIPCATCHER_JOURNEYS_PTY.md`) | Linux 6.12.94+, CPython 3.12.3, `os.cpu_count()` 4 | `/opt/cursor/artifacts/dipcatcher_journeys_pty_final.json` `47197c1f21264d95985f0eec866b00280285435b6410a69b7bbaed6e9f113f09` | `generated_at` `2026-10-12T13:17:07.421509+00:00` | schema `dipcatcher.journeys.v1`. passed 100, failed 0, unavailable 0, `pass_rate` 1.0. Critical 21/21, `failed_ids` `[]`. Suite sha256 `7ae3b65231d73a4190cd678606e4a39dc48f45e2555158025c31e9295e084ac2`. Each of the ten categories is 10/10 | P1 stays open. Scope string: not the five-user trial; no terminal recordings; not an installed-wheel run of the 100 journeys |
| P1 journeys, first recorded pty run | `1c2583ffcc2358f51933a9d136df313a792a6360` (dirty; the JSON lists many untracked paths) | same engine, earlier suite bytes | same kind of host (`os.cpu_count()` 4, CPython 3.12.3) | `/opt/cursor/artifacts/dipcatcher_journeys_pty_first_run_99_of_100.json` `269b09b9f3ab2c24c16f9ca2f9d35e299f09111ab8e270eb2321d4bede4cb7da` | `generated_at` `2026-10-12T08:46:40.750387+00:00` | passed 99, failed 1, `pass_rate` 0.99. Critical 20/21. Failed id `J082`, attribution `product defect`, reason `corrupt session file was reported but not quarantined`. Suite sha256 `cccb7768a2d779e5f7349c53caa005793f0328adfa7894afdef9d9fce245713a` (not the later digest) | Historical. The later 100/100 run is a different suite digest. This row is not a pass |
| P1 YAML journey engine | not a measured run | `make dipcatcher-journeys` is `pytest tests/fx1/journeys`. Freeze file `tests/fx1/journeys/journeys.sha256` digest `8e363775f2975727bcc6eb2b721f420488d8d6e2f6c0da25fe3d257bb621f3f7`, `count 29` files. `load_journeys()` on this tree returns 100 journeys (critical 44) | not run here | no JSON in `/opt/cursor/artifacts` | not run in this documentation pass | count only. No pass/fail number | P1 stays open. A frozen file exists; this pass did not execute the suite |
| fx1 lane | `c2e624f36` with `Makefile` and `pty_run.py` dirty (`/tmp/r3_fx1_runfixed1.log` prints `HEAD c2e624f36`) | three consecutive `pytest tests/fx1` runs, `-n 3`, one shared machine. Logs `r3_fx1_runfixed1.log`, `runfixed2`, `runfixed3` | Linux, CPython 3.12.3 | those `/tmp` logs, not committed | `2026-10-12T13:01:27+00:00` through `2026-10-12T13:06:14+00:00` | `4475 passed, 1 skipped, 1 warning` three times. Pytest times 92.02 s, 91.83 s, 93.76 s. Each log EXIT 0. The warning is `test_ops_io_adversarial.py` / duplicate zip name `dup.txt` | Not a gate closure. Not re-run at `ef35e0e04` in this pass. The dirty `pty_run.py` is the helper committed in `ef35e0e04` |
| `tests/unit/cli` | same dirty tree as the fx1 logs (`GIT_STATUS` in the cli logs) | two consecutive runs. Logs `/tmp/r3_cli_run1.log`, `r3_cli_run2.log` | same | those logs | `2026-10-12T13:08:07+00:00` to `2026-10-12T13:12:24+00:00` | `788 passed, 1 warning` twice. Pytest times 125.44 s and 124.90 s. EXIT 0. Warning: `quant_fund.cli.main` found in `sys.modules` before execution | Not re-run at the tip in this pass |
| lab unit suites at the merge | merge log, not the tip | `/tmp/gates_merge.log` section `lab suites`: `tests/unit/{cli,arch,docs,schemas}` plus the dependency-edge check, as that log ran it | same host | `/tmp/gates_merge.log` | log mtime `2026-10-12 11:13:01 +0000` | `857 passed, 1 warning` in 86.59 s. The same log's fx1 lane was `4431 passed, 1 skipped`, which is an earlier tree than the 4475 runs | Not re-measured after the merge. Do not treat 857 as the tip count |
| types, this tip | `ef35e0e04` | `mypy --strict src/fx1 src/dipcatcher_cli` | this machine | command output, not a file | 2026-10-12, this documentation pass | `Success: no issues found in 368 source files`. Exit 0 | Does not close P7. An earlier merge log reported 367 files for `src/fx1` alone |
| ruff, this tip | `ef35e0e04` | `ruff check src tests` and `ruff format --check src tests` | this machine | command output | 2026-10-12, this documentation pass | `All checks passed!` and `19232 files already formatted`. Both exit 0 | Does not close P7 |
| McCabe ratchet | `ef35e0e04` | `.venv/bin/python scripts/check_mccabe_ratchet.py` | this machine | command output | 2026-10-12, this documentation pass | Exit 1. No `src/fx1` or `src/dipcatcher_cli` name in the list. The 13 entries are lab-side and are listed under Known limitations in the CLI guide | Pre-existing ratchet drift. Not fixed here. P7 stays open |
| crown jewels | `ef35e0e04`, before the doc edits in this pass | `dipcatcher crown-jewels --check` | this machine | command output | 2026-10-12, this documentation pass | Exit 1. `DATA_LABEL=SYNTHETIC`. `jewel_mutated` on the 39 paths listed below | Needs a re-stamp with the signing key. Not done. Includes `Makefile`, `README.md`, `CHANGELOG.md`, `.gitignore`, and `AGENTS.md`, not only lab modules |
| docs epoch pin | `ef35e0e04` | `dipcatcher corpus-epoch --corpus-dir docs --glob '*' --check --heads-pin quality/epoch_heads.json --require-stamped --allow-member-updates` | this machine | command output | 2026-10-12, this documentation pass | Exit 1 both before and after the edits on this page. Before: drift on `ARCHITECTURE_ATLAS.md`, `DIP_CONCIERGE.md`, `FX1_ARCHITECTURE.md`, `FXI.md`, `OPERATIONS_RUNBOOK.md`, `architecture/manifest.json`, `architecture/module_deps.mmd`; unstamped `DIPCATCHER_JOURNEYS_PTY.md`, `DIPCATCHER_RELEVANCE.md`. After these edits, the same command also reports drift on `DIPCATCHER_PRODUCT_CONTRACT.md` and unstamped `DIPCATCHER_RELEASE_EVIDENCE.md` | The full corpus-epoch sweep was not run. Re-stamp needs the signing key. `tests/unit/docs` was failing because `docs/architecture/manifest.json` omitted the committed import `fx1.interactive.orchestration` → `fx1.interactive.procs` (package self-import count 179, now 180). This pass regenerated that manifest so the atlas check matches the tree. The new bytes are not epoch-stamped |

## Flake history

One lane run at tip `ce08665bb` (`-n 3`, one shared CPU) reported
`1 failed, 4474 passed, 1 skipped`: `tests/fx1/journeys/test_journey_gate.py::test_aggregate_gate`,
critical journey `verification-005`. The journey passes alone (8 of 8). The YAML
engine built its model client with a 2 s read timeout and a two-attempt retry, so a
loaded CPU could retry a call and consume the next scripted reply. The engine's
real-time bounds (client timeout, turn and job joins, in-flight wait, shutdown, the
research hold) were raised so only a hang can reach them; the stub-emitted 408 journeys
do not depend on them. After that change the YAML suite passed three times in a row
with two busy-loop processes competing (`114 passed` each) and the full lane passed
(`4475 passed, 1 skipped`, 92.95 s). An earlier flake in the journeys_pty suite (`J073`,
second Ctrl-C sent before the line reader was blocked in a read) was fixed the same way:
by waiting on the condition instead of the first byte of output.

## Independent review already in the tree

These commits are ancestors of `c2e624f36`. They are the review fixes
after the first journeys run. They are not a new measurement.

- `f96fc887d` dataset attach. A symlink at `data/<name>` that left `data/`
  was attached as a different file; resume now uses the same two bases as
  attach. An inspect profile binds an issue-time column only when the role
  is `exact_name`, the name is a safe token, and the column is in the file
  header. A stale-dataset refusal is an error, not exec exit 0. A summary
  that merely contains the words "point-in-time" no longer suppresses the
  unverified note. Chat and exec agree on backslash paths. The dataset note
  counts toward the context fit. Hashing does not follow a symlink. Resume
  checks kind and suffix. A served model id is one sanitized line.
- `caadedd0b` engines. The Linux parent-death wrapper no longer leaves
  `SIGPIPE`/`SIGXFSZ` ignored, no longer aborts when `prctl` cannot load,
  no longer treats a live PID 1 as dead, and resolves the child against the
  child's cwd and PATH. Redaction skips combining marks and unpaired
  surrogates and keeps NFKC expansions. Honesty treats `sharpe3.2`,
  `NAV1.9`, `pnl4.2`, and `sharpe_3.2` as headlines; the forbidden token
  sets are unchanged. `keep_sgr` keeps color and intensity only. TCP, proxy
  `CONNECT`, and TLS share `min(read timeout, 15 s)`. Fatal tty signals
  chain the previous handler. The chat console arms the terminal-stop scope.
- `d19a97464` approvals. A session grant for `harness:<capability>` no
  longer covers every later argv. Typed and legacy steps record
  `ActionBounds(argv, writes)`. Idempotent replay does not ask again. A
  non-interactive caller fails closed for a different argv.
- `c2e624f36` terminal stop. `SIGTERM`/`SIGHUP` kill the owned tree. A
  latched stop does not spend the retry budget on a second launch. A plain
  `terminate_all()` reaps and does not latch. The duplicate launch was what
  the chaos suite caught.

## `jewel_mutated` paths from `crown-jewels --check`

`.dockerignore`, `.env.example`, `.gitattributes`,
`.github/codeql/codeql-config.yml`, `.github/dependabot.yml`, `.gitignore`,
`.gitleaks.toml`, `.pre-commit-config.yaml`, `.test_durations`, `AGENTS.md`,
`APPLY.md`, `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, `Dockerfile`,
`HONESTY_RATING.md`, `INFLIGHT`, `Makefile`, `README.md`, `SECURITY.md`,
`day_grind_progress.md`, `docker-compose.yml`, `mkdocs.yml`,
`pyproject.toml`, `src/quant_fund/research/corpus_epoch.py`,
`src/quant_fund/research/corpus_inference.py`,
`src/quant_fund/research/crown_jewels.py`,
`src/quant_fund/research/custody.py`,
`src/quant_fund/research/epoch_delta.py`,
`src/quant_fund/research/epoch_merkle.py`,
`src/quant_fund/research/fuzz_drill.py`,
`src/quant_fund/research/lane_contracts.py`,
`src/quant_fund/research/ots_anchor.py`,
`src/quant_fund/research/receipt_v2.py`,
`src/quant_fund/research/release_attestation.py`,
`src/quant_fund/research/timestamp_anchor.py`,
`src/quant_fund/utils/hashing.py`, `tests/conftest.py`, `uv.lock`.

## How to reproduce

`make -n` on this tree prints:

```text
make dipcatcher-journeys
  pytest tests/fx1/journeys -q
make dipcatcher-journeys-pty
  python scripts/dipcatcher_journeys_pty.py --out ${EVIDENCE_DIR:-data/dipcatcher-evidence}/journeys_pty.json
make dipcatcher-relevance
  python scripts/dipcatcher_relevance.py --out ${EVIDENCE_DIR:-data/dipcatcher-evidence}/relevance.json
make dipcatcher-evidence
  python scripts/dipcatcher_latency.py --runs 30 --events 200 --out .../latency.json
  python scripts/dipcatcher_soak.py --duration-s 60 --out .../soak.json
  python scripts/dipcatcher_release_smoke.py --out .../release_smoke.json --checksums .../SHA256SUMS
make dipcatcher-gate
  ruff check, ruff format --check, mypy --strict src/fx1 src/dipcatcher_cli,
  then dipcatcher-test, both journey targets, and dipcatcher-relevance
```

`make dipcatcher-test` is `pytest tests/fx1 tests/unit/cli -q -n auto --dist loadfile`.
Default evidence directory is `data/dipcatcher-evidence` (`EVIDENCE_DIR`
overrides it). That directory is generated output. The numbers above are
the copies under `/opt/cursor/artifacts/`, not a fresh run of these targets.

Read a JSON file by its `schema` key (`dipcatcher.latency.v1`,
`dipcatcher.soak.v1`, `dipcatcher.release_smoke.v1`,
`dipcatcher.relevance.v1`, `dipcatcher.journeys.v1`). A true `ok` or
`pass` field means that script's own checks. It does not mean the contract
gate closed. `p5_gate_closed` and `reference_24h` are the fields that say
so for latency and soak.

## What this does not show

- No real `fx1` or `fx1-lite` endpoint. Model calls in these artifacts used
  the loopback stub, or did not contact a model (offline checks, the
  relevance planner, the wheel's `--help` / no-endpoint paths).
- No moderated five-user trial and no retained terminal recordings.
- No 24-hour soak. The soak row is two minutes on this machine.
- No macOS or Windows run. Those branches are covered by monkeypatched
  tests, not by a machine of that OS.
- No signed release artifact, no checksum published as a release, no
  rollback drill.
- No claim that the numeric P5 targets were met on reference hardware.
- The YAML journey suite was not executed for this table.
- McCabe, crown jewels, and the docs epoch pin are red. The corpus-epoch
  re-stamp was not run. It needs the signing key.
- `make dipcatcher-gate` was not run as one command.
