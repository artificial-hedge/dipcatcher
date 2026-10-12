# Dipcatcher journeys (gate P1)

Frozen evidence for the chat front door. The suite is an automated script
with a stub model. It is **not** the moderated five-user trial. Gate P1
stays **OPEN** until that trial and installed-wheel runs exist.

## What a journey is

Each journey is a `dipcatcher.journey.v1` record in
`tests/fx1/journeys/journeys.json` (ids `J001`–`J100`, immutable). The
result document records the SHA-256 of that file. A journey is data:

| Field | Meaning |
|---|---|
| `category` | One of the ten rows below, ten journeys each |
| `title` | A real researcher goal, in plain language |
| `critical` | Honesty, credential, cancellation, or recovery. Must pass even if 99/100 pass |
| `mode` | `inproc` (real `Concierge.handle_line`) or `pty` (real `python -m dipcatcher_cli chat`) |
| `requires` | `data`, `runs`, or `models-check`. If `/help` does not document the command, the journey is `unavailable` |
| `executes_lab` | The steps invoke a real lab harness command. At most 15 journeys do |
| `setup` | User files, a corrupt session, scripted stub replies, or an endpoint preset |
| `steps` | Ordered `{say, expect, forbid, wait}`. `secret` / `url` are prompt replies. `signal` is Ctrl-C or kill |
| `final` | Exit code, session/export/history/telemetry facts |

Onboarding journeys do not preset credentials. Elsewhere, `setup.endpoints`
may mean "this user already has an endpoint." The only placeholders are
`{{key}}`, `{{lite_key}}`, `{{stub_url}}`, and `{{work}}`. Fixture keys are
redacted from the result file.

## Completed in chat

The runner may only type lines and answer a `/keys set` secret prompt.
It does not write config mid-journey, export env vars mid-journey, call
expert subcommands, or edit the session store to make a step succeed.
Setup runs before the first prompt. A kill and `chat --resume` are the
crash, not a rescue. If a journey needs anything else, it fails.

`unavailable` is a failure in the pass rate, listed separately. It is not
an xfail.

## How to run

Full suite (exit 0 only when at least 99 pass **and** every critical
journey passes):

```bash
cd /workspace && NO_COLOR=1 TERM=xterm-256color \
  PYTHONPATH=/workspace:/workspace/src \
  .venv/bin/python scripts/dipcatcher_journeys.py --out result.json
```

One pytest case per journey. `JOURNEYS_FILTER` is a comma-separated list of
ids, categories, `inproc`, `pty`, or `critical`. Pty journeys are marked
`slow` (a `pty` marker is not registered in `pyproject.toml`).

```bash
cd /workspace && NO_COLOR=1 TERM=dumb FORCE_COLOR=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=src JOURNEYS_FILTER=J001,J011,J021 \
  .venv/bin/python -m pytest tests/fx1/journeys/test_journeys.py \
  -o addopts="" -q -p no:cacheprovider
```

Pty hang guards are at least 60s (startup wait 90s). Do not tighten them.

## Latest measured result

Two back-to-back runs, then a third run saved as the artifact, on
2026-10-12 against git `1c2583ffc` (dirty worktree; other agents were
editing `src/`). All three pass/fail sets were identical, including the
reason on `J082`.
`os.cpu_count()` reported 4. Python 3.12.3, Linux.

| | |
|---|---|
| Suite SHA-256 | `cccb7768a2d779e5f7349c53caa005793f0328adfa7894afdef9d9fce245713a` |
| Passed | 99 |
| Failed | 1 |
| Unavailable | 0 |
| Pass rate | 0.99 (99/100) |
| Critical | 20/21 passed. Failed: `J082` |
| Script exit | 1 (a critical journey failed) |
| Lab executions declared | 8 (`J021`–`J026`, `J030` dataset inspect; `J064` `verify-research`) |
| Pty / inproc | 24 / 76 |

| Category | n | Passed | Failed | Unavailable |
|---|---:|---:|---:|---:|
| onboarding | 10 | 10 | 0 | 0 |
| model-selection | 10 | 10 | 0 | 0 |
| data-discovery | 10 | 10 | 0 | 0 |
| research-planning | 10 | 10 | 0 | 0 |
| execution | 10 | 10 | 0 | 0 |
| follow-ups | 10 | 10 | 0 | 0 |
| verification | 10 | 10 | 0 | 0 |
| cancellation | 10 | 10 | 0 | 0 |
| recovery | 10 | 9 | 1 | 0 |
| export | 10 | 10 | 0 | 0 |

### Failures

| Id | Status | Reason | Attribution |
|---|---|---|---|
| J082 | failed | corrupt session file was reported but not quarantined | product defect |

`J082` is critical. `/sessions` prints `invalid session JSON` and the prompt
stays usable. `SessionStore.list` loads with `quarantine=False`, and
`SessionStore.diagnose` has no caller, so the corrupt file stays in the
session directory.

No journey was skipped, xfailed, or rewritten to avoid this result.
`/data`, `/runs`, and `/models check` were present in `/help` on this tree,
so those journeys ran instead of being marked unavailable.
