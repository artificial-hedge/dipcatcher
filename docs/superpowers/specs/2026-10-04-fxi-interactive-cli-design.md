# fxi — interactive CLI for fx-1 API keys

Date: 2026-10-04
Status: approved direction (auto mode; corrections welcome)
Path: architectural

## Context

`fx-1` is the in-tree product (`src/fx1`); `dipcatcher`/`quant_fund` is the
harness. Users currently interact through two Typer CLIs (`fx1`, `dipcatcher`)
that read credentials exclusively from environment variables. There is no
installer and no Homebrew distribution.

Two load-bearing facts discovered during exploration:

1. **`fx1-lite` does not exist** — zero references in the repo or history. The
   real two-key reality is:
   - `MOONSHOT_API_KEY` — outbound; powers the working `hosted_k3` backend
     (`src/fx1/serve/backends.py`), used by `fx1 eval`, hosted judges, and
     `fx1 infer`.
   - `FX1_API_KEY` — inbound; guards the loopback `fx1 strategy serve` pilot.
2. `HostedK3Backend(api_key=...)` already accepts an explicit key ahead of the
   env fallback, so no core change is needed to inject stored keys.

The distribution is proprietary (`Private :: Do Not Upload` — PyPI is blocked
deliberately), so curl/brew installation must pull a wheel from a release
asset, never from PyPI.

## Goals

- An interactive REPL, `fxi`, that stores fx-1 API key profiles securely and
  injects them into the harness (`fx1` / `dipcatcher` subprocesses) and into
  in-process model calls.
- One-shot mode: every REPL command also usable as `fxi <command>`.
- Installable via `curl -fsSL .../install.sh | sh` and via Homebrew (tap).
- Honest key semantics: the two profiles are documented as exactly the env
  vars they map to, not a fictional tier.

## Non-goals

- No new model tier, no hosted key-verification service, no PyPI upload.
- No changes to `src/fx1/serve/backends.py` or the harness CLIs.
- No new third-party dependencies (repo has no rich/textual; REPL is stdlib
  `cmd.Cmd`, CLI is Typer, already a dependency).

## Design

### Package layout

New package `src/fx1/interactive/` inside the existing `fx-1` distribution:

| Module | Responsibility |
|---|---|
| `profiles.py` | Key store + profile→env-var mapping + env injection |
| `actions.py` | `eval`, `chat`, `doctor`, `harness` helpers (shared by REPL and Typer) |
| `shell.py` | `cmd.Cmd` REPL |
| `app.py` | Typer app + `main()` entry (`fxi` console script) |

`pyproject.toml` gains one line in `[project.scripts]`:
`fxi = "fx1.interactive.app:main"`.

### Key profiles (`profiles.py`)

Fixed profile map (documented in help text):

| Profile | Env var injected | Powers |
|---|---|---|
| `fx1` | `MOONSHOT_API_KEY` | hosted K3 backend: `eval`, `chat`, hosted judges |
| `fx1-lite` | `FX1_API_KEY` | loopback `fx1 strategy serve` pilot |

- Store: `$FX1_CONFIG_DIR/credentials.json` or `~/.fx1/credentials.json`,
  mode `0600`, written atomically (tmp file + `os.replace` + `chmod`).
- Payload: `{"profiles": {"<profile>": {"key": ..., "env_var": ..., "set_at": ...}}}`.
- **Presence-only reporting everywhere**: lists and doctor output show
  `set`/`unset` and a last-4 fingerprint (`…wxyz`); raw keys are never
  printed, never logged, never placed in subprocess argv (env only).
- `keys set <profile>` prompts with `getpass` when no value is given.
- Resolution order for a command without explicit `--profile`: stored key for
  the active profile wins; otherwise the existing process env var is used
  (least surprise, mirrors today's behavior).
- `apply_profile()` mutates `os.environ` so child harness processes inherit
  the key — this is the literal "keys directly into the harness" path.

### Actions (`actions.py`)

- `run_eval(out, profile)` — subprocess `fx1 eval --backend hosted_k3
  [--out PATH]` with the resolved key in env. Reuses the exact CLI path the
  Makefile uses; no reimplementation of eval semantics. Output is the suite's
  proper scores only (honesty contract already enforced inside `fx1 eval`).
- `chat_loop(profile, model)` — in-process REPL-within-REPL:
  `HostedK3Backend(api_key=resolved_key, model=model)`, multi-turn messages,
  temperature pinned 0.0, `exit`/`quit` to leave. Fail-closed with a clear
  message when no key resolves.
- `doctor(profile)` — `fx1.doctor.collect_status()` plus: store path/permissions,
  per-profile presence + fingerprint, active profile, and a 3-second TCP
  reachability probe to the configured API host (no key burned).
- `harness_list()` / `harness_describe(name)` — in-process read of
  `fx1.harness.HARNESS_REGISTRY` (pure metadata).
- `harness_run(name, args)` — subprocess passthrough to the registered
  harness command with the session env.
- `verify(path)` — subprocess `dipcatcher verify-research <path>`.

### REPL (`shell.py`)

`cmd.Cmd` subclass. Prompt shows the active profile: `fxi[fx1] > `.
Commands: `keys`, `keys set`, `keys remove`, `use <profile>`, `profile`,
`doctor`, `eval`, `chat`, `harness list|describe|run`, `verify <path>`,
`run <fx1|dipcatcher|quant> <args…>`, `!<shell cmd>`, `help`, `exit`/`quit`.
Unknown input gets a hint, not a traceback. Tab completion for command names
via `completenames`.

### Typer app (`app.py`)

- `fxi` with no arguments launches the REPL (`main()` checks `len(sys.argv) == 1`);
  `fxi shell` does so explicitly.
- `fxi keys list|set|remove`, `fxi doctor`, `fxi eval [--out] [--profile]`,
  `fxi chat [--prompt TEXT] [--profile]`, `fxi harness list|describe|run`.
- Global `--profile` (app callback) selects the profile for the invocation.

### Installers

- `install.sh` (POSIX sh, pipeable): resolves a wheel from
  `$FXI_INSTALL_URL` (default: the `artificial-hedra/dipcatcher` GitHub
  release asset matching the version) or `--wheel <path>` for local/dev
  installs; requires Python ≥ 3.12; creates a venv in
  `${FXI_HOME:-~/.local/share/fxi}`; pip-installs the wheel; symlinks
  `fxi`, `fx1`, `dipcatcher`, `quant`, `verify-ledger`, `mc-engine` into
  `${FXI_BIN_DIR:-~/.local/bin}`; smoke-runs `fxi --version`; idempotent;
  `--uninstall` reverses it. Fails loudly with remediation hints, never
  `curl | sudo sh`.
- `packaging/homebrew/fxi.rb` — tap formula template installing the same
  wheel into a `libexec` venv. `url`/`sha256` are filled at release time;
  `license :cannot_represent` (proprietary, tap-only).

### Docs

`docs/FXI.md`: install (curl one-liner, brew tap), quickstart, key semantics
table, security model, release checklist (build wheel, compute sha256, bump
formula). README entry-points section gains `fxi`.

## Security & honesty

- Keys: `0600` store, atomic writes, getpass prompts, env-only subprocess
  injection, presence-only output (mirrors `fx1 doctor` and `AGENTS.md`).
- Fail-closed: missing key → clear error naming the profile and env var, never
  a silent fallthrough (mirrors `backends.py`).
- Honesty contract: the CLI adds no metrics of its own; eval/chat output flows
  through existing enforced paths. No live-trading claims.

## Testing

`tests/fx1/test_interactive.py` (fx1 lane, run via `make fx1-test`):
- store roundtrip in a tmp config dir; file mode `0600`; atomic-replace path
- profile map completeness; unknown profile rejected
- fingerprint shows last-4 only; raw key absent from `repr`/listings
- `apply_profile` injects the right env var; resolution order (stored > env)
- `chat_loop` against a stubbed backend (no network)
- `run_eval`/`harness_run` pass env through to subprocess (monkeypatched)
- REPL dispatch smoke: instantiate the shell, call `do_*` directly
- `main()` with no argv launches the shell (monkeypatched)

## Gates

`make lint`, `uv run mypy src/fx1`, `make fx1-test` (at least the new file:
`uv run pytest tests/fx1/test_interactive.py`), plus a manual end-to-end:
`fxi keys set fx1` → `fxi doctor` → `fxi chat` (real hosted call only if a key
is present) → `fxi eval --help` passthrough.

## Files touched

- `pyproject.toml` (one console-script line)
- `src/fx1/interactive/{__init__.py,profiles.py,actions.py,shell.py,app.py}` (new)
- `install.sh` (new), `packaging/homebrew/fxi.rb` (new), `docs/FXI.md` (new)
- `tests/fx1/test_interactive.py` (new)
- `README.md` (entry-points paragraph)

---

## Revision 2 (same day) — orbs, startup wizard, model-name semantics

Follow-up direction from the user, folded into the design:

1. **fx1 / fx1-lite are model names.** Profiles are model *endpoints*:
   `{api_key, base_url}` per model; the API `model` field is the profile
   name. Two additive core changes support this: `HostedK3Backend` honors a
   `FX1_BASE_URL` env override (arg > env > pinned URL), and `fx1 eval`
   gains an additive `--model` flag. Old `{key, env_var}` store entries are
   read-compat-normalized on load.
2. **Thinking-orbs animations.** `src/fx1/interactive/orb.py` ports
   [rareformlabs/thinking-orbs](https://github.com/rareformlabs/thinking-orbs)
   (MIT) to the terminal: six states (working/searching/solving/listening/
   composing/shaping), Braille-pixel dots, monochrome ink auto-detected via
   `COLORFGBG`, shared monotonic clock. Disabled for non-TTY, `NO_COLOR`,
   or `FXI_ORBS=off`. Used while the model composes a reply, while the
   wizard probes a host, and as the shell/wizard banner.
3. **Startup entry point.** Bare `dipcatcher` (harness Typer app) now runs
   `fx1.interactive.wizard.enter()` via an `invoke_without_command`
   callback (`no_args_is_help=False`): first run asks for model, API key
   (getpass, not echoed), base URL; probes the host; saves; then drops into
   `FxiShell`. Configured runs skip straight to the shell. Subcommands are
   unaffected. `quant_fund` already imports `fx1` at the package root, so
   the lazy import introduces no new dependency direction.
4. **UI.** Prompt is `fxi · <model>@<host> ›`; the wizard and shell open
   with a rounded-box banner orb (codex-inspired but orb/dot-based and
   monochrome rather than a plain spinner).

Additional files: `src/fx1/interactive/orb.py`, `src/fx1/interactive/wizard.py`,
`src/quant_fund/cli/_app.py` (callback), `src/fx1/serve/backends.py` +
`src/fx1/cli.py` (additive flags/env). Tests grew to ~50 cases including
orb rendering/disable rules, wizard flows, the `dipcatcher` hook, and
`--model` passthrough.
