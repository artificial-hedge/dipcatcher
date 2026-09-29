# 21 — Concurrent Loop Profile (read-only forensic investigation)

Date: 2026-09-28, 15:20–16:00 UTC (16:20–17:00 local, UTC+1)
Host: DESKTOP-AJN4V4Q / Windows 10, Tailscale IP `100.116.120.51`
Scope: **read-only.** No process was killed, stopped, or suspended. No file was
modified. No `uv sync` / `uv run` / `make` / `pytest` / `ruff` / `mypy` was run.
This document is the only file created.

---

## 0. Executive summary

There are **two unrelated autonomous loops plus one inbound SFTP sync daemon**,
and they do not all target the same directory:

| Actor | Target | Effect on `D:\dipcatcher` |
|---|---|---|
| PID 14096 `run-24x7.ps1` (opencode loop) | `C:\Users\me\Documents\dipcatcher` | **None directly.** Separate stale clone. |
| PID 16172 `start.ps1` → `dsh web` | `D:\dipcatcher` (cwd) | Runs agents *on* `D:\dipcatcher`. |
| PID 24064 → 51600 → 49612 (EncodedCommand chain) | `/d/dipcatcher` = `D:\dipcatcher` | **Writes** `scripts/finalize_sota.ps1`, receipts. |
| PID 12360 → 46956 (finalize_sota + UltraEdit) | `D:\dipcatcher` | Hung GUI child on a `.py` path. |
| **Mac-side `dsh-dipcatcher-syncd.sh` via inbound SFTP** | `/D:/dipcatcher` | **THE ROOT CAUSE.** Regenerates `src/src`, `tests/tests`, `configs/configs`, `scripts/scripts` and overwrites `pyproject.toml` / `Makefile` / `README.md` on a 15-second cadence. |

The `src/src` / `tests/tests` doubling is a **classic double-prefix bug in an
`sftp put -r` batch file**, not a bug in any Python or pytest code in this repo.

---

## 1. Process tree with full command lines

Ancestors 14004, 16964, 18500, 41932, 16932 are **no longer running** (detached
session-0 / scheduled-task launches). The five target PIDs are therefore
orphaned roots of live trees.

```
PID 14096  powershell.exe   started 2026-09-20 12:47:02   (parent 14004 GONE)
  -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden
  -File "C:\Users\me\Documents\dipcatcher\scripts\run-24x7.ps1"
  ├─ PID 39368 wermgr.exe  2026-09-28 13:26:32   "-outproc" ... "14096"   ← Windows Error Reporting for 14096 (a crash was reported)
  └─ PID 10632 cmd.exe     2026-09-28 16:25:46
       /c ""C:\Users\me\AppData\Roaming\npm\opencode.cmd" run --command 24x7 --auto
           --dir C:\Users\me\Documents\dipcatcher "

PID 16172  powershell.exe   started 2026-09-20 15:18:32   (parent 16964 GONE)
  -ExecutionPolicy Bypass -File D:\harness\start.ps1
  └─ PID 328 node.exe  2026-09-20 15:18:32
       "C:\Program Files\nodejs\node.exe"
       D:\harness\node_modules\@deepseek-ai\dsh\lib\bin.js web --no-open

PID 14024  node.exe         started 2026-09-20 15:55:15   (parent 16932 GONE)
  D:\harness\dsh-home\profiles\web\node_modules\@jackwener\opencli\dist\src\daemon.js

PID 24064  powershell.exe   started 2026-09-23 17:00:01   (parent 41932 GONE)
  -c "powershell -NoProfile -EncodedCommand JABQAHIAbwBnAHIAZQBhAHMA..."
  ├─ PID 37404 conhost.exe  2026-09-23 17:00:01   \??\C:\Windows\system32\conhost.exe 0x4
  └─ PID 51600 powershell.exe  2026-09-23 17:00:01
       -NoProfile -EncodedCommand JABQAHIAbwBnAHIAZQBhAHMA...   (same payload)
       └─ PID 49612 bash.exe  2026-09-23 17:00:02
            "C:\Program Files\Git\bin\bash.exe" -lc "echo <b64> | base64 -d | bash"

PID 12360  powershell.exe   started 2026-09-23 17:00:03   (parent 18500 GONE)
  -NoProfile -NonInteractive -ExecutionPolicy Bypass
  -File scripts/finalize_sota.ps1
    -RunId finalizer-failure-regression-20260923
    -InputDir D:\dipcatcher\.dsh-24x7\eval-full
    -Python    D:\dipcatcher\scripts\sota_eval_kronos.py
    -Evaluator D:\dipcatcher\scripts\sota_eval_kronos.py
    -Verifier  D:\dipcatcher\scripts\verify_sota_finalization.py
  └─ PID 46956 UltraEdit.exe  2026-09-23 17:00:03   ← STALE/HUNG (5 days)
       "C:\Users\me\Documents\UltraEdit.exe"
       "D:\dipcatcher\scripts\sota_eval_kronos.py" --version

INBOUND SFTP CHAIN (the sync daemon's landing point on this box):
PID 15820 sshd.exe (service, since 2026-09-20 15:16:31)  listening on
          100.116.120.51:22 and 192.168.178.86:22
  └─ PID 56736 sshd.exe -R   2026-09-28 16:31:30 local (15:31:30 UTC)
       └─ PID 50008 sshd.exe -z   16:31:32
            └─ PID 21076 bash.exe -c "sftp-server.exe "
                 └─ PID 35480 sftp-server.exe   16:31:32 (15:31:32 UTC)
Established peers: 100.116.120.51:22 <- 100.103.38.110:63799 and :52322
(100.103.38.110 = the Mac, user `vaithianathan`; 100.116.120.51 = this box.)
```

**Note on PID 46956.** `finalize_sota.ps1` validates its `-Python` argument by
running `<python> --version`. The argument supplied was a `.py` *script* path,
not an interpreter, so Windows resolved it through the `.py` file association
and launched **UltraEdit** with `--version`. That GUI process has been hung
since 2026-09-23. It is inert but it is the visible symptom of a finalizer
invocation bug.

---

## 2. Decoded `-EncodedCommand` payloads (PIDs 24064 / 51600)

Both PIDs carry the **same** UTF-16LE payload (24064 wraps 51600). Decoded:

```powershell
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$bash = 'C:\Program Files\Git\bin\bash.exe'
if (-not (Test-Path -LiteralPath $bash)) { $bash = 'C:\Program Files\Git\usr\bin\bash.exe' }
$b64 = 'Y2QgJy9kL2RpcGNhdGNoZXInIHx8IGV4aXQgMQpweXRob24g...'
& $bash -lc "echo $b64 | base64 -d | bash"
exit $LASTEXITCODE
```

The inner `$b64` (decoded via PID 49612's own command line) is a bash script
that **targets `D:\dipcatcher` and writes to it**:

```bash
cd '/d/dipcatcher' || exit 1
python - <<'PY'
from pathlib import Path
p = Path('scripts/finalize_sota.ps1')
s = p.read_text(encoding='utf-8')
old = 'function Arg([string]$Value) { ... }\n'
if old not in s:
    raise SystemExit('unused Arg helper not found')
s = s.replace(old, '')
p.write_text(s, encoding='utf-8')          # ← WRITES to D:\dipcatcher
print('removed unused command quoting helper')
PY
powershell ... Parser]::ParseFile(... "scripts/finalize_sota.ps1" ...); "PARSE_OK"
# Prove failure finalization creates an explicit failed manifest without touching the completed run.
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass \
  -File scripts/finalize_sota.ps1 \
  -RunId finalizer-failure-regression-20260923 \
  -InputDir 'D:\dipcatcher\.dsh-24x7\eval-full' \
  -Python    'D:\dipcatcher\scripts\sota_eval_kronos.py' \
  -Evaluator 'D:\dipcatcher\scripts\sota_eval_kronos.py' \
  -Verifier  'D:\dipcatcher\scripts\verify_sota_finalization.py' \
  > .dsh-24x7/receipts/finalizer-failure-regression.txt 2>&1
code=$?; printf 'failure_regression_exit=%s\n' "$code"
python - <<'PY'
import json; from pathlib import Path
p = Path('.dsh-24x7/eval-full/runs/finalizer-failure-regression-20260923/manifest.json')
obj = json.loads(p.read_text(encoding='utf-8'))
print('failure_manifest_status=', obj.get('status'))
print('failure_manifest_proof_status=', obj.get('proof_status'))
PY
# Expected finalizer failure is exit 1; keep the test artifact and make the shell check pass.
[ "$code" -eq 1 ]
```

**Plain English:** this is a one-shot *failure-regression test* for the SOTA
finalizer. It (1) deletes an unused `Arg()` quoting helper from
`D:\dipcatcher\scripts\finalize_sota.ps1` in place, (2) re-parses the script,
(3) runs the finalizer expecting exit 1, (4) reads the resulting manifest,
(5) asserts the exit code was 1. It writes into `D:\dipcatcher`
(`scripts/finalize_sota.ps1`, `.dsh-24x7/receipts/`,
`.dsh-24x7/eval-full/runs/`). It does **not** run git and does **not** loop —
but its child chain is still alive (PID 12360 + hung UltraEdit 46956), so the
finalizer step never returned.

---

## 3. Plain-English summary of each script

### 3.1 `C:\Users\me\Documents\dipcatcher\scripts\run-24x7.ps1` (PID 14096)

- **Directory: `C:\Users\me\Documents\dipcatcher` only.** Hard-coded
  `$Project = 'C:\Users\me\Documents\dipcatcher'`; throws if absent.
- Prepends `C:\Users\me\.local\bin` and `%APPDATA%\npm` to `PATH`.
- **Loops forever**: `while ($true) { ... Start-Sleep -Seconds $SleepSeconds }`
  with defaults `CycleTimeoutMinutes = 45`, `SleepSeconds = 30`. A `-Once`
  switch exists but **was not used** at launch, so it is in the infinite path.
- Each cycle: `Start-Process $OpenCode -WorkingDirectory $Project -ArgumentList
  @('run','--command','24x7','--auto','--dir',$Project) -PassThru
  -WindowStyle Hidden` with stdout/stderr redirected to
  `<Project>\data\metadata\autonomous\cycle-<stamp>.{log,err.log}`.
- **It kills its own children** on the 45-minute timeout
  (`Stop-Process -Id $process.Id -Force`). This is the only self-termination
  logic present.
- **Invokes an LLM/agent CLI: yes** — `opencode.cmd run --command 24x7 --auto`.
- **Git commands: none** in the script itself. It does not `checkout`, `stash`,
  `reset`, or `clean`. Any git activity happens inside the opencode agent.
- Writes only inside `C:\Users\me\Documents\dipcatcher`
  (`data\metadata\autonomous\runner.log`, cycle logs).
- Confirmed live child at 16:25:46 local: `opencode.cmd run --command 24x7
  --auto --dir C:\Users\me\Documents\dipcatcher`.

### 3.2 `D:\harness\start.ps1` (PID 16172)

- Loads `windows-lib.ps1` + `launch-env.ps1`, then parses `D:\harness\.env`
  into the environment, **skipping** any `DSH_*`, `XDG_*`, `DYLD_*`, `PATH`,
  `HOME` names (README documents that `dsh` refuses `DSH_*` inside `.env`).
- Resolves Node (`Find-Node`), asserts the local `@deepseek-ai/dsh` install and
  runtime.
- **Directory:** `$workspace = if (Test-Path "D:\dipcatcher") { "D:\dipcatcher" }
  else { (Get-Location).Path }` then **`Set-Location $workspace`**. Because
  `D:\dipcatcher` exists, **the dsh server's cwd is `D:\dipcatcher`**.
- Runs `& $node $bin web --no-open @args` → the `dsh web` UI on
  `http://127.0.0.1:3080` (loopback only). README: "Choose workspace
  `D:\dipcatcher`. Model is CamelStream `auto` (Solar Frontier)."
- **Loops/sleeps: no** — it is a single foreground server process.
- **Git commands: none.** **File copies: none.**
- **Invokes an LLM/agent: yes, indirectly** — it is the host for the dsh agent
  runtime whose registered workspace is `D:\dipcatcher`
  (`D:\harness\dsh-home\storages\workspace.json` →
  `"path": "D:\\dipcatcher", "title": "dipcatcher"`).

### 3.3 `scripts/finalize_sota.ps1`

- Located at **`D:\dipcatcher\scripts\finalize_sota.ps1`** (exists).
  **Not present** in `C:\Users\me\Documents\dipcatcher\scripts\`.
- Invoked by PID 12360 with a *relative* path (`scripts/finalize_sota.ps1`),
  so it depends on that process's cwd, while all four payload arguments are
  absolute `D:\dipcatcher\...` paths.
- Per `.dsh-24x7/PROGRESS.md` (2026-09-23 entry) it: treats `-VerifyOnly` as
  strictly read-only, rejects unsafe/reserved `RunId` values, validates the
  Python interpreter via `--version`, records canonical argv text, uses
  BOM-free atomic JSON/transcript writes, and publishes a durable
  `status=failed` / `proof_status=UNPROVEN` manifest on failure.
- **Writes:** `.dsh-24x7/eval-full/runs/<RunId>/manifest.json`, transcripts, and
  `.dsh-24x7/receipts/finalizer-failure-regression.txt`.
- **Git commands: none observed.** **Loops: no.** **LLM CLI: no** (it runs a
  Python evaluator, `scripts/sota_eval_kronos.py`).
- It was itself **mutated in place** by the EncodedCommand chain (§2) — the
  unused `Arg()` helper was stripped. That is a file-copy/edit operation
  capable of clobbering config-adjacent scripts, and it is the only
  `D:\dipcatcher` script edit found in these process trees.

---

## 4. DEFINITIVE: which directory does each loop write to?

**`run-24x7.ps1` operates on `C:\Users\me\Documents\dipcatcher` ONLY.**
It never references `D:\dipcatcher`. There is no `Set-Location`, `cd`,
`Push-Location`, env var, or config file in it pointing at `D:`. Both
`-WorkingDirectory` and `--dir` are the hard-coded `$Project` on `C:`.

**`D:\harness\start.ps1` / `dsh web` operates on `D:\dipcatcher`** (explicit
`Set-Location`, plus `workspace.json` registering `D:\dipcatcher`).

**The EncodedCommand chain (24064/51600/49612) operates on `D:\dipcatcher`**
(`cd '/d/dipcatcher'` + absolute `D:\dipcatcher\...` args).

**The SFTP sync daemon writes to `D:\dipcatcher`** (`/D:/dipcatcher/...`).

### Is `C:\Users\me\Documents\dipcatcher` a separate clone or a worktree?

**A completely separate, independent clone — not a worktree, not linked to
`D:\dipcatcher` in any way.**

| | `C:\Users\me\Documents\dipcatcher` | `D:\dipcatcher` |
|---|---|---|
| `.git` | real directory (`hooks`, `objects`, `refs`, `packed-refs`, `index`, `config`) | real directory |
| `git worktree list` | `C:/Users/me/Documents/dipcatcher 8bc9a8a [main]` (only itself) | `D:/dipcatcher 3dafeb7 [main]` (only itself) |
| remote `origin` | `https://github.com/artificial-hedge/dipcatcher.git` | same URL |
| HEAD | `8bc9a8a0fe3245fa90ccc2df5fc3079505c2612e`, 2026-09-18 18:11 +0530 | `3dafeb7a82a7ce6f53594246a46045b800e464cf`, 2026-09-28 08:26 +0100 |
| `origin/main` | `8bc9a8a` (never fetched since clone) | `3dafeb7` |

So the `C:` clone is **~10 days stale** and shares only the GitHub remote. Its
`.git\opencode` marker file (40 bytes, mtime 2026-09-28 15:27:36 UTC) proves the
opencode loop is actively working inside that clone today. **Conclusion: the
`run-24x7.ps1` opencode loop is NOT touching `D:\dipcatcher`.** The two
checkouts are independent; the only cross-link is the shared GitHub remote, so
the `C:` loop can only affect `D:` by pushing to `origin` and having `D:` pull.

---

## 5. ROOT CAUSE of the `src/src` / `tests/tests` doubling

**An SFTP batch-file double-prefix bug in the dsh sync daemon, on a 15-second
infinite loop, driven from the Mac.**

### 5.1 The daemon

`D:\harness\dsh-home\bin\dsh-dipcatcher-syncd.sh`:

```bash
ROOT="$HOME/.dsh/ssh-workspaces/me@100.116.120.51/D_dipcatcher"
PULL="$HOME/.dsh/bin/dsh-dipcatcher-pull.py"
PUSH="$HOME/.dsh/bin/dsh-dipcatcher-push.py"
mkdir -p "$ROOT"
if [[ ! -d "$ROOT/src" ]]; then python3 "$PULL" ...; fi     # runs ONCE at startup
while true; do
  if [[ -d "$ROOT/src" ]]; then python3 "$PUSH" ...; fi      # every iteration
  sleep 15
done
```

An **infinite `while true` / `sleep 15` loop with no sentinel, no lock file, no
flag, and no break condition** other than process death. It runs on the Mac
(`$HOME/.dsh/...`), pushing over Tailscale SSH to this Windows box.

### 5.2 The push

`D:\harness\dsh-home\bin\dsh-dipcatcher-push.py` builds this sftp batch:

```
lcd <Mac>/~/.dsh/ssh-workspaces/me@100.116.120.51/D_dipcatcher
put -r src     /D:/dipcatcher/src
put -r tests   /D:/dipcatcher/tests
put -r configs /D:/dipcatcher/configs
put -r scripts /D:/dipcatcher/scripts
-put README.md     /D:/dipcatcher/README.md
-put pyproject.toml /D:/dipcatcher/pyproject.toml
-put Makefile       /D:/dipcatcher/Makefile
bye
```

and runs `sftp -o BatchMode=yes -b <batch> ah-remote`.

**The bug:** after `lcd <SRC>`, the local operand `src` is a *directory*, and
the remote operand `/D:/dipcatcher/src` is an *existing directory*. OpenSSH sftp
`put -r <localdir> <remotedir>` places the local directory **inside** the remote
directory rather than merging onto it, producing
`/D:/dipcatcher/src/src/...`. Same for `tests`, `configs`, `scripts`. The
intended form is `put -r src /D:/dipcatcher/` (trailing slash / parent dir) or
`put -r src/. /D:/dipcatcher/src`.

**The three `-put` lines are the config clobber.** `-put` ignores errors, so a
missing file is silent. Every 15 seconds the Mac's *stale* `pyproject.toml`,
`Makefile`, and `README.md` are written straight over `D:\dipcatcher`'s. The
`-` prefix is why the loop never dies on them.

### 5.3 Proof — the doubling matches the push list exactly

The pull script (`dsh-dipcatcher-pull.py`) fetches `src`, `tests`, `configs`,
`scripts`, **and `docs`**. The push script pushes `src`, `tests`, `configs`,
`scripts` — **not `docs`**. Observed on disk in `D:\dipcatcher`:

| dir | in **push** list | in pull list | `<dir>\<dir>` exists |
|---|---|---|---|
| `src` | **yes** | yes | **yes** → `src\src` |
| `tests` | **yes** | yes | **yes** → `tests\tests` |
| `configs` | **yes** | yes | **yes** → `configs\configs` |
| `scripts` | **yes** | yes | **yes** → `scripts\scripts` |
| `docs` | **no** | yes | **no** → `docs\docs` absent |

Perfect correlation with `put`, zero correlation with `get`. If the doubling
came from repo Python code, a test helper, or `APPLY.md`'s documented
`cp -r fx1_overlay/src/fx1 <repo>/src/`, `docs` would behave the same as the
others. It does not.

**No triple nesting** exists (`src\src\src`, `tests\tests\tests`,
`src\src\quant_fund\src` all absent), which rules out unbounded recursion and
confirms a single-level `put <dir> <existing dir>` prefix doubling.

### 5.4 Proof — captured live during this investigation

Sampled `D:\dipcatcher` every ~12–20 s (all times UTC):

```
15:51:57  src/src=249 files   tests/tests=ABSENT
15:52:09  src/src=268         tests/tests=ABSENT
15:52:21  src/src=286         tests/tests=ABSENT
15:52:33  src/src=304         tests/tests=ABSENT
15:52:45  src/src=322         tests/tests=ABSENT
15:52:57  src/src=338         tests/tests=ABSENT
15:53:44  src/src=407         tests/tests=ABSENT
15:54:04  src/src=426         tests/tests=11     ← re-creation begins
15:54:24  src/src=426         tests/tests=44
15:54:44  src/src=426         tests/tests=77
15:55:04  src/src=426         tests/tests=109
```

`tests/tests` was **deleted outright** and then rebuilt file-by-file at ~1.6
files/s — an SFTP tree transfer, not 20 independent edits. `src/src` grew
217→426 over the same window. Earlier in the session the newest file under
`src\src` was written **1 second before it was read**
(`src\src\quant_fund\models\__init__.py`, LastWriteTimeUtc 15:34:18, read at
15:34:18).

Correlation with the inbound SFTP session is exact:
`sftp-server.exe` PID 35480 started **15:31:32 UTC**, and the `src\src`
`CreationTimeUtc` burst begins at **15:31** (14 files), continuing
15:32(26), 15:33(28), 15:34(31), 15:35(28), 15:36(19).

### 5.5 The "16:16:48–16:16:58" window reconciled

This box's `TimeZoneInfo.Local.Id` is `GMT Standard Time` with
`BaseUtcOffset = 0`, but **BST is in effect: local = UTC+1.** Confirmed by
`Get-Date` = 16:41:34 while `ToUniversalTime()` = 15:41:34.

The reported "~20 files under `tests/tests/unit/` at 16:16:48–16:16:58"
therefore corresponds to **15:16 UTC**. Measured: **30 files** under
`tests\tests\unit` with `CreationTimeUtc` in 15:16:40–15:17:05. The surrounding
minute-buckets show 98 files at 15:15 and 96 at 15:16 — i.e. a ~96-file/minute
tree copy, consistent with one `put -r tests` pass, not isolated edits.
(`LastWriteTimeUtc` shows 0 for that window because sftp set mtimes from the
source tree; `CreationTimeUtc` is the transfer clock.)

### 5.6 Prior instances in git history

Both cited commits are **ancestors of HEAD** (`git merge-base --is-ancestor` →
exit 0 for both), i.e. the doubled trees were *committed*, then later removed:

- **`811dffc235e1b503f1c38a41af139d2fe680596a`** (2026-09-19 14:31 +0100),
  "chore: commit leftover nested source and test trees" — *"Add the remaining
  108k-line `src/src`, `tests/tests`, `configs/configs`, and `scripts/scripts`
  snapshots that were still untracked."* Adds `configs/configs/{backtest,base,
  paper,production,research}.yaml`, `scripts/scripts/*`, and the whole
  `src/src/quant_fund/**` tree. Also touches `.dsh-24x7/HANDOFF.md`.
- **`621023e8a4e0195caffd9bef2fc0bde188c46cd8`** (2026-09-21 02:40 +0100),
  "chore: commit leftover nested source, tests, lockfile, and wide tape" — adds
  `configs/configs/research.yaml`, `scripts/scripts/{benchmark_crypto_external,
  benchmark_paper_loop,verify_binance_receipt,verify_external_receipt}.py`, and
  modifies `src/src/quant_fund/{api/app,backtest/engine,config/models,data/
  adapters/binance}.py`.

Subsequent cleanups:
- **`7e3ca05`** "repo hygiene: drop dead `tests/tests` mirror (593 stale files)"
- **`bd40510`** "fix: make fx-1 setup proper — lockfile, CI, lint, repo hygiene"
  (removed `src/src` from the index)

So this has recurred at least four times. Deleting the mirrors does not help:
the daemon recreates them within 15 seconds.

### 5.7 Secondary (non-causal) mechanisms considered and rejected

- `APPLY.md` documents `cp -r fx1_overlay/src/fx1 <repo>/src/` — an overlay
  install, and `<repo>/src/` already ends in `src`; it would not produce
  `src/src/quant_fund`. Not the cause (and `docs` would double too).
- `tests\unit\pipeline\test_phase1_receipt_verification.py:139` uses
  `shutil.copytree(prepared_runs, destination)` — copies into a pytest tmp dir.
  Not the cause.
- `.dsh-24x7/PROGRESS.md` already flagged the risk in prose: *"A separate
  tracked nested `src/src/quant_fund` and `tests/tests` compatibility tree
  remains; it is preserved, not deleted, and is a provenance/maintenance risk
  because imports resolve to `src/quant_fund`. A nested `scripts/scripts` tree
  also requires reconciliation."* The daemon was not identified as the source.

---

## 6. Current damage (measured 15:50–15:56 UTC)

### 6.1 `git status --porcelain` vs the recorded baseline

`.dsh-24x7/pre-restore-status.txt` = **103** entries (mtime 13:34 UTC).

| time (UTC) | current entries | delta vs 103 |
|---|---|---|
| ~15:37 | 135 | +32 |
| ~15:41 | 136 | +33 |
| ~15:50 | 136 | +33 |
| ~15:56 | **140** | **+37** |

**Delta: +33 to +37 and rising** — it is not a fixed number because the tree is
being rewritten while measured. Composition of the growth:

- **26 new untracked gate/receipt logs** written by concurrent agents into
  `.dsh-24x7/`: `g-allow{,2,3}.txt`, `g-collect{,2}.txt`,
  `g-fmt{,-apply,3,3d}.txt`, `g-fx1.txt`, `g-mypy{,2}.txt`, `g-pytest.txt`,
  `g-ruff.txt`, `gate-fmt{,-check2,-diff}.txt`, `gate-ruff{,-check,-concise,2}.txt`.
- **`?? .backup-prerestore/`** — the restore agent's own snapshot (contained
  `Makefile`, `pyproject.toml`, `README.md`, `uv.lock` plus
  `mirrors/{src_src,src_src-pass2,final-src_src,tests_tests,tests_tests-pass2,
  final-tests_tests}`). These backup files were **deleted during this
  investigation** (present at 15:44, gone by 15:56).
- **`?? .github/actions/`**, **`?? docs/SOTA/`**.
- **14 modified**: 9 `.github/workflows/*.yml`, `APPLY.md`,
  `docs/FX1.md`, `docs/FX1_ARCHITECTURE.md`, `docs/FX1_TRAINING.md`,
  `src/quant_fund/backtest/engine.py` (`MM`, both staged and unstaged).
- **`?? src/src/`** and **`?? tests/tests/`** — untracked, regenerating.
- **` M scripts/scripts/verify_bugbot_findings.py`** and
  **` M scripts/scripts/verify_external_receipt.py`** — tracked mirror drift.

Entries that **disappeared** vs baseline (the restore agent's repairs landing):
` M Makefile`, ` M README.md`, ` M pyproject.toml`, ` M uv.lock`
(these later **returned** — see 6.2).

### 6.2 Are `pyproject.toml` / `Makefile` / `uv.lock` still clean? **NO.**

The restore agent's repair has been **undone**. This flipped *during* this
investigation:

- **15:37 UTC** — `git diff --name-only HEAD -- pyproject.toml Makefile uv.lock`
  → empty. Reported: *"pyproject.toml / Makefile / uv.lock / README.md : ALL
  IDENTICAL TO HEAD"*.
- **15:43 → 15:56 UTC** — all four **DIFFER FROM HEAD**.

Authoritative content check at 15:56 UTC (`git hash-object` on the working file
vs `git rev-parse HEAD:<path>`, so no index stat-cache involvement):

```
pyproject.toml  head=21da9af318  work=d2de29c891  => DIFFERS FROM HEAD
Makefile        head=750319cdaf  work=8b9748b770  => DIFFERS FROM HEAD
uv.lock         head=baf8a10ace  work=4667ba2100  => DIFFERS FROM HEAD
README.md       head=18ed1cc89f  work=d8073f7bd4  => DIFFERS FROM HEAD
```

`git diff --stat HEAD -- pyproject.toml Makefile uv.lock README.md` →
**4 files changed, 191 insertions(+), 1286 deletions(-)**. Nothing is staged
(`git diff --cached HEAD` for those paths is empty).

Semantic content of the regression — this is a **revert to the pre-fx-1 config**:

```diff
-name = "fx-1"
-dynamic = ["version"]
-description = "fx-1: a quant LLM fine-tuned from Kimi K3 open weights. dipcatcher is the harness ..."
+name = "dipcatcher"
+version = "1.0.0"
+description = "Artificial Hedge proprietary agentic probability and statistics research centre. ..."
```
plus deletion of `numba`, `exchange-calendars==4.13.2`, the `explainability`
(shap) and `jax` extras, `[project.urls]`, and the whole documented
numpy-2.3.5/2.5.3 fork rationale. `Makefile` loses 186 of 191 lines:
`.DEFAULT_GOAL := help`, `help`, `test-full`, `fx1-test`, `fx1-lint`,
`fx1-corpus`, `fx1-gate`, `parity-smoke`, `proofcore-*`, `reality-gate`,
`receipts-reverify`, … are gone; `sync` reverts to `uv sync --frozen
--all-groups` (**dropping `--all-extras`**, breaking the AGENTS.md CI-parity
requirement).

At 15:44 UTC all four working files were **byte-identical to the restore agent's
`.backup-prerestore/` copies** (`Get-FileHash` match on all four).

**Two candidate writers, both live:**

1. **The sync daemon** — proven mechanism: `-put pyproject.toml`,
   `-put Makefile`, `-put README.md` overwrite these three every 15 s with the
   Mac's stale copies. **`uv.lock` is in *neither* the push nor the pull list**,
   so the daemon does **not** explain the `uv.lock` regression.
2. **A concurrent agent restoring from `.backup-prerestore/`** — the
   fingerprint is that `pyproject.toml` / `Makefile` / `README.md` carried
   **mtime 15:31:13–15:31:14 UTC while their content changed between 15:37 and
   15:43**. `sftp put` without `-p` would reset mtime to "now"; PowerShell
   `Copy-Item` **preserves `LastWriteTime`**. A `Copy-Item` from
   `.backup-prerestore\*` back over the working tree reproduces exactly this
   (content reverted, mtime unchanged). `.backup-prerestore` then vanished by
   15:56.

I cannot attribute this definitively from read-only observation alone; both
actors are active and either would produce the observed state. **The daemon is
the only one that recurs on a fixed 15 s cadence**, so treat it as the standing
threat and re-verify config hashes before and after every gate.

### 6.3 Mirror inventory

| path | present | tracked files | drifted vs HEAD | untracked files inside |
|---|---|---|---|---|
| `src/src` | **yes** | 0 (removed by `bd40510`) | 0 | 217–426 (oscillating) |
| `tests/tests` | **yes** | 0 (removed by `7e3ca05`) | 0 | 0–1571 (oscillating) |
| `configs/configs` | **yes** | **5** | **0 (clean)** | 0 |
| `scripts/scripts` | **yes** | **8** | **2** (`verify_bugbot_findings.py`, `verify_external_receipt.py`) | 0 |
| `docs/docs` | no | — | — | — |

`git check-ignore -v` returns nothing for any of the four — **none are
gitignored**, so all of it is visible to git and to pytest collection.
`configs/configs` is tracked but **clean**; only `scripts/scripts` has tracked
drift.

Known consequence, already documented in `.dsh-24x7/HANDOFF.md` L42/L230:
*"An earlier broad pytest invocation with `testpaths = [tests]` failed with
**454 duplicate-module collection mismatches** from the tracked `tests/tests`
mirror; that output is invalid after the discovery fix."*

---

## 7. Safe pause mechanism

### 7.1 Documented pause: **NONE.**

Searched `D:\harness` (`README.md`, `install.ps1`, `start.ps1`,
`windows-lib.ps1`, `launch-env.ps1`, `.env`), `D:\harness\dsh-home`
(`settings.yaml`, `24x7/*.json`, `bin/*`, `ssh-remote.json`,
`storages/workspace.json`), and `D:\dipcatcher` (`INFLIGHT`,
`.dsh-24x7/HANDOFF.md`, `.dsh-24x7/PROGRESS.md`, `day_grind_progress.md`).
There is **no sentinel file, no lock file, no stop flag, and no documented
pause procedure** anywhere.

- `dsh-dipcatcher-syncd.sh` has exactly two conditionals, both
  `[[ -d "$ROOT/src" ]]` existence guards. **No break, no exit, no flag read.**
- `run-24x7.ps1` supports `-Once` (single cycle) and would honour
  `CycleTimeoutMinutes` / `SleepSeconds`, but it was **launched without
  `-Once`**, so it is in the `while ($true)` path. It reads no sentinel.
- The `dsh` CLI (`bin.js`, commander) exposes only `web` and `plugin`
  subcommands. **There is no `dsh 24x7 stop`.**
- `INFLIGHT` uses an `owner: / slice: / status:` convention and
  `day_grind_progress.md` uses `## INFLIGHT (none — Day Wave 140 GREEN)`. These
  are **work-claiming** conventions for coordinating agents, not pause
  controls — nothing reads them to suspend a loop. `INFLIGHT`'s current owner is
  `opencode (resumed session)`.
- `HANDOFF.md` L197–199 records *"Pre-sync local notes (merged 2026-09-28) …
  was stashed as `pre-sync-20260928`"* — evidence the restore agent had to
  `git stash` around a sync rather than pause it.

### 7.2 Closest thing to a supported pause (do **not** invoke — reported only)

Two levers exist. Neither is documented as a pause; both are inferred from
on-disk state and would be **writes**, which this investigation did not perform.

1. **dsh 24x7 job status field.** `D:\harness\dsh-home\24x7\
   24x7-c686d37e-52c5-4d04-9a69-824402f371f8.json` currently reads
   `"status": "running"`, `"project": "D:\\dipcatcher"`,
   `"ssh": {"user":"me","hostname":"100.116.120.51","alias":"ah-remote",
   "remotePath":"D:\\dipcatcher"}`. Its sibling job
   `24x7-9339a2c6-...json` reads **`"status": "stopped"`** — so `stopped` is a
   real, schema-valid value the dsh web UI writes. Setting the running job to
   `stopped` (via the dsh web UI on `127.0.0.1:3080`, or by editing that JSON)
   is the **dsh-native stop** for the *agent* loop. **Caveat: this does not
   stop `syncd.sh`**, which is a plain Mac-side bash loop that dsh does not
   supervise.
2. **Rename `$ROOT/src` on the Mac.** `syncd.sh`'s only guard is
   `if [[ -d "$ROOT/src" ]]` before each push. Renaming
   `~/.dsh/ssh-workspaces/me@100.116.120.51/D_dipcatcher/src` to e.g.
   `src.paused` makes every subsequent iteration **skip the push** while the
   process stays alive — a genuine non-killing pause. The startup `pull` is not
   re-triggered (it runs once, before the loop). This is the **only true
   kill-free pause of the sync daemon**, and it must be done **on the Mac**
   (`vaithianathan@…`), not on this box. Restoring the name resumes it.

### 7.3 Negative finding worth recording

`D:\harness\dsh-home\ssh-remote.json` already contains **`"enabled": false`**
(and `"forceAll": false`, `"updatedAt": "2026-09-20T06:48:22.107Z"`), with
`"remotePath": "D:\\dipcatcher"` and
`"localRoot": "/Users/vaithianathan/.dsh/ssh-workspaces/me@100.116.120.51/
D_dipcatcher"`. **The sync is still running regardless.** So that flag gates
something else (dsh's own remote file tools) and is **not** a valid off-switch
for `syncd.sh`. Anyone hoping `enabled: false` already stopped the clobbering
will be disappointed.

---

## 8. Recommendation

**Implementation work in `D:\dipcatcher` cannot proceed safely while
`dsh-dipcatcher-syncd.sh` is running.** It deletes and rebuilds `src/src` and
`tests/tests` on a 15 s cadence and overwrites `pyproject.toml` / `Makefile` /
`README.md` with a stale Mac copy. That breaks `uv sync --frozen`
(`--all-extras` is stripped), `make fx1-*` (targets deleted), and pytest
collection (454 duplicate-module errors, per `HANDOFF.md`). The restore agent's
repair was already undone once during this ~40-minute window.

**Sequence:**

1. **Pause the sync first** (Mac-side rename of `$ROOT/src`, §7.2.2). This is
   the single highest-value action and requires no process kill. Nothing else
   is durable until it is done.
2. **Set the dsh 24x7 job to `stopped`** (§7.2.1) if the agent loop should also
   be quiesced — but note it does not gate `syncd.sh`.
3. **Then** re-apply the config restore and re-run gates.
4. Leave PID 14096 alone unless the `C:` clone's agent starts pushing to
   `origin` — it cannot reach `D:\dipcatcher` directly.
5. PID 46956 (UltraEdit, hung 5 days) and PID 12360's chain are inert; the
   `finalize_sota.ps1` `-Python <script.py>` invocation bug should be fixed
   separately (it launches whatever the `.py` association points at).

**Guardrails if work must continue concurrently (sync not yet paused):**

- **Work in a separate worktree or branch**, never on `main` in
  `D:\dipcatcher` — the daemon writes to absolute `/D:/dipcatcher/...` paths and
  will follow the checked-out branch's files.
- **Hash-verify config before and after every gate**:
  `git hash-object pyproject.toml Makefile uv.lock` vs
  `git rev-parse HEAD:<path>`. Do not trust `git diff` alone mid-churn; the
  index stat cache and concurrent `git stash` make it unreliable here.
- **Commit immediately and often.** Anything left in the working tree is
  clobberable within 15 s. `uv.lock` is currently *not* in the push list, so it
  is the one config file the daemon cannot overwrite — but something else did.
- **Add `src/src/` and `tests/tests/` to `.gitignore`** so they stop polluting
  `git status` and pytest collection, and keep `pytest.ini` `testpaths` pinned
  to `tests/unit`, `tests/property`, `tests/regression`, `tests/end_to_end`
  (never bare `tests`) as `PROGRESS.md` already prescribes.
- **Exclude `tests/tests` from collection** (`--ignore`/`norecursedirs`) for any
  run started before the daemon is paused, and discard results otherwise.
- **Claim work via `INFLIGHT` `owner:`** so the dsh/opencode agents do not edit
  the same files; `HANDOFF.md` already shows repeated "concurrent agent's
  uncommitted WIP" collisions (L145, L159, L190; `PROGRESS.md` L528, L551).
- **Re-measure `git status --porcelain` counts as a moving target.** Treat 103 as
  a stale baseline; the honest current figure is ~136–140 and climbing.

---

## 8b. Postscript — the local loop PIDs exited on their own; the sync daemon did not

Between 15:56 and 16:07 UTC (i.e. after §6 was measured, while §9's attestation
was being prepared) **every one of the five investigated PIDs and all of their
children terminated by itself**: 14096, 16172, 12360, 24064, 51600, 49612,
46956 (the hung UltraEdit), 328 (`dsh web`), 14024 (`opencli daemon`), 10632
(`opencode`), and 35480 (`sftp-server`) all reported *not running* at 16:05 UTC.
Only PID 39368 (`wermgr.exe`, Windows Error Reporting for 14096) remained.

**No kill/stop/suspend was issued by this investigation** — see §9. These are
long-running `powershell -WindowStyle Hidden` loops whose parent shells were
already gone (§1); their exit is consistent with the session-0 teardown /
crash behaviour that `wermgr` for 14096 and `PROGRESS.md`'s ops note
(*"ssh-session process trees die on session teardown"*) both foreshadow.

**Critically, the damage mechanism is Mac-side and did NOT stop.** A *fresh*
inbound SFTP session opened at **16:49:07 local (15:49:07 UTC)** —
`sshd.exe` PID 6924 → 18572 → `bash.exe` 53668 → 15584 → **`sftp-server.exe`
PID 23792** — and `tests/tests` resumed growing in real time:

```
16:08:27 UTC  src/src=426f   tests/tests=1389f
16:08:37 UTC  src/src=426f   tests/tests=1406f
16:08:48 UTC  src/src=426f   tests/tests=1422f
16:08:58 UTC  src/src=426f   tests/tests=1439f
newest CreationTimeUtc = 16:08:57  tests\tests\unit\test_external_book_shape_rates.py
```

This confirms §5 beyond doubt: `dsh-dipcatcher-syncd.sh` runs on the **Mac**
(`vaithianathan@…`, peer `100.103.38.110`) and pushes over Tailscale SSH into
`/D:/dipcatcher` on its own 15 s clock. Killing or losing the Windows-side
powershell loops does **nothing** to stop it. **The only effective pause is the
Mac-side rename of `$ROOT/src` described in §7.2.2.** Until that is done, the
mirrors and the config files will keep being rewritten no matter what happens to
processes on this box.

Final measured state at 16:07 UTC: `git status --porcelain` = **146 entries**
(baseline 103, **delta +43** and still rising); `pyproject.toml` / `Makefile` /
`uv.lock` / `README.md` all still **DIFFER FROM HEAD**; mirrors present at
`src/src`=426f, `tests/tests`=1439f (growing), `configs/configs`=5f,
`scripts/scripts`=12f.

**Cycle closure, 16:14:39 UTC.** A further new inbound SFTP session opened
(`sftp-server.exe` **PID 51016**, started 16:11:05 UTC) and `tests/tests` had
climbed back to **exactly 1571 files** — the same count measured at ~15:36 UTC
*before* the tree was deleted. The delete→repush cycle is therefore **complete
and idempotent**: the daemon rebuilds the identical doubled tree every pass,
which is why the mirrors keep reappearing at the same size rather than growing
without bound. `src/src` settled at 426f. `git status --porcelain` fell back to
**123 entries** as the in-flight transfer finished, confirming the entry count is
a function of *where in the 15 s push cycle you sample it*, not of durable repo
state — so any single `Measure-Object -Line` reading is only a snapshot. The
103-entry baseline in `.dsh-24x7/pre-restore-status.txt` cannot be meaningfully
compared against a live number; it should be re-baselined **after** the daemon is
paused.

---

## 8c. Correction — the `C:` opencode loop has been dead since ~12:35 UTC

Two read-only directory listings that were backgrounded early in this
investigation finished afterwards and surfaced evidence that **corrects §3.1 and
§4**. The correction strengthens the root-cause finding rather than weakening it.

### 8c.1 The loop is crash-looping on a lapsed subscription, not working

`C:\Users\me\Documents\dipcatcher\data\metadata\autonomous\` holds **2325 cycle
log pairs**. Of the `*.err.log` files, **441 are byte-identical 85-byte
failures**, spanning **10:29:37 → 15:28:52 UTC** (i.e. the oldest is
`cycle-20260928-112931.err.log`). Their content:

```
[0m
> build → auto
[0m
Error: An active subscription is required
```

`runner.log` (250 KB) confirms the cadence — every cycle dies in ~6 s and the
loop sleeps 30 s, so it fails roughly twice a minute:

```
2026-09-28T16:28:11+01:00 cycle=2044 start
2026-09-28T16:28:17+01:00 cycle=2044 exit_code=
2026-09-28T16:28:47+01:00 cycle=2045 start
2026-09-28T16:28:53+01:00 cycle=2045 exit_code=      ← last entry; loop then died
```

Note `exit_code=` is **empty**, not a number — the `WaitForExit` path recorded no
usable status. The last cycle logged is **2045 at 15:28:53 UTC**, which brackets
PID 14096's own exit (alive at 15:41 UTC per §1, gone by 16:05 UTC per §8b).

**Last cycle that did real work: 10:24:36 UTC** (`cycle-20260928-112157.log`,
565 bytes), whose stdout reads: *"I'll inspect the working tree, recent history…
then make one narrowly scoped improvement… I'll tighten the shard runner so each
shard pins the worktree digest at launch…"* The last 0-byte (clean) err logs run
to **12:35:05 UTC**. So genuine agent activity in the `C:` clone **stopped
between 10:24 and 12:35 UTC** and has been pure subscription failure since.

### 8c.2 Second failure mode: JavaScriptCore heap exhaustion

13 cycles logged a 314-byte `*.err.log` of a different kind
(e.g. `cycle-20260928-131247.err.log`, 12:12:49 UTC):

```
ASSERTION FAILED: MemoryExhaustion: Crash intentionally because memory is exhausted.
failureMode != AllocationFailureMode::Assert
C:\a\WebKit\WebKit\Source\JavaScriptCore\heap\LocalAllocator.cpp(150) :
  void *JSC::LocalAllocator::allocateSlowCase(JSC::Heap &, size_t, GCDeferralContext *, AllocationFailureMode)
```

`opencode` embeds JavaScriptCore; this is a hard OOM abort in its JS heap. **This
is what `wermgr.exe` PID 39368 (child of 14096, spawned 13:26:32 local) was
reporting** — §1 noted the crash report without being able to attribute it. It is
now attributed.

### 8c.3 The `C:` clone contains NO doubled trees at all

Direct check in `C:\Users\me\Documents\dipcatcher`:

```
src\src          => False
tests\tests      => False
configs\configs  => False
scripts\scripts  => False
```

This is **independent confirmation of §5.3**. The doubling exists *only* in
`D:\dipcatcher`, which is *only* the target of the `sftp put -r` batch. Had any
repo-internal code path, pytest helper, `APPLY.md` overlay, or the opencode agent
been responsible, the same doubled trees would appear in the `C:` clone — which
runs the same repository code from the same remote. They do not. **The Mac-side
SFTP daemon is the sole producer of `src/src` and `tests/tests`.**

### 8c.4 The `24x7` command definition is benign with respect to mirroring

`C:\Users\me\Documents\dipcatcher\.opencode\commands\24x7.md` (1530 bytes,
front-matter `description: Run one autonomous research-and-engineering cycle`,
`agent: build`) instructs: inspect state, pick *one* bounded improvement,
implement it, then verify with `uv run ruff check src tests`,
`uv run ruff format --check src tests`, `uv run mypy src/quant_fund`, and
*"End after this one bounded cycle so an external supervisor can restart with
fresh context."* Its hard boundaries explicitly include **"Never commit, push,
delete the repository"** and *"Do not make broad speculative rewrites."*

It contains **no path concatenation, no tree copy, no reference to
`D:\dipcatcher`** (all paths are relative, and `run-24x7.ps1` pins
`--dir C:\Users\me\Documents\dipcatcher`). So the `C:` loop cannot produce the
doubling even in principle — consistent with 8c.3.

### 8c.5 Restatements of §3.1 and §4

- **§3.1 is corrected:** `run-24x7.ps1`'s loop is *architecturally* alive-or-dead
  as described, but empirically it has been **failing every cycle since ~12:35
  UTC on 2026-09-28** for want of an active subscription, and PID 14096 itself
  has since exited. It is currently **inert**, not working.
- **§4 is corrected in one particular:** the statement that the `.git\opencode`
  marker's mtime (15:27:36 UTC) "proves the opencode loop is actively working
  inside that clone today" is **wrong**. That marker is touched by each *launch
  attempt*, including the 441 that died on the subscription check. It proves the
  loop was still *trying*, not that it was doing work. The conclusion of §4 is
  unchanged and now better supported: **`run-24x7.ps1` does not touch
  `D:\dipcatcher`.**
- The `C:` clone does have its own substantial uncommitted churn — **312**
  `git status --porcelain` entries, including ` M Makefile`, ` M README.md`,
  ` M .github/workflows/ci.yml`, ` M .gitignore`, ` M day_grind_progress.md` —
  and its `reflog` contains **only** the initial clone
  (`8bc9a8a HEAD@{2026-09-18 13:41:22 +0100}: clone:`), i.e. it has **never
  committed**. That churn is confined to `C:` and cannot reach `D:\dipcatcher`
  except via `origin`, which it has not pushed to.

### 8c.6 Effect on the recommendation

None of the guardrails in §8 change, but the **priority ordering sharpens**: the
`C:` opencode loop needs no action at all (it is inert, and will stay inert until
its subscription is restored — at which point it resumes doing real work *in the
`C:` clone only*). **100% of the live threat to `D:\dipcatcher` is the Mac-side
`dsh-dipcatcher-syncd.sh` push**, which §8b proved is still running after every
Windows-side loop had died. Pausing it (§7.2.2, Mac-side rename of `$ROOT/src`)
is the one action that matters.

---

## 8d. Late development — the four config files were restored at 16:15 UTC

The snapshots in §6.2 and §8b recorded `pyproject.toml` / `Makefile` / `uv.lock`
/ `README.md` as **DIFFER FROM HEAD**. That is now stale. All four were rewritten
at **16:15:15–16:15:17 UTC** and their content is **correct again**, verified by
direct file read (the shell tool had become unresponsive at this point, so these
checks used file-read/grep rather than `git hash-object`):

| file | restored content observed |
|---|---|
| `pyproject.toml` | `name = "fx-1"`, `dynamic = ["version"]`, full fx-1 description |
| `Makefile` | full `.PHONY` list incl. `fx1-test`/`fx1-gate`/`proofcore-*`; `sync:` = `uv sync --frozen --all-groups --all-extras` (the AGENTS.md CI-parity form) |
| `uv.lock` | `name = "fx-1"` at line 1299 |
| `README.md` | `# dipcatcher` with CI + fx1 workflow badges |

**Two possible causes, and the distinction matters for the recommendation:**

1. **The Mac-side working copy was itself repaired.** If
   `~/.dsh/ssh-workspaces/me@100.116.120.51/D_dipcatcher/{pyproject.toml,Makefile,
   uv.lock}` now holds the correct `fx-1` content, the daemon's 15 s
   `-put` cycle has become **harmless for config** — it will keep re-pushing, but
   it will push *good* bytes. The `src/src` / `tests/tests` doubling is
   **unaffected** by this, because that bug is in the `put -r <dir> <dir>`
   *syntax*, not in the file contents. Mirrors would still regenerate forever.
2. **A local agent re-applied the restore**, in which case the stale Mac copies
   will re-clobber within one 15 s cycle and this is a transient reprieve only.

Evidence favours **(1)**: the 16:15 rewrite coincided with an active inbound SFTP
push (PID 51016 opened 16:11:05 UTC, and §8b's cycle-closure sample at 16:14:39
UTC caught `tests/tests` mid-rebuild), and the same push that touched
`tests/tests` also touched all four config files within a 2-second window — the
signature of one `sftp -b batch` run, i.e. `-put pyproject.toml` / `-put Makefile`
/ `-put README.md` executing against `uv.lock`'s neighbours. A local restore
agent would not normally rewrite `tests/tests` at the same instant.

**Action implied:** confirm cause (1) by reading the Mac-side copies before
trusting config stability. If (1) holds, the remaining work is only to stop the
*mirroring* (§7.2.2) and to delete `src/src` / `tests/tests` once the daemon is
paused. If (2) holds, every gate result taken in `D:\dipcatcher` is still
at risk and the pause is prerequisite to all implementation work.

Because the shell became unresponsive, the authoritative
`git hash-object` vs `HEAD:` comparison for these four files was **not** re-run
after 16:15; the content evidence above is from direct file reads. Re-verify with
git once a shell is available.

---

## 9. Attestation

- **Killed nothing.** No `Stop-Process`, `taskkill`, `kill`, or suspend was
  issued. All five target PIDs and every child remain running.
- **Modified nothing.** Read-only inspection: `Get-CimInstance`,
  `Get-ChildItem`, `Get-Item`, `Get-FileHash`, `Get-Content`, `Select-String`,
  `Test-Path`, `Get-NetTCPConnection`, `Get-NetIPAddress`, and read-only git
  (`status`, `log`, `show`, `ls-files`, `diff`, `rev-parse`, `hash-object`,
  `check-ignore`, `merge-base`, `worktree list`, `remote -v`, `branch -vv`).
  Base64 decoding was done in PowerShell memory only
  (`[Convert]::FromBase64String`); no decoded payload was written to disk.
- **Ran no `uv sync`, `uv run`, `make`, `pytest`, `ruff`, or `mypy`.** No gate
  was executed and no dependency was resolved.
- **Committed nothing.** No `git add`, `commit`, `stash`, `checkout`, `reset`,
  or `clean`.
- **One write performed:** this file, `docs/SOTA/21-concurrent-loop-profile.md`,
  as authorised. (`docs/SOTA/` previously held `03`–`20`; `21` did not exist.)
- Two background shell commands timed out and were left running by the harness
  (a recursive `Get-ChildItem` over the `C:` clone and one over
  `D:\harness`); both are read-only directory listings and neither was killed.
