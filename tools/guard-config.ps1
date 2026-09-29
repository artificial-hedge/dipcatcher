#!/usr/bin/env pwsh
# guard-config.ps1 - protects build config from the inbound sftp clobber.
#
# ROOT CAUSE (docs/SOTA/21-concurrent-loop-profile.md): dsh-dipcatcher-syncd.sh
# runs on a Mac as `while true; do push; sleep 15; done` and `-put`s three
# literal files from the stale C:\Users\me\Documents\dipcatcher clone. Its
# `put -r src /D:/dipcatcher/src` also nests whole trees (src/src, tests/tests).
# Killing Windows processes does NOT stop it: the writer is off-machine and
# respawns inbound sftp sessions.
#
# WHY v3 (marker presence), not v2 (exact hash):
#   v2 matched the *completed* stub payload by SHA256. On 2026-09-29 ~13:40 UTC
#   the tree was found clobbered while v2 was running and had logged nothing.
#   Diagnosis: the files were PARTIAL MID-TRANSFER writes -- pyproject.toml was
#   3,790 bytes against the stub's 3,937, Makefile 871 against 2,461. Every one
#   of them hashed differently from the baked-in stub, so exact-hash matching
#   structurally cannot fire on a partial write. It also missed a 1-line
#   truncation seen at the same time.
#   v3 instead asserts that each file still CONTAINS a marker that only the real
#   project has. A truncated or stub-overwritten file lacks it; a legitimate
#   edit does not.
#
# MARKER SAFETY (zero false positives on real edits):
#   A worker adding a ruff rule to pyproject.toml (e.g. BLE001) cannot remove
#   `name = "fx-1"`. A worker restructuring Makefile targets would have to
#   delete the documented `fx1-gate:` gate, which AGENTS.md mandates. So these
#   markers survive every legitimate change while dying on any clobber.
#   This is what makes v3 safe where drift-detection is NOT: reverting whenever a
#   file differs from HEAD would destroy real edits.
#
# STOP IT PROPERLY (this script is a stopgap; pick one, on the Mac):
#   pkill -f dsh-dipcatcher-syncd.sh
#   mv "$ROOT/src" "$ROOT/src.paused"   # the daemon's only guard is [[ -d $ROOT/src ]]
#   ...or delete the three `-put` lines from dsh-dipcatcher-push.py
# Note: ssh-remote.json already reads "enabled": false and the sync runs anyway.
#
# Usage:  pwsh -File tools/guard-config.ps1                     # single check
#         pwsh -File tools/guard-config.ps1 -Loop               # watch every 10s
#         pwsh -File tools/guard-config.ps1 -Loop -IntervalSeconds 5
param(
    [switch]$Loop,
    [int]$IntervalSeconds = 10,
    [string]$RepoRoot = 'D:\dipcatcher'
)

Set-Location $RepoRoot
$log = Join-Path $RepoRoot 'tools\guard-config.log'

# Each guarded file must contain its marker. `FirstLine` compares line 1 exactly;
# `Contains` is a substring test. Verified 2026-09-29: every marker is present in
# HEAD and absent from the corresponding stub in the push source clone.
$guards = @(
    [pscustomobject]@{ File = 'pyproject.toml'; Mode = 'Contains';  Marker = 'name = "fx-1"' }
    [pscustomobject]@{ File = 'Makefile';       Mode = 'Contains';  Marker = 'fx1-gate:' }
    [pscustomobject]@{ File = 'README.md';      Mode = 'FirstLine'; Marker = '# dipcatcher' }
    [pscustomobject]@{ File = 'uv.lock';        Mode = 'Contains';  Marker = 'name = "fx-1"' }
)

function Test-MarkerOk {
    param($Guard)
    $path = Join-Path $RepoRoot $Guard.File
    if (-not (Test-Path $path)) { return $false }   # deleted mid-transfer
    try {
        if ($Guard.Mode -eq 'FirstLine') {
            $first = Get-Content $path -TotalCount 1 -ErrorAction Stop
            return ($first -ceq $Guard.Marker)
        }
        # Substring test over the whole file; -SimpleMatch avoids regex surprises.
        return [bool](Select-String -Path $path -Pattern $Guard.Marker -SimpleMatch -Quiet -ErrorAction Stop)
    } catch {
        return $false   # unreadable = treat as clobbered, let git checkout fix it
    }
}

function Invoke-Guard {
    $hit = @()
    foreach ($g in $guards) { if (-not (Test-MarkerOk $g)) { $hit += $g.File } }
    if ($hit.Count -eq 0) { return $false }

    foreach ($f in $hit) { git checkout -- $f 2>$null }

    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $restored = @()
    foreach ($f in $hit) {
        $w = git hash-object $f 2>$null
        $h = git rev-parse "HEAD:$f" 2>$null
        $restored += if ($w -and $h -and ($w -eq $h)) { "$f(OK)" } else { "$f(FAILED)" }
    }
    $msg = "[$ts] CLOBBER reverted (marker missing): $($restored -join ', ')"
    Add-Content -Path $log -Value $msg
    Write-Host $msg -ForegroundColor Yellow
    Write-Host "  -> daemon still pushing. On the Mac: pkill -f dsh-dipcatcher-syncd.sh" -ForegroundColor Yellow
    return $true
}

if (-not $Loop) {
    $null = Invoke-Guard
    # Non-zero exit advertises "a clobber was detected", for CI use.
    exit $(if (@($guards | Where-Object { -not (Test-MarkerOk $_) }).Count -gt 0) { 1 } else { 0 })
}

Write-Host "guard-config v3: marker-presence watch on $($guards.File -join ', ') every ${IntervalSeconds}s."
Write-Host "guard-config: stopgap only. Pause the Mac-side syncd to stop it at the source."
while ($true) {
    $null = Invoke-Guard
    Start-Sleep -Seconds $IntervalSeconds
}
