#!/usr/bin/env pwsh
# guard-config.ps1 — surgical protection against the inbound sftp config clobber.
#
# ROOT CAUSE (docs/SOTA/21-concurrent-loop-profile.md): dsh-dipcatcher-syncd.sh on
# the Mac runs `while true; do push; sleep 15; done` and pushes three literal
# files — pyproject.toml, Makefile, README.md — from the C:\Users\me\Documents\
# dipcatcher clone, whose pyproject.toml is a 132-line stub (`name = "dipcatcher"`)
# that predates this tree's real one. Killing Windows processes does NOT stop it;
# the writer is off-machine and respawns inbound sftp sessions.
#
# THIS IS A STOPGAP, not a fix. The fix is pausing the Mac daemon (see "STOP IT
# PROPERLY" below). This script only reverts the *specific known-bad* payload so
# concurrent agents can keep working; it never touches any other modification.
#
# SAFETY: restores ONLY when the live pyproject.toml carries the stub signature
# (`name = "dipcatcher"`). The real project is `name = "fx-1"`, so a legitimate
# edit can never match. If pyproject.toml is already clean, this script does
# nothing at all — no writes, no venv churn.
#
# STOP IT PROPERLY (pick one):
#   1. On the Mac:  kill/pause dsh-dipcatcher-syncd.sh  (its only guard is
#      `[[ -d "$ROOT/src" ]]`, so renaming $ROOT/src on the Mac also halts it)
#   2. Or set the dsh 24x7 job JSON "status": "running" -> "stopped"
#   3. Or remove the three `-put` lines from dsh-dipcatcher-push.py
#
# Usage:  pwsh -File tools/guard-config.ps1            # single check (CI-safe)
#         pwsh -File tools/guard-config.ps1 -Loop     # check every 20s
#         pwsh -File tools/guard-config.ps1 -Loop -IntervalSeconds 10
param(
    [switch]$Loop,
    [int]$IntervalSeconds = 20,
    [string]$RepoRoot = 'D:\dipcatcher'
)

Set-Location $RepoRoot
$log = Join-Path $RepoRoot 'tools\guard-config.log'
# The literal payload the daemon ships. Matched exactly; nothing else triggers.
$stubSignature = 'name = "dipcatcher"'
$guarded = @('pyproject.toml', 'Makefile', 'README.md', 'uv.lock')

function Test-Clobbered {
    $pp = Join-Path $RepoRoot 'pyproject.toml'
    if (-not (Test-Path $pp)) { return $false }
    # Only the first ~15 lines carry [project].name; avoid reading the whole file.
    $head = Get-Content $pp -TotalCount 15 -ErrorAction SilentlyContinue
    return ($head -match [regex]::Escape($stubSignature)) -as [bool]
}

function Invoke-Guard {
    if (-not (Test-Clobbered)) { return $false }

    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $drifted = @()
    foreach ($f in $guarded) {
        $p = Join-Path $RepoRoot $f
        if (-not (Test-Path $p)) { continue }
        $work = (git hash-object $f 2>$null)
        $head = (git rev-parse "HEAD:$f" 2>$null)
        if ($work -and $head -and ($work -ne $head)) { $drifted += $f }
    }

    foreach ($f in $guarded) { git checkout -- $f 2>$null }

    $msg = "[$ts] CLOBBER DETECTED + REVERTED (drifted: $($drifted -join ', '))"
    Add-Content -Path $log -Value $msg
    Write-Host $msg -ForegroundColor Yellow
    Write-Host "  -> Mac-side sync daemon is still pushing. Pause dsh-dipcatcher-syncd.sh." -ForegroundColor Yellow
    return $true
}

if (-not $Loop) {
    $null = Invoke-Guard
    if (Test-Clobbered) { exit 1 } else { exit 0 }
}

Write-Host "guard-config: watching every ${IntervalSeconds}s for the '$stubSignature' clobber."
Write-Host "guard-config: this is a stopgap. Pause the Mac-side syncd to stop it at the source."
$clobbers = 0
while ($true) {
    if (Invoke-Guard) { $clobbers++ }
    Start-Sleep -Seconds $IntervalSeconds
}
