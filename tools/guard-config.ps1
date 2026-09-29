#!/usr/bin/env pwsh
# guard-config.ps1 — surgical protection against the inbound sftp config clobber.
#
# ROOT CAUSE (docs/SOTA/21-concurrent-loop-profile.md): dsh-dipcatcher-syncd.sh
# runs on a Mac as `while true; do push; sleep 15; done`. Its push step issues
#   put -r src  /D:/dipcatcher/src
# with both operands directories, so sftp nests the source INSIDE the target
# (src/src, tests/tests). It also `-put`s three literal files from the
# C:\Users\me\Documents\dipcatcher clone, whose copies are stale stubs that
# predate this tree's real ones. Killing Windows processes does not stop it; the
# writer is off-machine and respawns inbound sftp sessions.
#
# DETECTION: exact SHA256 match against the captured stub payloads below.
#   * v1 of this script matched only a `name = "dipcatcher"` signature in
#     pyproject.toml. On 2026-09-29 12:04 UTC the daemon pushed ONLY Makefile
#     and README.md, so v1 did not fire and both stayed clobbered ~35 min.
#   * Hash matching has ZERO false positives: no legitimate edit can reproduce
#     these exact bytes. It is therefore safe where drift-detection is not --
#     reverting on any drift would destroy real work (e.g. the BLE001 ruff rule
#     an agent adds to pyproject.toml), whereas reverting only on an exact stub
#     hash never touches a genuine edit.
#
# STOP IT PROPERLY (this script is a stopgap; pick one, on the Mac):
#   pkill -f dsh-dipcatcher-syncd.sh
#   mv "$ROOT/src" "$ROOT/src.paused"   # the daemon's only guard is [[ -d $ROOT/src ]]
#   ...or delete the three `-put` lines from dsh-dipcatcher-push.py
# Note: ssh-remote.json already says "enabled": false and the sync runs anyway.
#
# Usage:  pwsh -File tools/guard-config.ps1                    # single check
#         pwsh -File tools/guard-config.ps1 -Loop              # watch every 20s
#         pwsh -File tools/guard-config.ps1 -Loop -IntervalSeconds 10
param(
    [switch]$Loop,
    [int]$IntervalSeconds = 20,
    [string]$RepoRoot = 'D:\dipcatcher'
)

Set-Location $RepoRoot
$log = Join-Path $RepoRoot 'tools\guard-config.log'

# Captured 2026-09-29 from C:\Users\me\Documents\dipcatcher (the push source).
# These are the known-bad payloads. Match is exact and full-length.
$stubs = [ordered]@{
    'pyproject.toml' = '4DE1732DAA8102F1'   # 132 lines, name = "dipcatcher"
    'Makefile'       = '0D52E4A54D448CBE'   #  51 lines, no fx1-* targets, sync lacks --all-extras
    'README.md'      = 'B5DEA43E36CAE17D'   #  71 lines
}

function Get-ShortHash {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return $null }
    return (Get-FileHash $Path -Algorithm SHA256).Hash.Substring(0, 16)
}

function Invoke-Guard {
    $hit = @()
    foreach ($name in $stubs.Keys) {
        if ((Get-ShortHash (Join-Path $RepoRoot $name)) -eq $stubs[$name]) { $hit += $name }
    }
    if ($hit.Count -eq 0) { return $false }

    foreach ($f in $hit) { git checkout -- $f 2>$null }

    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $restored = @()
    foreach ($f in $hit) {
        $w = git hash-object $f 2>$null
        $h = git rev-parse "HEAD:$f" 2>$null
        $restored += if ($w -and $h -and ($w -eq $h)) { "$f(OK)" } else { "$f(FAILED)" }
    }
    $msg = "[$ts] CLOBBER reverted: $($restored -join ', ')"
    Add-Content -Path $log -Value $msg
    Write-Host $msg -ForegroundColor Yellow
    Write-Host "  -> daemon still pushing. On the Mac: pkill -f dsh-dipcatcher-syncd.sh" -ForegroundColor Yellow
    return $true
}

if (-not $Loop) {
    $null = Invoke-Guard
    exit 0
}

Write-Host "guard-config: exact-hash watch on $($stubs.Keys -join ', ') every ${IntervalSeconds}s."
Write-Host "guard-config: stopgap only. Pause the Mac-side syncd to stop it at the source."
while ($true) {
    $null = Invoke-Guard
    Start-Sleep -Seconds $IntervalSeconds
}
