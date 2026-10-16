# Owner-operated Windows equivalent of Makefile's pin/checkpoint ceremony.
# Run only after the candidate tree and its upstream epoch history are settled.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$KeyFile,
    [switch]$AnchorCheckpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath(
    (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
)
$keyItem = Get-Item -LiteralPath $KeyFile -Force -ErrorAction Stop
if ($keyItem.PSIsContainer) {
    throw 'KeyFile must name a regular file outside the repository.'
}
if (($keyItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw 'KeyFile must not be a symlink or reparse point.'
}
$keyPath = [System.IO.Path]::GetFullPath($keyItem.FullName)
$keyRoot = [System.IO.Path]::GetPathRoot($keyPath)
$ancestor = [System.IO.Path]::GetDirectoryName($keyPath)
while ($ancestor) {
    $ancestorItem = Get-Item -LiteralPath $ancestor -Force -ErrorAction Stop
    if (($ancestorItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'KeyFile path must not traverse a symlink or reparse point.'
    }
    if ($ancestor.Equals($keyRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        break
    }
    $ancestor = [System.IO.Path]::GetDirectoryName($ancestor)
}
$repoPrefix = $repoRoot.TrimEnd([char[]]@('\', '/')) + [System.IO.Path]::DirectorySeparatorChar
if (
    $keyPath.Equals($repoRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $keyPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)
) {
    throw 'KeyFile must be kept outside the repository.'
}
# In a linked worktree, the primary checkout also lies outside $repoRoot.
# Exclude it so a key cannot accidentally be kept in the shared repository.
$commonGitDir = & git -C $repoRoot rev-parse --git-common-dir
if ($LASTEXITCODE -ne 0 -or -not $commonGitDir) {
    throw 'Cannot locate the shared Git directory for KeyFile validation.'
}
if (-not [System.IO.Path]::IsPathRooted($commonGitDir)) {
    $commonGitDir = Join-Path $repoRoot $commonGitDir
}
$commonGitDir = [System.IO.Path]::GetFullPath($commonGitDir)
if ([System.IO.Path]::GetFileName($commonGitDir).Equals('.git', [System.StringComparison]::OrdinalIgnoreCase)) {
    $primaryRoot = [System.IO.Path]::GetDirectoryName($commonGitDir)
    $primaryPrefix = $primaryRoot.TrimEnd([char[]]@('\', '/')) + [System.IO.Path]::DirectorySeparatorChar
    if (
        $keyPath.Equals($primaryRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
        $keyPath.StartsWith($primaryPrefix, [System.StringComparison]::OrdinalIgnoreCase)
    ) {
        throw 'KeyFile must be kept outside the repository.'
    }
}
if ($keyItem.Length -eq 0) {
    throw 'KeyFile is empty.'
}

function Invoke-Dipcatcher {
    param([string]$Stage, [string[]]$CliArgs)

    Write-Host $Stage
    & uv run --frozen dipcatcher @CliArgs
    $commandExit = $LASTEXITCODE
    if ($commandExit -ne 0) {
        throw "$Stage failed with exit code $commandExit."
    }
}

function Stamp-Corpus {
    param([string]$CorpusDir, [string]$Pattern)

    $cliArgs = @('corpus-epoch', '--corpus-dir', $CorpusDir)
    if ($Pattern) {
        $cliArgs += @('--glob', $Pattern)
    }
    $cliArgs += @('--out-dir', $CorpusDir, '--heads-pin', 'quality/epoch_heads.json')
    Invoke-Dipcatcher -Stage "Stamp $CorpusDir" -CliArgs $cliArgs
}

Push-Location -LiteralPath $repoRoot
try {
    Get-Command uv -ErrorAction Stop | Out-Null

    # Refuse a wrong, unregistered, duplicate, or below-threshold signer before
    # the first tracked pin changes. The check reads the key file but prints no key bytes.
    Invoke-Dipcatcher -Stage 'Validate quorum signer' -CliArgs @(
        'quorum-check-key', '--key-file', $keyPath
    )

    # Crown jewels include changed root files; pin them before the quality epoch.
    Invoke-Dipcatcher -Stage 'Pin crown jewels' -CliArgs @('crown-jewels', '--write')

    # Exact order and patterns from Makefile:stamp-epochs.
    Stamp-Corpus -CorpusDir 'receipts'
    Stamp-Corpus -CorpusDir 'verifier' -Pattern '*.md'
    Stamp-Corpus -CorpusDir 'quality'
    Stamp-Corpus -CorpusDir '.github/workflows' -Pattern '*.yml'
    Stamp-Corpus -CorpusDir 'configs' -Pattern '*'
    Stamp-Corpus -CorpusDir 'artifacts' -Pattern '*'
    Stamp-Corpus -CorpusDir '.dsh-24x7' -Pattern '*'
    Stamp-Corpus -CorpusDir 'data/metadata' -Pattern '*'
    foreach ($dir in @(
        'src', 'tests', 'scripts', 'docs', 'research', 'replay', 'reports',
        'notebooks', 'examples', 'clients', 'typings', 'spec', 'docker',
        'deploy', 'third_party', 'rust', 'web', '.box-soft-verify', '.cursor',
        '.github'
    )) {
        Stamp-Corpus -CorpusDir $dir -Pattern '*'
    }

    Invoke-Dipcatcher -Stage 'Quorum-sign pins' -CliArgs @(
        'quorum-sign', '--key-file', $keyPath
    )
    Invoke-Dipcatcher -Stage 'Anchor epoch heads' -CliArgs @(
        'anchor-timestamp', '--file', 'quality/epoch_heads.json'
    )
    Invoke-Dipcatcher -Stage 'Anchor crown jewels' -CliArgs @(
        'anchor-timestamp', '--file', 'quality/crown_jewels.json'
    )
    $checkpointArgs = @('checkpoint', '--key-file', $keyPath)
    if ($AnchorCheckpoint) {
        $checkpointArgs += '--anchor'
    }
    Invoke-Dipcatcher -Stage 'Sign checkpoint' -CliArgs $checkpointArgs

    Invoke-Dipcatcher -Stage 'Verify crown jewels' -CliArgs @('crown-jewels', '--check')
    Invoke-Dipcatcher -Stage 'Verify repository' -CliArgs @('verify-repo')
    Invoke-Dipcatcher -Stage 'Verify checkpoint chain' -CliArgs @('checkpoint-chain')
}
finally {
    Pop-Location
}
