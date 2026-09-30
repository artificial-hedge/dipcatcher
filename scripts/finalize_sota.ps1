[CmdletBinding()]
param(
  [string]$Python = "D:\dipcatcher\.venv\Scripts\python.exe",
  [string]$Evaluator = "D:\dipcatcher\scripts\sota_eval_kronos.py",
  [string]$Verifier = "D:\dipcatcher\scripts\verify_sota_finalization.py",
  [string]$InputDir = "D:\dipcatcher\.dsh-24x7\eval-full",
  [string]$RunId = "",
  [int]$Seed = 7,
  [int]$BootstrapCount = 2000,
  [switch]$VerifyOnly
)

# Explicit, fail-closed, provenance-preserving SOTA receipt finalizer.
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repo

function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
<<<<<<< Updated upstream
function Test-ReparsePath([string]$Path) { $current=[IO.Path]::GetFullPath($Path); while($current) { if(Test-Path -LiteralPath $current) { $item=Get-Item -LiteralPath $current -Force; if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return $true } }; $parent=[IO.Path]::GetDirectoryName($current); if(-not $parent -or $parent -eq $current) { break }; $current=$parent }; return $false }
=======
function Test-ReparsePath([string]$Path) { $current=[IO.Path]::GetFullPath($Path); while($current) { if(Test-Path -LiteralPath $current) { $item=Get-Item -LiteralPath $current -Force; if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { return $true } }; $parent=Split-Path -LiteralPath $current -Parent; if(-not $parent -or $parent -eq $current) { break }; $current=$parent }; return $false }
>>>>>>> Stashed changes
function Assert-NoReparsePath([string]$Path,[string]$Label) { if(Test-ReparsePath $Path) { throw "$Label uses a symlink or reparse point: $Path" } }
function Assert-RegularFile([string]$Path,[string]$Label) { if(-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Label is missing or not a regular file: $Path" }; $item=Get-Item -LiteralPath $Path -Force; if($item.PSIsContainer -or (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) { throw "$Label is missing or not a regular file: $Path" }; Assert-NoReparsePath $Path $Label }
function Write-Utf8NoBom([string]$Path, [string]$Text) {
  $encoding = New-Object System.Text.UTF8Encoding($false)
  [IO.File]::WriteAllText($Path, $Text, $encoding)
}
function Write-AtomicUtf8NoBom([string]$Path, [string]$Text) {
<<<<<<< Updated upstream
  $directory = [IO.Path]::GetDirectoryName($Path)
=======
  $directory = Split-Path -LiteralPath $Path -Parent
>>>>>>> Stashed changes
  $temporary = Join-Path $directory (([IO.Path]::GetFileName($Path)) + ".tmp-" + [Guid]::NewGuid().ToString("N"))
  try {
    Write-Utf8NoBom $temporary $Text
    Move-Item -LiteralPath $temporary -Destination $Path -Force
  } finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
  }
}
function Write-Json([string]$Path, $Value) {
<<<<<<< Updated upstream
  Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "WJ_pre:$(Get-Date -Format o)"
  if ($Value -is [System.Collections.IDictionary]) {
    foreach ($k in @($Value.Keys)) {
      Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "WJ_key_${k}:$(Get-Date -Format o)"
      $null = ($Value[$k] | ConvertTo-Json -Depth 40)
    }
  }
  $j = (($Value | ConvertTo-Json -Depth 40) + [Environment]::NewLine)
  Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "WJ_ser:$(Get-Date -Format o)"
  Write-AtomicUtf8NoBom $Path $j
  Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "WJ_atom:$(Get-Date -Format o)"
=======
  Write-AtomicUtf8NoBom $Path (($Value | ConvertTo-Json -Depth 40) + [Environment]::NewLine)
>>>>>>> Stashed changes
}
function Rel([string]$Path) {
  $full = (Resolve-Path -LiteralPath $Path).Path
  $root = $repo.TrimEnd('\') + '\'
  if ($full.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
    return $full.Substring($root.Length).Replace('\', '/')
  }
  return $full.Replace('\', '/')
}
function JsonHash($Value) {
  $bytes = [Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 30 -Compress))
  $sha = [Security.Cryptography.SHA256]::Create()
  try { ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() } finally { $sha.Dispose() }
}

if (-not $RunId) { $RunId = Get-Date -Format "yyyyMMdd-HHmmss" }
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*$" -or $RunId -in @(".", "..")) { throw "Unsafe RunId: $RunId" }
if ($RunId -match "^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$") { throw "Reserved Windows RunId: $RunId" }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Evaluator -PathType Leaf)) { throw "Evaluator not found: $Evaluator" }
if (-not (Test-Path -LiteralPath $Verifier -PathType Leaf)) { throw "Verifier not found: $Verifier" }
if (-not (Test-Path -LiteralPath $InputDir -PathType Container)) { throw "Input directory not found: $InputDir" }
$inputRoot = (Resolve-Path -LiteralPath $InputDir).Path
Assert-NoReparsePath $repo "Repository"
Assert-NoReparsePath $inputRoot "InputDir"
$repoRoot = [IO.Path]::GetFullPath($repo).TrimEnd("\") + "\"
if (-not ([IO.Path]::GetFullPath($inputRoot).StartsWith($repoRoot, [StringComparison]::OrdinalIgnoreCase))) { throw "InputDir must be inside repository: $InputDir" }
$runRoot = Join-Path $inputRoot (Join-Path "runs" $RunId)
if ($VerifyOnly) {
  if (-not (Test-Path -LiteralPath $runRoot -PathType Container)) { throw "VerifyOnly run directory not found: $runRoot" }
} else {
  if (Test-Path -LiteralPath $runRoot) { throw "Refusing to overwrite existing run directory: $runRoot" }
  New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
}
Assert-NoReparsePath $runRoot "Run directory"

if ($VerifyOnly) {
  $manifestPath = Join-Path $runRoot "manifest.json"
  if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "VerifyOnly manifest not found: $manifestPath" }
  & $Python (Resolve-Path -LiteralPath $Verifier).Path $manifestPath
  $verifyExit = $LASTEXITCODE
  if ($verifyExit -ne 0) { throw "Independent verification failed with exit code ${verifyExit}: $manifestPath" }
  Write-Output "verified_manifest: $manifestPath"
  Write-Output "verification_exit_code: 0"
  exit 0
}


$groups = @(
  [ordered]@{ name='daily_seed7'; prefix='d1'; output='d1_merged.json'; assets=@('adausdt_1d','avaxusdt_1d','bnbusdt_1d_deep','btcusdt_1d_deep','dogeusdt_1d','ethusdt_1d_deep','linkusdt_1d','ltcusdt_1d','solusdt_1d_deep','trxusdt_1d','xrpusdt_1d_deep') },
  [ordered]@{ name='four_hour_seed7'; prefix='h4'; output='h4_merged.json'; assets=@('bnbusdt_4h','btcusdt_4h_deep','ethusdt_4h_deep','solusdt_4h_deep','xrpusdt_4h') },
  [ordered]@{ name='daily_seed11'; prefix='seed11'; output='seed11_merged.json'; assets=@('btcusdt_1d_deep','ethusdt_1d_deep','solusdt_1d_deep') },
  [ordered]@{ name='daily_seed23'; prefix='seed23'; output='seed23_merged.json'; assets=@('btcusdt_1d_deep','ethusdt_1d_deep','solusdt_1d_deep') }
)

$records = @()
$currentGroup = $null
<<<<<<< Updated upstream
$verifierErrorText = $null
=======
>>>>>>> Stashed changes
try {
$pythonVersion = (& $Python --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersion -notmatch '^Python 3\.[0-9]+') { throw "Unable to validate Python interpreter: $Python ($pythonVersion)" }
$environment = [ordered]@{ cwd=$repo; os=[Environment]::OSVersion.VersionString; powershell=$PSVersionTable.PSVersion.ToString(); python=(Resolve-Path $Python).Path; python_version=$pythonVersion; python_sha256=(Hash $Python); evaluator=(Resolve-Path $Evaluator).Path; evaluator_sha256=(Hash $Evaluator); seed=$Seed; n_boot=$BootstrapCount }
foreach ($group in $groups) {
  $currentGroup = $group.name
  $shards = @()
  foreach ($asset in $group.assets) {
    $path = Join-Path $InputDir ("{0}_{1}.losses.npz" -f $group.prefix, $asset)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing explicit shard for $($group.name): $path" }
    Assert-RegularFile $path "Shard"
    $shards += [ordered]@{ path=(Resolve-Path $path).Path; relative=(Rel $path); bytes=(Get-Item $path).Length; sha256=(Hash $path) }
  }
  $output = Join-Path $runRoot $group.output
  $stdout = Join-Path $runRoot ("{0}.stdout.txt" -f $group.name)
  $stderr = Join-Path $runRoot ("{0}.stderr.txt" -f $group.name)
  $transcript = Join-Path $runRoot ("{0}.transcript.txt" -f $group.name)
  $args = @((Resolve-Path $Evaluator).Path,'--merge-parts') + @($shards | ForEach-Object { $_.path }) + @('--merge-out',$output,'--seed',[string]$Seed,'--n-boot',[string]$BootstrapCount)
  $commandText = (@($Python)+$args) -join ' '
  $commandRecord = Join-Path $runRoot ("{0}.command.json" -f $group.name)
  $record = [ordered]@{ group=$group.name; command=@($Python)+$args; command_text=$commandText; cwd=$repo; shards=$shards; stdout=$stdout; stderr=$stderr; transcript=$transcript; receipt=$output; command_record=$commandRecord; started_utc=(Get-Date).ToUniversalTime().ToString("o") }
  & $Python @args 1> $stdout 2> $stderr
  $record.exit_code = $LASTEXITCODE
  $outText = if (Test-Path $stdout) { Get-Content $stdout -Raw } else { '' }
  $errText = if (Test-Path $stderr) { Get-Content $stderr -Raw } else { '' }
  $transcriptText = @("COMMAND: $commandText", "EXIT_CODE: $($record.exit_code)", '--- STDOUT BEGIN ---', $outText, '--- STDOUT END ---', '--- STDERR BEGIN ---', $errText, '--- STDERR END ---') -join [Environment]::NewLine
  Write-AtomicUtf8NoBom $transcript ($transcriptText + [Environment]::NewLine)
  $record.stdout_sha256=Hash $stdout; $record.stderr_sha256=Hash $stderr; $record.transcript_sha256=Hash $transcript
  if ($record.exit_code -ne 0) { $record.status='failed'; Write-Json $commandRecord $record; throw "Merge failed for $($group.name), see $transcript" }
  Assert-RegularFile $output "Receipt"
  $losses = $output -replace "\.json$",".losses.npz"
  Assert-RegularFile $losses "Loss archive"
  $record.status="succeeded"; $record.receipt_sha256=Hash $output; $record.losses=$losses; $record.losses_sha256=Hash $losses
  Write-Json $commandRecord $record
  $records += $record
}

$expectedGroupNames = @($groups | ForEach-Object { $_.name })
$recordGroupNames = @($records | ForEach-Object { $_.group })
if ($recordGroupNames.Count -ne $expectedGroupNames.Count -or @($expectedGroupNames | Where-Object { $_ -notin $recordGroupNames }).Count -ne 0 -or @($recordGroupNames | Where-Object { $_ -notin $expectedGroupNames }).Count -ne 0) { throw "Finalizer group allowlist/count mismatch" }

$manifest = [ordered]@{
  schema='sota_finalization.v2'; status=if ($VerifyOnly) { 'verify_only' } else { 'complete' }; proof_status='UNPROVEN'; run_id=$RunId; created_at_utc=(Get-Date).ToUniversalTime().ToString('o'); repo=$repo; input_dir=(Resolve-Path $InputDir).Path; environment=$environment; environment_lock_sha256=(JsonHash $environment); evaluator_implementation_sha256=(Hash $Evaluator); groups=$records; interpretation='Hash-bound diagnostic real-data receipts; not universal SOTA proof, production proof, or independent third-party validation.'; limitations=@('Local model artifacts are not independently attested.','Bars are Binance archives rather than licensed point-in-time vendor vintages.','Merge integrity does not establish generalization beyond these workloads.','Industry-grade production evidence and live authorization are absent.')
}
$manifestPath = Join-Path $runRoot 'manifest.json'
Write-Json $manifestPath $manifest
<<<<<<< Updated upstream
$currentGroup = $null
$postStdout = Join-Path $runRoot 'postverify.stdout.txt'
$postStderr = Join-Path $runRoot 'postverify.stderr.txt'
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "A_preverify:$(Get-Date -Format o)"
  & $Python (Resolve-Path -LiteralPath $Verifier).Path $manifestPath 1> $postStdout 2> $postStderr
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "B_postverify:$(Get-Date -Format o)"
  $postVerifyExit = $LASTEXITCODE
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "C_exitcode:$(Get-Date -Format o)"
  if ($postVerifyExit -ne 0) {
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "D_innonzero:$(Get-Date -Format o)"
    if (Test-Path -LiteralPath $postStdout -PathType Leaf) {
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "E_precheck:$(Get-Date -Format o)"
      $verifierErrorText = (Get-Content -LiteralPath $postStdout -Raw)
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "F_gotcontent:$(Get-Date -Format o)"
      if ($verifierErrorText -and $verifierErrorText.Length -gt 4000) { $verifierErrorText = $verifierErrorText.Substring($verifierErrorText.Length - 4000) }
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "G_substrdone:$(Get-Date -Format o)"
    }
Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "H_prethrow:$(Get-Date -Format o)"
    throw ("Independent postcondition verification failed with exit code {0}: {1}" -f $postVerifyExit, $manifestPath)
}
=======
& $Python (Resolve-Path -LiteralPath $Verifier).Path $manifestPath
$postVerifyExit = $LASTEXITCODE
if ($postVerifyExit -ne 0) { throw ("Independent postcondition verification failed with exit code {0}: {1}" -f $postVerifyExit, $manifestPath) }
>>>>>>> Stashed changes
Write-Output "manifest: $manifestPath"
Write-Output "groups: $($records.Count)"
Write-Output "status: $($manifest.status)"
Write-Output "proof_status: UNPROVEN"
} catch {
<<<<<<< Updated upstream
  Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "I_catch:$(Get-Date -Format o)"
=======
>>>>>>> Stashed changes
  $message = $_.Exception.Message
  if (-not $VerifyOnly -and (Test-Path -LiteralPath $runRoot -PathType Container)) {
    $failure = [ordered]@{
      schema='sota_finalization.v2'
      status='failed'
      proof_status='UNPROVEN'
      run_id=$RunId
      created_at_utc=(Get-Date).ToUniversalTime().ToString('o')
      repo=$repo
      input_dir=$inputRoot
      failing_group=$currentGroup
      error=$message
<<<<<<< Updated upstream
      verifier_error=$verifierErrorText
=======
>>>>>>> Stashed changes
      groups=$records
      interpretation='Finalization failed before a complete diagnostic receipt was committed.'
    }
    $failurePath = Join-Path $runRoot 'manifest.json'
    try {
<<<<<<< Updated upstream
      Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "J_prewrite:$(Get-Date -Format o)"
      Write-Json $failurePath $failure
      Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "K_postwrite:$(Get-Date -Format o)"
=======
      Write-Json $failurePath $failure
>>>>>>> Stashed changes
    } catch {
      $failureFallback = [ordered]@{
        schema='sota_finalization.v2'
        status='failed'
        proof_status='UNPROVEN'
        run_id=$RunId
        created_at_utc=(Get-Date).ToUniversalTime().ToString('o')
        repo=$repo
        input_dir=$inputRoot
        failing_group=$currentGroup
        error=$message
        interpretation='Finalization failed before a complete diagnostic receipt was committed.'
      }
      try {
        Write-Utf8NoBom $failurePath (($failureFallback | ConvertTo-Json -Depth 20) + [Environment]::NewLine)
      } catch {
        $fallbackError = $_.Exception.Message
        Write-Output ("Unable to persist failure manifest: " + $fallbackError)
      }
    }
  }
<<<<<<< Updated upstream
  Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "L_preerror:$(Get-Date -Format o)"
  Write-Error $message
  Add-Content -LiteralPath C:\Users\me\fin_markers.log -Value "M_preexit:$(Get-Date -Format o)"
=======
  Write-Error $message
>>>>>>> Stashed changes
  exit 1
}
