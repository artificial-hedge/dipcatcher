$ErrorActionPreference = "Continue"
Set-Location D:\dipcatcher
$py = "D:\dipcatcher\.venv\Scripts\python.exe"
$outdir = "D:\dipcatcher\.dsh-24x7\lane-lgbmqv"
New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$shards = @(
  "d1_bnbusdt_1d_deep.v2aug.npz",
  "d1_btcusdt_1d_deep.v2aug.npz",
  "d1_ethusdt_1d_deep.v2aug.npz",
  "d1_solusdt_1d_deep.v2aug.npz",
  "d1_xrpusdt_1d_deep.v2aug.npz",
  "h4f_bnbusdt_4h_deep.v2aug.npz",
  "h4f_btcusdt_4h_deep.v2aug.npz",
  "h4f_ethusdt_4h_deep.v2aug.npz",
  "h4f_solusdt_4h_deep.v2aug.npz",
  "h4f_xrpusdt_4h_deep.v2aug.npz"
)
foreach ($s in $shards) {
  $out = Join-Path $outdir ($s -replace "\.npz$", ".lgbmqv.npz")
  if (Test-Path $out) { echo "skip $s (exists)"; continue }
  echo "=== $s start $(Get-Date -Format o)"
  & $py scripts\_lgbmqv_col.py --shard ".dsh-24x7\eval-full\$s" --bars-root data\raw\sources --out $out
  echo "=== $s done rc=$LASTEXITCODE $(Get-Date -Format o)"
}
echo "ALL_DONE"
