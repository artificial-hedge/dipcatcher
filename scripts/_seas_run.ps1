$ErrorActionPreference = "Stop"
Set-Location D:\dipcatcher
$py = "D:\dipcatcher\.venv\Scripts\python.exe"
$outdir = "D:\dipcatcher\.dsh-24x7\lane-seas"
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
  $base = $s -replace "\.v2aug\.npz$", ""
  $shard = "D:\dipcatcher\.dsh-24x7\eval-full\$s"
  $out = "$outdir\$base.seascol.npz"
  & $py scripts\_seas_col.py --shard $shard --bars-root data\raw\sources --out $out
  if ($LASTEXITCODE -ne 0) { throw "failed on $s" }
}
Write-Output "SEAS_DONE"
