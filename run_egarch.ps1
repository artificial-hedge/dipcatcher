$ErrorActionPreference = "Stop"
Set-Location D:/dipcatcher
New-Item -ItemType Directory -Force -Path .dsh-24x7/lane-egarch | Out-Null
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
  $out = ".dsh-24x7/lane-egarch/egarch_$s"
  Write-Output "=== $s ==="
  .venv/Scripts/python.exe scripts/_egarch_col.py --shard ".dsh-24x7/eval-full/$s" --bars-root data/raw/sources --out $out
  if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $s exit=$LASTEXITCODE" }
}
Write-Output "ALL_DONE"
