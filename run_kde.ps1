$ErrorActionPreference = "Stop"
$py = "D:\dipcatcher\.venv\Scripts\python.exe"
$shards = @(
  "d1_bnbusdt_1d_deep",
  "d1_btcusdt_1d_deep",
  "d1_ethusdt_1d_deep",
  "d1_solusdt_1d_deep",
  "d1_xrpusdt_1d_deep",
  "h4f_bnbusdt_4h_deep",
  "h4f_btcusdt_4h_deep",
  "h4f_ethusdt_4h_deep",
  "h4f_solusdt_4h_deep",
  "h4f_xrpusdt_4h_deep"
)
New-Item -ItemType Directory -Force -Path "D:\dipcatcher\.dsh-24x7\lane-kde" | Out-Null
foreach ($s in $shards) {
  $shard = "D:\dipcatcher\.dsh-24x7\eval-full\$s.v2aug.npz"
  $out = "D:\dipcatcher\.dsh-24x7\lane-kde\$s.kde.npz"
  Write-Output "=== $s ==="
  & $py D:\dipcatcher\scripts\_kde_col.py --shard $shard --bars-root D:\dipcatcher\data\raw\sources --out $out
  if ($LASTEXITCODE -ne 0) { throw "kde col failed on $s" }
}
Write-Output "ALL_DONE"
