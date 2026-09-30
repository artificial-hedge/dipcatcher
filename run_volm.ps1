$ErrorActionPreference = 'Stop'
Set-Location D:\dipcatcher
New-Item -ItemType Directory -Force .dsh-24x7\lane-volm | Out-Null
$shards = @(
  'd1_bnbusdt_1d_deep','d1_btcusdt_1d_deep','d1_ethusdt_1d_deep','d1_solusdt_1d_deep','d1_xrpusdt_1d_deep',
  'h4f_bnbusdt_4h_deep','h4f_btcusdt_4h_deep','h4f_ethusdt_4h_deep','h4f_solusdt_4h_deep','h4f_xrpusdt_4h_deep'
)
foreach ($s in $shards) {
  $shard = ".dsh-24x7\eval-full\$s.v2aug.npz"
  $out = ".dsh-24x7\lane-volm\$s.volm.npz"
  & .venv\Scripts\python.exe scripts\_volm_col.py --shard $shard --bars-root data\raw\sources --out $out
  if ($LASTEXITCODE -ne 0) { Write-Error "FAILED $s"; exit 1 }
}
Write-Output "ALL_DONE"
