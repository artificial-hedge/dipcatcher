$ErrorActionPreference = 'Stop'
cd D:\dipcatcher
$shards = @(
  'd1_bnbusdt_1d_deep','d1_ethusdt_1d_deep','d1_solusdt_1d_deep','d1_xrpusdt_1d_deep',
  'h4f_bnbusdt_4h_deep','h4f_btcusdt_4h_deep','h4f_ethusdt_4h_deep','h4f_solusdt_4h_deep','h4f_xrpusdt_4h_deep'
)
foreach ($s in $shards) {
  .venv\Scripts\python.exe scripts\_har_col.py --shard ".dsh-24x7/eval-full/$s.v2aug.npz" --bars-root data/raw/sources --out ".dsh-24x7/lane-har/$s.harcol.npz"
}
Write-Output 'HAR_DONE'
