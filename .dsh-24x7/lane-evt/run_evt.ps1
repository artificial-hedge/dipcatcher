Set-Location D:\dipcatcher
$py = 'D:\dipcatcher\.venv\Scripts\python.exe'
$shards = @('d1_bnbusdt_1d_deep','d1_btcusdt_1d_deep','d1_ethusdt_1d_deep','d1_solusdt_1d_deep','d1_xrpusdt_1d_deep','h4f_bnbusdt_4h_deep','h4f_btcusdt_4h_deep','h4f_ethusdt_4h_deep','h4f_solusdt_4h_deep','h4f_xrpusdt_4h_deep')
foreach ($s in $shards) {
  $shard = ".dsh-24x7\eval-full\$s.v2aug.npz"
  $out = ".dsh-24x7\lane-evt\$s.dip_evt.npz"
  & $py scripts\_evt_col.py --shard $shard --bars-root data\raw\sources --out $out
}
Write-Output 'EVT_LANE_DONE'
