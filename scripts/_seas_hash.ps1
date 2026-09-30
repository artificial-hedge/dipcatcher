$ErrorActionPreference = "Stop"
Set-Location D:\dipcatcher\.dsh-24x7\eval-full
$names = @(
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
foreach ($n in $names) {
  $h = (Get-FileHash -Algorithm SHA256 $n).Hash.ToLower()
  Write-Output "$n $h"
}
