import sys, time
sys.path.insert(0, r"D:\dipcatcher\src")
import numpy as np, polars as pl
import quant_fund.backtest.fast_replay as fr
from quant_fund.backtest.engine import run_backtest
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig

SIDS = ["adausdt","avaxusdt","bnbusdt","btcusdt","dogeusdt","ethusdt","linkusdt","ltcusdt","solusdt","trxusdt","xrpusdt"]
frames = {}
for s in SIDS:
    f = pl.read_parquet(rf"D:\dipcatcher\data\raw\sources\{s}_1d.parquet").sort("event_time")
    if "close_total_return" not in f.columns:
        f = f.with_columns(pl.col("close").alias("close_total_return"))
    frames[f["security_id"][0]] = f
closes = {sid: f["close"].to_numpy() for sid, f in frames.items()}
times = next(iter(frames.values()))["event_time"].to_list()
rows = []
n = len(frames)
for i in range(len(times)):
    if i + 1 >= 20:
        gated = [s for s, c in closes.items() if c[i] > float(np.mean(c[i+1-20:i+1]))]
        for sid in gated:
            rows.append({"event_time": times[i], "security_id": sid, "target_weight": 0.9/n})
weights = pl.DataFrame(rows, schema={"event_time": pl.Datetime("us","UTC"),"security_id": pl.String,"target_weight": pl.Float64})
bars = pl.concat(list(frames.values()))
cfg = AppConfig(
    costs=CostConfig(commission_bps=10.0, half_spread_bps=0.0, impact_y=0.0,
                     bps_per_turnover=0.0, borrow_bps_per_year=0.0,
                     frictionless=False, participation_limit=1.0),
    risk_gate=RiskGateConfig(max_order_notional=1e12, max_gross=100.0,
                             max_net=100.0, max_name=1.0,
                             max_participation=1.0, max_predicted_vol=100.0,
                             stale_price_bars=3, stale_model_hours=1e9))

# warm
r = fr.run_backtest_fast(bars, weights, cfg, initial_nav=1e6)

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
for _ in range(5):
    fr.run_backtest_fast(bars, weights, cfg, initial_nav=1e6)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
print(s.getvalue())
