import sys, datetime as dt
sys.path.insert(0, r"D:\dipcatcher\src")
import polars as pl
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig
from quant_fund.backtest.engine import run_backtest, StaleValuationError

n = 40
bars = pl.DataFrame({
    "security_id": ["x"] * n,
    "event_time": [dt.datetime(2024,1,1)+dt.timedelta(days=i) for i in range(n)],
    "open": [100.0+i for i in range(n)],
    "close": [100.5+i for i in range(n)],
    "close_total_return": [100.5+i for i in range(n)],
    "volume": [1e6]*n,
    "source": ["file"]*n,
})
dead = [dt.datetime(2024,1,1)+dt.timedelta(days=i) for i in range(30, n)]
stale = bars.with_columns(
    pl.when(pl.col("event_time").is_in(dead)).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col("close")).alias("close"),
    pl.when(pl.col("event_time").is_in(dead)).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col("close_total_return")).alias("close_total_return"),
)
w = pl.DataFrame({
    "event_time": [dt.datetime(2024,1,1)+dt.timedelta(days=i) for i in range(n-1)],
    "security_id": ["x"]*(n-1),
    "target_weight": [0.5]*(n-1),
})
cfg = AppConfig(
    costs=CostConfig(commission_bps=10.0, half_spread_bps=0.0, impact_y=0.0,
                     bps_per_turnover=0.0, borrow_bps_per_year=0.0,
                     frictionless=False, participation_limit=1.0),
    risk_gate=RiskGateConfig(max_order_notional=1e12, max_gross=100.0,
                             max_net=100.0, max_name=1.0, max_participation=1.0,
                             max_predicted_vol=100.0, stale_price_bars=3,
                             stale_model_hours=1e9))
try:
    r = run_backtest(stale, w, cfg, initial_nav=1e6)
    print("COMPLETED — nav rows:", r.equity.height, "fills:", r.fills.height)
    print(r.fills)
except StaleValuationError as e:
    print("STALE:", e)
except Exception as e:
    print(type(e).__name__, ":", e)
