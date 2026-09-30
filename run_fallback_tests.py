import sys
sys.path.insert(0, r"D:\dipcatcher\src")
import quant_fund.backtest.fast_replay as fr
fr._nb_driver = None  # force interpreted fallback (numba-absent path)
import quant_fund.backtest._fast_kernel as k
k.HAVE_NUMBA = False
import pytest
sys.exit(pytest.main(["tests/unit/test_fast_replay.py", "-x", "-q"]))
