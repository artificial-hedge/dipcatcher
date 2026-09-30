import sys, traceback
sys.path.insert(0, r"D:\dipcatcher\src")
try:
    import numba
    print("numba:", numba.__version__)
except Exception:
    print("numba import FAILED:")
    traceback.print_exc()
try:
    import quant_fund.backtest._fast_kernel as k
    print("kernel module imported; HAVE_NUMBA =", k.HAVE_NUMBA)
    print("replay_driver:", k.replay_driver)
except Exception:
    print("kernel import FAILED:")
    traceback.print_exc()
import quant_fund.backtest.fast_replay as fr
print("fr._nb_driver =", fr._nb_driver)
