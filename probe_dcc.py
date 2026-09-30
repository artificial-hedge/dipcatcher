import sys
sys.path.insert(0, r"D:\dipcatcher\src")
import numpy as np
from quant_fund.models.covariance import dcc_gaussian
rng = np.random.default_rng(11)
try:
    sigma, params = dcc_gaussian(rng.normal(size=(240, 4)) * 0.01)
    print("OK sigma finite:", np.isfinite(sigma).all())
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:140])
import arch
print("arch", arch.__version__)
