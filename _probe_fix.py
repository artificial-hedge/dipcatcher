import sys, numpy as np
sys.path.insert(0,"src")
from quant_fund.models.covariance import dcc_gaussian, dcc_student_t, adcc, agdcc

def cand(rows, assets, seed, corr):
    rng = np.random.default_rng(seed)
    x = 0.001 + rng.normal(scale=0.01, size=(rows, assets))
    if corr: x[:,1] += 0.5*x[:,0]
    return x

best = []
for seed in (21, 31, 7, 0, 5):
  for (rows, assets) in ((240,3),(300,3),(200,2),(300,2),(240,2)):
    for corr in (False, True):
        x = cand(rows,assets,seed,corr)
        try:
            h,p = dcc_gaussian(x)
            ok = np.isfinite(h).all() and p["success"]==1.0
            if ok: best.append((seed,rows,assets,corr,float(p["a"]),float(p["b"])))
        except Exception as e:
            pass
print("WORKING fixtures (seed,rows,assets,corr,a,b):")
for b in best: print("  ",b)
