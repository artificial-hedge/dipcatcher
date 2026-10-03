"""Arrhenius accelerated-life model from stress-temperature test data.

Fits log-life = log(C) + Ea/(k_B T) by least squares on the reciprocal
Kelvin scale, then extrapolates the use-condition life. Bench compares
the fitted activation energy and the extrapolated use life against the
planted values in the shared fixture.
"""

import numpy as np

from quant_fund.models._rel_synth import ARR_C, ARR_EA, ARR_USE, KB, arrhenius_lives


def _fit(temps: np.ndarray, lives: np.ndarray) -> tuple[float, float]:
    x = 1.0 / (KB * temps)
    y = np.log(lives)
    a = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(np.exp(coef[0])), float(coef[1])


def bench_life_stress(seed: int = 4909) -> dict[str, float]:
    temps, lives = arrhenius_lives(seed)
    c_hat, ea_hat = _fit(temps, lives)
    use_hat = c_hat * np.exp(ea_hat / (KB * ARR_USE))
    use_true = ARR_C * np.exp(ARR_EA / (KB * ARR_USE))
    return {
        "synthetic_arr_ea": ea_hat,
        "synthetic_arr_ea_err": abs(ea_hat - ARR_EA),
        "synthetic_arr_use_life": use_hat,
        "synthetic_arr_use_rel": abs(use_hat / use_true - 1.0),
        "synthetic_arr_af_400": float(np.exp(ea_hat * (1 / (KB * ARR_USE) - 1 / (KB * 400.0)))),
    }
