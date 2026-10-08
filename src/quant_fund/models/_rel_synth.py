"""Shared synthetic fixtures for the wave-209 reliability canon (SYNTHETIC)."""

import numpy as np

WB_BETA = 1.8
WB_ETA = 1200.0
WB_CENSOR = 900.0

ARR_EA = 0.65  # eV activation energy
ARR_C = 40.0  # prefactor hours
KB = 8.617e-5  # eV/K
ARR_TEMPS = np.array([373.15, 398.15, 423.15])  # stress temps, K
ARR_USE = 333.15  # use-condition temp, K

# fault tree: TOP = OR(AND(a,b), AND(c,d), e)
TREE_P = {"a": 0.10, "b": 0.20, "c": 0.15, "d": 0.10, "e": 0.05}

# repairable parallel system: two identical units, one repair crew
RAM_LAM = 0.01
RAM_MU = 0.20

RBD_T = 100.0
RBD_LAM = np.array([0.004, 0.006, 0.005])


def weibull_lifetimes(seed: int, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Right-censored Weibull lifetimes; returns (times, failed flags)."""
    rng = np.random.default_rng(seed)
    t = WB_ETA * (-np.log(rng.uniform(size=n))) ** (1 / WB_BETA)
    failed = t <= WB_CENSOR
    return np.minimum(t, WB_CENSOR), failed


def arrhenius_lives(seed: int, n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Lives at each stress temp: log-normal around C*exp(Ea/kT)."""
    rng = np.random.default_rng(seed)
    out = []
    for temp in ARR_TEMPS:
        mean = ARR_C * np.exp(ARR_EA / (KB * temp))
        out.append(mean * np.exp(rng.normal(0.0, 0.30, n)))
    return np.repeat(ARR_TEMPS, n), np.concatenate(out)


def fault_tree() -> dict:
    """TOP = OR(AND(a,b), AND(c,d), e) — a 3-branch fault tree."""
    return {
        "gate": "or",
        "children": [
            {"gate": "and", "children": ["a", "b"]},
            {"gate": "and", "children": ["c", "d"]},
            "e",
        ],
    }


def fault_tree_top_prob() -> float:
    p = TREE_P
    ab, cd, e = p["a"] * p["b"], p["c"] * p["d"], p["e"]
    return 1.0 - (1 - ab) * (1 - cd) * (1 - e)


def mc_system_state(rng: np.random.Generator, up: int = 2) -> int:
    """Number of up units for RAM system sample (unused helper kept small)."""
    return up
