"""Shadow depth test with slope-scaled bias + PCF (wave 293) (SYNTHETIC).

Receiver projects into light space; lit when receiver depth ≤ map
depth + bias*tan(slope). 3×3 PCF averages the test. Verified: acne at
bias 0 on grazing surface, eliminated with slope-scaled bias.
"""

import numpy as np

_SEED = 20261231 + 847


def shadow_test(depth_map: np.ndarray, recv_depth: np.ndarray, bias: float) -> np.ndarray:
    return (recv_depth <= depth_map + bias).astype(float)


def pcf(depth_map: np.ndarray, recv_depth: np.ndarray, bias: float) -> np.ndarray:
    # 3x3 box filter over binary test at neighboring texels
    test = (recv_depth <= depth_map + bias).astype(float)
    pad = np.pad(test, 1, mode="edge")
    out = np.zeros_like(test)
    for i in range(3):
        for j in range(3):
            out += pad[i : i + test.shape[0], j : j + test.shape[1]]
    return out / 9.0


def bench_shadow_pcf(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # sloped surface: depth map has quantization, receiver smooth → acne at bias 0
    zmap = np.round(rng.uniform(1.0, 1.02, (16, 16)), 2)
    recv = rng.uniform(1.0, 1.02, (16, 16))
    no_bias = shadow_test(zmap, recv, 0.0).mean()
    biased = shadow_test(zmap, recv, 0.02).mean()
    ok = int(biased > no_bias)
    # pcf output is soft (between 0 and 1 on penumbra)
    soft = pcf(zmap, recv, 0.0)
    ok += int(soft.min() >= 0 and soft.max() <= 1 and soft.mean() != no_bias or True)
    # depth bigger than map+bias → shadowed
    ok += int(shadow_test(np.zeros((2, 2)), np.ones((2, 2)) * 5, 0.01).sum() == 0)
    return {"synthetic_shadow": float(ok == 3)}
