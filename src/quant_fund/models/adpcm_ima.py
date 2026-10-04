"""IMA ADPCM: 4-bit adaptive differential PCM encoder/decoder.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 903

_STEP = np.array(
    [
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        16,
        17,
        19,
        21,
        23,
        25,
        28,
        31,
        34,
        37,
        41,
        45,
        50,
        55,
        60,
        66,
        73,
        80,
        88,
        97,
        107,
        118,
        130,
        143,
        157,
        173,
        190,
        209,
        230,
        253,
        279,
        307,
        337,
        371,
        408,
        449,
        494,
        544,
        598,
        658,
        724,
        796,
        876,
        963,
        1060,
        1166,
        1282,
        1411,
        1552,
        1707,
        1878,
        2066,
        2272,
        2499,
        2749,
        3024,
        3327,
        3660,
        4026,
        4428,
        4871,
        5358,
        5894,
        6484,
        7132,
        7845,
        8630,
        9493,
        10442,
        11487,
        12635,
        13899,
        15289,
        16818,
        18500,
        20350,
        22385,
        24623,
        27086,
        29794,
        32767,
    ],
    dtype=np.float64,
)
_IDX = np.array([-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8], dtype=np.float64)


def adpcm_encode(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    codes = np.zeros(n, dtype=np.int64)
    pred, idx = 0.0, 0.0
    for i in range(n):
        step = _STEP[int(idx)]
        d = x[i] - pred
        code = 0
        if d < 0:
            code, d = 8, -d
        if d >= step:
            code |= 4
            d -= step
        if d >= step / 2:
            code |= 2
            d -= step / 2
        if d >= step / 4:
            code |= 1
        dq = step / 8
        if code & 4:
            dq += step
        if code & 2:
            dq += step / 2
        if code & 1:
            dq += step / 4
        pred += dq if not (code & 8) else -dq
        pred = float(np.clip(pred, -32768.0, 32767.0))
        idx = float(np.clip(idx + _IDX[code], 0.0, 88.0))
        codes[i] = code
    return codes


def adpcm_decode(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    out = np.zeros(codes.size)
    pred, idx = 0.0, 0.0
    for i, c in enumerate(codes):
        step = _STEP[int(idx)]
        dq = step / 8
        if c & 4:
            dq += step
        if c & 2:
            dq += step / 2
        if c & 1:
            dq += step / 4
        pred += dq if not (c & 8) else -dq
        pred = float(np.clip(pred, -32768.0, 32767.0))
        idx = float(np.clip(idx + _IDX[c], 0.0, 88.0))
        out[i] = pred
    return np.asarray(out, dtype=np.float64)


def bench_adpcm_ima(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    # smooth AR(1) signal: ADPCM tracks well
    x = np.zeros(4000)
    for i in range(1, 4000):
        x[i] = 0.97 * x[i - 1] + 400.0 * rng.normal()
    x *= 0.9 * 32767.0 / np.max(np.abs(x))
    x -= np.mean(x)
    xh = adpcm_decode(adpcm_encode(x))
    err = float(np.sqrt(np.mean((x - xh) ** 2)))
    sn = float(10 * np.log10(np.mean(x**2) / (err**2 + 1e-9)))
    score += 1.0 if sn > 15.0 else 0.0
    # 4 bits per sample: codes all in [0,15]
    c = adpcm_encode(x)
    score += 1.0 if np.all((c >= 0) & (c <= 15)) else 0.0
    # step adaptation: louder segment → bigger steps (compare reconstruction granularity)
    quiet = np.full(500, 200.0)
    loud = np.full(500, 20000.0)
    ch = adpcm_decode(adpcm_encode(np.concatenate([quiet, loud])))
    # overshoot/step size at loud part start vs steady quiet part
    dq_quiet = np.abs(np.diff(ch[:400])).max()
    dq_loud = np.abs(np.diff(ch[500:900])).max()
    score += 1.0 if dq_loud > 10.0 * dq_quiet else 0.0
    # DC reconstruction accuracy at steady state
    score += 1.0 if abs(ch[-1] - 20000.0) < 3000.0 else 0.0
    return {"synthetic_adpcm_ima": score / 4.0}
