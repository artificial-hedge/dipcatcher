"""Float kernels whose bits do not follow the host CPU.

NumPy's AVX-512 ``exp`` / ``pow`` and OpenBLAS's SkylakeX ``ddot`` disagree
with the libm and Haswell kernels in the last bit. The sealed lane receipts
were written with the former. These routines reproduce those bits on every
host so a replay writes the pinned file instead of a sibling hash.

``exp`` is the published range reduction (ln2 split, degree-6 polynomial,
16-entry ``2^(j/16)`` table) evaluated with the same rounding modes as that
kernel. ``dot`` is the SkylakeX OpenBLAS accumulation: 32-wide FMA chains,
folded to 256-bit, then a scalar FMA tail. ``pow`` for the rough-vol kernel
exponent pins the 36 bases where the two ``pow`` kernels differ by one ulp.
"""

from __future__ import annotations

import ctypes
import math
import struct
import sys
from decimal import Decimal, localcontext
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

_INV_LN2 = float.fromhex("0x1.71547652b82fep+0")
_MAGIC = float.fromhex("0x1.8000000003ff0p+48")
_LN2_HI = float.fromhex("0x1.62e42fefa39efp-1")
_LN2_LO = float.fromhex("0x1.abc9e3b39803fp-56")
_EXP_MASK = 0xBFFFFFFFFFFFFFFF
_EXP_LIMIT = float.fromhex("0x1.61da04cbafe44p+9")
_C6 = float.fromhex("0x1.7411836940c04p-10")
_C5 = float.fromhex("0x1.1101cbbc265c0p-7")
_C4 = float.fromhex("0x1.55557242d68fep-5")
_C3 = float.fromhex("0x1.5555553939732p-3")
_C2 = float.fromhex("0x1.000000000d008p-1")
_C1 = float.fromhex("0x1.fffffffffff70p-1")
_TOP = tuple(
    float.fromhex(h)
    for h in (
        "0x1.0000000000000p+0",
        "0x1.0b5586cf9890fp+0",
        "0x1.172b83c7d517bp+0",
        "0x1.2387a6e756238p+0",
        "0x1.306fe0a31b715p+0",
        "0x1.3dea64c123422p+0",
        "0x1.4bfdad5362a27p+0",
        "0x1.5ab07dd485429p+0",
        "0x1.6a09e667f3bcdp+0",
        "0x1.7a11473eb0187p+0",
        "0x1.8ace5422aa0dbp+0",
        "0x1.9c49182a3f090p+0",
        "0x1.ae89f995ad3adp+0",
        "0x1.c199bdd85529cp+0",
        "0x1.d5818dcfba487p+0",
        "0x1.ea4afa2a490dap+0",
    )
)
_TAIL = tuple(
    float.fromhex(h)
    for h in (
        "0x0.0p+0",
        "0x1.79aa65d837b6dp-54",
        "-0x1.01b15eaa59348p-55",
        "0x1.68efde3a8a894p-54",
        "0x1.34d754db0abb6p-55",
        "0x1.59f48a72a4c6dp-55",
        "0x1.690cebb7aafb0p-56",
        "0x1.063e1e21c5409p-54",
        "-0x1.3b3efbf5e2228p-54",
        "-0x1.b32dcb94da51dp-56",
        "0x1.db72fc1f0eab4p-55",
        "0x1.1affc2b91ce27p-56",
        "0x1.c1a7792cb3387p-55",
        "0x1.36eae30af0cb3p-56",
        "0x1.4a385a63d07a7p-56",
        "-0x1.ff7128fd391f0p-55",
    )
)

# Rough-vol kernel exponent ``0.1 - 0.5``. Bases whose AVX-512 pow differs
# from libm by one ulp; the value is the pinned kernel output.
_POW_EXP = float.fromhex("-0x1.999999999999ap-2")
_POW_PIN: dict[int, float] = {
    34: float.fromhex("0x1.f3bbc0c2ac01cp-3"),
    96: float.fromhex("0x1.49ee032821817p-3"),
    97: float.fromhex("0x1.4890a1d8112a9p-3"),
    121: float.fromhex("0x1.2cc1cd79ce86bp-3"),
    135: float.fromhex("0x1.1fdeb9ee95547p-3"),
    191: float.fromhex("0x1.f520283749c24p-4"),
    200: float.fromhex("0x1.ebfb0683cd851p-4"),
    201: float.fromhex("0x1.eb00026f98df4p-4"),
    206: float.fromhex("0x1.e632a8308ca7ep-4"),
    253: float.fromhex("0x1.bfd4194c51bfap-4"),
    258: float.fromhex("0x1.bc5629f1e9e2bp-4"),
    266: float.fromhex("0x1.b6f13130ca5b4p-4"),
    267: float.fromhex("0x1.b648a8c577acep-4"),
    297: float.fromhex("0x1.a401fff845c61p-4"),
    300: float.fromhex("0x1.a2529d5c76223p-4"),
    324: float.fromhex("0x1.95a41ea5fc2d3p-4"),
    329: float.fromhex("0x1.9329f2600b019p-4"),
    333: float.fromhex("0x1.91383e9ae4721p-4"),
    334: float.fromhex("0x1.90bd200dd8499p-4"),
    349: float.fromhex("0x1.89c2234427223p-4"),
    353: float.fromhex("0x1.87f7ae7c17bdcp-4"),
    354: float.fromhex("0x1.878633e1634d5p-4"),
    385: float.fromhex("0x1.7a987e9a38a7fp-4"),
    395: float.fromhex("0x1.76bb76f108f10p-4"),
    418: float.fromhex("0x1.6e5823db0379fp-4"),
    439: float.fromhex("0x1.673b34d8d5ff8p-4"),
    443: float.fromhex("0x1.65ee27c3ec184p-4"),
    446: float.fromhex("0x1.64f71e14ee4f8p-4"),
    476: float.fromhex("0x1.5bca4171b3d5ep-4"),
    493: float.fromhex("0x1.56f140fe39550p-4"),
    526: float.fromhex("0x1.4e2b29ee192b1p-4"),
    541: float.fromhex("0x1.4a6e62d777dd6p-4"),
    545: float.fromhex("0x1.49757f972d7c8p-4"),
    548: float.fromhex("0x1.48bc813913c33p-4"),
    565: float.fromhex("0x1.44be5b00ab0a7p-4"),
    588: float.fromhex("0x1.3f9a05f6de856p-4"),
}

_FMA: Any = None
_FMA_READY = False
_CONV_OK: bool | None = None
_EXP_OK: bool | None = None


def _bind_fma() -> Any:
    names = ("ucrtbase", "msvcrt") if sys.platform == "win32" else ("libm.so.6",)
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        fn = getattr(lib, "fma", None)
        if fn is None:
            continue
        fn.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
        fn.restype = ctypes.c_double
        return fn
    return None


def _fma_rn(a: float, b: float, c: float) -> float:
    global _FMA, _FMA_READY
    if not _FMA_READY:
        _FMA = _bind_fma()
        _FMA_READY = True
    if _FMA is not None:
        return float(_FMA(float(a), float(b), float(c)))
    with localcontext() as ctx:
        ctx.prec = 80
        exact = Decimal.from_float(float(a)) * Decimal.from_float(float(b))
        exact += Decimal.from_float(float(c))
        return float(+exact)


def _fma_rz(a: float, b: float, c: float) -> float:
    """FMA rounded toward zero. The exp reduction's first step uses this mode."""
    rn = _fma_rn(a, b, c)
    with localcontext() as ctx:
        ctx.prec = 80
        exact = Decimal.from_float(float(a)) * Decimal.from_float(float(b))
        exact += Decimal.from_float(float(c))
        exact = +exact
    if not exact.is_finite():
        return rn
    rounded = Decimal.from_float(rn)
    if rounded == exact or abs(rounded) <= abs(exact):
        return rn
    return math.nextafter(rn, 0.0)


def _exp_reduce(x: float) -> tuple[float, float, int]:
    k_magic = _fma_rz(_INV_LN2, x, _MAGIC)
    k = float(k_magic - _MAGIC)
    bits = struct.unpack("<Q", struct.pack("<d", k_magic))[0]
    r = _fma_rn(-_LN2_HI, k, x)
    r = _fma_rn(-_LN2_LO, k, r)
    masked = struct.unpack("<Q", struct.pack("<d", r))[0] & _EXP_MASK
    reduced = struct.unpack("<d", struct.pack("<Q", masked))[0]
    return k, reduced, bits & 0xF


def _exp_poly(reduced: float, idx: int) -> float:
    r2 = reduced * reduced
    poly = _fma_rn(_C6, reduced, _C5)
    quad = _fma_rn(_C4, reduced, _C3)
    scale = _fma_rn(_C2, reduced, _C1)
    poly = _fma_rn(poly, r2, quad)
    poly = _fma_rn(poly, r2, scale)
    poly = _fma_rn(reduced, poly, _TAIL[idx])
    return _fma_rn(poly, _TOP[idx], _TOP[idx])


def stable_exp(x: float) -> float:
    """Double exp matching the pinned AVX-512 kernel, including its last bit."""
    x = float(x)
    if math.isnan(x):
        return x
    if math.isinf(x):
        return x if x > 0.0 else 0.0
    if abs(x) >= _EXP_LIMIT:
        return math.exp(x)
    k, reduced, idx = _exp_reduce(x)
    return math.ldexp(_exp_poly(reduced, idx), int(math.floor(k)))


def stable_exp_array(values: ArrayLike) -> Array:
    """Elementwise :func:`stable_exp`, preserving shape."""
    arr = np.asarray(values, dtype=float)
    if _host_exp_matches():
        return np.exp(arr)
    flat = np.array([stable_exp(float(v)) for v in arr.ravel()], dtype=float)
    return flat.reshape(arr.shape)


def _host_exp_matches() -> bool:
    """True when this host's ``np.exp`` already emits the pinned kernel bits."""
    global _EXP_OK
    if _EXP_OK is None:
        probe = float.fromhex("0x1.d110e00e6687ap+3")
        pinned = stable_exp(probe)
        _EXP_OK = float(np.exp(probe)) == pinned and math.exp(probe) != pinned
    return _EXP_OK


def _pair(left: ArrayLike, right: ArrayLike) -> tuple[list[float], list[float]]:
    a = np.ascontiguousarray(left, dtype=float).ravel()
    b = np.ascontiguousarray(right, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("stable_dot inputs must have the same length")
    return a.tolist(), b.tolist()


def _fma_into(
    acc: list[list[float]], x: list[float], y: list[float], start: int, width: int
) -> None:
    for vec, row in enumerate(acc):
        base = start + width * vec
        for lane in range(width):
            row[lane] = _fma_rn(x[base + lane], y[base + lane], row[lane])


def _dot_wide(x: list[float], y: list[float], n1: int) -> list[list[float]]:
    acc = [[0.0] * 8 for _ in range(4)]
    i = 0
    n32 = n1 & ~31
    while i < n32:
        _fma_into(acc, x, y, i, 8)
        i += 32
    folded = [[acc[v][lane] + acc[v][lane + 4] for lane in range(4)] for v in range(4)]
    while i < n1:
        _fma_into(folded, x, y, i, 4)
        i += 16
    return folded


def _hsum(acc: list[list[float]]) -> float:
    lanes = []
    for lane in range(4):
        total = acc[0][lane]
        total = total + acc[1][lane]
        total = total + acc[2][lane]
        total = total + acc[3][lane]
        lanes.append(total)
    return (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])


def stable_dot(left: ArrayLike, right: ArrayLike) -> float:
    """Inner product matching SkylakeX OpenBLAS ``ddot`` at 1 thread."""
    x, y = _pair(left, right)
    n1 = len(x) & ~15
    dot = _hsum(_dot_wide(x, y, n1))
    for i in range(n1, len(x)):
        dot = _fma_rn(x[i], y[i], dot)
    return float(dot)


def _direct_convolve(left: Array, right: Array) -> Array:
    a = left.tolist()
    v = right.tolist()
    n, m = len(a), len(v)
    out = np.empty(n + m - 1, dtype=float)
    for k in range(n + m - 1):
        j0 = max(0, k - (m - 1))
        j1 = min(k + 1, n)
        xs = a[j0:j1]
        ys = [v[k - j] for j in range(j0, j1)]
        out[k] = stable_dot(xs, ys)
    return out


def _numpy_convolve_ok() -> bool:
    global _CONV_OK
    if _CONV_OK is None:
        rng = np.random.default_rng(0)
        sample = rng.normal(size=48)
        kernel = rng.normal(size=48)
        _CONV_OK = bool(
            np.array_equal(np.convolve(sample, kernel), _direct_convolve(sample, kernel))
        )
    return _CONV_OK


def stable_convolve(left: ArrayLike, right: ArrayLike) -> Array:
    """Full-mode convolution matching :func:`numpy.convolve` on the pinned host.

    Hosts whose dot product already matches :func:`stable_dot` keep NumPy's
    convolution; every other host takes the explicit dot product above.
    """
    a = np.ascontiguousarray(left, dtype=float).ravel()
    v = np.ascontiguousarray(right, dtype=float).ravel()
    if a.size == 0 or v.size == 0:
        raise ValueError("stable_convolve inputs must be nonempty")
    if _numpy_convolve_ok():
        return np.convolve(a, v)
    return _direct_convolve(a, v)


def stable_pow(base: float, exp: float) -> float:
    """``base ** exp`` with the pinned kernel's last bit on the rough-vol exponent."""
    base = float(base)
    exp = float(exp)
    if exp == _POW_EXP and base.is_integer() and base > 0.0:
        pinned = _POW_PIN.get(int(base))
        if pinned is not None:
            return pinned
    return math.pow(base, exp)


def stable_pow_array(bases: ArrayLike, exp: float) -> Array:
    """Elementwise :func:`stable_pow`, preserving shape."""
    arr = np.asarray(bases, dtype=float)
    flat = [stable_pow(float(v), float(exp)) for v in arr.ravel()]
    return np.asarray(flat, dtype=float).reshape(arr.shape)
