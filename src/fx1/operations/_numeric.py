"""Unregistered numeric infrastructure; not an independent harness capability."""

from math import inf, isqrt, ldexp
from sys import float_info

_MAX_RATIO_BITS = 16_384


def correctly_rounded_sqrt(numerator: int, denominator: int) -> float:
    """Round sqrt(numerator/denominator) once to nearest binary64, ties to even.

    Both arguments must be built-in integers with at most 16384 bits;
    numerator must be nonnegative and denominator positive. Zero and infinity
    are returned for genuine rounded underflow and overflow, respectively.
    Callers that require a finite nonzero result must reject those outcomes.

    An integer unit is the ULP in the root's binade, clamped to 2**-1074 for
    subnormals. The floor root in those units is obtained with integer square
    root. Comparing the exact ratio against the squared half-integer midpoint
    determines rounding without a floating division or square-root operation.
    The rounded integer has at most 53 significant bits, so its conversion and
    power-of-two scaling introduce no further rounding for finite results.
    """
    if type(numerator) is not int or type(denominator) is not int:
        raise TypeError("square-root ratio arguments must be built-in integers")
    if numerator < 0 or denominator <= 0:
        raise ValueError("square-root ratio requires numerator >= 0 and denominator > 0")
    if numerator.bit_length() > _MAX_RATIO_BITS or denominator.bit_length() > _MAX_RATIO_BITS:
        raise ValueError("square-root ratio arguments exceed the 16384-bit bound")
    if (float_info.radix, float_info.mant_dig, float_info.min_exp, float_info.max_exp) != (
        2,
        53,
        -1021,
        1024,
    ):
        raise RuntimeError("correctly rounded rational square roots require binary64 floats")
    if numerator == 0:
        return 0.0

    # The bit-length difference is either floor(log2(n/d)) or one greater.
    floor_log2 = numerator.bit_length() - denominator.bit_length()
    if floor_log2 >= 0:
        if numerator < denominator << floor_log2:
            floor_log2 -= 1
    elif numerator << -floor_log2 < denominator:
        floor_log2 -= 1
    unit_exponent = max(floor_log2 // 2 - 52, -1074)
    shift = 2 * unit_exponent
    if shift >= 0:
        scaled_numerator, scaled_denominator = numerator, denominator << shift
    else:
        scaled_numerator, scaled_denominator = numerator << -shift, denominator

    lower = isqrt(scaled_numerator // scaled_denominator)
    midpoint_left = 4 * scaled_numerator
    midpoint_right = scaled_denominator * (2 * lower + 1) ** 2
    if midpoint_left > midpoint_right or (midpoint_left == midpoint_right and lower % 2):
        lower += 1
    try:
        return ldexp(float(lower), unit_exponent)
    except OverflowError:
        return inf
