"""`log` and `exp` to full float64 precision.

Not upstream code. `std.math.log` and `std.math.exp` in this toolchain are the
SLEEF u10 kernels, accurate to about one part in 1e10: `log(5)` comes out as
1.6094379121731952 where numpy reports 1.6094379124341003. Both LDP and NetLSD
put those values on a histogram or into an exponent, and a 2.6e-10 gap is
enough to move a bin, so the series are spelled out here. `frexp`, `ldexp` and
`sqrt` are exact and are used as given.

`natural_log` is accurate to within one ulp of `math.log`, not bit equal to it:
measured over 2020 arguments it lands one ulp away on 45 of them, always on the
low side. `NodeSketch` feeds `-log` of a uniform draw into an argmin, so the
tests compare its embedding against real upstream output rather than trusting
this bound.
"""

from std.math import frexp, ldexp
from std.utils.numerics import inf

# ln2 split so that `k * LN2_HI` is exact for any integer `k` a graph can
# produce, with `LN2_HI + LN2_LO` equal to ln2.
comptime LN2_HI = 0.693145751953125
comptime LN2_LO = 1.4286068202862268e-06
comptime SQRT_HALF = 0.707106781186547524400844362104849039


def natural_log(x: Float64) -> Float64:
    """`math.log`, to within one ulp.

    The power of two is split off and the mantissa is folded into
    `[sqrt(0.5), sqrt(2))`, so that `t = (m - 1) / (m + 1)` stays under 0.172 and
    `log(m) = 2 * atanh(t) = 2 * (t + t^3/3 + t^5/5 + ...)` truncates below
    1e-20 after twelve terms. The terms are summed with Kahan compensation,
    because twelve halvings of an ulp in the running total are a visible part of
    the answer, and the final add is a TwoSum against the split `ln2` so the
    exponent's own rounding does not leak in. The callers pass a node degree, a
    node count or a uniform draw; `0` is handled because `NodeSketch` takes
    `-log` of a draw that is exactly zero once in 2**53, and numpy's answer
    there is `-inf`. Negative arguments are not handled.
    """
    if x == 0.0:
        return -inf[DType.float64]()
    var split = frexp(x)
    var m = Float64(split[0])
    var k = Float64(split[1])
    if m < SQRT_HALF:
        m = m * 2.0
        k = k - 1.0
    var t = (m - 1.0) / (m + 1.0)
    var t2 = t * t
    var power = 1.0
    var series = 0.0
    var carry = 0.0
    var i = 1
    while i <= 12:
        power = power * t2
        var adjusted = power / Float64(2 * i + 1) - carry
        var total = series + adjusted
        carry = (total - series) - adjusted
        series = total
        i += 1
    var s = 2.0 * t * (1.0 + series)
    var head = k * LN2_HI
    var tail = k * LN2_LO
    var result = s + head
    var bv = result - s
    var err = (s - (result - bv)) + (head - bv)
    return result + (err + tail)


def natural_exp(x: Float64) -> Float64:
    """`math.exp`, to a few times 1e-14 relative at the large-magnitude end.

    Range reduction to `r = x - k*ln2` with `|r| <= ln2/2`, then the Taylor
    series of `exp(r)`, whose terms fall below 1e-17 after twenty. NetLSD
    evaluates this at `t * eigenvalue` with `t` up to 1e2 and eigenvalues up to
    2, so the unreduced argument is large enough that reducing first matters.
    The reduction cancels two operands of size `|x|`, so `r` carries about one
    ulp of `x` and `exp` inherits it: measured worst case is 2e-14 at x = -200
    and 8e-14 at x = -700, against numpy's 1e-16 for the same range. That is
    still four orders below the SLEEF bound this module exists to beat, and
    NetLSD's parity is asserted against real upstream output either way.
    """
    if x > 709.78:
        # ln(DBL_MAX) is 709.7827; past it the exact result overflows, and
        # numpy returns inf there too.
        return inf[DType.float64]()
    if x < -745.0:
        return 0.0
    var k = round_half(x / (LN2_HI + LN2_LO))
    var r = x - (Float64(k) * LN2_HI + Float64(k) * LN2_LO)
    var term = 1.0
    var total = 1.0
    var i = 1
    while i <= 20:
        term = term * r / Float64(i)
        total = total + term
        i += 1
    return ldexp(Float64(total), k)


def round_half(x: Float64) -> Int32:
    """Round to the nearest integer, halves away from zero, as `ldexp` wants."""
    if x >= 0.0:
        return Int32(Int(x + 0.5))
    return Int32(Int(x - 0.5))
