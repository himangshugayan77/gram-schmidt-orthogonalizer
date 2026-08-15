#!/usr/bin/env python3
"""Bit-accurate software model of the gs_top Modified Gram-Schmidt engine.

Every arithmetic operation here mirrors the RTL exactly -- same rounding, same
saturation, same Newton-Raphson recurrence, same order of operations.  Running
this model and the Verilog simulation on the same input must produce
*identical* bit patterns; tb/tb_gs_top.v and model/check_qr.py rely on that to
catch datapath regressions that a loose numerical tolerance would hide.

Also provides a float reference (mgs_float) for measuring how much accuracy the
fixed-point implementation actually loses.
"""
from __future__ import annotations

import math

from gen_seed_rom import seed_table

# --------------------------------------------------------------------------
# default configuration -- keep in sync with rtl/gs_top.v defaults
# --------------------------------------------------------------------------
DEFAULTS = dict(
    W=18,        # datapath width
    F=16,        # datapath fractional bits
    ACC_W=48,    # accumulator width
    SC_W=32,     # scalar width
    SC_F=20,     # scalar fractional bits
    NR_ITER=3,
    # matches rtl/gs_top.v: EPS_SHIFT = 2*F - 2*(SC_W-1-SC_F)
    EPS_NORM2=1 << (2*16 - 2*(32-1-20)),
)

MF = 30          # rsqrt mantissa fractional bits
MW = 32          # rsqrt mantissa width
LUT_BITS = 6
_SEED = seed_table(LUT_BITS)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def rnd_sat(x: int, shift: int, out_w: int) -> tuple[int, bool]:
    """round-half-up by `shift`, then saturate to a signed out_w value.

    Mirrors rtl/round_sat.v.  Python's >> on negative ints is an arithmetic
    shift, matching Verilog's >>> on a signed operand.
    """
    half = 1 << (shift - 1) if shift > 0 else 0
    v = (x + half) >> shift
    hi = (1 << (out_w - 1)) - 1
    lo = -(1 << (out_w - 1))
    if v > hi:
        return hi, True
    if v < lo:
        return lo, True
    return v, False


def sat(x: int, out_w: int) -> tuple[int, bool]:
    hi = (1 << (out_w - 1)) - 1
    lo = -(1 << (out_w - 1))
    if x > hi:
        return hi, True
    if x < lo:
        return lo, True
    return x, False


def rsqrt_fx(x: int, X_F: int, Y_W: int, Y_F: int, S_W: int, S_F: int,
             nr_iter: int = 3):
    """Bit-accurate model of rtl/rsqrt_unit.v.

    Returns (y, s, zero_flag, sat_flag) with
        y ~ 1/sqrt(x) as an unsigned Y_W-bit Q(Y_W-Y_F).Y_F value
        s ~   sqrt(x) as an unsigned S_W-bit Q(S_W-S_F).S_F value
    """
    if x == 0:
        return (1 << (Y_W - 1)) - 1, 0, True, True

    # --- normalise: x = m * 2^e, e even, m in [1,4), Mm = m * 2^MF ----------
    p = x.bit_length() - 1
    par = (p ^ X_F) & 1
    e = p - X_F - par
    h = e >> 1
    lsh = (MF + par) - p
    Mm = (x << lsh) if lsh >= 0 else (x >> -lsh)
    Mm &= (1 << MW) - 1

    # --- seed --------------------------------------------------------------
    Y = _SEED[Mm >> (MW - LUT_BITS)] << (MF - 15)

    # --- Newton-Raphson:  y <- y*(3 - m*y^2)/2 -----------------------------
    mask34 = (1 << 34) - 1
    for _ in range(nr_iter):
        t1 = ((Y * Y) >> MF) & mask34
        t2 = ((Mm * t1) >> MF) & mask34
        t3 = (3 * (1 << MF) - t2) & mask34
        Y = (Y * t3) >> (MF + 1)
        if Y > (1 << MF):
            Y = 1 << MF

    pr = Mm * Y

    # --- denormalise -------------------------------------------------------
    y_sh = (MF - Y_F) + h
    if y_sh >= 0:
        half = (1 << (y_sh - 1)) if y_sh > 0 else 0
        y = (Y + half) >> y_sh
    else:
        y = Y << (-y_sh)
    y_ovf = y > (1 << (Y_W - 1)) - 1
    if y_ovf:
        y = (1 << (Y_W - 1)) - 1

    s_sh = (2 * MF - S_F) - h
    s = (pr + (1 << (s_sh - 1))) >> s_sh
    s_ovf = s > (1 << (S_W - 1)) - 1
    if s_ovf:
        s = (1 << (S_W - 1)) - 1

    return y, s, False, (y_ovf or s_ovf)


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------
def mgs_fixed(A: list[list[int]], cfg: dict | None = None):
    """Modified Gram-Schmidt on an M x N integer (Q(W-F).F) matrix.

    A is given as A[i][j] (row i, column j) of raw integers.
    Returns (Q, R, rank_def, ovf) with Q, R raw integers in the same format.
    """
    c = dict(DEFAULTS)
    if cfg:
        c.update(cfg)
    W, F, SC_F, SC_W = c["W"], c["F"], c["SC_F"], c["SC_W"]
    M = len(A)
    N = len(A[0])

    V = [row[:] for row in A]                       # in-place residuals
    Q = [[0] * N for _ in range(M)]
    R = [[0] * N for _ in range(N)]
    rank_def = False
    ovf = False

    for j in range(N):
        # ---- squared norm (exact accumulation) ----------------------------
        norm2 = sum(V[i][j] * V[i][j] for i in range(M))

        if norm2 < c["EPS_NORM2"]:
            rank_def = True
            y = 0
            R[j][j] = 0
        else:
            y, s, _z, sf = rsqrt_fx(norm2, 2 * F, SC_W, SC_F, W, F, c["NR_ITER"])
            ovf |= sf
            R[j][j] = s

        # ---- normalise ----------------------------------------------------
        for i in range(M):
            q, o = rnd_sat(V[i][j] * y, SC_F, W)
            Q[i][j] = q
            ovf |= o
        for i in range(M):
            V[i][j] = Q[i][j]

        # ---- remove the projection from every later column ----------------
        for k in range(j + 1, N):
            acc = sum(Q[i][j] * V[i][k] for i in range(M))
            r, o = rnd_sat(acc, F, W)
            ovf |= o
            R[j][k] = r
            scalar = r << (SC_F - F)
            for i in range(M):
                t, o1 = rnd_sat(Q[i][j] * scalar, SC_F, W)
                d, o2 = sat(V[i][k] - t, W)
                V[i][k] = d
                ovf |= o1 | o2

    return Q, R, rank_def, ovf


# --------------------------------------------------------------------------
# float reference
# --------------------------------------------------------------------------
def mgs_float(A):
    M = len(A)
    N = len(A[0])
    V = [[float(A[i][j]) for j in range(N)] for i in range(M)]
    Q = [[0.0] * N for _ in range(M)]
    R = [[0.0] * N for _ in range(N)]
    for j in range(N):
        nrm = math.sqrt(sum(V[i][j] ** 2 for i in range(M)))
        R[j][j] = nrm
        for i in range(M):
            Q[i][j] = V[i][j] / nrm if nrm > 0 else 0.0
        for k in range(j + 1, N):
            r = sum(Q[i][j] * V[i][k] for i in range(M))
            R[j][k] = r
            for i in range(M):
                V[i][k] -= r * Q[i][j]
    return Q, R


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def to_real(v: int, F: int = 16) -> float:
    return v / float(1 << F)


def to_fixed(v: float, W: int = 18, F: int = 16) -> int:
    q = int(round(v * (1 << F)))
    hi = (1 << (W - 1)) - 1
    lo = -(1 << (W - 1))
    return max(lo, min(hi, q))


def to_hex(v: int, W: int) -> str:
    return "%0*x" % ((W + 3) // 4, v & ((1 << W) - 1))


def from_hex(s: str, W: int) -> int:
    v = int(s, 16) & ((1 << W) - 1)
    return v - (1 << W) if v >> (W - 1) else v


if __name__ == "__main__":
    # smoke test: 8x4 well-conditioned matrix
    import random
    random.seed(1)
    M, N = 8, 4
    A = [[to_fixed(random.uniform(-0.3, 0.3)) for _ in range(N)] for _ in range(M)]
    Q, R, rd, ov = mgs_fixed(A)
    # orthogonality
    worst = 0.0
    for a in range(N):
        for b in range(N):
            d = sum(to_real(Q[i][a]) * to_real(Q[i][b]) for i in range(M))
            d -= 1.0 if a == b else 0.0
            worst = max(worst, abs(d))
    rec = 0.0
    for i in range(M):
        for jj in range(N):
            s = sum(to_real(Q[i][t]) * to_real(R[t][jj]) for t in range(N))
            rec = max(rec, abs(s - to_real(A[i][jj])))
    print("rank_def=%s ovf=%s" % (rd, ov))
    print("max |QtQ - I|  = %.3e" % worst)
    print("max |A - QR|   = %.3e" % rec)
