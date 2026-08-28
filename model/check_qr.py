#!/usr/bin/env python3
"""Independent post-simulation check of the RTL results.

Reads a.hex / q.hex / r.hex from the simulation directory and verifies, without
reusing any of the RTL's own arithmetic:

    * A = Q R                 (reconstruction error)
    * Q^T Q = I               (loss of orthogonality)
    * R upper triangular, non-negative diagonal
    * hardware output == bit-accurate model output
    * how the fixed-point result compares with a float MGS and with numpy's
      Householder QR (the numerically ideal answer)

    python3 check_qr.py --m 8 --n 4 --dir ../sim/build
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from gs_model import DEFAULTS, from_hex, mgs_fixed, mgs_float


def read_hex(path: str, n: int, W: int) -> list[int]:
    vals = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                vals.append(from_hex(line, W))
    if len(vals) != n:
        raise SystemExit("%s: expected %d words, found %d" % (path, n, len(vals)))
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--w", type=int, default=DEFAULTS["W"])
    ap.add_argument("--f", type=int, default=DEFAULTS["F"])
    ap.add_argument("--dir", default=".")
    ap.add_argument("--tol-lsb", type=float, default=64.0)
    args = ap.parse_args()

    M, N, W, F = args.m, args.n, args.w, args.f
    d = args.dir
    scale = float(1 << F)

    a_raw = read_hex(os.path.join(d, "a.hex"), M * N, W)
    q_raw = read_hex(os.path.join(d, "q.hex"), M * N, W)
    r_raw = read_hex(os.path.join(d, "r.hex"), N * N, W)

    A_int = [[a_raw[j * M + i] for j in range(N)] for i in range(M)]
    A = np.array(A_int, dtype=float) / scale
    Q = np.array([[q_raw[j * M + i] for j in range(N)] for i in range(M)], dtype=float) / scale
    R = np.array([[r_raw[i * N + j] for j in range(N)] for i in range(N)], dtype=float) / scale

    fails = []
    lsb = 1.0 / scale

    recon = np.max(np.abs(A - Q @ R)) if M and N else 0.0
    ortho = np.max(np.abs(Q.T @ Q - np.eye(N)))
    lower = np.max(np.abs(np.tril(R, -1))) if N > 1 else 0.0
    diag_min = float(np.min(np.diag(R)))

    # dependent columns legitimately produce a zero column / zero diagonal
    rank_def = bool(np.any(np.abs(np.diag(R)) < 1e-6))

    # --- bit-exact agreement with the model ---------------------------------
    cfg = dict(DEFAULTS)
    cfg.update(W=W, F=F)
    Qm, Rm, rd_m, ovf_m = mgs_fixed(A_int, cfg)
    q_model = [Qm[i][j] for j in range(N) for i in range(M)]
    r_model = [Rm[i][j] for i in range(N) for j in range(N)]
    exact_q = q_model == q_raw
    exact_r = r_model == r_raw

    # --- references ----------------------------------------------------------
    Qf, Rf = mgs_float(A_int)
    Qf = np.array(Qf) / 1.0
    Qf = Qf  # already normalised, entries are unitless
    ortho_f = np.max(np.abs(Qf.T @ Qf - np.eye(N)))
    Qh, _ = np.linalg.qr(A)
    ortho_h = np.max(np.abs(Qh.T @ Qh - np.eye(N)))
    cond = np.linalg.cond(A)
    # MGS loses orthogonality like u*cond(A) = (cond/2) LSB here; allow 8x that.
    tol = max(args.tol_lsb, 4.0 * cond) * lsb

    print("  matrix        : %dx%d, cond(A) = %.3e" % (M, N, cond))
    print("  error budget  : %.0f LSB" % (tol / lsb))
    print("  max |A - QR|  : %.3e  (%.1f LSB)" % (recon, recon / lsb))
    print("  max |QtQ - I| : %.3e  (%.1f LSB)   float MGS %.2e, LAPACK %.2e"
          % (ortho, ortho / lsb, ortho_f, ortho_h))
    print("  R lower tri   : %.3e     min diag %.6f" % (lower, diag_min))
    print("  model match   : Q %s, R %s   (model rank_def=%d ovf=%d)"
          % ("ok" if exact_q else "MISMATCH", "ok" if exact_r else "MISMATCH",
             rd_m, ovf_m))

    if not exact_q:
        fails.append("Q does not match the bit-accurate model")
    if not exact_r:
        fails.append("R does not match the bit-accurate model")
    if lower > 0.0:
        fails.append("R is not upper triangular")
    if diag_min < 0.0:
        fails.append("R has a negative diagonal entry")
    if not rank_def:
        if recon > tol:
            fails.append("reconstruction error %.3e > tol %.3e" % (recon, tol))
        if ortho > tol:
            fails.append("orthogonality error %.3e > tol %.3e" % (ortho, tol))

    if fails:
        for f in fails:
            print("  FAIL: %s" % f)
        print("  CHECK FAILED")
        return 1
    print("  CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
