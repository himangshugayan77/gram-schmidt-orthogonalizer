#!/usr/bin/env python3
"""Generate stimulus for tb_gs_top and the matching bit-accurate expectation.

    python3 gen_stimulus.py --m 8 --n 4 --case random --out ../sim/build

Writes into the output directory:
    a.hex        M*N lines, column-major, W-bit hex   (testbench input)
    exp_q.hex    expected Q, column-major             (bit-accurate model)
    exp_r.hex    expected R, row-major, N*N lines
    meta.txt     M N P W F and the model's flags

Cases
    random     well-conditioned, entries ~ U(-c, c)
    illcond    prescribed condition number ~1e3 via an SVD construction
    extreme    condition number ~1e6, below the fixed-point noise floor;
               the engine is expected to declare rank deficiency
    rankdef    column 2 is an exact copy of column 0
    identity   scaled identity, exercises the trivial path
    hadamard   orthogonal columns already -> R must be diagonal
"""
from __future__ import annotations

import argparse
import os
import random

from gs_model import DEFAULTS, mgs_fixed, to_fixed, to_hex


def make_matrix(case: str, M: int, N: int, W: int, F: int, seed: int):
    rng = random.Random(seed)
    scale = 0.9 / (M ** 0.5)          # keeps ||a_j|| < 1 with margin
    A = [[0] * N for _ in range(M)]

    if case == "random":
        for i in range(M):
            for j in range(N):
                A[i][j] = to_fixed(rng.uniform(-scale, scale), W, F)

    elif case in ("illcond", "extreme"):
        # Build A = U diag(s) V^T with logarithmically spaced singular values,
        # so cond(A) is prescribed rather than accidental.
        import numpy as np
        target = 1.0e3 if case == "illcond" else 1.0e6
        nrng = np.random.default_rng(seed)
        U, _ = np.linalg.qr(nrng.standard_normal((M, M)))
        V, _ = np.linalg.qr(nrng.standard_normal((N, N)))
        sv = np.logspace(0.0, -np.log10(target), N) if N > 1 else np.ones(1)
        Ar = U[:, :N] @ np.diag(sv) @ V.T
        Ar = Ar * (0.9 / np.max(np.abs(Ar)))
        for i in range(M):
            for j in range(N):
                A[i][j] = to_fixed(float(Ar[i][j]), W, F)

    elif case == "rankdef":
        for i in range(M):
            for j in range(N):
                A[i][j] = to_fixed(rng.uniform(-scale, scale), W, F)
        if N >= 3:
            for i in range(M):
                A[i][2] = A[i][0]      # exactly dependent column

    elif case == "identity":
        for i in range(M):
            for j in range(N):
                A[i][j] = to_fixed(0.5 if i == j else 0.0, W, F)

    elif case == "hadamard":
        for i in range(M):
            for j in range(N):
                bits = bin(i & j).count("1")
                A[i][j] = to_fixed((1 if bits % 2 == 0 else -1) * scale, W, F)

    else:
        raise SystemExit("unknown case: %s" % case)

    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--w", type=int, default=DEFAULTS["W"])
    ap.add_argument("--f", type=int, default=DEFAULTS["F"])
    ap.add_argument("--case", default="random")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="../sim/build")
    args = ap.parse_args()

    M, N, W, F = args.m, args.n, args.w, args.f
    if M % args.p:
        raise SystemExit("M must be a multiple of P")
    if N > M:
        raise SystemExit("N must be <= M")

    os.makedirs(args.out, exist_ok=True)
    A = make_matrix(args.case, M, N, W, F, args.seed)

    cfg = dict(DEFAULTS)
    cfg.update(W=W, F=F)
    Q, R, rank_def, ovf = mgs_fixed(A, cfg)

    with open(os.path.join(args.out, "a.hex"), "w") as fh:
        for j in range(N):                 # column-major
            for i in range(M):
                fh.write(to_hex(A[i][j], W) + "\n")

    with open(os.path.join(args.out, "exp_q.hex"), "w") as fh:
        for j in range(N):
            for i in range(M):
                fh.write(to_hex(Q[i][j], W) + "\n")

    with open(os.path.join(args.out, "exp_r.hex"), "w") as fh:
        for i in range(N):                 # row-major
            for j in range(N):
                fh.write(to_hex(R[i][j], W) + "\n")

    # A condition-number-aware error budget.  MGS loses orthogonality like
    # u*cond(A); with u = 2^-(F+1) that is cond/2 LSB, so 4*cond LSB leaves an
    # 8x margin over the textbook bound while still being a real constraint.
    try:
        import numpy as np
        Af = np.array([[A[i][j] for j in range(N)] for i in range(M)], dtype=float)
        cond = float(np.linalg.cond(Af)) if N > 1 else 1.0
    except Exception:
        cond = 1.0
    tol_lsb = max(64.0, 4.0 * cond)
    if rank_def:
        tol_lsb = 0.0          # metrics are meaningless for a dropped column

    with open(os.path.join(args.out, "meta.txt"), "w") as fh:
        fh.write("M %d\nN %d\nP %d\nW %d\nF %d\ncase %s\nrank_def %d\novf %d\n"
                 "cond %.6e\ntol_lsb %.1f\n"
                 % (M, N, args.p, W, F, args.case, int(rank_def), int(ovf),
                    cond, tol_lsb))

    print("generated %s: M=%d N=%d P=%d W=%d F=%d case=%s cond=%.3e "
          "rank_def=%d ovf=%d tol=%.0f LSB"
          % (args.out, M, N, args.p, W, F, args.case, cond, rank_def, ovf, tol_lsb))


if __name__ == "__main__":
    main()
