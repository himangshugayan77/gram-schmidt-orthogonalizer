#!/usr/bin/env python3
"""Sweep geometries, lane counts and matrix corner cases through the RTL.

Each configuration regenerates stimulus + the bit-accurate expectation, rebuilds
the simulation with the matching parameters, and requires BOTH the Verilog
self-check and the independent numpy check to pass.

    python3 regress.py            # default sweep
    python3 regress.py --quick    # short sweep, used by CI
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RTL = os.path.join(ROOT, "rtl")
TB = os.path.join(ROOT, "tb")
MODEL = os.path.join(ROOT, "model")

# (M, N, P, case)
FULL = [
    (8,  4, 1, "random"),
    (8,  4, 2, "random"),
    (8,  4, 4, "random"),
    (8,  4, 8, "random"),
    (4,  4, 2, "random"),
    (16, 8, 4, "random"),
    (16, 3, 2, "random"),
    (32, 4, 8, "random"),
    (12, 3, 3, "random"),
    (64, 8, 8, "random"),
    (8,  1, 2, "random"),
    (8,  4, 2, "illcond"),
    (16, 4, 4, "illcond"),
    (8,  4, 2, "extreme"),
    (16, 8, 4, "extreme"),
    (8,  4, 2, "rankdef"),
    (16, 6, 2, "rankdef"),
    (8,  4, 2, "identity"),
    (8,  4, 2, "hadamard"),
    (8,  8, 2, "hadamard"),
    (8,  4, 2, "random"),      # different seeds
    (8,  4, 2, "random"),
    (8,  4, 2, "random"),
]

QUICK = [
    (8,  4, 1, "random"),
    (8,  4, 4, "random"),
    (16, 8, 4, "random"),
    (8,  4, 2, "illcond"),
    (8,  4, 2, "rankdef"),
    (8,  4, 2, "hadamard"),
]


def read_meta(workdir):
    meta = {}
    path = os.path.join(workdir, "meta.txt")
    if os.path.exists(path):
        for line in open(path):
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def one(m, n, p, case, seed, workdir, keep_log=False):
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)

    r = run([sys.executable, "gen_stimulus.py", "--m", str(m), "--n", str(n),
             "--p", str(p), "--case", case, "--seed", str(seed),
             "--out", workdir], cwd=MODEL)
    if r.returncode:
        return False, "stimulus generation failed:\n" + r.stderr, {}

    meta = read_meta(workdir)
    tol = max(64.0, float(meta.get("tol_lsb", 64.0)))

    srcs = sorted(os.path.join(RTL, f) for f in os.listdir(RTL) if f.endswith(".v"))
    r = run(["iverilog", "-g2005", "-o", os.path.join(workdir, "tb.vvp"),
             "-DGS_M=%d" % m, "-DGS_N=%d" % n, "-DGS_P=%d" % p, "-DGS_RUNS=2",
             "-DGS_TOL_LSB=%.1f" % tol]
            + srcs + [os.path.join(TB, "tb_gs_top.v")])
    if r.returncode:
        return False, "compile failed:\n" + r.stderr, {}

    r = run(["vvp", "tb.vvp"], cwd=workdir)
    log = r.stdout + r.stderr
    if "TEST PASSED" not in log:
        return False, log, {}

    stats = {}
    mm = re.search(r"completed in (\d+) cycles", log)
    if mm:
        stats["cycles"] = int(mm.group(1))
    mm = re.search(r"max \|Q'Q - I\| = \S+\s+\(([\d.]+) LSB\)", log)
    if mm:
        stats["ortho_lsb"] = float(mm.group(1))
    stats["rank_def"] = "rank_def=1" in log

    r = run([sys.executable, os.path.join(MODEL, "check_qr.py"),
             "--m", str(m), "--n", str(n), "--dir", "."], cwd=workdir)
    log2 = r.stdout + r.stderr
    if "CHECK PASSED" not in log2:
        return False, log + "\n" + log2, stats
    mm = re.search(r"cond\(A\) = (\S+)", log2)
    if mm:
        stats["cond"] = float(mm.group(1))
    return True, log + log2 if keep_log else "", stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep per-case work dirs")
    args = ap.parse_args()

    cases = QUICK if args.quick else FULL
    workroot = os.path.join(HERE, "build_regress")
    os.makedirs(workroot, exist_ok=True)

    print("%-4s %-4s %-3s %-9s %-8s %-10s %-9s %s"
          % ("M", "N", "P", "case", "cycles", "cond(A)", "ortho", "result"))
    print("-" * 74)

    npass = nfail = 0
    for idx, (m, n, p, case) in enumerate(cases):
        wd = os.path.join(workroot, "c%02d" % idx)
        ok, log, st = one(m, n, p, case, seed=idx + 1, workdir=wd)
        cond = "%.2e" % st["cond"] if "cond" in st else "-"
        cyc = st.get("cycles", "-")
        orth = "%.1f LSB" % st["ortho_lsb"] if "ortho_lsb" in st else "-"
        tag = "PASS" + (" (rank def)" if st.get("rank_def") else "")
        print("%-4d %-4d %-3d %-9s %-8s %-10s %-9s %s"
              % (m, n, p, case, cyc, cond, orth, tag if ok else "FAIL"))
        if ok:
            npass += 1
        else:
            nfail += 1
            print(log)
        if not args.keep:
            shutil.rmtree(wd, ignore_errors=True)

    print("-" * 74)
    print("%d passed, %d failed" % (npass, nfail))
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
