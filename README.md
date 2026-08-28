# Fixed-Point Gram-Schmidt QR Decomposition Engine

Synthesisable Verilog-2001 implementation of a **Modified Gram-Schmidt**
orthogonaliser. Streams in an `M x N` matrix `A` and streams out `Q` (`M x N`,
orthonormal columns) and `R` (`N x N`, upper triangular) with `A = Q*R`.

Fully parameterised in dimensions, lane parallelism, word width and fractional
bits. Verified against a bit-accurate Python model across 23 configurations.

## Status

| item                        | state                                            |
|-----------------------------|--------------------------------------------------|
| RTL                         | complete, 8 modules                              |
| Verification                | 23/23 regression configs pass, exact model match  |
| rsqrt unit                  | 616 directed tests, worst error 1.13 ulp          |
| Synthesis elaboration       | clean under Yosys (`check -assert`, no latches)   |
| Cycle-count model           | exact against simulation                          |

## Quick start

Requires `iverilog` and `python3` with `numpy`.

```sh
cd sim
make test                # generate stimulus, build, simulate, self-check
make rsqrt               # unit-test the reciprocal-square-root block
make regress             # full sweep over geometries and corner cases
make test M=16 N=8 P=4   # any geometry
```

`make test` ends in `TEST PASSED` only if the RTL output matches the
bit-accurate model **word for word** and the numerical checks pass.

## Repository map

```
rtl/
  gs_top.v            top level: MGS sequencer, 15-state FSM, memories
  vec_mac.v           P-lane dot product, exact ACC_W accumulation
  vec_axpy.v          P-lane scale / scale-and-subtract (shared array)
  rsqrt_unit.v        Newton-Raphson; returns BOTH 1/||v|| and ||v||
  rsqrt_seed_rom.v    64 x 16 seed table (generated)
  round_sat.v         round-half-up + signed saturate, with overflow flag
  ram_2r1w.v          2-read/1-write vector memory (mirrored banks)
  ram_1r1w.v          R matrix memory
model/
  gs_model.py         bit-accurate reference (the golden model)
  gen_stimulus.py     stimulus + expected output generator
  gen_seed_rom.py     regenerates rsqrt_seed_rom.v
  check_qr.py         independent numpy check of A=QR, Q'Q=I
tb/
  tb_gs_top.v         full self-checking testbench, random stalls
  tb_rsqrt.v          616-point rsqrt unit test
sim/
  Makefile            build and run targets
  regress.py          geometry and corner-case sweep
syn/
  synth_vivado.tcl    out-of-context synthesis script
  synth_yosys.ys      open-source elaboration and area check
  constraints/gs.xdc  clock and I/O timing constraints
docs/
  architecture.md     block diagram, FSM, cycle model, resources
  fixed_point.md      format contract, guard bits, error analysis
```

## Parameters

| name        | default | meaning |
|-------------|---------|---------|
| `M`         | 8       | rows (vector length) |
| `N`         | 4       | columns, `N <= M` |
| `P`         | 2       | lanes; `M` must be a multiple of `P` |
| `W`         | 18      | datapath word width |
| `F`         | 16      | datapath fractional bits |
| `ACC_W`     | 48      | dot-product accumulator, needs `>= 2W + log2(M)` |
| `SC_W`      | 32      | scalar width |
| `SC_F`      | 20      | scalar fractional bits |
| `NR_ITER`   | 3       | Newton-Raphson iterations |
| `EPS_SHIFT` | derived | rank-deficiency threshold, see `docs/fixed_point.md` |

`M % P == 0` and `N <= M` are checked at elaboration time.

## Interface

All three data ports are AXI-Stream style (`tdata`/`tvalid`/`tready`,
transfer on `tvalid && tready` at the rising edge).

| port      | dir | order        | beats |
|-----------|-----|--------------|-------|
| `s_a_*`   | in  | column-major | `M*N` |
| `m_q_*`   | out | column-major | `M*N` |
| `m_r_*`   | out | row-major    | `N*N` |

`R` is emitted in full, including the strict lower triangle, which is exactly
zero. Control: pulse `start`, wait for `busy` to fall and the one-cycle `done`
pulse. `rank_def` and `ovf_flag` are sticky per decomposition.

Back-to-back decompositions need no reset; the testbench asserts that two
consecutive runs on the same input produce identical output.

## Input scaling

Scale `A` so every column has `||a_j|| < 2` (sufficient: `|a_ij| <= 1/sqrt(M)`).
This one condition bounds the entire datapath, because `|q_ij| <= 1` always and
`|r_kj| <= ||a_j||`. Violations saturate rather than wrap and raise `ovf_flag`.

Useful accuracy requires `cond(A)` below roughly `4e3` at default parameters.
Beyond that the engine raises `rank_def` and zeroes the offending column rather
than emitting a plausible but wrong `Q`. See `docs/fixed_point.md`.

## Performance

```
cycles = (M/P + 6)*N*(N+1) + 26*N + 3*(M/P)*N + 2*M*N + 5*N^2 + 1
```

Exact against simulation. Selected points, and the measured orthogonality error:

| M  | N  | P | cycles | `\|\|Q'Q-I\|\|` |
|----|----|---|--------|----------------|
| 8  | 4  | 1 | 625    | 4.4 LSB |
| 8  | 4  | 2 | 497    | 1.8 LSB |
| 8  | 4  | 8 | 401    | 1.6 LSB |
| 16 | 8  | 4 | 1601   | 4.6 LSB |
| 64 | 16 | 8 | 7937   | 2.9 LSB |

Multiplier count is `2P + 1`; on Xilinx roughly `3P + 4` DSP48E1.

## Design notes

Three choices carry most of the design:

- **Modified, not Classical Gram-Schmidt.** Orthogonality degrades like
  `u*cond(A)` instead of `u*cond(A)^2`. The measured `cond=1e3` case gives
  102 LSB of error where CGS would give roughly `5e5` LSB, i.e. nothing usable.
  The price is that the inner loop must be sequential.
- **One multiplier array for two operations.** Normalisation and projection
  removal are both vector-times-scalar, so `vec_axpy` serves both and the DSP
  cost of the update path is halved.
- **No divider and no separate square root.** `rsqrt_unit` recovers
  `||v|| = m * (1/sqrt(m)) * 2^h` from the same multiplier the Newton-Raphson
  iterations use, so one unit produces both values the algorithm needs.

## Verification

The golden model in `model/gs_model.py` reproduces the hardware bit for bit,
including rounding, saturation, the seed table and every Newton-Raphson step.
Equality with it is an exact check, not a tolerance check. On top of that,
`check_qr.py` independently verifies `A = QR`, `Q'Q = I`, upper-triangularity
and a non-negative diagonal in numpy, with a condition-number-aware budget of
`max(64, 4*cond(A))` LSB, and compares against both floating-point MGS and
LAPACK.

The testbench drives random input bubbles and random output backpressure by
default (`-DGS_NOSTALL` disables this for clean cycle measurements).

Regression coverage: `P` in `{1,2,4,8}`, geometries from `4x4` to `64x16`,
including `N=1` and `M=N`, and the corner cases `random`, `illcond`
(`cond=1e3`), `extreme` (`cond=1e6`, must flag `rank_def`), `rankdef` (exactly
duplicated column), `identity` and `hadamard`.

### Bugs this regression caught

- Lane counters were sized from the chunk count rather than the lane count, so
  any configuration with `P > M/P` hung. Found at `M=8, P=8`.
- The `1/||v||` scalar saturated on ill-conditioned input, silently producing a
  non-unit `Q`. Fixed by widening the scalar and deriving the rank-deficiency
  threshold from the saturation point, so the engine now reports the condition
  instead of hiding it.

Two earlier "failures" were defects in the testbench rather than the RTL
(`$itor` truncating to 32 bits; sampling the AXI handshake on the wrong clock
edge). Both are documented in comments at the point where they bit, because
both are easy to reintroduce.

## Synthesis

```sh
cd syn
yosys synth_yosys.ys                     # open-source elaboration + area
vivado -mode batch -source synth_vivado.tcl   # out-of-context, needs Vivado


```
## Author

**[Himangshu Gayan]**  
Hardware Engineer | Power Electronics | ARM SoC Design

- 📧 Email: [himangshugayan7@gmail.com]

## License

MIT. See `LICENSE`.
