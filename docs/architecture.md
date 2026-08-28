# Architecture

## 1. What the engine computes

Given a real matrix `A` of size `M x N` (`N <= M`), the engine produces `Q` (`M x N`,
orthonormal columns) and `R` (`N x N`, upper triangular, non-negative diagonal)
such that `A = Q*R`.

## 2. Why Modified Gram-Schmidt

Classical Gram-Schmidt (CGS) computes every projection coefficient against the
**original** column `a_j`:

```
for j: v = a_j;  for k<j: r_kj = q_k' * a_j ;  v -= r_kj * q_k
```

Rounding errors already present in `q_k` are therefore never seen by the
remaining coefficients, and they accumulate. The measured loss of orthogonality
for CGS grows like `u * cond(A)^2`.

Modified Gram-Schmidt (MGS) subtracts from the **running residual**:

```
for j: v = a_j;  for k<j: r_kj = q_k' * v ;  v -= r_kj * q_k
```

Each coefficient is computed against a vector that already had the previous
projections removed, so the error feeds back and is self-correcting to first
order. The bound improves to `u * cond(A)`, i.e. the squared term disappears.

This is not a free choice: MGS forces the `k` loop to be **sequential**, because
`r_kj` cannot be computed until the `k-1` update has landed. CGS could compute
all `j-1` coefficients in parallel. The engine accepts the serialisation to buy
the accuracy, and recovers throughput through lane parallelism `P` instead.

MGS also updates in place, so only one `M x N` vector memory is needed rather
than separate source and destination matrices.

## 3. Top-level block diagram

```
                  s_a_tdata/tvalid/tready  (A, column-major, M*N beats)
                                |
                                v
                        +---------------+
                        |  lane packer  |  P words -> one P*W memory word
                        +-------+-------+
                                |
                +---------------v----------------+
                |   vector RAM  ram_2r1w         |   depth (M/P)*N, width P*W
                |   2 read ports, 1 write port   |   holds A, then V, then Q
                +---+-----------------------+----+
             port a |                       | port b
                    |                       |
          +---------v---------+   +---------v----------+
          |     vec_mac       |   |      vec_axpy      |
          |  P multipliers    |   |  P multipliers     |
          |  adder tree       |   |  round_sat x P     |
          |  ACC_W accumulator|   |                    |
          +---------+---------+   +---------+----------+
                    |                       |
        ||v||^2 or  |                       | dst = src*scalar        (NORMALISE)
        q_k'*v      |                       | dst = dst - src*scalar  (PROJECT)
                    |                       +----> back to vector RAM
                    |
         +----------v-----------+
         |     rsqrt_unit       |  one shared 34x34 multiplier
         |  normalise -> seed   |  returns BOTH  y = 1/||v||  (Q12.20)
         |  -> NR_ITER x NR     |              and s = ||v||  (Q2.16)
         |  -> denormalise      |
         +----------+-----------+
                    |
                    +--> R diagonal          +--> scalar to vec_axpy
                    
          +-------------------+
          |  R RAM ram_1r1w   |  N*N words, row-major, zeroed before use
          +---------+---------+
                    |
                    v
       m_r_tdata/tvalid/tready   (R, row-major, N*N beats)
       m_q_tdata/tvalid/tready   (Q, column-major, M*N beats)
```

Two structural decisions carry most of the area saving:

**One multiplier array for two jobs.** Normalisation (`q = v * (1/||v||)`) and
projection removal (`v = v - r * q`) are both "vector times scalar". `vec_axpy`
implements `dst = dst_in*use_dst - round_sat(src*scalar)`, so the same `P`
multipliers serve both phases. Only the `use_dst` bit and the scalar source
change.

**No divider, no separate square root.** The engine needs `1/||v||` to normalise
and `||v||` to store as `R[j][j]`. A naive design instantiates a square root
**and** a divider. Instead `rsqrt_unit` computes `y = 1/sqrt(m)` by
Newton-Raphson and then recovers the norm with one extra multiply:

```
||v|| = m * (1/sqrt(m)) * 2^h        since  m/sqrt(m) = sqrt(m)
```

Both results come out of the same 34x34 multiplier that the NR iterations
already use, sequenced by a two-phase FSM. Cost: one multiplier, no divider.

## 4. Control FSM

15 states. `ST_*_D` states are drain states that wait out the pipeline latency
of the arithmetic block just started.

```
ST_IDLE
   | start
ST_RCLR      zero the R RAM so the strict lower triangle streams out as
   |         exact zeros without any special-case logic at the output
ST_LOAD      accept M*N beats of A, packing P at a time
   |
   +---------------------------------------------+
   v                                             |
ST_NORM  --> ST_NORM_D      ||v_j||^2 over M/P chunks, DRAIN cycles to flush
   |                                             |
ST_RSQ                      rsqrt_unit: y = 1/||v||, s = ||v||
   |                        if ||v||^2 < 2^EPS_SHIFT -> R[j][j]=0, q_j=0,
   |                        sticky rank_def, skip the scale
ST_SCALE --> ST_SCALE_D     q_j = v_j * y
   |
   +--> for k = j+1 .. N-1:
   |      ST_PROJ  --> ST_PROJ_D    r_jk = q_j' * v_k
   |      ST_UPD   --> ST_UPD_D     v_k -= r_jk * q_j
   |
   +-- next j ------------------------------------+
   |
ST_OUTQ      stream Q, column-major
ST_OUTR      stream R, row-major
ST_FIN       assert done for one cycle
```

### Write-address delay line

`vec_axpy` results arrive 3 cycles after the read address is issued (1 cycle RAM
read latency + 2 cycles through the multiplier and `round_sat`). The write
address is therefore delayed through `wa1 -> wa2 -> wa3` so the result lands in
the slot it came from. `iss_dot` and `iss_axpy` are registered one extra time to
align the enable with the RAM data. Getting this delay line off by one is the
classic failure mode of an in-place update engine, so the depth is derived from
the same constants that define the module latencies rather than hard-coded.

## 5. Cycle count

With `CHUNKS = M/P`, `DRAIN = 6`, and rsqrt latency 26 (24 internal + 2
handshake):

```
cycles = (CHUNKS + DRAIN) * N * (N+1)     <- dot products and updates
       + 26 * N                           <- one rsqrt per column
       + 3 * CHUNKS * N                   <- Q output addressing
       + 2 * M * N                        <- load A, stream Q
       + 5 * N^2                          <- clear R, stream R
       + 1
```

This formula is **exact**, verified against simulation with no stalls:

| M  | N  | P | predicted | measured |
|----|----|---|-----------|----------|
| 8  | 4  | 1 | 625       | 625      |
| 8  | 4  | 2 | 497       | 497      |
| 8  | 4  | 4 | 433       | 433      |
| 8  | 4  | 8 | 401       | 401      |
| 16 | 8  | 2 | 1985      | 1985     |
| 16 | 8  | 4 | 1601      | 1601     |
| 32 | 16 | 4 | 6913      | 6913     |
| 64 | 8  | 8 | 2753      | 2753     |
| 64 | 16 | 8 | 7937      | 7937     |

Note how weakly small cases scale with `P`: at `M=8, N=4` the `2*M*N` I/O term
and the `26*N` rsqrt term dominate, so going from `P=1` to `P=8` only buys 1.6x.
Lane parallelism pays off when `M/P` still dominates `DRAIN = 6`, i.e. for tall
matrices. At `M=64, N=16` the compute term is 87% of the total.

## 6. Resource estimate

Multiplier count is exactly `2P + 1`, confirmed by Yosys cell counts
(`$mul = 5` at `P = 2`):

| block      | multipliers | shape       | Xilinx DSP48E1 each |
|------------|-------------|-------------|---------------------|
| `vec_mac`  | `P`         | `W x W`     | 1                   |
| `vec_axpy` | `P`         | `W x SC_W`  | 2 (32b > 25b port)  |
| `rsqrt`    | 1           | 34 x 34     | 4                   |

So roughly **`3P + 4` DSP48E1**: 10 at `P=2`, 28 at `P=8`.

Memory:

- vector RAM: `2 * (M/P) * N * P * W` bits (two mirrored banks give the second
  read port; BRAM-inferable, read-first)
- R RAM: `N * N * W` bits
- seed ROM: `64 * 16` bits

Registers: about 700 flops at `M=8, N=4, P=2` outside the memories.

## 7. Timing

The critical path is inside `rsqrt_unit`: the 34x34 multiplier feeding the
`3 - m*y^2` subtraction. The two-phase FSM already places registers on both
sides of the multiplier, so on a mid-speed FPGA this closes comfortably around
150-200 MHz. To push higher, pipeline the 34x34 multiplier into two stages and
extend the phase counter accordingly; this costs 2 cycles per column
(`26 -> 28` in the formula above) and nothing else, because nothing else in the
design depends on the internal timing of the rsqrt.

The `vec_mac` adder tree is `log2(P)` deep and is not critical for `P <= 8`.
