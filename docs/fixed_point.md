# Fixed-point contract and error analysis

Everything here is expressed in the default configuration `W=18, F=16`
(Q2.16 data), `ACC_W=48`, `SC_W=32, SC_F=20` (Q12.20 scalars). All of it is
parameterised; the derivations show what changes if you move a parameter.

## 1. Formats

| signal              | format  | range                | resolution |
|---------------------|---------|----------------------|------------|
| `A`, `V`, `Q`, `R`  | Q2.16   | `[-2, 2)`            | 2^-16      |
| dot accumulator     | Q16.32  | exact                | 2^-32      |
| `1/||v||`, `||v||`  | Q12.20  | `[-2048, 2048)`      | 2^-20      |
| rsqrt mantissa      | Q2.30   | `[1, 4)`             | 2^-30      |

`Q(I).(F)` means `I` integer bits including sign, `F` fractional bits, stored
as a two's-complement integer of value `x * 2^F`.

## 2. Input scaling contract

**Scale `A` so that every column satisfies `||a_j|| < 2`.** The simple
sufficient condition is `|a_ij| <= 1/sqrt(M)`.

Why this is the binding constraint: `Q` has orthonormal columns, so every entry
of `Q` satisfies `|q_ij| <= 1` and is always representable in Q2.16. `R` entries
are bounded by `|r_kj| <= ||a_j||`, so bounding the column norm bounds `R` too.
The intermediate residuals `v` only shrink (`||v|| <= ||a_j||`), so nothing
inside the recursion can exceed the input bound either. One condition on the
input therefore guarantees the whole datapath.

Violating it does not corrupt silently: `round_sat` saturates rather than
wrapping, and any saturation raises the sticky `ovf_flag` output.

## 3. Why the accumulator is 48 bits

A dot product of two Q2.16 values gives Q4.32 per product term. Summing `M`
of them adds `log2(M)` bits of headroom:

```
ACC_W >= 2*W + ceil(log2(M))
```

At `W=18, M=64` that is `36 + 6 = 42`, so `ACC_W=48` leaves margin to `M=4096`.
Because the accumulator is wide enough to hold every partial sum exactly,
**dot products are computed with zero intermediate rounding** — the only
rounding in a dot product happens once, when the result is converted back to
Q2.16. This matters: a naive design that rounds after each MAC would inject `M`
independent rounding errors into every projection coefficient.

## 4. Why the scalars are Q12.20

The normalisation scalar is `y = 1/||v||`, so its magnitude grows without bound
as a column becomes dependent on its predecessors. Two competing requirements:

- **Range.** `y` must represent `1/||v||` for the smallest residual norm the
  engine intends to accept. Q12.20 saturates at 2048, covering
  `||v|| >= 1/2048 ~ 4.9e-4`.
- **Precision.** `q = v * y` is rounded to 16 fractional bits, so `y` only needs
  enough *relative* precision that its own error is invisible after that
  rounding. At the typical case `y ~ 1`, Q12.20 gives a relative precision of
  about `2^-20 ~ 1e-6`, roughly 16x finer than the Q2.16 output can resolve.

The original design used Q8.24, which saturates at 128. That was a genuine bug
found in regression: with `cond(A) ~ 2.7e4`, residual norms fell to 0.0024 and
the required scale factor was 455. The scalar saturated, `q_j` came out
non-unit, and orthogonality collapsed to 0.9 — while the RTL still matched the
bit-accurate model exactly, because the model had the same format. Widening to
Q12.20 trades 4 fractional bits (which were never observable) for 16x more
range.

## 5. Rank deficiency threshold

A column is declared linearly dependent when

```
||v||^2 < 2^EPS_SHIFT ,     EPS_SHIFT = 2*F - 2*(SC_W - 1 - SC_F)
```

which with the defaults is `2^10`, i.e. `||v|| < 2^-11 ~ 4.9e-4`.

This threshold is **derived, not chosen**: it is exactly the point at which
`1/||v||` stops fitting in the scalar format. Below it the engine cannot
normalise correctly, so it must not try. On detection it sets `R[j][j] = 0`,
writes `q_j = 0`, and raises the sticky `rank_def` output. The alternative — the
old behaviour — was to saturate and emit a plausible-looking but wrong `Q`,
which is the worst possible failure mode for a numerical block.

If you widen `SC_W` or reduce `SC_F`, the threshold automatically follows.
Raising `EPS_SHIFT` manually to reject weaker columns is safe. Lowering it
without widening the scalar re-introduces the saturation bug, which is why the
parameter carries a warning in the source.

## 6. Accuracy

For MGS the loss of orthogonality is bounded by `~ u * cond(A)` where `u` is the
unit roundoff. With `F=16`, `u = 2^-17`, so in units of an output LSB (`2^-16`):

```
||Q'Q - I||  ~  cond(A) / 2   LSB
```

Measured against that prediction:

| case                | cond(A) | predicted | measured  |
|---------------------|---------|-----------|-----------|
| identity            | 1.0     | 0.5 LSB   | 0.0 LSB   |
| hadamard            | 1.0     | 0.5 LSB   | 2.7 LSB   |
| random 8x4          | 3.3     | 1.7 LSB   | 1.8 LSB   |
| random 16x8         | 3.4     | 1.7 LSB   | 4.6 LSB   |
| random 64x8         | 1.8     | 0.9 LSB   | 2.9 LSB   |
| ill-conditioned 8x4 | 1.0e3   | 500 LSB   | 102 LSB   |
| ill-conditioned16x4 | 1.0e3   | 500 LSB   | 79 LSB    |

The measurements sit at or below the bound across four orders of magnitude of
conditioning, which is the signature of MGS rather than CGS. Under CGS the
`cond^2` term would have put the `cond=1e3` cases around `5e5` LSB — far past
full scale, i.e. total loss of orthogonality.

Reconstruction error `||A - QR||` stays near 1 LSB in all well-conditioned cases,
as expected: reconstruction is backward-stable for both CGS and MGS and does not
depend on conditioning.

## 7. Practical accuracy ceiling

Useful results require `cond(A)` comfortably below `2 * 2^(SC_W-1-SC_F) ~ 4e3`;
past that the engine flags `rank_def` instead of returning a wrong answer.
Independently, the input quantisation itself limits what is knowable: quantising
`A` to Q2.16 perturbs its singular values by about `sqrt(M*N) * 2^-17`, so a
matrix whose smallest singular value is below that is numerically
indistinguishable from a rank-deficient one no matter how the hardware behaves.

To go beyond, in increasing order of cost:

1. Raise `SC_W` (range and threshold move together, ~2 DSP per lane).
2. Raise `W` and `F` (whole datapath widens).
3. Use Householder reflections instead of Gram-Schmidt — unconditionally
   backward stable, orthogonality independent of `cond(A)`, at the cost of a
   much heavier and less streamable datapath.

## 8. Rounding mode

`round_sat` performs round-half-up (add `2^(SHIFT-1)`, arithmetic shift right)
followed by signed saturation, and exports an `ovf` bit. Round-half-up carries a
small positive bias compared to round-half-to-even; it is used because it costs
one adder rather than a comparison tree, and the bias is well below the
orthogonality error at every configuration measured above. If you need an
unbiased result, `round_sat` is the single place to change.
