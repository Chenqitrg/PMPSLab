# PeriodicMPS — Current Specification

## State representation

```julia
struct PeriodicMPS{A <: GenericMPSTensor} <: MPSKit.AbstractMPS
    tensors::PeriodicVector{A}
end
```

`PeriodicMPS` is the only finite-ring state type. It is finite
(`isfinite == true`), uses `PeriodicChainStyle`, and stores the tensors whose
closed contraction defines the physical state. There is no
`CanonicalPeriodicMPS`, `AL/AR/C` storage layer, cut closure, or finite-chain
`canonicalize` API.

## Projector gauge fixing

```julia
@kwdef struct ProjectorCanonical{T<:TruncationStrategy,O} <:
        AbstractLoopCanonical
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trscheme::T = TruncationStrategy()
    alg_orth::O = MPSKit.Defaults.alg_orth()
end

gaugefix!(mps, alg) -> mps
```

The algorithm reuses the mathematical regularization scheme of an InfiniteMPS
without treating its `AL`, `AR`, or `C` as the finite state.

At one reference bond, the left gauge is the fixed point of a full-cell QR
sweep,

```math
L_{i-1} A_i = A_{L,i} L_i,
```

and the right gauge is the fixed point of the corresponding reverse LQ sweep,

```math
A_i R_i = R_{i-1} A_{R,i}.
```

The solve uses direct periodic QR/LQ power iteration throughout. The adapter
disables MPSKit's optional mixed-transfer eigensolver acceleration and fixes
verbosity to `VERBOSE_NONE`; projector gauge fixing does not emit or return the
solver's temporary native log.

Only one left and one right fixed-point problem are solved. MPSKit's internal
solvers return all bond gauges from those two solves, so no per-bond restart is
needed. The work is `O(iter*N)`, not `O(iter*N^2)`.

MPSKit fixes one overall unit-cell scale by normalizing its closing gauge. For
projector construction, every returned `L[b]` and `R[b]` is normalized after the
solve and stored in a `HalfPeriodicVector`, matching the old per-reference-cut
entanglement-filter convention.

## Projectors and in-place update

At bond `b = i + 1/2`, define the mixed gauge

```math
C_b = L_b R_b.
```

Its SVD produces projectors `PR[b]` and `PL[b]` on the same bond. If the mixed
gauge is full rank and no truncation is requested,

```math
PR_b PL_b \simeq I,
```

For a rank-deficient or truncated gauge, `PR_b PL_b` is instead the projector
onto the supported or retained virtual subspace. This nontrivial projection is
intentional: it removes unused or discarded virtual directions. In the
rank-deficient untruncated case it still preserves the represented finite-ring
state exactly; explicit truncation may change that state.

The state remains a `PeriodicMPS`, updated locally as

```math
A_i \leftarrow PL_{i-1/2} A_i PR_{i+1/2}.
```

With a truncating `TruncationStrategy`, the same operation is the approximate
entanglement filter: the bond dimension and physical finite-ring state may
change. Truncation is controlled explicitly by the algorithm rather than by a
canonical storage constructor.

`AL`, `AR`, and the normalized InfiniteMPS `C` are temporary fixed-point
objects only. In particular, MPSKit normalizes its boundary gauge during
InfiniteMPS regularization, so its returned unit-cell tensors cannot directly
replace the finite-ring tensors or preserve the finite-ring amplitude.

## 2-to-1 approximation

`OneLoopAnsatz` first performs its local SVD split and then calls
`gaugefix!(mps, ProjectorCanonical(...))` in place. The later split/ALS machinery
continues to work exclusively with `PeriodicMPS` values.

### Split cost log and callback

`split(mps, alg)` currently returns `(mpsB, log)`. Each `SiteLog.error` records
the cheap local quantity produced by `solvenew`,

```math
c_{\mathrm{local}} = \operatorname{Re}
\left(\langle A|A\rangle - \operatorname{tr}(W x^\dagger)\right).
```

This equals the global squared distance when the local normal equation and its
mixed environment are consistent. `SplitContext` maintains `braketAB` uniformly
as `\langle B|A\rangle`; completed B pairs are therefore written with
`site_transfer_matrix(B₁, B₂, A)`. Exact-solver regression tests require the
local estimate to agree with a fresh complete-loop contraction after every
site and require non-increase for both circular and zigzag sweeps.

The public interface provides a synchronous optional callback after each
complete sweep:

```julia
split(mps, alg; callback = nothing)
callback(log)
cost = last(log.site_logs).error
```

No separate progress type or duplicate cost storage is needed. The error in the
last `SiteLog` is the cost after the most recent update,

```math
c = \langle A|A\rangle + \langle B|B\rangle
    - 2\operatorname{Re}\langle B|A\rangle.
```

The callback fires at a change of `SweepState.n` and once after the final
sweep. It receives the same mutable `SplitLog` that is eventually returned and
adds no contraction or MPS copy. Rendering, channels, progress bars, stopping
policy, and cancellation remain frontend responsibilities.

## Norm semantics

```julia
squared_norm(mps) = real(dot(mps, mps))
norm(mps) = sqrt(squared_norm(mps))
squared_distance(a, b) =
    squared_norm(a) + squared_norm(b) - 2real(dot(a, b))
```

## Required tests

- All bond gauges are obtained by one direct MPSKit left solve and one direct
  MPSKit right solve, without copying the input tensors.
- With no truncation, full-rank adjacent projectors cancel. Rank-deficient
  projectors act nontrivially only outside the represented support, and `dense`
  remains unchanged.
- One- and multi-physical-leg tensors, real/complex scalars, and graded spaces
  use the same generic QR/LQ and absorption implementation.
- With truncation, errors are measured from a complete finite-loop contraction.
- For `split` with an exact local solver, compare the local update against the
  complete loop cost after every site and require non-increase before enabling
  the sweep callback.
