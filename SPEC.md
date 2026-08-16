# PeriodicMPS — Current Specification

## State representation

```julia
struct PeriodicMPS{A <: GenericMPSTensor} <: MPSKit.AbstractMPS
    tensors::PeriodicVector{A}
end
```

`PeriodicMPS` is the only finite-periodic state type. It is finite
(`isfinite == true`), uses `PeriodicChainStyle`, and stores the tensors whose
closed contraction defines the physical state. There is no
`CanonicalPeriodicMPS`, `AL/AR/C` storage layer, cut closure, or finite-chain
public canonical-state API.

## Projector gauge fixing

```julia
@kwdef struct ProjectorGaugeFix{T<:TruncationStrategy,O} <:
        AbstractGaugeFix
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trunc::T = TruncationStrategy()
    orthogonalization_algorithm::O = MPSKit.Defaults.alg_orth()
end

gaugefix!(mps, algorithm) -> mps
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

`gaugefix!` first calls MPSKit's in-place `makefullrank!`. This is a local
QR/LQ structural cleanup: it changes incompatible virtual-space dimensions
without forming a dense state. The non-mutating `find_projectors` helper does
not perform this step and rejects a structurally rank-deficient input before
allocating MPSKit's fixed workspace.

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
intentional: it removes unused or discarded virtual directions. It need not be
an exact gauge transformation on the finite periodic state. In particular,
removing a closed redundant double line can change a global scalar even when
the projective physical state is unchanged; explicit truncation can also
change the projective state.

The state remains a `PeriodicMPS`, updated locally as

```math
A_i \leftarrow PL_{i-1/2} A_i PR_{i+1/2}.
```

With a truncating `TruncationStrategy`, the same operation is the approximate
entanglement filter: the bond dimension and physical periodic state may
change. Truncation is controlled explicitly by the algorithm rather than by a
canonical storage constructor.

`AL`, `AR`, and the normalized InfiniteMPS `C` are temporary fixed-point
objects only. In particular, MPSKit normalizes its boundary gauge during
InfiniteMPS regularization, so its returned unit-cell tensors cannot directly
replace the finite-periodic tensors or preserve their amplitude.

## Algorithm configuration values

Algorithm configurations subtype `AbstractAlgorithmConfig` and have value
semantics. Their fields are not mutated after construction. `@setconfig` derives
a new configuration by rebuilding every affected node through its keyword
constructor, so constructor validation still applies and a replacement may
change a field's concrete strategy type.

Paths use `/` from the root configuration to a leaf:

```julia
updated = @setconfig algorithm begin
    max_rg_steps = 4
    split/sweep/max_sweeps = 60
    split/stopping/atol = 1e-10
end
```

Assignments are applied from top to bottom. Unmentioned branches are reused;
the original configuration remains unchanged. Replacing a whole nested strategy
is supported, for example `split/stopping = NoStopping()`.

## Two-physical-leg target approximation

`PeriodicGaugeAnsatz` first performs its local SVD split and then calls
`gaugefix!(mps, ProjectorGaugeFix(...))` in place. The later split/ALS machinery
continues to work exclusively with `PeriodicMPS` values.

### Split cost log and observer

`split(target, algorithm)` currently returns `(approximation, log)`. Each
`SiteLog.cost` records
the cheap local quantity produced by `_solve_local_update`,

```math
c_{\mathrm{local}} = \operatorname{Re}
\left(\langle A|A\rangle - \operatorname{tr}(W x^\dagger)\right).
```

This equals the global squared distance when the local normal equation and its
mixed environment are exact and consistent. `SplitEnvironmentPolicy` selects
the transfer representation independently for the normal `⟨B|B⟩` and mixed
`⟨B|A⟩` caches. Both use `DenseTransferPolicy` by default;
`TruncatedTransferPolicy` keeps accumulated products in factored SVD form and
reapplies its truncation strategy after multiplication. `KrylovSolve` consumes
a truncated normal environment through its factored matrix action. Direct
solvers are also valid and materialize that local environment immediately before
solving. A truncated mixed environment is contracted into the right-hand side.

`SplitContext` maintains the mixed cache uniformly as `\langle B|A\rangle`;
its transfer factors are derived from completed B pairs only when cache
maintenance needs them. Exact dense-solver regression tests require the local
estimate to agree with a fresh complete-period contraction after every site and
require non-increase for both circular and zigzag sweeps. With a truncated
working environment, the logged value is an approximate local cost; explicit
`squared_distance` remains an exact dense finite-period check.

The public interface accepts an optional observer that receives each completed
site event:

```julia
split(target, algorithm; observer = event -> println(event.log.cost))
```

Omitting `observer` selects the split progress observer in an interactive Julia
session and remains silent in a non-interactive script. Passing
`observer=nothing` is always silent. A custom observer replaces the default and
receives the same [`SplitSiteCompleted`](@ref) event emitted by [`step!`](@ref).
It adds no contraction or MPS copy.

The cost in the last `SiteLog` is the objective value after the most recent update,

```math
c = \langle A|A\rangle + \langle B|B\rangle
    - 2\operatorname{Re}\langle B|A\rangle.
```

The default observer renders site progress and one persistent summary for each
completed sweep. Numerical stopping remains part of `ALSSplit`; pause and
cancellation remain execution-session responsibilities.

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
- `find_projectors` rejects structurally rank-deficient inputs; `gaugefix!`
  applies exact local full-rank cleanup before allocating the solver workspace.
- With no truncation, full-rank adjacent projectors cancel. Rank-deficient
  projectors are tested both on exactly preserved states and on CDL states
  where every corner bond is compressed and a closed global scalar may change.
- One- and multi-physical-leg tensors, real/complex scalars, and graded spaces
  use the same generic QR/LQ and absorption implementation.
- With truncation, errors are measured from a complete finite-period contraction.
- For `split` with an exact local solver, compare the local update against the
  complete periodic cost after every site and require non-increase before enabling
  the sweep observer.
