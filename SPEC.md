# PeriodicMPS — Specification

Gauge fixing of Matrix Product States on a loop.

## Type Hierarchy

Follows MPSKit's pattern: the number of physical legs (`site`) is encoded in the tensor type parameter,
not in the state type.

```julia
using MPSKit: AbstractMPS, GenericMPSTensor, MPSTensor, PeriodicVector

# Subtype MPSKit.AbstractMPS directly
struct PeriodicMPS{A <: GenericMPSTensor} <: AbstractMPS
    tensors::PeriodicVector{A}
end

# Convenience aliases
const TwoSitePeriodicMPS{S} = PeriodicMPS{MPSTensor{S}}              # GenericMPSTensor{S, 2}
const OneSitePeriodicMPS{S}  = PeriodicMPS{GenericMPSTensor{S, 1}}
```

- `A`: tensor type, e.g. `MPSTensor{S}` (2-site) or `GenericMPSTensor{S, 1}` (1-site)
- `S`: symmetry sector type (from TensorKit, extracted via `spacetype`)
- Element type `E` accessed via `scalartype` / `storagetype` (MPSKit interface)
- Container: `PeriodicVector` from MPSKit — wrap-around indexing (`tensors[0] == tensors[end]`)
- No `periodicity` type parameter — loop length N is `length(tensors)`

## Constructors

```julia
# @unchecked — trust caller, no validation (performance path)
PeriodicMPS{A}(tensors::PeriodicVector{A}) where {A <: GenericMPSTensor}

# @checked — validates leg-matching around the loop (calls assertconsistency)
PeriodicMPS(tensors::Vector{<:GenericMPSTensor})
PeriodicMPS(tensors::PeriodicVector{<:GenericMPSTensor})
```

Convenience random initializers:
```julia
PeriodicMPS(physicalspaces::Vector{S}, virtualspaces::Vector{S}) where {S}
PeriodicMPS(As::Vector{<:MPSTensor})  # common 2-site case
```

## Interface (inherited from `AbstractMPS`)

Must implement: `left_virtualspace`, `right_virtualspace`, `physicalspace`, `eachsite`.

Provided by `AbstractMPS`: `site_type`, `bond_type`, `spacetype`, `sectortype`, `eltype`, `checkbounds`,
`scalartype`.

## Consistency Check

```julia
function assertconsistency(mps::PeriodicMPS)
```

- Checks: for each site `i`, `right_virtualspace(mps, i) == left_virtualspace(mps, i+1)` (with wrap-around at `i == N`). Uses TensorKit space compatibility.
- Throws a descriptive `SpaceMismatch` on failure (not a `Bool` return).

## Convergence Log

All iterative algorithms return a `ConvergenceLog` alongside their primary output.

```julia
@kwdef struct ConvergenceLog
    converged::Bool                 # true if converged within tol
    iterations::Int                 # number of sweeps taken
    sweep_times::Vector{Float64}    # wall-clock time per sweep (seconds)
    error_history::Vector{Float64}  # error at each iteration
    cost_history::Vector{Float64}   # cost at each iteration (if applicable)
end
```

## Canonicalization — Gauge Fixing

A benchmarking framework for loop gauge fixing. Multiple algorithm variants compete on the same
task: bring a `PeriodicMPS` into a uniform canonical form by absorbing transfer matrix fixed-point
projectors into the site tensors.

```julia
# Dispatch interface
function canonicalize!(mps::PeriodicMPS, alg::AbstractLoopCanonical) -> ConvergenceLog
```

### Algorithm Hierarchy

```julia
abstract type AbstractLoopCanonical <: Algorithm end

# TNRKit's projector approach — find_projectors → absorb into tensors
@kwdef struct ProjectorCanonical <: AbstractLoopCanonical
    tol::Float64              # convergence tolerance
    maxiter::Int              # max iterations
    trscheme::TruncationScheme # truncation for projector SVD
    alg_orth                  # orthogonalization algorithm (default: QR with positive Householder)
    alg_eigsolve              # eigensolver algorithm for fixed point
    eig_miniter::Int          # min iterations before using eigensolver
end

# Nuclear norm regularization approach
@kwdef struct NuclearNormCanonical <: AbstractLoopCanonical
    tol::Float64
    maxiter::Int
    trscheme::TruncationScheme
    ρ::Float64
    ξ_init::Float64
    ξ_min::Float64
    # ...
end
```

### Common Behavior

- Mutates `mps.tensors` in-place.
- Targets TNRKit's uniform canonical form: find the fixed-point left/right environments of the full-loop transfer matrix, decompose into projectors `PL, PR` at each bond, absorb projectors into site tensors.
- Iterative algorithm: fixed-point eigensolve step → orth sweep around the loop with `left_orth!`/`right_orth!` → convergence check.
- Returns `ConvergenceLog` with iteration data for benchmarking.
- Each variant may use different mechanics (projector decomposition, nuclear norm projection, etc.).

### Dependencies

- `fixedpoint` from MPSKit (wraps KrylovKit `eigsolve`)
- `left_orth!`/`right_orth!` from MatrixAlgebraKit
- `TransferMatrix` / `ProductTransferMatrix` from MPSKit

## Observables

### Loop Entanglement Entropy

```julia
function loop_entropy(mps::PeriodicMPS) -> Float64
```

- Gauge-invariant observable per [Evenbly 2018](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235155).
- Forms the double-layer transfer matrix at each site: `T[i] = tensors[i] ⊗ conj(tensors[i])`.
- The full-loop transfer matrix product is independent of where the cut is made
  (`det(AB - λI) = det(BA - λI)` — all cuts give the same spectrum).
- Returns the von Neumann entropy of the transfer matrix eigenvalues.

This is *not* bipartite entanglement entropy — it measures short-range entanglement in the loop MPS
via the transfer matrix spectrum, without requiring a canonical form.

### Expectation Values

```julia
function expect(mps::PeriodicMPS, operator, site::Int)
```

- Inserts a local operator at `site` and computes the loop trace via `@plansor`.

### Norm

```julia
function norm(mps::PeriodicMPS)
```

- Squared norm of the loop state (trace of the full transfer matrix product).

## 2-to-1 Site Approximation

A benchmarking framework for loop optimization. Multiple algorithm variants compete on the same
task: given a `TwoSitePeriodicMPS`, produce a `OneSitePeriodicMPS` that best approximates the
loop state.

```julia
# Dispatch interface
function approximate(mps::TwoSitePeriodicMPS, alg::AbstractLoopApproximation) -> (OneSitePeriodicMPS, ConvergenceLog)
```

### Algorithm Hierarchy

```julia
abstract type AbstractLoopApproximation <: Algorithm end

# TNRKit's current approach — Krylov linsolve sweep (LU decomposition)
@kwdef struct LULoopApprox <: AbstractLoopApproximation
    tol::Float64              # convergence tolerance for cost function
    maxiter::Int              # max sweeps
    trscheme::TruncationScheme # truncation for SVD step
    krylovdim::Int            # Krylov subspace dimension
    maxkryloviter::Int        # max Krylov iterations per site
end

# ZigZag two-site sweeping pattern
@kwdef struct ZigZagLoopApprox <: AbstractLoopApproximation
    tol::Float64
    maxiter::Int
    trscheme::TruncationScheme
    # ...
end

# Alternating direction (AD) method
@kwdef struct ADLoopApprox <: AbstractLoopApproximation
    tol::Float64
    maxiter::Int
    # ...
end

# Nuclear norm minimization
@kwdef struct NuclearNormLoopApprox <: AbstractLoopApproximation
    tol::Float64
    maxiter::Int
    # ...
end
```

### Common Behavior

- Optimization target: minimize `||loop_state(2-site) - loop_state(1-site_approximation)||²`.
- Cost function: `C + TNT - WdT - dWT` (L2 distance, lazy-computed — constant terms cached, per-site terms incremental, TNRKit pattern).
- Returns a new `OneSitePeriodicMPS` (pure, non-mutating).
- Each algorithm may use different internal mechanics (Krylov solves, SVD, alternating optimization, nuclear norm projection, etc.).

## Module Structure

```
src/
  PeriodicMPS.jl        # module entry, exports
  types.jl              # PeriodicMPS, type aliases, constructors
  assert.jl             # assertconsistency
  canonicalize.jl       # LoopCanonical, canonicalize!
  observables.jl        # entropy, expect, norm
  approximate/
    interface.jl        # AbstractLoopApproximation, approximate dispatch
    lu.jl               # LULoopApprox (TNRKit Krylov sweep)
    zigzag.jl           # ZigZagLoopApprox
    ad.jl               # ADLoopApprox
    nuclearnorm.jl      # NuclearNormLoopApprox
```

## Dependencies

- **TensorKit.jl** — tensor types, symmetries, `@plansor`
- **MPSKit.jl** — `AbstractMPS`, `GenericMPSTensor`, `MPSTensor`, `PeriodicVector`, `TransferMatrix`, `fixedpoint`, `regauge!`, `entanglement_spectrum`, `entropy`, `@plansor`
- **KrylovKit.jl** — `eigsolve`, `linsolve` (via MPSKit's `fixedpoint` and direct use in `approximate`)
- **MatrixAlgebraKit** — `left_orth!`, `right_orth!`, `svd_trunc` (via MPSKit)

## Testing

Standard `@testset` + `Test`:
- Unit tests: `assertconsistency` with valid/invalid tensors
- Integration tests: small random `PeriodicMPS` → `canonicalize!` → verify projector idempotence and norm conservation
- 2-to-1: small random `TwoSitePeriodicMPS` → `approximate` → verify cost function decreases

## Build Order

1. Types + constructors
2. `assertconsistency`
3. `canonicalize!`
4. Observables
5. `approximate` (2-to-1)
