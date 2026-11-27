# PMPSLab

Periodic MPS construction, contraction, transfer matrices, gauge fixing and optimization in Julia. ALS splitting supports nuclear-norm regularization, per-site progress, manual stepping and numerical snapshots through AlgorithmsInterface.

## Use

```julia
using PMPSLab
mps = one_mps(ComplexF64, 4, 2, 1)
@assert length(mps) == 4
state = dense(mps)
paired = contract_pairs(mps)
```

## Optimize

```julia
using PMPSLab
import AlgorithmsInterface as AI
import LatticeIteratorLab, TensorLinearSolveLab

target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
algorithm = ALSSplit(
    initialization=PMPSLab.SVDInitialization(),
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
    solver=TensorLinearSolveLab.CholeskySolve())
result = AI.solve(PMPSLab.SplitProblem(target), algorithm)
result.mps       # Optimized MPS with paired single-physical-leg sites.
result.progress  # Last completed site update.
```

`ALSSplit <: AlgorithmsInterface.Algorithm` and
`SplitState <: AlgorithmsInterface.State`. The convenience function
`split(target, algorithm)` calls `AI.solve`. For control between site updates,
use `AI.initialize_state`, `AI.is_finished!`, `AI.increment!` and `AI.step!`.
See [execution and restart](docs/optimization/execution-extensions.md) for a complete loop.

Capture numerical restart data with `PMPSLab.snapshot(state)` before execution,
between completed updates, or after completion:

```julia
state = PMPSLab.initialize_split(target, algorithm)
saved = PMPSLab.snapshot(state)
restored = PMPSLab.restore_split(saved, target)
```

The snapshot owns its tensors and contains no numerical caches. Restore with the
original target to rebuild those caches. Saving the snapshot to disk is the
caller's responsibility. DrWatson can cache complete runs in an experiment project;
it is not required by PMPSLab.

## Review order

Start with the [example index](examples/README.md). The representation examples each
demonstrate one feature and point to the implementation to review. Begin with
construction, then periodic indexing.

```sh
julia --project=. examples/01_construction.jl
```

1. `representation/container.jl`: indexing, storage, and consistency.
2. `representation/constructors.jl`, `representation/cdl.jl`: construction and examples.
3. `contractions/contraction.jl`: direct contraction and pairing.
4. `representation/lift.jl`: MPSKit conversion.
5. `gauge/`: projector construction and in-place gauge fixing.
6. `transfers/` and `environments/`: local operators, periodic caches, overlaps and norms.
7. `optimization/`: split policies, local equations and site updates.
8. `execution/`: A.I. integration and numerical restart.

All source files are in `src/`; focused regression tests live in `test/`.

## Site and virtual-space convention

Space queries are the functions from MPSKit:

```julia
using MPSKit: left_virtualspace, right_virtualspace

right_virtualspace(mps, i) == left_virtualspace(mps, i + 1)
right_spaces = right_virtualspace(mps)  # periodically indexed, one entry per site
```

For vector-based `rand` and `one_mps` constructors, `V[i]` is the **right** virtual
space of site `i`, matching MPSKit's `InfiniteMPS` constructor. The left space is
`V[i-1]` with periodic wrapping. Ordinary vectors and MPSKit `PeriodicVector`s are
accepted. For nondual input spaces, `right_virtualspace(mps, i) == V[i]`;
construction standardizes dual virtual spaces using gauge transformations.

`insert_gauge!(mps, i, R, L)` inserts matrices on the right bond of integer site
`i`, between sites `i` and `i+1`. There is no half-integer bond-label API.
Replace the former `HalfPeriodicVector` with `PeriodicVector` or a plain vector,
`virtualspaces(mps)` with `right_virtualspace(mps)`, and `mps[half(2i+1)]` with
`right_virtualspace(mps, i)`. Replace a former `virtualspace(mps)` query with an
explicit `left_virtualspace(mps, i)` or `right_virtualspace(mps, i)` query.

Saved configurations using the removed bond-vector type require their original
environment; automatic checkpoint migration is not provided. Package versions
remain unchanged.

## Dependencies and local tests

Uses TensorKit and MPSKit for tensors and canonicalization, MatrixAlgebraKit for
factorizations, PeriodicProductsLab for cached products, LatticeIteratorLab for sweep
ordering, TensorLinearSolveLab for local solves, and AlgorithmsInterface for
execution and stopping. Keep the Lab dependencies
as sibling checkouts. Julia 1.12 or later is required.

From this repository:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Package boundary

GaugeFixingLab, TransferEnvironmentsLab, TransferMatricesLab and MPSSplitLab
are consolidated into PMPSLab. Their implementations, configuration identities,
tests and examples use the PMPSLab namespace. The original local Git repositories
are preserved in `../.retired-labs/`; active environments do not load them.
No compatibility aliases or serialized-checkpoint migration are provided.

LoopTNRLab remains responsible for lattice operations and RG. Legacy configuration
registries, generator sessions and recorded-run APIs have been retired. Existing
serialized records require their original environment. ExperimentLab remains a
transitive dependency of the sweep and solver packages until those migrate. See [the architecture](docs/architecture.md)
and [the example index](examples/README.md) for review paths.

PMPSLab retains UUID `671cd7e2-a4a1-40fd-ad7d-501014e12597` and version `0.1.0`.

## Origin

This research code builds on TensorKit.jl and MPSKit.jl. We are grateful to their
authors and contributors, and to the
[TNRKit.jl](https://github.com/QuantumKitHub/TNRKit.jl) community, whose LoopTNR
work has been a valuable reference for the periodic MPS and optimization components.

The idea of separating algorithm configuration from the iterator that drives
execution comes from Zhengyuan Yue's [TRG implementation](https://github.com/Yue-Zhengyuan/TNRKit.jl/blob/9cc5088bcca55ff9c5a72403764acfe8e9ceeb21/src/schemes/trg.jl)
and its [Renormalizer iterator](https://github.com/Yue-Zhengyuan/TNRKit.jl/blob/9cc5088bcca55ff9c5a72403764acfe8e9ceeb21/src/schemes/renormalizer.jl)
in his TNRKit fork. We are grateful to him for sharing this design and for the
helpful discussions that informed this work.

We also thank Zhengyuan Yue for his discussions on
[MPS conventions and algorithm interfaces](https://github.com/QuantumKitHub/TNRKit.jl/pull/104)
and his [review and suggestions on nuclear-norm regularization](https://github.com/QuantumKitHub/TNRKit.jl/pull/144#discussion_r2922211049).

Earlier standalone package history is preserved in the retired repositories.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
The `pseudopow` helper includes code adapted from TNRKit.jl; its upstream
copyright and MIT license are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
