# PMPSLab

Finite periodic MPS storage, tensor spaces, constructors, direct contraction, gauge insertion, and lifting to MPSKit representations.

## Use

```julia
using PMPSLab
mps = one_mps(ComplexF64, 4, 2, 1)
@assert length(mps) == 4
state = dense(mps)
paired = contract_pairs(mps)
```

## Review order

Start with the [example index](examples/README.md). Nine independent files each
demonstrate one feature and point to the implementation to review. Begin with
construction, then periodic indexing.

```sh
julia --project=. examples/01_construction.jl
```

1. `container.jl`: indexing, storage, and consistency.
2. `constructors.jl`, `cdl.jl`: construction and examples.
3. `contraction.jl`: direct contraction and pairing.
4. `lift.jl`: MPSKit conversion.

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

Uses TensorKit, MPSKit, and Random. Transfer environments and gauge-fixing algorithms live in separate sibling packages.

From this repository:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Ownership and package rename

This package owns `PeriodicMPS`, its indexing and constructors, tensor-space
queries, direct contraction, gauge insertion, and MPSKit conversion. Optimization
belongs to [MPSSplitLab](../MPSSplitLab); load both for splitting workflows.
See the [package architecture](../MPSSplitLab/docs/architecture.md).

Formerly named `MPSRepresentationsLab`, it retains UUID
`671cd7e2-a4a1-40fd-ad7d-501014e12597`. The former optimizer called PMPSLab is now
MPSSplitLab and retains its own UUID. This package does not expose split algorithms
or execution controls. Version remains `0.1.0`.

Checkpoints created before the rename require their original package environment;
no automatic migration of serialized module paths is provided.

## Origin

Extracted from the earlier PMPSLab umbrella package without changing tensor equations.

This research code builds on TensorKit.jl and MPSKit.jl; the numerical work also
draws on TNRKit.jl where applicable. Credit remains with those projects and their
authors and contributors. See the sibling MPSSplitLab README for the original attribution.
