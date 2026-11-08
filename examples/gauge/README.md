# PMPSLab review examples

Each file runs independently and covers one feature. Start with 01–04 to follow
how matrices are inserted, constructed, and applied to an MPS. Then read 05 for
approximate compression and 06 for exact structural cleanup.

| File | Feature | Source to inspect |
| --- | --- | --- |
| [01_insert_gauge.jl](01_insert_gauge.jl) | Insert inverse matrices on the last-to-first bond without changing the state | [PMPSLab/container.jl](../../src/representation/container.jl), `insert_gauge!` |
| [02_projector_decomposition.jl](02_projector_decomposition.jl) | Construct `(PR, PL)` from two gauge matrices | `src/gauge/projector.jl`, `P_decomp` |
| [03_find_projectors.jl](03_find_projectors.jl) | Find one pair per bond without modifying the input | `src/gauge/projector.jl`, `find_projectors` |
| [04_gaugefix.jl](04_gaugefix.jl) | Compare explicit insertion with the complete in-place workflow | `src/gauge/interface.jl`, `gaugefix!` |
| [05_bond_compression.jl](05_bond_compression.jl) | Limit virtual dimensions and measure the resulting state change | `ProjectorGaugeFix(trunc=...)` |
| [06_structural_cleanup.jl](06_structural_cleanup.jl) | Remove structurally redundant dimensions before gauge solving | `gaugefix!`, MPSKit's `makefullrank!` |

Run from the PMPSLab repository:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/gauge/01_insert_gauge.jl
```

In your editor, select PMPSLab as the active Julia project. Examples use
only their own imports and end with `nothing`. Random examples use the default
random number generator; printed values vary, while the tested relationships
remain the same.

## Bond convention and ownership

Bond `i` is the bond to the right of site `i`, including the last-to-first bond.
For old bond space `V` and new bond space `W`, `PR[i]` has shape `V ← W` and
`PL[i]` has shape `W ← V`. Inserting all pairs gives the virtual-leg update
`A[i] → PL[i-1] * A[i] * PR[i]`, with the physical legs left intact.

- PMPSLab owns `insert_gauge!`: validate spaces and absorb the supplied matrices.
- PMPSLab chooses the matrices through `P_decomp` and `find_projectors`.
- `gaugefix!` performs structural cleanup, finds projectors, and inserts them.
- MPSKit supplies canonicalization and the shared left/right space conventions.

`find_projectors` runs left and right periodic canonicalization in temporary
storage to obtain the gauges used to construct the projectors. `gaugefix!` calls
it once, then absorbs the projectors. Its preceding structural cleanup only
performs local QR/LQ transformations where dimensions are incompatible; it does
not repeat the periodic gauge solve. The dimension restriction of
`find_projectors` comes from the current solver workspace, not from its
non-mutating interface or a requirement that the MPS be injective.

An inverse pair preserves the state. A full-rank, well-conditioned decomposition
without truncation also gives `PR * PL ≈ I`. Rank-deficient or truncated pairs
can instead project onto a subspace. Exact structural cleanup removes redundant
dimensions without changing the state; requested truncation can change it.

The examples compare small dense states directly through PMPSLab and TensorKit.
They do not require PMPSLab, optimization, or execution APIs.

[07_configuration_tree.jl](07_configuration_tree.jl) separately demonstrates the
new tree constructor and structured-data round trip for projector settings.
