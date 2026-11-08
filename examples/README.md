# PMPSLab examples

Each numbered file demonstrates one feature. Every file imports its own
dependencies and creates its own inputs, so you can run any example independently.
Start with construction and indexing; then choose the operation you want to review.

Run from the PMPSLab repository directory:

```sh
julia --project=. examples/01_construction.jl
```

| File | Feature | Implementation to review |
| --- | --- | --- |
| [01_construction.jl](01_construction.jl) | Random, all-ones, and multiple-physical-leg construction | [constructors.jl](../src/representation/constructors.jl) |
| [02_periodic_indexing.jl](02_periodic_indexing.jl) | Periodic site indices and MPSKit left/right space queries | [container.jl](../src/representation/container.jl) |
| [03_dense_contraction.jl](03_dense_contraction.jl) | Full contraction checked against ordinary matrix traces | [contraction.jl](../src/contractions/contraction.jl) |
| [04_pair_contraction.jl](04_pair_contraction.jl) | Adjacent pairing and physical-leg ordering | [contraction.jl](../src/contractions/contraction.jl) |
| [05_gauge_insertion.jl](05_gauge_insertion.jl) | Cancellation of a unitary gauge pair | [container.jl](../src/representation/container.jl) |
| [06_validation.jl](06_validation.jl) | Invalid bond dimensions, pairing, and noninteger site indices | [container.jl](../src/representation/container.jl), [contraction.jl](../src/contractions/contraction.jl) |
| [07_tensor_mapping.jl](07_tensor_mapping.jl) | Applying a function to each tensor | [container.jl](../src/representation/container.jl) |
| [08_finite_lift.jl](08_finite_lift.jl) | Repeating a finite unit cell | [lift.jl](../src/representation/lift.jl) |
| [09_symmetry.jl](09_symmetry.jl) | Z₂-graded spaces and state-preserving pairing | [constructors.jl](../src/representation/constructors.jl), [contraction.jl](../src/contractions/contraction.jl) |

The `@test` statements check the behavior demonstrated by each example.
`@test_throws` verifies an expected error without stopping the example.
The examples use tiny chains because full dense contraction grows exponentially
with the number of physical legs.

Each file creates `rng = Xoshiro(20261102)` for reproducibility. The generator is
optional: `rand(PeriodicMPS, ComplexF64, N, d, D)` uses Julia's default generator.

To run all examples together, the original walkthrough path remains a short runner:

```sh
julia --project=. examples/review_pmps.jl
```

## Numerical operations

After the representation examples, review [gauge fixing](gauge/README.md) and
[transfer environments](environments/README.md). Each example runs independently
from the PMPSLab project and can be evaluated at the REPL.

## Transfers and optimization

- [Local transfers](transfers/README.md): construction, composition and truncation.
- [Optimization](optimization/README.md): initialization, site updates, convergence,
  nuclear-norm regularization, and ExperimentLab-controlled execution.

All examples use this repository's root environment. No retired sibling package
is needed. Start the optimization review with
[01_simple_split.jl](optimization/01_simple_split.jl).
