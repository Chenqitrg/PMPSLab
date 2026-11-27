# Architecture

PMPSLab owns periodic MPS values and the numerical operations that act on them.
Representation, transfer matrices and optimization share one Julia module, with
separate source directories for review.

| Directory | Responsibility |
| --- | --- |
| `src/representation/` | Storage, periodic indexing, spaces and constructors |
| `src/contractions/` | Dense contraction and adjacent-site pairing |
| `src/transfers/` | Dense/factored local transfer matrices and truncation |
| `src/environments/` | Periodic transfer products, MPS caches, overlaps and norms |
| `src/gauge/` | Projector construction and gauge fixing |
| `src/optimization/algorithm/` | Initialization, stopping and regularization policies |
| `src/optimization/als/` | Local equations, workspaces and site updates |
| `src/execution/` | Native A.I. initialization, execution and numerical snapshots |
| `src/configuration.jl` | Value hashing for typed options |

Local transfer constructors still accept site tensors; they do not require a
PeriodicMPS. Environments adapt MPS values to PeriodicProductsLab's generic
product caches. Optimization uses those operations without cross-package bridges.

## External responsibilities

- **AlgorithmsInterface** supplies the algorithm/state lifecycle and stopping criteria.
- **DrWatson** can cache experiment results and record source metadata outside PMPSLab.
- **PeriodicProductsLab** owns ordered products, cached environments and invalidation.
- **LatticeIteratorLab** supplies update positions and sweep ordering.
- **TensorLinearSolveLab** supplies local tensor equation solvers.
- **LoopTNRLab** composes MPS optimization with lattice filtering and RG steps.

PMPSLab stores `SplitProgress` after each completed site update. Applications can
inspect the state, measure tensors or save a snapshot between calls to `AI.step!`.
A snapshot captures tensors, traversal and stopping history; caches are rebuilt on
restore. `AI.solve!` resets a workspace, so resume with a snapshot or the manual loop.

The legacy generator, configuration registry and recording integration have been
removed. DrWatson provides caching, not a replacement execution controller.
See [execution details](optimization/execution-extensions.md).

## Review path

Start with [construction](../examples/01_construction.jl), then
[local transfers](../examples/transfers/README.md),
[environments](../examples/environments/README.md), and
[a simple split](../examples/optimization/01_simple_split.jl).
For algorithm code, read [the split review guide](optimization/split_review.md).

The default exports cover common MPS operations and `split`/`ALSSplit`.
Specialized transfer representations, initialization policies, regularization,
and initialization/recovery APIs are public qualified names, such as
`PMPSLab.SVDInitialization` and `PMPSLab.SplitProblem`.
