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
| `src/execution/` | Split log events, blocking entry point and numerical snapshots |
| `src/configuration.jl` | JSON names and composed configuration registry |

Local transfer constructors still accept site tensors; they do not require a
PeriodicMPS. Environments adapt MPS values to PeriodicProductsLab's generic
product caches. Optimization uses those operations without cross-package bridges.

## External responsibilities

- **ExperimentLab** owns configuration interchange, sessions, pause/resume,
  data records, storage and record-bound logs. It has no tensor dependencies.
- **PeriodicProductsLab** owns ordered products, cached environments and invalidation.
- **LatticeIteratorLab** supplies update positions and sweep ordering.
- **TensorLinearSolveLab** supplies local tensor equation solvers.
- **LoopTNRLab** composes MPS optimization with lattice filtering and RG steps.

PMPSLab yields `SplitProgress` after each completed site update. LoopTNRLab
forwards those events unchanged and emits its own RG/filtering events.
ExperimentLab controls execution at the yield boundaries. A numerical snapshot
captures the tensors and iteration state; derived caches are rebuilt on restore.

## Review path

Start with [construction](../examples/01_construction.jl), then
[local transfers](../examples/transfers/README.md),
[environments](../examples/environments/README.md), and
[a simple split](../examples/optimization/01_simple_split.jl).
For algorithm code, read [the split review guide](optimization/split_review.md).

The default exports cover common MPS operations and `split`/`ALSSplit`.
Specialized transfer representations, initialization policies, regularization,
and generator/recovery APIs are public qualified names, such as
`PMPSLab.SVDInitialization` and `ExperimentLab.algorithm_steps`.
