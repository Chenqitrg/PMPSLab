# Review the numerical split first

The ALS generator shows the numerical flow; ExperimentLab supplies its execution driver.
Start with [example 01](../../examples/optimization/01_simple_split.jl), then
[example 03](../../examples/optimization/03_one_site_update.jl), which inspects the first completed site through an ExperimentLab callback.
Example 01 names the packages supplying the data, sweep and solver. Example 03
inspects the actual transfer values and caches used by the split. Continue with
[07](../../examples/optimization/07_exact_error.jl) for compressed transfer policies and
[08](../../examples/optimization/08_gauge_initialization.jl) for delegated gauge fixing.

## What became smaller?

The `algorithm/` directory defines choices, `als/` implements numerical updates,
and `execution/` connects them to ExperimentLab. The old execution extension
and run wrapper have been removed.
Run `julia --project=. -e 'using Pkg; Pkg.test(test_args=["numerical"])'` to check the numerical code alone.
Execution integration tests live separately in `test/execution/`.

Generic traversal and tensor linear solves now live in LatticeIteratorLab and
TensorLinearSolveLab. Their numerical implementations remain independent of session control.
Diagnostic history belongs to ExperimentLab. The runtime model has three types:

- `SplitState`: tensors, caches, traversal and internal stopping-rule memory.
- `SplitProgress`: one immutable completed-update summary, also used for logging.
- `SplitSnapshot`: detached numerical restart data without caches.

The return value is simply `(mps=..., progress=...)`.

## First pass: the optimization flow

Read these three files in order:

| File | Question |
| --- | --- |
| [run.jl](../../src/execution/run.jl) | How does `split` run to completion? |
| [iteration.jl](../../src/optimization/als/iteration.jl) | How does ALS iterate, check convergence and yield between sites? |
| [update.jl](../../src/optimization/als/update.jl) | How is the tensor solved, replaced, and its cache refreshed? |

`split_steps` directly implements the ALS loop. It updates a site, refreshes
caches, checks sweep-boundary convergence, logs a `SplitProgress` and yields it.
In the same file, `_exact_error` measures at sweep boundaries and
`_finish_update!` updates stopping-rule memory and constructs the progress value.
Read the main loop first, then these helpers. `state.progress`, the callback
argument and the log payload describe the same completed update.
ExperimentLab controls whether the next iteration may run. There is no separate
`step!`, `isfinished` or result-extraction protocol. `split` starts the same
generator and waits for its result. Numerical state never stores update history.
The local solve, quadratic cost and cache refresh are now together in `update.jl`.

## Second pass: the tensor mathematics

- [workspace.jl](../../src/optimization/als/workspace.jl): numerical state, initialization, target/partner indexing and cached environments.
- [local_equation.jl](../../src/optimization/als/local_equation.jl): normal operator and right-hand side.
- [TensorLinearSolveLab/solve.jl](../../../TensorLinearSolveLab/src/solve.jl): the selected local linear solver.
- [initialization.jl](../../src/optimization/algorithm/initialization.jl): starting approximation.

Then consult [LatticeIteratorLab/sweep.jl](../../../LatticeIteratorLab/src/sweep.jl) and
[stopping.jl](../../src/optimization/algorithm/stopping.jl) for the particular policies you use.
The [example sequence](../../examples/optimization/README.md) separates these features.

For JSON interchange, read [example 12](../../examples/optimization/12_configuration_json.jl),
then [configuration.jl](../../src/configuration.jl). This file names the
configuration types and composes the sibling registries; it adds no parameters
to the numerical structs. ExperimentLab applies that registry when writing
configuration files or recording algorithms and measurements.

Example 03 checks the local cost against an independent whole-period distance.
The full quadratic objective remains valid for incomplete solves with exact
environments. Compressed environments give an estimate; example 07 demonstrates
`error_measure=:exact` and its additional global contraction per completed sweep.
Example 06 compares copied and borrowed initial tensors.

## Optional: controlled execution and recovery

Read [iteration.jl](../../src/optimization/als/iteration.jl) for the generator and
[run.jl](../../src/execution/run.jl) for logging and the blocking entry point.
Then read [snapshot.jl](../../src/execution/snapshot.jl) for numerical
restart validation and cache rebuilding. Generic persistence belongs to ExperimentLab.
See [the integration guide](execution-extensions.md).

Read `state.progress` for numerical metadata. Read
`ExperimentLab.run_status(session)` for execution status. Inspect mutable tensors
only after an acknowledged pause, inside `onstep`, or after completion.

## Nuclear-norm regularization

Start with [example 09](../../examples/optimization/09_nuclear_norm.jl).
[algorithm/regularization.jl](../../src/optimization/algorithm/regularization.jl) contains the
coupled penalty schedule. [als/regularization.jl](../../src/optimization/als/regularization.jl)
contains the normalized local equation, sector-wise soft threshold and dual
update. The existing generator and caches serve both regularized and ordinary
ALS. Snapshot payloads additionally carry NNR auxiliary/dual arrays when enabled.

## Existing numerical limits

`update!` is not transactional: a cache-refresh failure can happen after tensor
replacement. Recover from a known snapshot if using controlled execution.
The cache adapter translates LatticeIteratorLab positions to PeriodicProductsLab's
existing cache-state API. The two lower-level packages do not depend on each other.
