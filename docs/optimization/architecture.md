# Optimization architecture

See the [package architecture](../architecture.md) for the complete source layout.
Optimization is part of PMPSLab; it shares MPS values, transfers, environments
and gauge operations directly.

Read `src/optimization/algorithm/config.jl`, then
`src/optimization/als/iteration.jl`. Follow `update!` into `update.jl`,
`workspace.jl` and `local_equation.jl` in the same directory.
`src/execution/run.jl` registers site events and provides the blocking shortcut;
`src/execution/snapshot.jl` captures numerical state and rebuilds caches.

The generator is driven by ExperimentLab. `split(target, algorithm)` waits for
completion. `start_run(algorithm_steps(state))` adds pause/resume control;
`start_run(store, algorithm_ref, algorithm_steps(state))` also associates logs with
an execution. Snapshot capture remains an explicit callback choice.
See [execution integration](execution-extensions.md).
