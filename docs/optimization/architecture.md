# Optimization architecture

PMPSLab owns tensors, local split equations, transfer environments and numerical
restart data. AlgorithmsInterface owns the common problem/algorithm/state protocol
and solver driver. Experiment scripts own persistence, measurements and history.

Read `src/optimization/interface.jl` for the A.I. types, then
`src/optimization/algorithm/config.jl` for `ALSSplit`. The site update is
`AI.step!` in `src/optimization/als/iteration.jl`; it delegates tensor solves and
cache refresh to `update.jl`. `src/execution/run.jl` implements initialization,
stopping queries, finalization and the `split` convenience call.

Stopping criteria live in `src/optimization/algorithm/stopping.jl`. Their mutable
history belongs to the state, including when criteria are nested in A.I. groups.
`src/execution/snapshot.jl` copies that history and reconstructs traversal and caches.

See [execution integration](execution-extensions.md) for manual control, callbacks,
checkpoint boundaries and the distinction between reset and resume.
