# Reviewing the split implementation

Start with [one site update](../../examples/optimization/03_one_site_update.jl).
It calls the A.I. interface once, checks which tensor changed, and compares the
reported cost with an independent dense contraction.

1. Read `src/optimization/interface.jl`: `SplitProblem`, `AbstractSplit` and
   `AbstractSplitState` implement the A.I. hierarchy.
2. Read `algorithm/config.jl` and `als/workspace.jl`: configuration, current
   `iterate`, numerical caches, sweep cursor and progress.
3. Read `als/iteration.jl`: one `AI.step!` commits one site update. Follow `update!`
   into `update.jl` and `local_equation.jl` for local solves and cache refresh.
4. Read `algorithm/stopping.jl` and `execution/run.jl`: cost criteria check completed
   sweeps; native A.I. criteria may stop at any site. The sweep schedule remains a
   hard limit. Stopping queries mark convergence before the next numerical update.
5. Read `execution/snapshot.jl`: tensors and criterion history are copied;
   traversal positions and caches are rebuilt on restoration.

`SplitProgress` is an immutable description of the last completed update.
Diagnostic history belongs to the caller. `exact_error` is only present at a sweep
boundary where it was actually measured. A.I.'s `iteration` counter includes the
currently attempted update; `completed_updates` counts committed tensor changes.

NNR uses the same state and step implementation. `als/regularization.jl` owns the
auxiliary/dual buffers, normalized local equations and soft thresholding. Tensors
remain in their original units. Multiply the normalized reported costs by
`progress.cost_scale` to recover physical squared-error units.

Tests cover sweep order, explicit initial ownership, compressed environments,
stopping exceptions, checkpoint restoration at every forward/backward boundary,
NNR buffers, and direct A.I. execution. Existing ExperimentLab recording and tree
configuration tests have been replaced by native execution and ordinary data tests.
No new public PMPS stepping protocol is introduced.
