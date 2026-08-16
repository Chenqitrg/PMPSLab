# PMPSLab

Experimental tools for finite periodic matrix-product states, projector-based
periodic gauge fixing, 2-to-1 splitting, and resumable local execution.

## Credits and upstream foundations

PMPSLab is downstream research software built on the work of
[TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl),
[MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl), and
[TNRKit.jl](https://github.com/QuantumKitHub/TNRKit.jl), and on the work of
their authors and contributors. TensorKit provides the symmetry-aware tensor,
space, contraction, and factorization foundations used throughout PMPSLab.
MPSKit provides core MPS and algorithm interfaces, periodic indexing, and the
canonicalization machinery on which PMPSLab builds. Although TNRKit is not a
runtime package dependency, PMPSLab's Loop-TNR design and parts of its
numerical implementation draw on the algorithms and earlier implementation
work in TNRKit. PMPSLab should therefore be understood as building on these
projects, not as an independent replacement for them.

## Scope

PMPSLab is a tensor-network research toolbox rather than a prescribed
end-to-end application. It exposes periodic MPS containers, transfer products
and caches, gauge fixing, truncation policies, local solvers, sweep iterators,
split runs, progress events, cooperative pause/cancel control, and durable
checkpoints as composable pieces.

The current execution layer is intentionally local. It does not provide job
IDs, a persistent job registry, remote workers, automatic retry, or a network
control protocol. Those concerns belong above the numerical run interface.

## Start the project

From the repository root, start Julia with this environment:

```sh
julia --project=.
```

The examples below use:

```julia
using PMPSLab
using MPSKit: TruncationStrategy
using KrylovKit: GMRES
```

## 1. Construct a periodic MPS

Create a period-five MPS whose tensors each have two physical legs. Its virtual
bond dimension is four and each physical leg has dimension two:

```julia
A = rand(PeriodicMPS, ComplexF64, 5, (2, 2), 4)

physicalspaces(A)
@show length(A)
@show assertconsistency(A)
```

Expected output:

```text
length(A) = 5
assertconsistency(A) = nothing
```

A one-physical-leg periodic MPS can be generated with `rand`:

```julia
mps = rand(PeriodicMPS, ComplexF64, 10, 2, 4)
```

Pass an RNG first when the generated tensors must be reproducible:

```julia
using Random

mps = rand(Xoshiro(42), PeriodicMPS, ComplexF64, 10, 2, 4)
```

Tensor unit cells follow the same convention:

```julia
using TensorKit

gate_space = ℂ^2 ⊗ ℂ^3 ← ℂ^3 ⊗ ℂ^2
gates = rand(UniformGates, ComplexF64, gate_space)
```

## 2. Gauge-fix a periodic MPS

`gaugefix!` computes periodic bond gauges and absorbs the resulting projectors
into the finite MPS in place:

```julia
gaugefix!(mps, ProjectorCanonical())
@show assertconsistency(mps)
```

Expected output:

```text
assertconsistency(mps) = nothing
```

PMPSLab keeps the original finite `PeriodicMPS` representation; it does not
introduce a separate canonical periodic-MPS storage type.

## 3. Run a basic 2-to-1 split

`ALSSplit` performs an alternating least-squares split, approximating every
two-physical-leg tensor in `A` by two one-physical-leg tensors. The
approximation therefore contains
`2 * length(A)` sites:

```julia
algorithm = ALSSplit(;
    ansatz=SVDAnsatz(),
    sweep=CircularSweep(; N=2 * length(A), maximal_sweep=4),
    solve=CholeskySolve(),
)

B, log = split(A, algorithm)

@show length(B)
@show length(log.site_logs)
@show log.converged
cost_matches = isapprox(
    last(log.site_logs).error,
    squared_distance(A, B);
    rtol=1e-10,
    atol=1e-10,
)
@show cost_matches
```

Expected output:

```text
length(B) = 10
length(log.site_logs) = 40
log.converged = false
cost_matches = true
```

`split` is the simplest blocking API. Internally it uses the same `SplitRun`
and one-site `step!` implementation as controllable execution.

## 4. Truncate the initial approximation

Pass a `TruncationStrategy` to `SVDAnsatz` to limit the virtual spaces generated
by the initial two-to-one tensor decompositions:

```julia
algorithm = ALSSplit(;
    ansatz=SVDAnsatz(; trunc=TruncationStrategy(maxrank=3)),
    sweep=CircularSweep(; N=2 * length(A), maximal_sweep=20),
    solve=CholeskySolve(),
)

B, log = split(A, algorithm)

@show length(B)
@show length(log.site_logs)
```

Expected output:

```text
length(B) = 10
length(log.site_logs) = 200
```

This truncates the approximation tensors. It is separate from truncating the
transfer environments used during each local solve.

## 5. Stop when the cost stabilizes

`maximal_sweep` is always a hard upper bound. Add `CostConvergence` to stop at a
complete circular lap or zigzag round trip when consecutive sweep-end costs are
approximately equal:

```julia
algorithm = ALSSplit(;
    ansatz=SVDAnsatz(; trunc=TruncationStrategy(maxrank=3)),
    sweep=CircularSweep(; N=2 * length(A), maximal_sweep=500),
    solve=CholeskySolve(),
    stopping=CostConvergence(;
        rtol=1e-8,
        atol=1e-12,
        min_sweeps=5,
        consecutive_sweeps=2,
    ),
)

B, log = split(A, algorithm)

completed_sweeps = last(log.site_logs).state.n + 1
@show completed_sweeps
@show log.converged
```

Expected output shape:

```text
completed_sweeps = <an integer from 5 through 500>
log.converged = true  # if the cost stabilized before the hard cap
```

`log.converged == true` means the stopping criterion ended the run. A false
value means the sweep iterator was exhausted without satisfying that criterion,
or no criterion was configured.

## 6. Observe a controllable run

In an interactive Julia REPL, `split` and `start_run` select a default split
observer when the `observer` keyword is omitted. It emits structured progress
for every completed site and keeps one permanent summary line per completed
sweep. VS Code displays the progress events in its progress UI; other loggers
may choose their own rendering.

The simplest usage needs no logging configuration:

```julia
B, log = split(A, algorithm)
```

While the blocking split runs, the REPL reports sweep summaries such as:

```text
[ Info: ALS split started
[ Info: ALS split sweep 1 completed: cost=<Float64>, time=<Float64>s
[ Info: ALS split sweep 2 completed: cost=<Float64>, time=<Float64>s
...
[ Info: ALS split completed after <integer> site updates
```

For asynchronous execution, initialize a run and place it inside a
`LocalRunSession`:

```julia
session = start_run(initialize_run(A, algorithm))
B, log = fetch(session)
```

The asynchronous run uses the same default observer and log format.

Non-interactive scripts remain silent by default. Pass `observer=nothing` to
disable the default observer explicitly in a REPL:

```julia
session = start_run(initialize_run(A, algorithm); observer=nothing)
```

A custom observer replaces the default output completely:

```julia
session = start_run(
    initialize_run(A, algorithm);
    observer = event -> println(
        "sweep=$(event.log.state.n + 1) ",
        "site=$(event.log.state.site) ",
        "cost=$(event.log.error)",
    ),
)

B, log = fetch(session)
```

The custom observer above prints one line per completed site, for example:

```text
sweep=1 site=1 cost=<Float64>
sweep=1 site=2 cost=<Float64>
...
```

The exact costs depend on `A`; the sweep and site fields follow the configured
iterator exactly.

`fetch(session)` waits for completion, returns the algorithm result, and
rethrows a captured worker exception if the run failed. `wait(session)` only
waits for the current driver task; it can also return after a pause or
cancellation. A failed session is terminal; start a fresh run or restore a
previous checkpoint rather than retrying its in-memory numerical state.

## 7. Pause and continue in memory

`pause!` is a non-blocking request. The current site update is allowed to
finish, after which `wait` returns with the run at a consistent site boundary.
This example requests the pause from the observer after the first completed
site, avoiding a race with a very short run:

```julia
session_ref = Ref{LocalRunSession}()
observer = event -> begin
    event.log.state == SweepState{Forward}(1, 0) && pause!(session_ref[])
    nothing
end
session = LocalRunSession(initialize_run(A, algorithm); observer)
session_ref[] = session

start_run!(session)
wait(session)
run_status(session) == PMPSLab.LOCAL_PAUSED || error("run did not pause")
@show run_status(session)

resume!(session)  # continues the same SplitRun and keeps its caches
B, log = fetch(session)
@show run_status(session)
```

Expected output:

```text
run_status(session) = PMPSLab.LOCAL_PAUSED
run_status(session) = PMPSLab.LOCAL_COMPLETED
```

When `pause!` is issued externally instead, a very short computation may finish
before the request arrives. Completion then wins and `fetch` returns the result.

## 8. Save and restore across processes

For a durable stop, `cancel_and_checkpoint!` waits for the next safe boundary
and then atomically publishes a checkpoint. If the final step completes first,
it reports that no checkpoint was needed:

```julia
session = start_run(initialize_run(A, algorithm))
outcome = cancel_and_checkpoint!(session, "split.jld2")
@show outcome

if outcome == PMPSLab.LOCAL_CHECKPOINT_WRITTEN
    saved = read_checkpoint("split.jld2")
    restored = restore_run(saved, A, algorithm)
    B, log = fetch(start_run(restored))
else
    B, log = fetch(session)
end
```

Expected output for a cancellation that reaches a safe boundary:

```text
outcome = PMPSLab.LOCAL_CHECKPOINT_WRITTEN
```

If the run completes first, the output is
`PMPSLab.LOCAL_CHECKPOINT_SKIPPED_COMPLETED` and no file is created.

After an actual process restart, reconstruct the original target values `A` and
the complete `algorithm` configuration before calling `restore_run`. Restore
checks target tensor spaces, ansatz, sweep, solver, transfer policies, stopping
criterion, cursor history, and convergence history. It cannot detect different
target values with identical spaces. The checkpoint does not store the target
tensors, Julia tasks, locks, observers, or derived transfer caches, and is
accepted only by the same PMPSLab package version.

For a still-running long task that should remain resumable in the current
process, the blocking durable-pause helper combines the request, wait, snapshot,
and file publication:

```julia
pause_session = start_run(initialize_run(A, algorithm))
pause_and_checkpoint!(pause_session, "paused-split.jld2")
resume!(pause_session)
```

The resumed in-memory session reuses its existing caches; restoring from the
file constructs a new run and rebuilds them.

## 9. Configure truncated transfer environments

Dense transfer environments are the default. For larger systems, normal and
mixed environments can independently use truncated factored products:

```julia
truncated = TruncatedTransferPolicy(;
    trunc=TruncationStrategy(maxrank=8),
)
environments = SplitEnvironmentPolicy(;
    norm=truncated,
    mixed=truncated,
)

algorithm = ALSSplit(;
    ansatz=SVDAnsatz(; trunc=TruncationStrategy(maxrank=3)),
    sweep=CircularSweep(; N=2 * length(A), maximal_sweep=500),
    solve=KrylovSolve(; alg=GMRES(; maxiter=100, tol=1e-10)),
    environments,
    stopping=CostConvergence(;
        rtol=1e-8,
        atol=1e-12,
        min_sweeps=5,
        consecutive_sweeps=2,
    ),
)

@show algorithm.environments.norm isa TruncatedTransferPolicy
@show algorithm.environments.mixed isa TruncatedTransferPolicy
```

Expected output:

```text
algorithm.environments.norm isa TruncatedTransferPolicy = true
algorithm.environments.mixed isa TruncatedTransferPolicy = true
```

With a truncated normal environment, `KrylovSolve` applies the factored normal
operator without densifying it. Direct solvers remain supported, but they
materialize that local operator before solving. Krylov non-convergence emits a
warning; the current split implementation continues with the latest iterate.

## 10. Run LoopTNR

`LoopTNR` repeatedly turns a tensor cell into its designed eight-site split
problem, optimizes that split, coarse-grains it back to a tensor cell, and then
normalizes and measures the new RG boundary. The blocking API returns a named
result rather than a tuple. All accepted input cell layouts are copied into a
canonical `PlaquetteGates`; run, snapshot, and result cells therefore always
use that representation:

```julia
tnr = default_looptnr(; maximal_rg=2, dcut=8)
result = renormalize(gates, tnr)

@show result.cell
@show length(result.measurement_results)  # RG 0, 1, and 2
@show length(result.log.rg_logs)
```

The default uses an eight-site circular ALS split with at most 40 sweeps,
dense normal and mixed environments, an LDLT local solve, and a periodic-gauge
ansatz. Its preliminary SVD retains at most `2dcut` states before projector
canonicalization reduces the universal bond dimension to `dcut`. After five
complete sweeps it stops when both `abs(cost) < 1e-9` and the relative
sweep-to-sweep cost change is below `1e-2`. Every RG boundary is normalized by
the coefficient infinity norm; fixed-point transfer measurements remain
disabled until their eigensolver is available.

Algorithm configurations are immutable values. Use `@setconfig` to derive a
modified configuration. Its `/`-separated paths follow the displayed
configuration tree from root to leaf:

```julia
tnr = @setconfig tnr begin
    maximal_rg = 4
    split/sweep/maximal_sweep = 60
    split/stopping/atol = 1e-10
end
```

The original `tnr` is unchanged unless the returned value is assigned back to
it. A whole strategy can also be replaced, even when that changes the concrete
configuration type:

```julia
tnr = @setconfig tnr split/stopping = NoStopping()
```

In an interactive session the default observer uses one replaceable progress
entry for the current RG split, spanning all of its split sweeps, and leaves
concise sweep and RG summaries behind:

```text
[ Info: LoopTNR started · RG 0/2 normalization pending
[ Info: LoopTNR RG 0/2 initialized · scale <Real>
[ Info: LoopTNR RG 1/2 · split sweep 1 completed · cost <Float64> · time <Float64> s
[ Info: LoopTNR RG 1/2 completed · cost <Float64> · split <Float64> s · coarse grain <Float64> s · scale <Real>
...
[ Info: LoopTNR completed · RG 2/2 · 3 measurements
```

Typing a run at the REPL shows its next safe action without expanding tensor
storage. During an active split it has this shape:

```text
LoopTNRRun(RG 1/2 · split sweep 3 · next site 5/8)
  committed cell: 2×2 PlaquetteGates(ComplexF64, ComplexSpace)
  RG logs: 1
  measurements: 1
  active split: 20/320 site updates
```

LoopTNR uses the same controllable execution interface as a standalone split:

```julia
run = initialize_run(gates, tnr)
session_ref = Ref{LocalRunSession}()
pause_requested = Ref(false)
observer = event -> begin
    if !pause_requested[] &&
            event isa LoopTNRSplitStepCompleted &&
            event.event.log.state.site == 4
        pause_requested[] = true
        pause!(session_ref[])
    end
    nothing
end

session = LocalRunSession(run; observer)
session_ref[] = session
start_run!(session)

wait(session)  # stops after the current split site or complete RG action
run_status(session) == PMPSLab.LOCAL_PAUSED || error("run did not pause")
display(run)   # reports the exact continuation point
resume!(session)
result = fetch(session)
```

Here the observer requests the pause once, after split site 4 completes. This
avoids racing an external `pause!` call against a short run. An external pause
request is still valid, but completion wins if the run finishes first.

A custom observer replaces the display and receives both nested split-site
events and committed RG events. Physical measurement results are carried by
the RG event, separately from the control-only logs:

```julia
observer = event -> begin
    event isa LoopTNRCoarseGrained || return
    measurement = event.result
    println("RG $(event.rg): normalization scale = ",
            measurement.normalization.scale)
end

result = renormalize(gates, tnr; observer)
```

Durable pause stores the committed outer cell, its measurement history, and an
active split snapshot when the pause lands inside a split. Restoring a LoopTNR
checkpoint therefore needs the algorithm configuration, but not the original
input cell:

```julia
session = start_run(initialize_run(gates, tnr))
outcome = pause_and_checkpoint!(session, "paused-looptnr.jld2")

if outcome == PMPSLab.LOCAL_CHECKPOINT_WRITTEN
    saved = read_checkpoint("paused-looptnr.jld2")
    restored = restore_run(saved, tnr)
    result = fetch(start_run(restored))
else
    result = fetch(session)
end
```

## Cost and progress semantics

With dense environments and an exact local solver, each `SiteLog.error` agrees
with the complete finite-period cost up to numerical roundoff:

```math
\lVert A-B\rVert^2 = \langle A|A\rangle + \langle B|B\rangle
                     - 2\operatorname{Re}\langle B|A\rangle.
```

With truncated working environments, `SiteLog.error` is the corresponding
approximate local cost. `squared_distance(A, B)` always performs the exact dense
finite-period check.

`SiteLog.state.site` is the updated site, `SiteLog.state.n` is the zero-based
sweep number, and `log.sweep_time` accumulates time spent inside completed site
updates. Time spent paused is excluded.

## Design documents

- [`design/resumable-execution.md`](design/resumable-execution.md) describes the
  run, session, snapshot, and checkpoint boundaries.
- [`design/durable-execution-plan.md`](design/durable-execution-plan.md) records
  the completed durable local-execution design.
- The [LoopTNR acceleration research plan](design/looptnr-als-acceleration-research-plan.md)
  records deferred convergence-acceleration research.
- [`SPEC.md`](SPEC.md) documents the numerical periodic-MPS background; some
  older execution-interface discussion there is historical.
