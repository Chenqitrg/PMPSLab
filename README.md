# PMPSLab

Experimental tools for finite periodic matrix-product states, projector-based
periodic gauge fixing, 2-to-1 splitting, and resumable local execution.

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
A = random_mps2(ComplexF64, 5, 2, 4)

physicalspaces(A)
@show length(A)
@show assertconsistency(A)
```

Expected output:

```text
length(A) = 5
assertconsistency(A) = nothing
```

A one-physical-leg periodic MPS can be created with `random_mps`:

```julia
mps = random_mps(ComplexF64, 10, 2, 4)
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

`PeriodicSplit` approximates every two-physical-leg tensor in `A` by two
one-physical-leg tensors. The approximation therefore contains
`2 * length(A)` sites:

```julia
algorithm = PeriodicSplit(
    SVDAnsatz(),
    CircularSweep(2 * length(A); maximal_sweep=4),
    CholeskySolve(),
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
algorithm = PeriodicSplit(
    SVDAnsatz(TruncationStrategy(maxrank=3)),
    CircularSweep(2 * length(A); maximal_sweep=20),
    CholeskySolve(),
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
algorithm = PeriodicSplit(
    SVDAnsatz(TruncationStrategy(maxrank=3)),
    CircularSweep(2 * length(A); maximal_sweep=500),
    CholeskySolve();
    stopping=CostConvergence(
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

For per-site progress and asynchronous execution, initialize a run and place it
inside a `LocalRunSession`:

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

The observer prints one line per completed site, for example:

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
truncated = TruncatedTransferPolicy(
    TruncationStrategy(maxrank=8),
)
environments = SplitEnvironmentPolicy(
    norm=truncated,
    mixed=truncated,
)

algorithm = PeriodicSplit(
    SVDAnsatz(TruncationStrategy(maxrank=3)),
    CircularSweep(2 * length(A); maximal_sweep=500),
    KrylovSolve(GMRES(; maxiter=100, tol=1e-10));
    environments,
    stopping=CostConvergence(
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
