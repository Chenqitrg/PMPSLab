# Resumable Algorithm Execution

## Status

This document proposes a platform-independent execution model for pausing,
checkpointing, restoring, and cancelling long-running PMPSLab algorithms. It
does not define a job service, frontend protocol, or storage backend.

The first implementation target is the existing 2-to-1 periodic MPS `split`
algorithm. The design must also compose with future algorithms, especially a
LoopTNR run whose renormalization-group steps may contain nested `split` runs.

## Motivation

A single local update may take up to approximately one minute. A user must be
able to request a pause while that update is running. The update is allowed to
finish, after which execution stops at the next consistent site boundary. The
frontend reports progress at site granularity, so no finer-grained interruption
is required.

Paused jobs may retain large tensor networks and contraction caches. A pause
must therefore be durable: the current mathematical state can be written to a
checkpoint and later restored without keeping a worker task or its tensors in
memory. Resumed execution should agree with uninterrupted execution within
machine precision.

The eventual deployment platform already provides general job dispatch and
backend infrastructure. PMPSLab should expose the numerical state and stepping
protocol needed by that platform rather than implementing its own scheduler,
HTTP API, job registry, authentication, or storage service.

## Requirements

### Functional requirements

1. A pause request takes effect after the currently executing safe step.
2. For `split`, one safe step is one complete site update.
3. A paused run can be checkpointed, unloaded from memory, restored, and
   continued from the next site.
4. Restore must not repeat or skip a site, including at circular wrap-around,
   zigzag turns, and sweep boundaries.
5. Restored results must agree with uninterrupted results within machine
   precision.
6. Cancellation is cooperative and takes effect at the same safe boundaries.
7. Cancellation supports both saving and discarding the current progress.
8. Algorithm parameters are immutable after a run starts.
9. Checkpoints remain valid until an explicit external deletion policy removes
   them.
10. The same execution protocol must support future algorithms and nested
    algorithm runs.
11. A Julia REPL user can start a run in the background and request pause,
    resume, or cancellation with ordinary Julia commands.

### Non-functional requirements

1. The numerical core must not depend on a frontend or deployment platform.
2. Checkpoint encoding must operate on `IO`, not filesystem paths.
3. Runtime-only objects such as tasks, locks, channels, callbacks, and platform
   handles must never be part of a checkpoint.
4. Derived contraction caches may be rebuilt on restore when this only changes
   results within machine precision.
5. Existing synchronous APIs, especially `split(mps, alg; callback=...)`, must
   remain available as compatibility wrappers.

## Non-goals

This project will not initially implement:

- job identifiers or a persistent job registry;
- HTTP, WebSocket, or frontend state synchronization;
- user authentication or multi-user resource isolation;
- worker allocation, CPU affinity, or memory quotas;
- frontend disconnect detection;
- checkpoint placement, retention, or object-store integration;
- hard interruption inside TensorKit contractions, BLAS calls, or local
  solvers;
- checkpoint migration across arbitrary Julia, PMPSLab, or dependency versions;
- bitwise reproducibility across different hardware and threading setups.

These concerns belong to the hosting platform or to a later platform adapter.

## Design principles

### Explicit state instead of a suspended call stack

An active algorithm must be represented by an ordinary Julia object containing
all state required to execute its next safe step. A Julia `Task` is an execution
mechanism, not algorithm state, and cannot be the unit of persistence.

Pausing therefore means:

1. finish the current safe step;
2. create a snapshot;
3. allow the worker invocation to return;
4. optionally discard the in-memory run;
5. create a new run from the snapshot when execution resumes.

No thread or task needs to remain blocked while a job is paused.

### Inputs, contexts, runs, and snapshots are distinct

The execution model separates four kinds of data:

- **Inputs and algorithm configuration** define a particular computation. Their
  types and arity are algorithm-specific and are handled by multiple dispatch.
- An **algorithm context** contains mutable numerical working state, including
  tensors and derived contraction caches.
- A **run** coordinates a context with its algorithm configuration and execution
  state such as the next safe step, convergence log, and elapsed time.
- A **snapshot** contains the minimum durable state needed to reconstruct a
  run from the same immutable inputs and algorithm configuration.

This separation avoids repeatedly storing immutable inputs and avoids treating
large, derived caches as permanent data.

### Safe steps are transactional boundaries

`step!` may mutate many related objects internally. It must update the durable
cursor only after the whole step succeeds. If a process fails during a step,
the most recent earlier checkpoint remains the recovery point.

The initial implementation does not attempt copy-on-write rollback of a failed
site update because copying tensor-network state for every site would be too
expensive.

### Nested algorithms use explicit child runs

A parent algorithm must not hide a complete child algorithm behind a blocking
call if the child needs its own pause points. Instead, the parent stores the
child run explicitly and delegates one safe step at a time. A checkpoint of the
root run then captures the full nested computation.

## Core interface

The platform-independent interface is:

```julia
abstract type AbstractAlgorithmRun end
abstract type AbstractAlgorithmSnapshot end
abstract type AlgorithmEvent end

initialize_run(inputs..., algorithm)::AbstractAlgorithmRun
step!(run::AbstractAlgorithmRun)::AlgorithmEvent
isfinished(run::AbstractAlgorithmRun)::Bool
run_result(run::AbstractAlgorithmRun)

snapshot(run::AbstractAlgorithmRun)::AbstractAlgorithmSnapshot
restore_run(
    snapshot::AbstractAlgorithmSnapshot,
    inputs...,
    algorithm,
)::AbstractAlgorithmRun
```

The methods have the following contracts.

### `initialize_run`

Constructs a fresh run from algorithm-specific immutable inputs and an
algorithm configuration, including all derived transfer matrices and caches.
Different algorithms may define different input types and argument counts using
multiple dispatch. Random initialization must happen here exactly once. Its
result becomes part of the run state and eventually the snapshot; restore must
not repeat random initialization.

### `step!`

Advances the run from one consistent boundary to the next and returns an
immutable progress event. If the method returns successfully, the run is safe
to inspect and snapshot.

### `isfinished` and `run_result`

`isfinished` is a read-only completion check. `run_result` is valid only after
completion and returns the algorithm's existing public result shape.

### `snapshot`

Creates a durable-state representation. A snapshot must not contain callbacks,
observers, tasks, locks, platform commands, or open storage handles.

### `restore_run`

Validates the snapshot against the immutable inputs and algorithm configuration,
reconstructs derived state, and returns a run positioned at exactly the next
safe step.

## Split execution model

### Initialization inputs

The split run is initialized directly through multiple dispatch:

```julia
initialize_run(mps::PeriodicMPS, algorithm::PeriodicSplit)
```

The target MPS and complete `PeriodicSplit` configuration are immutable for the
lifetime of the run. They do not require a common problem wrapper.

### Run

```julia
mutable struct SplitRun{M,A,C,P} <: AbstractAlgorithmRun
    target::M
    algorithm::A
    context::C
    pending_iteration::Union{Nothing,P}
    log::SplitLog
end
```

`SplitContext` continues to own the numerical working state: the current
approximation, transfer matrices, and partial-product caches. `SplitRun` owns
the resumable execution cursor and reporting state.

`log.sweep_time` accumulates the completed site timings and therefore excludes
time spent paused between safe steps. Pause timestamps and total paused time
belong to the execution adapter rather than the numerical run.

### Step semantics

Conceptually, one split step is:

```julia
function step!(run::SplitRun)
    state, next_iterator_state = run.pending_iteration
    following_iteration = iterate(run.algorithm.sweep, next_iterator_state)
    started_at = time()
    _, cost = update!(run.context, run.algorithm.solve, state)

    site_log = SiteLog(state, cost, time() - started_at)
    push!(run.log.site_logs, site_log)
    run.log.sweep_time += site_log.time

    # Commit the durable cursor only after the site update and log append
    # have both succeeded.
    run.pending_iteration = following_iteration

    return SplitSiteCompleted(site_log)
end
```

Initialization prefetches the first iterator item. Consequently,
`isfinished(run)` is equivalent to `isnothing(run.pending_iteration)`, and the
run becomes complete as part of its final real site step rather than requiring
an extra completion-only call to `step!`.

`pending_iteration` stores the unmodified result of Julia's `iterate`: the next
`SweepState` to execute and the opaque iterator state for the call after it.
Persisting only the just-completed `SweepState` would cause duplicate work
after restore, especially at a zigzag turn.

The final implementation must preserve the existing sweep callback semantics:
one callback after each complete sweep and one after the final sweep, without
duplicate calls across a restore boundary.

### Snapshot

```julia
struct SplitSnapshot{M,L} <: AbstractAlgorithmSnapshot
    schema_version::Int
    mpsB::M
    iterator_state::SweepState
    log::L
    active_sweep::Union{Nothing,Int}
    completed::Bool
end
```

The snapshot stores the current approximation but not the derived
`PeriodicTransferMatrix` and `PeriodicCache` values. Restore reconstructs a
`SplitContext` from the immutable target MPS and the stored approximation.

This may change the order in which some floating-point contractions are
recomputed, so equality is specified within machine precision rather than
bitwise equality.

## Progress events

Events are immutable numerical-core values. A platform adapter may translate
them into JSON, database records, or WebSocket messages.

The initial event set is:

```julia
struct SplitSiteCompleted{L} <: AlgorithmEvent
    log::L
end

struct SweepCompleted <: AlgorithmEvent
    sweep::Int
    error::Float64
    compute_time::Float64
end

struct AlgorithmCompleted <: AlgorithmEvent
    compute_time::Float64
end
```

Pause lifecycle events such as `PauseRequested`, `Checkpointing`, and `Paused`
belong to the platform execution adapter because they describe job state, not a
numerical transition. PMPSLab supplies the site, sweep, and RG events that make
safe boundaries observable.

For nested algorithms, the event path must preserve parent context. One
possible representation is:

```julia
struct ProgressFrame
    algorithm::Symbol
    phase::Symbol
    current::Int
    total::Union{Nothing,Int}
end

struct ProgressEvent{E<:AlgorithmEvent}
    path::Vector{ProgressFrame}
    event::E
end
```

For example, the frontend may receive a path equivalent to:

```text
LoopTNR RG 4/20
  Split sweep 2/5
    Site 7/16 completed
```

The exact JSON representation remains a platform-adapter decision.

## Nested LoopTNR execution

A future LoopTNR run should expose its phase and current child run explicitly:

```julia
@enum LoopTNRPhase begin
    PREPARE_RG
    OPTIMIZE
    COARSE_GRAIN
    FINISH_RG
end

mutable struct LoopTNRRun{I,A,S,C} <: AbstractAlgorithmRun
    input::I
    algorithm::A
    state::S
    rg_index::Int
    phase::LoopTNRPhase
    child::C
    completed::Bool
end
```

When `phase == OPTIMIZE`, `step!(run::LoopTNRRun)` delegates one step to the
child. When the child completes, the parent commits its result and advances to
the next phase. The outer cursor must advance only after the child result has
been committed successfully.

A root LoopTNR snapshot contains a snapshot of its active child, so a pause may
occur inside a split site sequence or between RG steps without special handling
by the platform.

The first implementation should include a small synthetic composite run test
before the real LoopTNR implementation exists. This will verify that events,
snapshots, and restore compose through a parent-child boundary.

## Checkpoint encoding

### IO-based API

Checkpoint encoding must not assume a local filesystem:

```julia
write_checkpoint(io::IO, snapshot::AbstractAlgorithmSnapshot)
read_checkpoint(io::IO)::AbstractAlgorithmSnapshot
```

Tests use `IOBuffer`. A future platform adapter may supply a file, object-store
stream, artifact handle, or another `IO` implementation.

### Header

Each encoded checkpoint begins with a PMPSLab-owned header:

```julia
struct CheckpointHeader
    magic::NTuple{8,UInt8}
    schema_version::Int
    algorithm::Symbol
    package_version::VersionNumber
end
```

The decoder must reject an invalid magic value, an unsupported schema version,
or an algorithm/snapshot mismatch with a descriptive error.

The deployment platform may additionally record the Git commit, Manifest hash,
and container image digest. PMPSLab does not need to duplicate those fields if
the platform already guarantees the execution environment, but schema version
ownership must remain in PMPSLab.

### Initial codec

The first implementation may use Julia's `Serialization` standard library
behind the PMPSLab header. This is suitable for same-environment recovery and
preserves tensor data without designing a complete portable TensorKit schema.

The codec is not promised to support arbitrary dependency upgrades. If
long-term cross-version migration becomes necessary, it can be replaced or
augmented without changing the run/snapshot interfaces.

## Pause and cancellation integration boundary

PMPSLab does not need a production job manager. A platform adapter eventually
drives the core protocol and checks its own command token after each event:

```julia
while !isfinished(run)
    event = step!(run)
    emit!(platform_runtime, event)

    command = command(platform_runtime)
    if command == PAUSE
        save_checkpoint!(platform_runtime, snapshot(run))
        return Paused()
    elseif command == CANCEL_SAVE
        save_checkpoint!(platform_runtime, snapshot(run))
        return CancelledSaved()
    elseif command == CANCEL_DISCARD
        return CancelledDiscarded()
    end
end
```

The worker invocation returns after a durable pause, allowing the platform to
release CPU and memory. Resume creates a new run with `restore_run` and launches a
new worker invocation. PMPSLab does not prescribe how the platform represents
these outcomes.

A lightweight local adapter provides `LocalRunSession`, `start_run`, `pause!`,
`resume!`, and `cancel!` for development and Julia REPL use. The session keeps
only an in-memory run until checkpoint support is implemented and must not grow
into a second production scheduler.

```julia
run = initialize_run(inputs..., algorithm)
session = start_run(run)

pause!(session)   # becomes PAUSED after the current safe step
resume!(session)
cancel!(session)  # discards progress until checkpoint support is added
```

With multiple Julia threads, a local session may execute on a default worker
thread so the REPL remains responsive during a step. With one thread, the local
driver explicitly yields after every safe step; commands typed while a step is
running are therefore handled before a later step begins, but the REPL may not
visibly respond until the current step completes. This matches the site-boundary
pause contract.

## Failure model

### Requested pause or cancellation

The current safe step finishes, its cursor is committed, and a snapshot can be
created. A `CANCEL_DISCARD` request needs no snapshot.

### Exception during `step!`

The current in-memory run may be partially mutated. It must not be snapshotted
as a successful boundary. Recovery uses the last previously committed durable
checkpoint.

### Frontend disconnect

The hosting platform may translate a disconnect into a pause request. As long
as the backend worker remains alive, the current step can finish and a new
checkpoint can be written.

### Worker or machine failure

A crashed process cannot create a new checkpoint after failure. Recovery is
limited to checkpoints written before the failure. If crash recovery beyond
explicit pauses is later required, the platform can request checkpointing on
events such as `SweepCompleted` or `RGCompleted` without changing the algorithm
protocol.

## Numerical consistency

The acceptance criterion is numerical agreement within machine precision, not
bitwise identity. Tests compare the final MPS, costs, and relevant observables
with suitable `isapprox` tolerances.

The algorithm-specific inputs and algorithm configuration are immutable. Restore must not
allow a caller to change the ansatz, solver, sweep order, truncation policy, or
maximum sweep count. A snapshot must be rejected when its dimensions, spaces,
algorithm kind, or schema are incompatible with the supplied inputs and
configuration.

Random initial approximations are stored through `mpsB`; they are never
regenerated during restore. Any future algorithm that uses randomness after a
safe boundary must also snapshot its RNG state.

## Testing strategy

### Compatibility tests

- The public `split(mps, alg; callback=...)` result remains numerically
  unchanged after it becomes a wrapper around `SplitRun`.
- Callback timing and count remain unchanged.
- Existing circular and zigzag update-cost tests continue to pass.

### Stepper tests

- `step!` visits exactly the same `SweepState` sequence as the configured
  iterator.
- A cursor advances only after a successful update.
- Completion is idempotent and `run_result` is rejected before completion.
- Site, sweep, and completion events contain the expected state and cost.

### Restore tests

For every possible safe boundary in a small run:

1. run an uninterrupted reference calculation;
2. execute to the selected boundary;
3. create a snapshot;
4. encode it to an `IOBuffer`;
5. decode and restore a new run;
6. finish the restored run;
7. compare it with the uninterrupted result within machine precision.

The matrix must include:

- `CircularSweep` and `ZigzagSweep`;
- a zigzag turning point;
- a circular wrap-around;
- a sweep boundary;
- the final site;
- `LUSolve`, `CholeskySolve`, and `KrylovSolve`;
- real, complex, and graded tensor spaces where practical;
- malformed and incompatible checkpoint headers.

### Composition tests

A synthetic parent run with a resumable child verifies that:

- parent and child cursors are both restored;
- nested progress events retain their path;
- a child result is committed exactly once;
- pausing before and after child completion produces the same final result as
  uninterrupted execution.

## Implementation plan

### Phase 1: Explicit split execution

1. Add the abstract run/snapshot/event interfaces.
2. Introduce `SplitRun` with a `SplitContext` numerical workspace.
3. Move one-site loop behavior into `step!`.
4. Reimplement `split` as a synchronous compatibility wrapper.
5. Verify that all existing tests remain green.

### Phase 2: Split snapshots

1. Add `SplitSnapshot`.
2. Rebuild `SplitContext` from immutable input and snapshot state.
3. Add per-boundary restore tests.
4. Verify numerical agreement within machine precision.

### Phase 3: Checkpoint codec

1. Add the versioned checkpoint header.
2. Implement `IO`-based encoding and decoding.
3. Add corruption and version-mismatch tests.
4. Document the initial same-environment compatibility guarantee.

### Phase 4: Composition proof

1. Add a synthetic parent-child run in tests.
2. Verify nested stepping, progress, snapshot, and restore.
3. Refine the protocol only where composition exposes a concrete need.

### Phase 5: Platform adapter

This phase waits for the hosting platform API. It translates platform commands,
progress reporting, and checkpoint storage into the interfaces defined above.
No numerical algorithm should require redesign at this stage.

## Deferred decisions

The following decisions should be made only after the platform backend is
available:

- the concrete pause and cancellation token API;
- how progress events are serialized for the frontend;
- whether the platform streams checkpoint data or provides a path;
- maximum checkpoint size and write timeout;
- when an in-memory paused run is evicted;
- how frontend disconnects map to commands;
- whether sweep or RG events trigger automatic crash-recovery checkpoints;
- how platform job metadata pins the Julia environment and PMPSLab version.

None of these decisions blocks Phases 1 through 4.
