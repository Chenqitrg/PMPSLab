# Durable Local Execution Plan

## Status and Scope

This is an implementation plan for the remaining local durable-execution work
around `PeriodicSplit`. It is deliberately narrower than a job platform.

The completed baseline is:

- `SplitRun` advances one complete site through `step!`;
- `LocalRunSession` supports in-memory `pause!`, `resume!`, and `cancel!`;
- `snapshot`, `write_checkpoint`, `read_checkpoint`, and `restore_run` exist;
- checkpoints are atomically published by rename;
- a restored split run rebuilds derived transfer matrices and caches.

This plan adds:

1. a snapshot identity that records target spaces and the complete
   `PeriodicSplit` configuration;
2. explicit local helpers for durable pause and durable cancellation;
3. focused tests and user documentation for the resulting lifecycle.

This plan does not add job IDs, a persistent registry, automatic checkpoint
discovery, HTTP/WebSocket transport, object storage, worker assignment, or
cross-version checkpoint migration. Those remain hosting-platform concerns.

## Existing Ownership Boundaries

The implementation must preserve the established separation:

| Layer | Owns | Must not own |
| --- | --- | --- |
| `src/split/run.jl` | numerical cursor, approximation, numerical log, split-specific snapshot validation | `Task`, locks, callback, checkpoint path |
| `src/execution/local_session.jl` | local task, session state, observer, run driver | control policy, checkpoint orchestration |
| `src/execution/local_control.jl` | cooperative pause, resume, and cancellation transitions | file I/O, numerical state |
| `src/execution/local_checkpoint.jl` | quiescence checks and durable session operations | checkpoint encoding, numerical cache format |
| `src/execution/checkpoint.jl` | versioned JLD2 envelope and atomic file publication | algorithm-specific numerical validation |
| future platform adapter | job lifecycle, durable storage policy, eviction, retry | tensor-update implementation |

`SplitSnapshot` continues to exclude caches, tasks, locks, callbacks, and
checkpoint paths. A durable checkpoint is a snapshot plus the JLD2 envelope;
it is not a serialized `LocalRunSession`.

## Invariants

1. A successful `step!` is the only operation that advances the numerical
   cursor. Its returned event marks a snapshot-safe boundary.
2. `pause!` and `cancel!` never interrupt a site kernel. A request made during
   a step takes effect before any later step starts.
3. `pause_and_checkpoint!` and `cancel_and_checkpoint!` may write only after
   the session has reached a quiescent safe boundary.
4. A short pause retains the existing `SplitRun`, including its caches. A
   subsequent `resume!` must not call `restore_run`.
5. A write failure never discards the in-memory run or changes a paused/
   cancelled session into a completed one.
6. A checkpoint may be restored only with compatible target tensor spaces and
   the exact immutable split configuration from which it was captured. The
   caller remains responsible for supplying the original target values.
7. The current local adapter has one controlling caller. Its observer may call
   `pause!` or `cancel!`; durable helper methods must therefore not hold a lock
   across `wait(session)` or file I/O.

The last invariant is important. Holding a new session-wide lock while waiting
would deadlock if an observer tries to request pause or cancellation from the
driving task. A platform adapter needing concurrent command submission should
provide a request mailbox above this local API rather than expanding
`LocalRunSession` into a scheduler.

## Public API

Keep the existing non-blocking control commands unchanged:

```julia
pause!(session::LocalRunSession) -> LocalRunSession
resume!(session::LocalRunSession) -> LocalRunSession
cancel!(session::LocalRunSession) -> LocalRunSession
```

Add three explicit, blocking durable helpers:

```julia
checkpoint!(session::LocalRunSession, path::AbstractString) -> LocalRunSession

pause_and_checkpoint!(
    session::LocalRunSession,
    path::AbstractString,
) -> LocalRunSession

cancel_and_checkpoint!(
    session::LocalRunSession,
    path::AbstractString,
) -> LocalRunSession
```

Do not add `checkpoint_path` as a keyword to `cancel!`. The current `cancel!`
returns immediately; making only its keyword form wait for a task and perform
disk I/O would make one API have two surprising timing models. A separate
verb makes the durable, blocking behavior obvious and matches the existing
explicit `write_checkpoint` API.

`checkpoint!` is the primitive. It accepts only a quiescent
`LOCAL_PAUSED` or `LOCAL_CANCELLED` session. It captures `snapshot(session.run)`
and calls `write_checkpoint(path, saved)`. It does not release `session.run`.

The low-level `snapshot(run)` method remains public for algorithm composition.
It retains its contract: callers must invoke it only at an algorithm-defined
safe boundary. The new session helpers guarantee that boundary for normal local
use; they do not attempt to prohibit expert callers from using the lower-level
API.

## State Transitions

### `checkpoint!`

| Initial status | Result | File |
| --- | --- | --- |
| `LOCAL_PAUSED` | remain paused | write checkpoint |
| `LOCAL_CANCELLED` | remain cancelled | write checkpoint |
| any other status | throw `ArgumentError` | no write |

Check the status and create the snapshot while holding `session.lock`. The
driving task transitions to `LOCAL_PAUSED` or `LOCAL_CANCELLED` only after it
has stopped scheduling numerical work, and retaining the lock through
`snapshot` prevents a concurrent `resume!` from starting another task. Release
the lock before `write_checkpoint`, since file I/O does not need to block
session control.

If `write_checkpoint` throws, leave both the status and `session.run` intact.
The caller can retry `checkpoint!` with another path, resume a paused session,
or drop the session after inspecting the failure.

### `pause_and_checkpoint!`

| Initial status | Required behavior |
| --- | --- |
| `LOCAL_RUNNING` | request pause, wait for the driving task, then checkpoint |
| `LOCAL_PAUSE_REQUESTED` | wait for the existing request, then checkpoint |
| `LOCAL_PAUSED` | checkpoint immediately without rebuilding caches |
| `LOCAL_READY`, `LOCAL_CANCEL_REQUESTED`, `LOCAL_CANCELLED`, `LOCAL_COMPLETED`, `LOCAL_FAILED` | throw `ArgumentError`; do not write |

Implementation sequence:

1. validate that the initial status is one of the three accepted states;
2. call `pause!(session)` when it is running or pause-requested;
3. call `wait(session)` without holding `session.lock`;
4. require `run_status(session) == LOCAL_PAUSED`;
5. delegate to `checkpoint!(session, path)`.

If a concurrent controller cancels or resumes the run while this helper waits,
the final-status check fails rather than writing a racing snapshot. Document
that normal local use has one controller; callers that deliberately coordinate
multiple controllers must retry after resolving the final status.

### `cancel_and_checkpoint!`

| Initial status | Required behavior |
| --- | --- |
| `LOCAL_RUNNING`, `LOCAL_PAUSE_REQUESTED` | request cancellation, wait at the next safe boundary, then checkpoint |
| `LOCAL_READY`, `LOCAL_PAUSED` | cancel immediately, then checkpoint the current safe boundary |
| `LOCAL_CANCEL_REQUESTED` | wait for the existing request, then checkpoint if cancelled |
| `LOCAL_CANCELLED` | checkpoint immediately |
| `LOCAL_COMPLETED` | return completed session; do not create a cancellation checkpoint |
| `LOCAL_FAILED` | throw `ArgumentError`; do not write |

Implementation sequence:

1. under `session.lock`, reject `LOCAL_FAILED`, return immediately for
   `LOCAL_COMPLETED`, or atomically move a running/paused/ready state to the
   corresponding cancellation state;
2. retain an existing cancellation request without changing it;
3. call `wait(session)` without holding `session.lock`;
4. if completion won during the final site update, return the completed session
   without creating a file;
5. otherwise require `LOCAL_CANCELLED` and delegate to `checkpoint!`.

If checkpoint publication fails after cancellation, the session remains
cancelled but still retains its `run`. Retrying `checkpoint!` is valid. It is
not resumable through `resume!`; recovery proceeds by restoring the successful
checkpoint into a new run.

## Phase 1: Checkpoint Identity

### Goal

Reject restores with incompatible target spaces or a different algorithm
configuration without scanning the target tensor values.

### Data Model

Add a split-specific, serializable value type in `src/split/identity.jl`:

```julia
struct SplitProblemIdentity{S,A}
    target_spaces::S
    algorithm::A
end
```

Add `identity::SplitProblemIdentity` to `SplitSnapshot`. Capture it from
`run.context.mpsA` and `run.algorithm` in `snapshot(run)`; validate it before
copying `saved.mpsB` or rebuilding a `SplitContext` in `restore_run`.

### Identity Contract

Introduce a private `_split_problem_identity(mpsA, algorithm)` helper with
these requirements:

1. It records the target tensor codomains and domains explicitly.
2. It does not inspect or hash target tensor values; supplying the original
   immutable target remains the caller's responsibility.
3. It stores a detached copy of the complete `PeriodicSplit` configuration and
   compares it with the standard `==` value semantics supplied by
   `AbstractAlgorithmConfig`.
4. It does not depend on Julia `Serialization`, hashing, `objectid`, or pointer
   addresses.

The supported configuration graph is the existing split ansatz, sweep, solver,
canonicalization, and dense/truncated transfer-policy values. These immutable
configuration types inherit `AbstractAlgorithmConfig`, which defines `==`,
`isequal`, and `hash` from their concrete type and fields. Checkpoint code must
not implement a second reflection-based equality protocol.

### Restore Semantics

`restore_run(saved, mpsA, algorithm)` must:

1. retain the existing MPS-length and sweep-domain checks for useful diagnostics;
2. compare the supplied target spaces and algorithm to `saved.identity`;
3. throw `ArgumentError` before allocating derived transfer matrices or caches
   when it differs;
4. retain `_validate_split_cursor` as an independent structural validation.

This does not make a mutable caller-owned target MPS safe to modify while a
run is active. The existing API treats it as immutable input. Document that
callers must not mutate `mpsA` or `algorithm` after `initialize_run`.

### Checkpoint Compatibility

Keep `_CHECKPOINT_SCHEMA_VERSION` at `1`. This code has not been deployed and
there is no published checkpoint compatibility surface to preserve yet. Update
the checkpoint tests to cover the identity field through a normal schema `1`
round trip and retain rejection of unsupported schema values.

## Phase 2: Session Checkpoint Helpers

### Implementation Locations

- `src/execution/local_checkpoint.jl`: `checkpoint!`,
  `pause_and_checkpoint!`, and `cancel_and_checkpoint!`.
- `src/PMPSLab.jl`: export the three new functions.
- `src/execution/checkpoint.jl`: no session state, status transition, or
  numerical logic; retain only file codec concerns.

No new `Task`, channel, timer, global registry, or checkpoint-path field is
added to `LocalRunSession`. The caller supplies the path for each durable
operation, as it already does for `write_checkpoint`.

### Error Behavior

Use `ArgumentError` for an unsupported session status, matching the current
session API. Preserve the original exception from `snapshot` or
`write_checkpoint`; do not wrap it in a generic session error that hides the
path or filesystem cause.

The helpers return the original `LocalRunSession` on success, allowing the
normal Julia style:

```julia
pause_and_checkpoint!(session, "split.jld2")
resume!(session)

cancel_and_checkpoint!(session, "cancelled-split.jld2")
saved = read_checkpoint("cancelled-split.jld2")
resumed = restore_run(saved, target_mps, algorithm)
```

### Memory Lifecycle

`pause_and_checkpoint!` intentionally retains the context for fast short-pause
resume. To release a long-paused run, the owner writes the checkpoint, drops
its references to the `LocalRunSession`/`SplitRun`, and optionally asks Julia
to collect garbage. This plan does not change the `run::R` field into an
optional field or introduce an unloadable session state; doing so would be a
separate platform-lifecycle design change.

## Phase 3: Tests

### Identity Tests

Add to `test/split/snapshot_test.jl` and `test/checkpoint_test.jl`:

1. matching in-memory snapshot restore succeeds;
2. matching JLD2 round-trip restore succeeds;
3. target MPS with identical spaces but different tensor values is accepted;
4. target MPS with incompatible tensor spaces is rejected;
5. changed `LUSolve`/`CholeskySolve`, ansatz, sweep, and each transfer-policy
   field are rejected;
6. identity mismatch fails before `_build_split_context` is called;
7. the schema `1` checkpoint metadata is accepted and unsupported schema values
   are rejected.

### Durable-Pause Tests

Extend `test/execution_test.jl` with a deterministic dummy run that blocks one
`step!` on a Base `Channel`. This permits testing a real pause request while a
step is active without relying on timing or `sleep`:

1. start the run, wait until its first step reports entry, then invoke
   `pause_and_checkpoint!` from another task;
2. assert no checkpoint exists before the channel releases the step;
3. release the step, await the durable helper, and assert
   `LOCAL_PAUSED` plus exactly one completed dummy step;
4. read and restore the checkpoint, then verify the restored run begins at the
   next step;
5. call `resume!` on the original session and verify it continues the original
   in-memory run;
6. use a nonexistent destination directory to verify failed publication leaves
   the session paused and resumable;
7. verify `pause_and_checkpoint!` rejects ready, cancelled, failed, and
   completed sessions, and reports completion if the final step wins the
   pause race;
8. verify `checkpoint!` rejects a running, ready, completed, and failed
   session, while accepting paused and cancelled sessions.

### Durable-Cancellation Tests

Use the same deterministic run to cover:

1. cancellation during an active step writes only after that step completes;
2. the saved snapshot contains the first uncompleted cursor;
3. cancellation from ready and paused states creates a restorable snapshot;
4. cancellation during the final step produces `LOCAL_COMPLETED` and no file;
5. a failed write leaves the session cancelled and allows a later
   `checkpoint!` retry;
6. the existing `cancel!(session)` remains non-blocking and creates no file.

### Regression Suite

Run, in order:

```sh
julia --project=. test/execution_test.jl
julia --project=. test/checkpoint_test.jl
julia --project=. test/runtests.jl
git diff --check
```

The focused test files currently rely on imports provided by their aggregate
test entry points. If direct invocation is not supported by the repository,
run the smallest aggregate entry point instead and do not alter test bootstrap
solely for this plan.

## Phase 4: Documentation

Update `README.md` with two explicit flows.

Short in-memory pause:

```julia
pause!(session)
wait(session)
resume!(session)
```

Durable pause and later process restart:

```julia
pause_and_checkpoint!(session, "split.jld2")

saved = read_checkpoint("split.jld2")
run = restore_run(saved, target_mps, algorithm)
session = start_run(run)
```

Document that a durable pause writes the numerical state but does not preserve a
Julia task, observer, checkpoint path, target object, or algorithm object.
Those immutable inputs are supplied to `restore_run`. Target spaces and the
algorithm configuration must match the checkpoint identity; supplying the
original target tensor values remains the caller's responsibility.

Update `design/resumable-execution.md` to mark the local checkpoint codec and
durable local helpers accurately, while leaving platform responsibilities and
nested LoopTNR execution as deferred.

## Delivery Sequence

1. Implement `SplitProblemIdentity` from target spaces and algorithm configuration.
2. Update JLD2 round-trip/metadata tests for the identity field.
3. Add `checkpoint!` and its quiescent-state tests.
4. Add `pause_and_checkpoint!` with deterministic active-step tests.
5. Add `cancel_and_checkpoint!` with final-step and failure-retry tests.
6. Update exports and README/design documentation.
7. Run focused tests, full tests, and `git diff --check`.

Do not combine this work with cache representation changes, LoopTNR, Krylov
policy changes, or a platform adapter. Each phase should remain independently
reviewable.
