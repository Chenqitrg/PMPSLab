# Execution integration

Both `split` and `start_run(split_steps(state))` use ExperimentLab's driver.
PMPSLab expresses ALS directly as a generator; it has no separate stepping protocol,
session state machine,
observer protocol, diagnostic history container or filesystem implementation.

Other optimizers implement the [split interface](../../src/optimization/interface.jl): an
`AbstractSplit` configuration, an `AbstractSplitState`, initialization, a generator,
and snapshot restoration. Their events need not have ALS site/sweep fields; their
final result must contain `mps`. Implement `ExperimentLab.copy_state` when restart
requires a smaller representation than the default deep copy. When extending a
generator, first `import PMPSLab: split_steps`, then define
`@resumable function split_steps(...)`; the macro requires an unqualified name.
The details below describe the current ALS implementation.

`split_steps` updates one site, refreshes caches, records numerical progress,
logs a `SplitProgress` through Julia Logging, then yields it. At termination it returns `(mps=..., progress=...)`. The recorded entry point
`start_run(store, algorithm_ref, split_steps(state))` owns execution records and
JSONL logs. Use `onstep(session, update)` to capture numerical state or request a
pause. ExperimentLab handles failures and finalization.

`SplitProgress` is the same immutable value seen in `state.progress`, `onstep`
and logs. `exact_error` is `nothing` unless this step actually measured it.
The previous stopping error and patience counter stay in `SplitState`, outside
public progress. Numerical convergence never reads logs.

Use `capture(ref, state)` directly. PMPSLab specializes
`ExperimentLab.copy_state` to produce a detached cache-free `SplitSnapshot`.
Restoration is `restore_split(saved, target)`; the snapshot carries its own
configuration. It does not replay old logs. This preserves per-site restart,
including backward scans and convergence patience, with one tensor copy at capture.

Snapshots must be captured inside the callback, after an acknowledged pause, or
after completion. The original target is required when restoring; matching spaces
do not imply matching tensor values.

See [the recorded example](../../examples/optimization/execution/02_recorded_split.jl).

For a selected-site pause, call `pause!(session)` inside `onstep` when the event's
site and sweep match. That site is complete; the next one has not started.
Wait until `run_status(session) == ExperimentLab.PAUSED` before inspecting tensors
from the REPL. `resume!` continues the same generator. `fetch` waits for termination
and does not implicitly resume it. See [the pause example](../../examples/optimization/execution/01_controlled_execution.jl).

A snapshot contains the completed update count, not an independently saved
iterator cursor or suspended generator. Restoration replays only LatticeIteratorLab
positions to reconstruct the cursor, then rebuilds caches and starts a fresh
generator there. This traversal is linear in the completed update count and
does not repeat tensor operations or logs. Do not drive two generators sharing the same `SplitState`.
