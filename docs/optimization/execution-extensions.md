# Execution through AlgorithmsInterface

`SplitProblem <: AlgorithmsInterface.Problem` holds the read-only target.
`ALSSplit <: AlgorithmsInterface.Algorithm` holds the policies and
`stopping_criterion`. `SplitState <: AlgorithmsInterface.State` owns `iterate`,
`iteration`, `stopping_criterion_state`, numerical caches and the sweep cursor.

`split(target, algorithm)` calls `AlgorithmsInterface.solve`. An explicit initial
MPS is accepted by `split(target, initial, algorithm)` or by
`initialize_state(problem, algorithm; iterate=initial)`. Initial tensors are copied
unless `copy_initial=false` is requested.

## One completed site

The A.I. driver checks `is_finished!`, increments `iteration`, then calls `step!`.
PMPS solves one site, refreshes the affected caches, and commits `state.progress`.
`progress.completed_updates` counts completed work; `iteration` also identifies
an attempted update that may fail. The step function does not log or save data.

A caller can drive this interface directly while upstream `eachstep` is pending:

```julia
import AlgorithmsInterface as AI
problem = PMPSLab.SplitProblem(target)
state = AI.initialize_state(problem, algorithm)
while !AI.is_finished!(problem, algorithm, state)
    AI.increment!(state)
    AI.step!(problem, algorithm, state)
    # Inspect state.progress, measure state.iterate, or capture a snapshot here.
    state.progress.completed_updates == 2 && break
end
```

Continue with the same loop, omitting the break. There is no background task to
pause or cancel. A.I. `CallbackAction`s are available at `:PostStep` for observation;
put persistence directly in the calling loop when write failures must propagate.
Only one driver may advance a workspace at a time.

## Stopping

`stopping_criterion` accepts A.I. criteria and groups. PMPS's `CostTolerance` and
`CostConvergence` evaluate only completed sweeps, using `local_cost` or `exact_error`
according to `error_measure`. Each criterion owns its error and patience history.
The finite sweep schedule is a hard limit even if the criterion remains inactive.
`NoStopping()` means run that schedule without convergence-based early stopping.

Stopping is checked before the next update. Consequently a `:PostStep` observer
sees the completed numerical update before that update's convergence check;
`progress.converged` is set by the subsequent stopping check. Repeating a check
without another update does not advance patience again. A stopping-rule exception
leaves the completed update and next traversal position available for inspection.
Its history is committed only after a successful check, so retrying cannot skip
the failed check. Native convergence criteria may also stop before the first site
or between sites; their checkpoints remain restartable.

## Snapshots and restart

`PMPSLab.snapshot(state)` copies the current MPS, algorithm, stopping history and
regularization buffers. It excludes caches and the target. Capture before execution,
between completed updates, or after completion. `restore_split(saved, target)`
requires the original target and rebuilds the caches. Matching spaces do not prove
that a different target has the same tensor values.

```julia
saved = PMPSLab.snapshot(state)
result = AI.solve(problem, algorithm; snapshot=saved)
```

A snapshot taken before a stopping check retains that pending check. Restoration
reconstructs only sweep positions; it does not replay numerical updates. NNR
snapshots also preserve detached auxiliary and dual tensors.

`AI.solve!(problem, algorithm, state)` resets the workspace from its current MPS
(or a supplied `iterate`), including counters and stopping history. It is not a
resume operation. Use the existing state loop or the `snapshot` keyword to resume.

## Experiment data

DrWatson belongs in the experiment environment. It can call `split` through
`produce_or_load`, save a snapshot with `wsave`, and add Git provenance with
`tagsave`. PMPS has no storage-backend dependency. See the existing
[recorded-run example](../../examples/optimization/execution/02_recorded_split.jl).

ExperimentLab configuration registries, `@config`, recorded sessions and generator
APIs are retired from PMPS. Historical configuration files and snapshots need their
original environment; automatic migration of that storage format is not provided.
The main call-site changes are:

| Former use | Current use |
| --- | --- |
| `ALSSplit(stopping=rule, ...)` | `ALSSplit(stopping_criterion=rule, ...)` |
| `state.approximation` | `state.iterate` |
| Generator plus `start_run` / `fetch` | `AI.solve` or the explicit loop above |
| Recorded runs and `capture` | Application-owned results and `PMPSLab.snapshot` |
| Registered configuration JSON | Ordinary constructors, application settings in JSON, or JLD2 objects |

LatticeIteratorLab and TensorLinearSolveLab still depend on ExperimentLab
transitively; removing those dependencies is a separate package change.

The A.I. dependency is pinned to main commit `abe2c54` in `Project.toml`. Scoped
logging, `eachstep`, and editable tree input/display are not prerequisites for this
implementation and can be adopted after their upstream PRs settle.
