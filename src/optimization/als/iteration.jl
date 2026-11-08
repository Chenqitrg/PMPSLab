# Measure only at completed sweep boundaries; never repeat a stale exact error.
function _exact_error(run::SplitState, sweep_completed::Bool)
    if sweep_completed && run.algorithm.error_measure === :exact
        return Float64(squared_distance(
            run.target, run.approximation,
        ) / _cost_scale(run))
    end
    return nothing
end

# Commit one completed numerical update. A user stopping rule may throw; the
# finally block still records which site was completed before Session marks failure.
function _finish_update!(
        run::SplitState,
        position::LatticeIteratorLab.SweepState,
        following::SplitPendingIteration,
        local_cost::Real,
        started::Float64,
    )
    sweep_completed = LatticeIteratorLab.completes_sweep(position, following)
    exact_error = _exact_error(run, sweep_completed)
    completed_updates = run.progress.completed_updates + 1
    completed_sweeps = run.progress.completed_sweeps + Int(sweep_completed)
    penalty = _penalty(run)
    converged = false
    run.pending_iteration = following

    try
        if sweep_completed
            previous_error = run.last_error
            run.last_error = run.algorithm.error_measure === :exact ? exact_error : Float64(local_cost)
            converged, run.stable_comparisons = check_convergence(
                run.algorithm.stopping, previous_error, run.last_error,
                completed_sweeps, run.stable_comparisons,
            )
        end
    finally
        run.progress = SplitProgress(;
            completed_updates,
            completed_sweeps,
            site=position.site,
            sweep=position.completed_sweeps + 1,
            direction=position isa LatticeIteratorLab.SweepState{LatticeIteratorLab.Forward} ? "forward" : "backward",
            local_cost,
            exact_error,
            sweep_completed,
            converged,
            duration=time() - started,
            penalty,
            cost_scale=_cost_scale(run),
        )
    end
    if converged
        run.pending_iteration = nothing
    end
    return run.progress
end

"""
    algorithm_steps(state::SplitState)

Run ALS, yielding `state.progress` after each complete site update and cache
refresh. ExperimentLab handles pause/cancel requests between yields. Capture
`state` at those boundaries to restart from disk. Returns `(mps, progress)` as
a named tuple; the MPS is the workspace's approximation, without an extra copy.
"""
@resumable function algorithm_steps(run::SplitState)::SplitProgress
    while !isnothing(run.pending_iteration)
        position, cursor = run.pending_iteration
        following = iterate(run.sweep, cursor)
        started = time()

        _, local_cost = update!(run, position)
        progress = _finish_update!(run, position, following, local_cost, started)
        _log_split_progress(progress)
        @yield progress
    end
    return (mps=run.approximation, progress=run.progress)
end
