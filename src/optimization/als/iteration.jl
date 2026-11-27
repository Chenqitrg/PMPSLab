# Measure only at completed sweep boundaries; never repeat a stale exact error.
function _exact_error(run::SplitState, sweep_completed::Bool)::Union{Nothing,Float64}
    if sweep_completed && run.algorithm.error_measure === :exact
        return Float64(squared_distance(run.target, run.iterate) / _cost_scale(run))
    end
    return nothing
end

# Commit the numerical update before A.I. checks stopping on its next loop pass.
# A failing stopping rule therefore cannot hide a completed tensor update.
function _finish_update!(
        run::SplitState,
        position::LatticeIteratorLab.SweepState,
        following::SplitPendingIteration,
        local_cost::Real,
        started::Float64,
    )::SplitProgress
    sweep_completed = LatticeIteratorLab.completes_sweep(position, following)
    exact_error = _exact_error(run, sweep_completed)
    run.pending_iteration = following
    run.progress = SplitProgress(;
        completed_updates=run.progress.completed_updates + 1,
        completed_sweeps=run.progress.completed_sweeps + Int(sweep_completed),
        site=position.site,
        sweep=position.completed_sweeps + 1,
        direction=position isa LatticeIteratorLab.SweepState{LatticeIteratorLab.Forward} ? "forward" : "backward",
        local_cost,
        exact_error,
        sweep_completed,
        duration=time() - started,
        penalty=_penalty(run),
        cost_scale=_cost_scale(run),
    )
    return run.progress
end

"""Perform one site update. The A.I. driver checks stopping and increments first."""
function AI.step!(problem::SplitProblem, algorithm::ALSSplit, run::SplitState)::SplitState
    problem.target === run.target && algorithm == run.algorithm ||
        throw(ArgumentError("problem and algorithm must match the initialized state"))
    isnothing(run.pending_iteration) && throw(ArgumentError("split has no pending update"))
    position, cursor = run.pending_iteration
    following = iterate(run.sweep, cursor)
    started = time()

    _, local_cost = update!(run, position)
    _finish_update!(run, position, following, local_cost, started)
    return run
end
