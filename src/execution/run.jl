"""
    AI.initialize_state(problem::SplitProblem, algorithm::ALSSplit; iterate=nothing, snapshot=nothing)

Initialize from the algorithm's policy, an explicit MPS, or a PMPS snapshot.
`iterate` is copied by default. A snapshot resumes its counters and stopping history;
provide the original target in `problem`.
"""
function AI.initialize_state(
        problem::SplitProblem, algorithm::ALSSplit;
        iterate::Union{Nothing,PeriodicMPS}=nothing,
        snapshot::Union{Nothing,SplitSnapshot}=nothing,
        copy_initial::Bool=true,
    )::SplitState
    if !isnothing(snapshot)
        isnothing(iterate) || throw(ArgumentError("provide either iterate or snapshot"))
        return restore_split(snapshot, problem.target; algorithm)
    end
    isnothing(iterate) && return initialize_split(problem.target, algorithm)
    return initialize_split(problem.target, iterate, algorithm; copy_initial)
end

"""Reset an existing workspace from an MPS, rebuilding caches and stopping history."""
function AI.initialize_state!(
        problem::SplitProblem, algorithm::ALSSplit, state::S;
        iterate::PeriodicMPS=state.iterate, copy_initial::Bool=true,
    )::S where {S<:SplitState}
    fresh = AI.initialize_state(problem, algorithm; iterate, copy_initial)
    typeof(fresh) === S || throw(ArgumentError("reset requires the same workspace type; initialize a new state"))
    for field in fieldnames(S)
        setfield!(state, field, getfield(fresh, field))
    end
    return state
end

# Progress describes the completed update; only the stopping check marks convergence.
function _converged(progress::SplitProgress)::SplitProgress
    return SplitProgress(
        progress.completed_updates, progress.completed_sweeps, progress.site,
        progress.sweep, progress.direction, progress.local_cost, progress.exact_error,
        progress.sweep_completed, true, progress.duration, progress.penalty, progress.cost_scale,
    )
end

function AI.is_finished!(problem::SplitProblem, algorithm::ALSSplit, state::SplitState)::Bool
    finished = if state.checked_iteration != state.progress.completed_updates
        stopped = AI.is_finished!(problem, algorithm, state, algorithm.stopping_criterion, state.stopping_criterion_state)
        state.checked_iteration = state.progress.completed_updates
        AI.indicates_convergence(algorithm, state) && (state.progress = _converged(state.progress))
        stopped || isnothing(state.pending_iteration)
    else
        AI.is_finished(problem, algorithm, state)
    end
    finished && (state.pending_iteration = nothing)
    return finished
end

function AI.is_finished(problem::SplitProblem, algorithm::ALSSplit, state::SplitState)::Bool
    return isnothing(state.pending_iteration) || AI.is_finished(
        problem, algorithm, state, algorithm.stopping_criterion, state.stopping_criterion_state,
    )
end

function AI.get_reason(algorithm::ALSSplit, state::SplitState)::Union{Nothing,String}
    reason = AI.get_reason(algorithm.stopping_criterion, state.stopping_criterion_state)
    !isnothing(reason) && return reason
    return state.progress.completed_updates == length(state.sweep) ? "Completed the configured traversal.\n" : nothing
end

function AI.finalize_state!(::SplitProblem, ::ALSSplit, state::SplitState)::NamedTuple{(:mps,:progress)}
    return (mps=state.iterate, progress=state.progress)
end

"""
    split(target, algorithm)
    split(target, initial, algorithm; copy_initial=true)

Run `AlgorithmsInterface.solve` and return `(mps, progress)`. An explicit initial
MPS fixes the virtual spaces and is copied by default. Use A.I.'s state/step methods
for measurements and checkpoints between site updates.
"""
function split(target::PeriodicMPS, algorithm::AbstractSplit)::NamedTuple{(:mps,:progress)}
    return AI.solve(SplitProblem(target), algorithm)
end

function split(
        target::PeriodicMPS, initial::PeriodicMPS, algorithm::AbstractSplit;
        copy_initial::Bool=true,
    )::NamedTuple{(:mps,:progress)}
    return AI.solve(SplitProblem(target), algorithm; iterate=initial, copy_initial)
end
