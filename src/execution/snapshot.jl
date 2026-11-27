"""Restart data containing tensors, traversal and stopping history; no caches."""
struct SplitSnapshot{M<:PeriodicMPS,A<:ALSSplit,T,R,C<:AI.StoppingCriterionState}
    approximation::M
    algorithm::A
    target_spaces::T
    progress::SplitProgress
    stopping_criterion_state::C
    checked_iteration::Int
    regularization::R
end

"""
    snapshot(state::SplitState)

Copy the approximation, algorithm and stopping/regularization state into a
`SplitSnapshot`. Capture before execution, between completed site updates, or
after completion; do not copy a state while another task is updating it.

The snapshot owns its tensors and excludes numerical caches and the target.
Use `restore_split(saved, target)` with the original target to rebuild a workspace.
Persistence belongs to the caller.
"""
function snapshot(state::SplitState)::SplitSnapshot
    return SplitSnapshot(
        PeriodicMPS([copy(t) for t in state.iterate]),
        deepcopy(state.algorithm),
        [(codomain(t), domain(t)) for t in state.target],
        state.progress,
        deepcopy(state.stopping_criterion_state),
        state.checked_iteration,
        deepcopy(state.regularization),
    )
end

# Reconstruct the cursor from the completed update count. Only the lightweight
# traversal is replayed: no tensor updates, measurements, callbacks or logs.
# Custom sweeps can use nonconsecutive integer cursors; LatticeIteratorLab owns their meaning.
function _restore_iteration(saved::SplitSnapshot, sweep::S)::SplitPendingIteration where {S}
    progress = saved.progress
    0 <= progress.completed_updates <= length(sweep) ||
        throw(ArgumentError("invalid snapshot update count"))
    -1 <= saved.checked_iteration <= progress.completed_updates ||
        throw(ArgumentError("invalid snapshot stopping-check count"))
    _validate_stopping_state(saved.stopping_criterion_state, progress)

    pending = iterate(sweep)
    completed_sweeps = 0
    local position, boundary
    for _ in 1:progress.completed_updates
        isnothing(pending) && throw(ArgumentError("snapshot exceeds traversal"))
        position, cursor = pending
        pending = iterate(sweep, cursor)
        boundary = LatticeIteratorLab.completes_sweep(position, pending)
        completed_sweeps += Int(boundary)
    end
    progress.completed_sweeps == completed_sweeps ||
        throw(ArgumentError("snapshot sweep count does not match traversal"))

    if progress.completed_updates == 0
        initial = progress.converged ? _converged(SplitProgress()) : SplitProgress()
        progress == initial || throw(ArgumentError("invalid initial progress"))
    else
        direction = position isa LatticeIteratorLab.SweepState{LatticeIteratorLab.Forward} ? "forward" : "backward"
        if progress.site != position.site || progress.sweep != position.completed_sweeps + 1 ||
           progress.direction != direction || progress.sweep_completed != boundary
            throw(ArgumentError("snapshot progress does not match traversal"))
        end
    end
    return progress.converged ? nothing : pending
end

"""Restore at a completed update boundary using the original read-only target."""
function restore_split(saved::SplitSnapshot, target::PeriodicMPS; algorithm::ALSSplit=saved.algorithm)::SplitState
    algorithm == saved.algorithm || throw(ArgumentError("split configuration mismatch"))
    target_spaces = [(codomain(t), domain(t)) for t in target]
    saved.target_spaces == target_spaces ||
        throw(DimensionMismatch("snapshot target spaces do not match"))
    sweep = LatticeIteratorLab.sweep_states(saved.algorithm.sweep, length(saved.approximation))
    pending = _restore_iteration(saved, sweep)
    _validate_initial(target, saved.approximation)
    target_squared_norm = expectation(target, target)
    regularization = _restore_regularization(
        saved.algorithm.regularization, saved.regularization, saved.approximation, target_squared_norm,
    )
    approximation = PeriodicMPS([copy(t) for t in saved.approximation])
    run = _build_split_state(target, approximation, saved.algorithm, target_squared_norm, regularization)
    run.pending_iteration = pending
    run.progress = saved.progress
    typeof(run.stopping_criterion_state) === typeof(saved.stopping_criterion_state) ||
        throw(ArgumentError("snapshot stopping state does not match its configuration"))
    run.stopping_criterion_state = deepcopy(saved.stopping_criterion_state)
    run.iteration = saved.progress.completed_updates
    run.checked_iteration = saved.checked_iteration
    AI.is_finished(SplitProblem(target), algorithm, run) && (run.pending_iteration = nothing)
    return run
end

_restore_regularization(::NoRegularization, ::Nothing, ::PeriodicMPS, ::Number) = nothing
function _restore_regularization(
        ::NuclearNorm, saved::NuclearNormState, approximation::PeriodicMPS, squared_norm::Number,
    )
    _validate_nnr_norm(squared_norm)
    for tensors in (saved.auxiliary, saved.dual)
        length(tensors) == length(approximation) ||
            throw(ArgumentError("snapshot regularization buffer length does not match the MPS"))
        all(space(tensors[i]) == space(approximation[i]) for i in eachindex(tensors)) ||
            throw(DimensionMismatch("snapshot regularization spaces do not match the MPS"))
    end
    return deepcopy(saved)
end

_restore_regularization(::AbstractRegularization, buffers, ::PeriodicMPS, ::Number) =
    throw(ArgumentError("snapshot regularization buffers do not match its configuration"))

# Unknown A.I. criteria own their validation; validate the history PMPS itself owns.
_validate_stopping_state(::AI.StoppingCriterionState, ::SplitProgress)::Nothing = nothing

function _validate_stopping_state(history::CostStoppingState, progress::SplitProgress)::Nothing
    0 <= history.last_sweep <= progress.completed_sweeps ||
        throw(ArgumentError("snapshot convergence sweep does not match traversal"))
    0 <= history.stable_comparisons <= max(0, history.last_sweep - 1) ||
        throw(ArgumentError("invalid snapshot convergence count"))
    -1 <= history.at_iteration <= progress.completed_updates ||
        throw(ArgumentError("invalid snapshot stopping iteration"))
    (history.last_sweep == 0) == isnothing(history.last_error) ||
        throw(ArgumentError("snapshot error does not match its convergence history"))
    return nothing
end

function _validate_stopping_state(history::AI.GroupStoppingCriterionState, progress::SplitProgress)::Nothing
    for child in history.criteria_states
        _validate_stopping_state(child, progress)
    end
    return nothing
end
