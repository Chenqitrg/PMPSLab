"""Restart data without caches, suspended generators or diagnostic history."""
struct SplitSnapshot{M<:PeriodicMPS,A<:ALSSplit,T,R}
    approximation::M
    algorithm::A
    target_spaces::T
    progress::SplitProgress
    last_error::Union{Nothing,Float64}
    stable_comparisons::Int
    regularization::R
end

# capture(ref, state) calls this directly: tensor buffers are copied exactly once.
function ExperimentLab.copy_state(state::SplitState)
    return SplitSnapshot(
        PeriodicMPS([copy(t) for t in state.approximation]),
        deepcopy(state.algorithm),
        [(codomain(t), domain(t)) for t in state.target],
        state.progress,
        state.last_error,
        state.stable_comparisons,
        deepcopy(state.regularization),
    )
end

# Reconstruct the cursor from the completed update count. Only the lightweight
# traversal is replayed: no tensor updates, measurements, callbacks or logs.
# Custom sweeps can use nonconsecutive integer cursors; LatticeIteratorLab owns their meaning.
function _restore_iteration(saved::SplitSnapshot, sweep)
    progress = saved.progress
    0 <= progress.completed_updates <= length(sweep) ||
        throw(ArgumentError("invalid snapshot update count"))
    0 <= saved.stable_comparisons <= max(0, progress.completed_sweeps - 1) ||
        throw(ArgumentError("invalid snapshot convergence count"))

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
        progress == SplitProgress() || throw(ArgumentError("invalid initial progress"))
        isnothing(saved.last_error) && saved.stable_comparisons == 0 ||
            throw(ArgumentError("initial snapshot has stopping history"))
    else
        direction = position isa LatticeIteratorLab.SweepState{LatticeIteratorLab.Forward} ? "forward" : "backward"
        if progress.site != position.site || progress.sweep != position.completed_sweeps + 1 ||
           progress.direction != direction || progress.sweep_completed != boundary
            throw(ArgumentError("snapshot progress does not match traversal"))
        end
        progress.converged && !boundary &&
            throw(ArgumentError("convergence requires a completed sweep"))
    end
    return progress.converged ? nothing : pending
end

"""Restore at a completed update boundary using the original read-only target."""
function restore_split(saved::SplitSnapshot, target::PeriodicMPS; algorithm::ALSSplit=saved.algorithm)
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
    run.last_error = saved.last_error
    run.stable_comparisons = saved.stable_comparisons
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
