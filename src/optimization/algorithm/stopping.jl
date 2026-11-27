"""
    AbstractStoppingCriterion

Base type for optional algorithm stopping criteria.
Concrete criteria implement [`check_convergence`](@ref) at completed sweep boundaries.
"""
abstract type AbstractStoppingCriterion <: AI.StoppingCriterion end

"""
    NoStopping()

Disable convergence-based early stopping and run the configured sweep to
completion.
"""
Configurations.@option struct NoStopping <: AbstractStoppingCriterion end

"""
    CostTolerance(; atol, rtol, min_sweeps=2)

Stop at a complete sweep boundary once both the absolute cost and its relative
change from the preceding sweep satisfy their tolerances. The criterion is
evaluated only after at least `min_sweeps` complete sweeps.
"""
Configurations.@option struct CostTolerance <: AbstractStoppingCriterion
    atol::Float64
    rtol::Float64
    min_sweeps::Int = 2

    function CostTolerance(atol::Real, rtol::Real, min_sweeps::Integer)
        absolute_tolerance = Float64(atol)
        relative_tolerance = Float64(rtol)
        minimum_sweeps = Int(min_sweeps)

        isfinite(absolute_tolerance) && absolute_tolerance > 0 ||
            throw(ArgumentError(
                "atol must be finite and positive, got $atol"
            ))
        isfinite(relative_tolerance) && relative_tolerance > 0 ||
            throw(ArgumentError(
                "rtol must be finite and positive, got $rtol"
            ))
        minimum_sweeps >= 2 || throw(ArgumentError(
            "min_sweeps must be at least 2, got $min_sweeps"
        ))
        return new(absolute_tolerance, relative_tolerance, minimum_sweeps)
    end
end

"""
    CostConvergence(; atol=0, rtol=1e-8, min_sweeps=2, consecutive_sweeps=2)

Stop after the cost at complete sweep boundaries remains approximately
unchanged for `consecutive_sweeps` comparisons. At least `min_sweeps` complete
sweeps are always performed.
"""
Configurations.@option struct CostConvergence <: AbstractStoppingCriterion
    atol::Float64 = 0.0
    rtol::Float64 = 1e-8
    min_sweeps::Int = 2
    consecutive_sweeps::Int = 2

    function CostConvergence(
            atol::Real,
            rtol::Real,
            min_sweeps::Integer,
            consecutive_sweeps::Integer,
        )
        absolute_tolerance = Float64(atol)
        relative_tolerance = Float64(rtol)
        minimum_sweeps = Int(min_sweeps)
        required_comparisons = Int(consecutive_sweeps)

        isfinite(absolute_tolerance) && absolute_tolerance >= 0 || throw(ArgumentError(
            "atol must be finite and non-negative, got $atol"
        ))
        isfinite(relative_tolerance) && relative_tolerance >= 0 || throw(ArgumentError(
            "rtol must be finite and non-negative, got $rtol"
        ))
        minimum_sweeps >= 2 || throw(ArgumentError(
            "min_sweeps must be at least 2, got $min_sweeps"
        ))
        required_comparisons >= 1 || throw(ArgumentError(
            "consecutive_sweeps must be positive, got $consecutive_sweeps"
        ))
        return new(
            absolute_tolerance,
            relative_tolerance,
            minimum_sweeps,
            required_comparisons,
        )
    end
end

"""
    check_convergence(criterion, previous, current, sweeps, stable)

Return `(converged, stable_comparisons)` at a completed sweep. `previous` is
nothing on the first sweep. Extensions need only the previous selected error,
current error, total sweep count and their consecutive-comparison counter.
"""
check_convergence(::NoStopping, previous, current, sweeps, stable) = (false, 0)

function check_convergence(criterion::CostTolerance, previous, current, sweeps, stable)
    if previous === nothing
        return false, 0
    end

    change = abs(current - previous)
    if iszero(previous)
        relative_change = iszero(change) ? 0.0 : Inf
    else
        relative_change = change / abs(previous)
    end

    converged = sweeps >= criterion.min_sweeps &&
        isfinite(previous) && isfinite(current) &&
        abs(current) < criterion.atol &&
        relative_change < criterion.rtol
    return converged, 0
end

function check_convergence(criterion::CostConvergence, previous, current, sweeps, stable)
    matched = previous !== nothing &&
        isfinite(previous) && isfinite(current) &&
        isapprox(previous, current; atol=criterion.atol, rtol=criterion.rtol)

    stable_comparisons = matched ? stable + 1 : 0
    converged = sweeps >= criterion.min_sweeps &&
        stable_comparisons >= criterion.consecutive_sweeps
    return converged, stable_comparisons
end

# Each criterion owns its history, including when nested in an A.I. criterion group.
mutable struct CostStoppingState <: AI.StoppingCriterionState
    at_iteration::Int
    last_error::Union{Nothing,Float64}
    stable_comparisons::Int
    last_sweep::Int
end

function AI.initialize_state(
        ::SplitProblem, ::AbstractSplit, ::AbstractStoppingCriterion,
    )::CostStoppingState
    return CostStoppingState(-1, nothing, 0, 0)
end

function AI.initialize_state!(
        ::SplitProblem, ::AbstractSplit, ::AbstractStoppingCriterion, state::CostStoppingState,
    )::CostStoppingState
    state.at_iteration = -1
    state.last_error = nothing
    state.stable_comparisons = 0
    state.last_sweep = 0
    return state
end

AI.indicates_convergence(::Type{CostTolerance})::Bool = true
AI.indicates_convergence(::Type{CostConvergence})::Bool = true

function AI.is_finished!(
        ::SplitProblem, algorithm::AbstractSplit, state::AbstractSplitState,
        criterion::AbstractStoppingCriterion, history::CostStoppingState,
    )::Bool
    AI.is_active(criterion, history) && return true
    progress = state.progress
    if progress.sweep_completed && progress.completed_sweeps > history.last_sweep
        current = algorithm.error_measure === :exact ? progress.exact_error : progress.local_cost
        converged, stable = check_convergence(
            criterion, history.last_error, current, progress.completed_sweeps, history.stable_comparisons,
        )
        history.last_error = current
        history.last_sweep = progress.completed_sweeps
        history.stable_comparisons = stable
        converged && (history.at_iteration = state.iteration)
    end
    return AI.is_active(criterion, history)
end

function AI.is_finished!(
        ::SplitProblem, ::AbstractSplit, state::AbstractSplitState,
        criterion::NoStopping, history::CostStoppingState,
    )::Bool
    isnothing(state.pending_iteration) && (history.at_iteration = state.iteration)
    return AI.is_active(criterion, history)
end

AI.is_finished(::SplitProblem, ::AbstractSplit, ::AbstractSplitState,
    criterion::AbstractStoppingCriterion, history::CostStoppingState)::Bool = AI.is_active(criterion, history)

function AI.get_reason(criterion::AbstractStoppingCriterion, history::CostStoppingState)::Union{Nothing,String}
    AI.is_active(criterion, history) || return nothing
    return "Cost converged after $(history.last_sweep) sweeps.\n"
end

function AI.get_reason(criterion::NoStopping, history::CostStoppingState)::Union{Nothing,String}
    AI.is_active(criterion, history) || return nothing
    return "Completed the configured traversal after $(history.at_iteration) site updates.\n"
end

function Base.summary(io::IO, criterion::AbstractStoppingCriterion, history::CostStoppingState)::Nothing
    print(io, nameof(typeof(criterion)), ": ", AI.is_active(criterion, history) ? "reached" : "not reached")
    return nothing
end
