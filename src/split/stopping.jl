"""
    AbstractStoppingCriterion

Base type for optional algorithm stopping criteria.
Concrete criteria implement [`should_stop`](@ref) for their progress-log type.
"""
abstract type AbstractStoppingCriterion <: AbstractAlgorithmConfig end

"""
    NoStopping()

Disable convergence-based early stopping and run the configured sweep to
completion.
"""
struct NoStopping <: AbstractStoppingCriterion end

"""
    CostTolerance(; atol, rtol, min_sweeps=2)

Stop at a complete sweep boundary once both the absolute cost and its relative
change from the preceding sweep satisfy their tolerances. The criterion is
evaluated only after at least `min_sweeps` complete sweeps.
"""
struct CostTolerance <: AbstractStoppingCriterion
    atol::Float64
    rtol::Float64
    min_sweeps::Int

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

CostTolerance(;
        atol::Real,
        rtol::Real,
        min_sweeps::Integer = 2,
) = CostTolerance(atol, rtol, min_sweeps)

"""
    CostConvergence(; atol=0, rtol=1e-8, min_sweeps=2, consecutive_sweeps=2)

Stop after the cost at complete sweep boundaries remains approximately
unchanged for `consecutive_sweeps` comparisons. At least `min_sweeps` complete
sweeps are always performed.
"""
struct CostConvergence <: AbstractStoppingCriterion
    atol::Float64
    rtol::Float64
    min_sweeps::Int
    consecutive_sweeps::Int

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

function CostConvergence(;
        atol::Real = 0,
        rtol::Real = 1e-8,
        min_sweeps::Integer = 2,
        consecutive_sweeps::Integer = 2,
    )
    return CostConvergence(atol, rtol, min_sweeps, consecutive_sweeps)
end

"""
    should_stop(criterion, progress)

Return whether `criterion` is satisfied by algorithm-specific `progress`.
[`NoStopping`](@ref) disables early stopping.
"""
should_stop(::NoStopping, ::Any) = false

function should_stop(
        criterion::CostTolerance,
        site_logs::AbstractVector{<:SplitSiteLog},
    )
    isempty(site_logs) && return false

    latest = last(site_logs)
    latest.state.completed_sweeps + 1 >= criterion.min_sweeps || return false
    isfinite(latest.cost) || return false
    abs(latest.cost) < criterion.atol || return false

    previous = nothing
    for index in (length(site_logs) - 1):-1:1
        candidate = site_logs[index]
        candidate.state.completed_sweeps == latest.state.completed_sweeps && continue
        candidate.state.completed_sweeps == latest.state.completed_sweeps - 1 || return false
        previous = candidate
        break
    end
    isnothing(previous) && return false
    isfinite(previous.cost) || return false

    cost_change = abs(latest.cost - previous.cost)
    relative_cost_change = iszero(previous.cost) ?
        (iszero(cost_change) ? 0.0 : Inf) :
        cost_change / abs(previous.cost)
    return relative_cost_change < criterion.rtol
end

function should_stop(
        criterion::CostConvergence,
        site_logs::AbstractVector{<:SplitSiteLog},
    )
    isempty(site_logs) && return false

    latest = last(site_logs)
    latest.state.completed_sweeps + 1 >= criterion.min_sweeps || return false

    comparisons = 0
    latest_sweep = latest.state.completed_sweeps
    for index in (length(site_logs) - 1):-1:1
        earlier = site_logs[index]
        earlier.state.completed_sweeps == latest_sweep && continue
        earlier.state.completed_sweeps == latest_sweep - 1 || return false
        isfinite(earlier.cost) && isfinite(latest.cost) || return false
        isapprox(
            earlier.cost,
            latest.cost;
            atol=criterion.atol,
            rtol=criterion.rtol,
        ) || return false

        comparisons += 1
        comparisons == criterion.consecutive_sweeps && return true
        latest = earlier
        latest_sweep = earlier.state.completed_sweeps
    end
    return false
end
