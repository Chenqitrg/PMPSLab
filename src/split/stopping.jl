"""
    AbstractStoppingCriterion

Base type for optional algorithm stopping criteria.
Concrete criteria implement [`should_stop`](@ref) for their progress-log type.
"""
abstract type AbstractStoppingCriterion <: AbstractAlgorithmConfig end

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
`nothing` disables early stopping.
"""
should_stop(::Nothing, ::Any) = false

function should_stop(
        criterion::CostConvergence,
        site_logs::AbstractVector{<:SplitSiteLog},
    )
    isempty(site_logs) && return false

    latest = last(site_logs)
    latest.state.n + 1 >= criterion.min_sweeps || return false

    comparisons = 0
    latest_sweep = latest.state.n
    for index in (length(site_logs) - 1):-1:1
        earlier = site_logs[index]
        earlier.state.n == latest_sweep && continue
        earlier.state.n == latest_sweep - 1 || return false
        isfinite(earlier.error) && isfinite(latest.error) || return false
        isapprox(
            earlier.error,
            latest.error;
            atol=criterion.atol,
            rtol=criterion.rtol,
        ) || return false

        comparisons += 1
        comparisons == criterion.consecutive_sweeps && return true
        latest = earlier
        latest_sweep = earlier.state.n
    end
    return false
end
