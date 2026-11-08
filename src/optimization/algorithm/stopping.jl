"""
    AbstractStoppingCriterion

Base type for optional algorithm stopping criteria.
Concrete criteria implement [`check_convergence`](@ref) at completed sweep boundaries.
"""
abstract type AbstractStoppingCriterion <: ExperimentLab.AbstractAlgorithmConfig end

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
