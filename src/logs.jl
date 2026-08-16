# ------------------------------------------------------------------------------
# Convergence logs — layered, immutable inner → mutable outer
# ------------------------------------------------------------------------------

abstract type ConvergenceLog end

# One immutable control/convergence snapshot. Physical measurement results live
# in a separate MeasurementHistory.
struct SiteLog{S}
    state::S
    cost::Float64
    duration::Float64
end
SiteLog(state::S, cost::Real, duration::Real) where {S} =
    SiteLog{S}(state, Float64(cost), Float64(duration))

# One run's records — element type T is algorithm-specific.
mutable struct SweepLog{T} <: ConvergenceLog
    site_logs::Vector{T}
    duration::Float64
    converged::Bool
end
SweepLog{T}() where {T} = SweepLog{T}(T[], 0.0, false)
SweepLog(site_logs::Vector{T}, duration::Real, converged::Bool) where {T} =
    SweepLog{T}(site_logs, Float64(duration), converged)

const SplitSiteLog = SiteLog{S} where {S<:SweepState}
const SplitLog = SweepLog{SplitSiteLog}
