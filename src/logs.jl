# ------------------------------------------------------------------------------
# Convergence logs — layered, immutable inner → mutable outer
# ------------------------------------------------------------------------------

abstract type ConvergenceLog end

# One immutable convergence snapshot. `state` is algorithm-specific.
struct SiteLog{S,E}
    state::S
    error::Float64
    time::Float64
    observable::E
end
SiteLog(state::S, error::Real, time::Real) where {S} =
    SiteLog(state, Float64(error), Float64(time), nothing)
SiteLog(
    state::S, error::Real, time::Real, observable::E
) where {S,E} =
    SiteLog{S,E}(
        state, Float64(error), Float64(time), observable
    )

# One run's records — element type T is algorithm-specific.
mutable struct SweepLog{T} <: ConvergenceLog
    site_logs::Vector{T}
    sweep_time::Float64
    converged::Bool
end
SweepLog{T}() where {T} = SweepLog{T}(T[], 0.0, false)
SweepLog(site_logs::Vector{T}, sweep_time::Real, converged::Bool) where {T} =
    SweepLog{T}(site_logs, Float64(sweep_time), converged)

const SplitSiteLog = SiteLog{S,Nothing} where {S<:SweepState}
const SplitLog = SweepLog{SplitSiteLog}
