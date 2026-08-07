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
SiteLog(state, error::Real, time::Real) =
    SiteLog(state, Float64(error), Float64(time), nothing)
SiteLog(state, error::Real, time::Real, observable) =
    SiteLog{typeof(state),typeof(observable)}(
        state, Float64(error), Float64(time), observable
    )

# One run's records — element type T is algorithm-specific.
mutable struct SweepLog{T} <: ConvergenceLog
    site_logs::Vector{T}
    sweep_time::Float64
    converged::Bool
    SweepLog{T}() where T = new(T[], 0.0, false)
    SweepLog{T}(site_logs::Vector{T}, sweep_time::Float64, converged::Bool) where {T} =
        new(site_logs, sweep_time, converged)
end
SweepLog(site_logs::Vector{T}, sweep_time::Real, converged::Bool) where {T} =
    SweepLog{T}(site_logs, Float64(sweep_time), converged)
const SplitLog = SweepLog{SiteLog{SweepState,Nothing}}
