# ------------------------------------------------------------------------------
# Convergence logs — layered, immutable inner → mutable outer
# ------------------------------------------------------------------------------

abstract type ConvergenceLog end

# One periodic QR/LQ fixed-point solve.
struct PowerIterLog
    pos::HalfInt
    errors::Vector{Float64}
    # MPSKit's private uniform solvers do not expose their internal iteration
    # count, so direct-adapter runs record `nothing` here.
    steps::Union{Int,Nothing}
    converged::Bool
end

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
SweepLog(site_logs::Vector{PowerIterLog}, sweep_time::Real) =
    SweepLog(site_logs, sweep_time, all(log -> log.converged, site_logs))

mutable struct ProjectorLog <: ConvergenceLog
    sweeps::Vector{SweepLog{PowerIterLog}}
    converged::Bool
    ProjectorLog() = new(SweepLog{PowerIterLog}[], false)
end

const SplitLog = SweepLog{SiteLog{SweepState,Nothing}}
