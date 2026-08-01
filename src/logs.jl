# ------------------------------------------------------------------------------
# Convergence logs — layered, immutable inner → mutable outer
# ------------------------------------------------------------------------------

abstract type ConvergenceLog end

# Per-site power-iteration log (immutable snapshot)
struct PowerIterLog
    pos::HalfInt
    errors::Vector{Float64}
    steps::Int
    converged::Bool
end

# One full projector-computation sweep (immutable)
struct SweepLog
    site_logs::Vector{PowerIterLog}
    sweep_time::Float64
end

# Top-level: accumulates sweeps across algorithm steps (mutable)
mutable struct ProjectorLog <: ConvergenceLog
    sweeps::Vector{SweepLog}
    converged::Bool
    ProjectorLog() = new(SweepLog[], false)
end

# Split log (mutable)
@kwdef mutable struct SplitLog <: ConvergenceLog
    sweeps::Vector{SweepLog} = SweepLog[]
    converged::Bool = false
    cost_history::Vector{Float64} = Float64[]
end
