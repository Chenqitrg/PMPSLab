using MPSKit: Algorithm, GenericMPSTensor, TruncationStrategy, PeriodicVector

# ------------------------------------------------------------------------------
# Half-integer index for bond/link positions
# ------------------------------------------------------------------------------

struct HalfInt
    n::Int  # represents n + 1/2, i.e. the bond between site n and site n+1
end
Base.convert(::Type{HalfInt}, n::Int) = HalfInt(n)
Base.hash(h::HalfInt, h0::UInt) = hash(h.n, h0)
Base.:(==)(a::HalfInt, b::HalfInt) = a.n == b.n
Base.:+(h::HalfInt, n::Int) = HalfInt(h.n + n)
Base.:-(h::HalfInt, n::Int) = HalfInt(h.n - n)

# ------------------------------------------------------------------------------
# PeriodicMPS — periodic boundary MPS
# ------------------------------------------------------------------------------

struct PeriodicMPS{A <: GenericMPSTensor}
    tensors::PeriodicVector{A}
end
Base.length(mps::PeriodicMPS) = length(mps.tensors)
Base.getindex(mps::PeriodicMPS, i::Int) = mps.tensors[i]
Base.setindex!(mps::PeriodicMPS, v, i::Int) = (mps.tensors[i] = v)
Base.iterate(mps::PeriodicMPS, args...) = iterate(mps.tensors, args...)



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

# ------------------------------------------------------------------------------
# Algorithm hierarchy
# ------------------------------------------------------------------------------

abstract type AbstractLoopCanonical <: Algorithm end

@kwdef struct ProjectorCanonical <: AbstractLoopCanonical
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trscheme::TruncationStrategy = TruncationStrategy()
end

@kwdef struct NuclearNormCanonical <: AbstractLoopCanonical
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trscheme::TruncationStrategy = TruncationStrategy()
    ρ::Float64 = 0.8
    ξ_init::Float64 = 1e-5
    ξ_min::Float64 = 1e-7
end

# ------------------------------------------------------------------------------
# 2-to-1 site approximation
# ------------------------------------------------------------------------------

abstract type AbstractLoopApproximation <: Algorithm end

@kwdef mutable struct ApproximationLog <: ConvergenceLog
    sweeps::Vector{SweepLog} = SweepLog[]
    converged::Bool = false
    cost_history::Vector{Float64} = Float64[]
end
