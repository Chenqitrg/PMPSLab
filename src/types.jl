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
# Convergence log
# ------------------------------------------------------------------------------

# Each algorithm has its own log type — not a single generic struct.
# This allows future front-end progress visualization per phase.
abstract type ConvergenceLog end

@kwdef mutable struct ProjectorLog <: ConvergenceLog
    converged::Bool = false
    iterations::Int = 0
    sweep_times::Vector{Float64} = Float64[]
    error_history::Dict{HalfInt, Vector{Float64}} = Dict{HalfInt, Vector{Float64}}()
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
    converged::Bool = false
    iterations::Int = 0
    sweep_times::Vector{Float64} = Float64[]
    cost_history::Vector{Float64} = Float64[]
end
