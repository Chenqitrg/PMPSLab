using TensorKit: left_orth, right_orth
using MPSKit: Algorithm, GenericMPSTensor, TruncationStrategy,
    left_virtualspace, right_virtualspace, physicalspace

# ------------------------------------------------------------------------------
# Half-integer index for bond/link positions
# ------------------------------------------------------------------------------

struct HalfInt
    n::Int  # represents n + 1/2, i.e. the bond between site n and site n+1
end
Base.convert(::Type{HalfInt}, n::Int) = HalfInt(n)
Base.:+(h::HalfInt, n::Int) = HalfInt(h.n + n)
Base.:-(h::HalfInt, n::Int) = HalfInt(h.n - n)

# ------------------------------------------------------------------------------
# Stub types (will move to types.jl later)
# ------------------------------------------------------------------------------

struct PeriodicMPS{A <: GenericMPSTensor}
    tensors::PeriodicVector{A}
end
Base.length(mps::PeriodicMPS) = length(mps.tensors)

@kwdef struct ConvergenceLog
    converged::Bool
    iterations::Int
    sweep_times::Vector{Float64}
    error_history::Vector{Float64}
    cost_history::Vector{Float64}
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

function canonicalize!(mps::PeriodicMPS, alg::AbstractLoopCanonical)::ConvergenceLog
    # 1. find_projectors at each site
    # 2. absorb projectors into tensors (mutate mps.tensors in-place)
    # 3. return ConvergenceLog
end
