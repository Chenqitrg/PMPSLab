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
Base.getindex(mps::PeriodicMPS, i::Int) = mps.tensors[i]
Base.setindex!(mps::PeriodicMPS, v, i::Int) = (mps.tensors[i] = v)

using MPSKit: MPSTensor
const TwoSiteMPS{S} = PeriodicMPS{GenericMPSTensor{S, 3}}              # GenericMPSTensor{S, 2}
const OneSiteMPS{S}  = PeriodicMPS{GenericMPSTensor{S, 2}}

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

# Absorb projectors into 2-site MPS tensor: T_i = PL_i * T_i * PR_i
function _absorb_projectors!(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, PL_vec, PR_vec) where {E,S}
    N = length(mps)
    for i in 1:N
        @plansor mps[i][-1 -2 -3; -4] := PL_vec[i][-1; 1] * mps[i][1 -2 -3; 4] * PR_vec[i][4; -4]
    end
end

# Absorb projectors into 1-site MPS tensor
function _absorb_projectors!(mps::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}}, PL_vec, PR_vec) where {E,S}
    N = length(mps)
    for i in 1:N
        @plansor mps[i][-1 -2; -3] := PL_vec[i][-1; 1] * mps[i][1 -2; 2] * PR_vec[i][2; -3]
    end
end

function canonicalize!(mps::PeriodicMPS, alg::ProjectorCanonical)
    t_start = time()
    PR_vec, PL_vec = find_projectors(mps, alg)
    _absorb_projectors!(mps, PL_vec, PR_vec)
    elapsed = time() - t_start
    return ConvergenceLog(; converged=true, iterations=0,
        sweep_times=[elapsed], error_history=Float64[], cost_history=Float64[])
end
