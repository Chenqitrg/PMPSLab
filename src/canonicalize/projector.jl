# ------------------------------------------------------------------------------
# Projector-based periodic gauge fixing
# ------------------------------------------------------------------------------

abstract type AbstractLoopCanonical <: Algorithm end

@kwdef struct ProjectorCanonical{
        T<:TruncationStrategy, O
    } <: AbstractLoopCanonical
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trscheme::T = TruncationStrategy()
    alg_orth::O = MPSKit.Defaults.alg_orth()
end

# Tensor-layout helpers supporting any number of physical output legs.
_projector_tail(A::AbstractTensorMap{E,S,NO,NI}) where {E,S,NO,NI} =
    repartition(A, 1, NO + NI - 1)
_projector_front(A::AbstractTensorMap{E,S,NO,NI}) where {E,S,NO,NI} =
    repartition(A, NO + NI - 1, 1)
_projector_mul_front(
        C::AbstractTensorMap{E,S,1,1}, A::AbstractTensorMap{E,S,N,1}
    ) where {E,S,N} =
    _projector_front(C * _projector_tail(A))

_left_algorithm(alg::ProjectorCanonical) = MPSKit.LeftCanonical(;
    alg.tol, alg.maxiter, verbosity = MPSKit.VERBOSE_NONE,
    alg.alg_orth, eig_miniter = typemax(Int)
)
_right_algorithm(alg::ProjectorCanonical) = MPSKit.RightCanonical(;
    alg.tol, alg.maxiter, verbosity = MPSKit.VERBOSE_NONE,
    alg.alg_orth, eig_miniter = typemax(Int)
)

function _gauge_seed(
    mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}}
    ) where {E,S,N}
    return id(E, left_virtualspace(mps[1]))
end

function _mpskit_gauges(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        alg::ProjectorCanonical
    ) where {E,S,N}
    A = mps.tensors
    Aorth = similar.(A)
    seed = _gauge_seed(mps)

    L = similar(A, typeof(seed))
    MPSKit.uniform_leftorth!(
        (Aorth, L), A, seed, _left_algorithm(alg)
    )
    foreach(normalize!, L)

    R = similar(A, typeof(seed))
    MPSKit.uniform_rightorth!(
        (Aorth, R), A, seed, _right_algorithm(alg)
    )
    foreach(normalize!, R)

    return HalfPeriodicVector(L), HalfPeriodicVector(R)
end

"""
    P_decomp(R, L, trunc) -> (PR, PL)

Factor the combined periodic gauge `L * R`. If that gauge is full rank and no
truncation is requested, the adjacent projectors satisfy `PR * PL ≈ I`. For
a rank-deficient or truncated gauge, their product is the projector onto the
supported or retained virtual subspace; that nontrivial projection is the
intended virtual-bond compression.
"""
function P_decomp(
        R::AbstractTensorMap{E,A,1,1}, L::AbstractTensorMap{E,A,1,1},
        trunc::TruncationStrategy
    ) where {E,A}
    U, S, Vh, _ = svd_trunc(L * R; trunc)
    inverse_sqrt = pseudopow(S, -0.5)
    PR = R * Vh' * inverse_sqrt
    PL = inverse_sqrt * U' * L
    return PR, PL
end

"""
    find_projectors(mps, alg) -> (PR, PL)

Run MPSKit's left and right periodic gauge solvers directly on the read-only
`mps.tensors` container. A single internal orthogonal-tensor buffer is reused
for the left and right solves. The normalized gauges and resulting projectors
use `HalfPeriodicVector` bond indexing.
"""
function find_projectors(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        alg::ProjectorCanonical
    ) where {E,S,N}
    L, R = _mpskit_gauges(mps, alg)

    projectors = map(eachindex(mps)) do bond
        pos = HalfInt(bond)
        P_decomp(R[pos], L[pos], alg.trscheme)
    end
    return HalfPeriodicVector(first.(projectors)), HalfPeriodicVector(last.(projectors))
end
