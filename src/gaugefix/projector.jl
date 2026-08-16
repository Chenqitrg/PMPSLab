# ------------------------------------------------------------------------------
# Projector-based periodic gauge fixing
# ------------------------------------------------------------------------------

abstract type AbstractGaugeFix <: AbstractAlgorithmConfig end

"""
    ProjectorGaugeFix(;
        tol=1e-12,
        maxiter=100,
        trunc=TruncationStrategy(),
        orthogonalization_algorithm=MPSKit.Defaults.alg_orth(),
    )

Configure periodic left/right canonicalization and the truncation used to
construct the resulting bond projectors.
"""
@kwdef struct ProjectorGaugeFix{
        T<:TruncationStrategy, O
    } <: AbstractGaugeFix
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trunc::T = TruncationStrategy()
    orthogonalization_algorithm::O = MPSKit.Defaults.alg_orth()
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

_left_algorithm(algorithm::ProjectorGaugeFix) = MPSKit.LeftCanonical(;
    algorithm.tol, algorithm.maxiter, verbosity = MPSKit.VERBOSE_NONE,
    alg_orth=algorithm.orthogonalization_algorithm,
    eig_miniter=typemax(Int),
)
_right_algorithm(algorithm::ProjectorGaugeFix) = MPSKit.RightCanonical(;
    algorithm.tol, algorithm.maxiter, verbosity = MPSKit.VERBOSE_NONE,
    alg_orth=algorithm.orthogonalization_algorithm,
    eig_miniter=typemax(Int),
)

function _gauge_seed(
    mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}}
    ) where {E,S,N}
    return id(E, left_virtualspace(first(mps)))
end

function _mpskit_gauges(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        algorithm::ProjectorGaugeFix,
    ) where {E,S,N}
    A = parent(mps)
    Aorth = similar.(A)
    seed = _gauge_seed(mps)

    L = similar(A, typeof(seed))
    MPSKit.uniform_leftorth!(
        (Aorth, L), A, seed, _left_algorithm(algorithm)
    )
    foreach(normalize!, L)

    R = similar(A, typeof(seed))
    MPSKit.uniform_rightorth!(
        (Aorth, R), A, seed, _right_algorithm(algorithm)
    )
    foreach(normalize!, R)

    return HalfPeriodicVector(L), HalfPeriodicVector(R)
end

function _require_structurally_fullrank(mps::PeriodicMPS)
    invalid_site = findfirst(eachindex(mps)) do i
        !MPSKit.isfullrank(mps[i])
    end
    isnothing(invalid_site) && return nothing

    throw(ArgumentError(
        "find_projectors requires structurally full-rank local tensor spaces; " *
        "site $invalid_site violates this condition. Use gaugefix! to perform " *
        "the exact in-place structural cleanup before projector construction."
    ))
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
        R::AbstractTensorMap{E,Sp,1,1}, L::AbstractTensorMap{E,Sp,1,1},
        trunc::TruncationStrategy
    ) where {E,Sp}
    U, S, Vh, _ = svd_trunc(L * R; trunc)
    inverse_sqrt = pseudopow(S, -0.5)
    PR = R * Vh' * inverse_sqrt
    PL = inverse_sqrt * U' * L
    return PR, PL
end

"""
    find_projectors(mps, algorithm) -> (PR, PL)

Run MPSKit's left and right periodic gauge solvers directly on the read-only
`mps.tensors` container. A single internal orthogonal-tensor buffer is reused
for the left and right solves. The normalized gauges and resulting projectors
use `HalfPeriodicVector` bond indexing. Because this function does not mutate
its input, `mps` must already have structurally full-rank local tensor spaces;
use `gaugefix!` when exact structural cleanup may be required.
"""
function find_projectors(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,N,1}},
        algorithm::ProjectorGaugeFix,
    ) where {E,S,N}
    _require_structurally_fullrank(mps)
    L, R = _mpskit_gauges(mps, algorithm)

    projectors = map(eachindex(mps)) do bond
        pos = half(2bond + 1)
        P_decomp(R[pos], L[pos], algorithm.trunc)
    end
    return HalfPeriodicVector(first.(projectors)), HalfPeriodicVector(last.(projectors))
end
