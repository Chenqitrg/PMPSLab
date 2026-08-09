# Random and uniform one-site and two-site MPS constructors.

function _check_matching_periods(P, V)
    length(P) == length(V) || throw(DimensionMismatch(
        "physical and virtual spaces must have the same length: " *
        "got $(length(P)) and $(length(V))"
    ))
    return eachindex(P)
end

"""
    random_mps(elt, N, d, D) -> PeriodicMPS

Random periodic MPS with `N` sites: physical space `ℂ^d`, virtual space
`ℂ^D`. Every site carries the same spaces, so the period closes.

    random_mps(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual), e.g. Z2Irrep.
"""
function random_mps(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    tensors = [randn(elt, ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps(
        elt::Type{<:Number},
        N::Int,
        P::ElementarySpace,
        V::ElementarySpace,
    )
    tensors = [rand(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps(
        elt::Type{<:Number},
        P::PeriodicVector{<:ElementarySpace},
        V::HalfPeriodicVector{<:ElementarySpace},
    )
    tensors = [
        rand(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
        for i in _check_matching_periods(P, V)
    ]
    return PeriodicMPS(tensors)
end

"""
    one_mps(elt, N, d, D) -> PeriodicMPS

Uniform periodic MPS whose tensors are all-ones matrices: physical space
`ℂ^d`, virtual space `ℂ^D`. Deterministic, in contrast to `random_mps`.

    one_mps(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual).
"""
function one_mps(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    tensors = [ones(elt, ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function one_mps(
        elt::Type{<:Number},
        N::Int,
        P::ElementarySpace,
        V::ElementarySpace,
    )
    tensors = [ones(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function one_mps(
        elt::Type{<:Number},
        P::PeriodicVector{<:ElementarySpace},
        V::HalfPeriodicVector{<:ElementarySpace},
    )
    tensors = [
        ones(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
        for i in _check_matching_periods(P, V)
    ]
    return PeriodicMPS(tensors)
end

"""
    random_mps2(elt, N, d, D) -> PeriodicMPS

Random two-site periodic MPS with `N` sites: each site carries two physical
legs `ℂ^d` (periodic order: site a, site b) and virtual space `ℂ^D`. The tensors
live in `ℂ^D ⊗ ℂ^d ⊗ ℂ^d ← ℂ^D`, as produced by `pair_contraction`.

    random_mps2(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual), e.g. Z2Irrep.
"""
function random_mps2(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    tensors = [randn(elt, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps2(
        elt::Type{<:Number},
        N::Int,
        P::ElementarySpace,
        V::ElementarySpace,
    )
    tensors = [rand(elt, V ⊗ P ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps2(
        elt::Type{<:Number},
        P::PeriodicVector{<:ProductSpace},
        V::HalfPeriodicVector{<:ElementarySpace},
    )
    tensors = [
        rand(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
        for i in _check_matching_periods(P, V)
    ]
    return PeriodicMPS(tensors)
end

"""
    one_mps2(elt, N, d, D) -> PeriodicMPS

Uniform two-site periodic MPS whose tensors are all-ones matrices (two
physical legs `ℂ^d` each, virtual space `ℂ^D`). Deterministic.

    one_mps2(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual).
"""
function one_mps2(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    tensors = [ones(elt, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function one_mps2(
        elt::Type{<:Number},
        N::Int,
        P::ElementarySpace,
        V::ElementarySpace,
    )
    tensors = [ones(elt, V ⊗ P ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function one_mps2(
        elt::Type{<:Number},
        P::PeriodicVector{<:ProductSpace},
        V::HalfPeriodicVector{<:ElementarySpace},
    )
    tensors = [
        ones(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
        for i in _check_matching_periods(P, V)
    ]
    return PeriodicMPS(tensors)
end
