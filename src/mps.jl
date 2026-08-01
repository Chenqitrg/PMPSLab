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
Base.eltype(::Type{<:PeriodicMPS{A}}) where {A} = A

# ------------------------------------------------------------------------------
# Constructors — random and uniform (all-ones) periodic MPS
# (mirrors MPSKit's `FiniteMPS(f, elt, L, d, D)` / `FiniteMPS(ones, …)`.)
# ------------------------------------------------------------------------------

"""
    random_mps(elt, N, d, D) -> PeriodicMPS

Random periodic MPS with `N` sites: physical space `ℂ^d`, virtual space
`ℂ^D`. Every site carries the same spaces, so the loop closes.

    random_mps(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual), e.g. Z2Irrep.
"""
function random_mps(elt, N::Int, d::Int, D::Int)
    tensors = [TensorMap(randn(elt, D * d, D), ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

function random_mps(elt, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [rand(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

"""
    one_mps(elt, N, d, D) -> PeriodicMPS

Uniform periodic MPS whose tensors are all-ones matrices: physical space
`ℂ^d`, virtual space `ℂ^D`. Deterministic, in contrast to `random_mps`.

    one_mps(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual).
"""
function one_mps(elt, N::Int, d::Int, D::Int)
    tensors = [TensorMap(ones(elt, D * d, D), ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

function one_mps(elt, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [ones(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

"""
    random_mps2(elt, N, d, D) -> PeriodicMPS

Random two-site periodic MPS with `N` sites: each site carries two physical
legs `ℂ^d` (loop order: site a, site b) and virtual space `ℂ^D`. The tensors
live in `ℂ^D ⊗ ℂ^d ⊗ ℂ^d ← ℂ^D`, as produced by `pair_contraction`.

    random_mps2(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual), e.g. Z2Irrep.
"""
function random_mps2(elt, N::Int, d::Int, D::Int)
    tensors = [TensorMap(randn(elt, D * d * d, D), ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

function random_mps2(elt, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [rand(elt, V ⊗ P ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

"""
    one_mps2(elt, N, d, D) -> PeriodicMPS

Uniform two-site periodic MPS whose tensors are all-ones matrices (two
physical legs `ℂ^d` each, virtual space `ℂ^D`). Deterministic.

    one_mps2(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual).
"""
function one_mps2(elt, N::Int, d::Int, D::Int)
    tensors = [TensorMap(ones(elt, D * d * d, D), ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

function one_mps2(elt, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [ones(elt, V ⊗ P ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(PeriodicVector(tensors))
end

# ------------------------------------------------------------------------------
# lift — replicate unit cell (finite → larger finite, or ∞ → infinite)
# ------------------------------------------------------------------------------

"""
    lift(mps::PeriodicMPS, fold::Int) -> PeriodicMPS

Replicate the unit cell `fold` times by concatenation.
Corresponds to lifting a loop S¹ → cover.
"""
function lift(mps::PeriodicMPS, fold::Int)
    tensors = repeat(collect(mps), fold)
    return PeriodicMPS(PeriodicVector(tensors))
end

"""
    lift(mps::PeriodicMPS, ::Val{Inf}) -> InfiniteMPS

Lift to an infinite MPS (MPSKit.InfiniteMPS) by repeating the unit cell.
"""
function lift(mps::PeriodicMPS, ::Val{Inf})
    return MPSKit.InfiniteMPS(collect(mps))
end
