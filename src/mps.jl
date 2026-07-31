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
