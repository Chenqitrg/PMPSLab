# ------------------------------------------------------------------------------
# HalfInt — half-integer bond index
# ------------------------------------------------------------------------------

struct HalfInt
    n::Int  # represents n + 1/2, i.e. the bond between site n and site n+1
end
Base.convert(::Type{HalfInt}, n::Int) = HalfInt(n)
function HalfInt(x::Rational{Int})
    den(x) in (1, 2) || throw(ArgumentError("HalfInt: denominator must be 1 or 2, got $x"))
    return HalfInt(div(num(x), den(x)))
end
Base.hash(h::HalfInt, h0::UInt) = hash(h.n, h0)
Base.:(==)(a::HalfInt, b::HalfInt) = a.n == b.n
Base.:+(h::HalfInt, n::Int) = HalfInt(h.n + n)
Base.:-(h::HalfInt, n::Int) = HalfInt(h.n - n)

# ------------------------------------------------------------------------------
# HalfPeriodicVector — N bond elements, indexed by HalfInt
# ------------------------------------------------------------------------------

struct HalfPeriodicVector{A}
    elements::PeriodicVector{A}
end
Base.length(v::HalfPeriodicVector) = length(v.elements)
Base.getindex(v::HalfPeriodicVector, pos::HalfInt) = v.elements[pos.n]
Base.iterate(v::HalfPeriodicVector, i::Int = 1) =
    i > length(v) ? nothing : (v.elements[i], i + 1)
Base.eltype(::Type{HalfPeriodicVector{A}}) where {A} = A

# ------------------------------------------------------------------------------
# PeriodicMPS — periodic boundary MPS
# ------------------------------------------------------------------------------

struct PeriodicMPS{A <: GenericMPSTensor}
    tensors::PeriodicVector{A}
end
PeriodicMPS(tensors::AbstractVector{<:GenericMPSTensor}) =
    PeriodicMPS(PeriodicVector(tensors))
Base.length(mps::PeriodicMPS) = length(mps.tensors)
Base.getindex(mps::PeriodicMPS, i::Int) = mps.tensors[i]
Base.setindex!(mps::PeriodicMPS, v, i::Int) = (mps.tensors[i] = v)
Base.iterate(mps::PeriodicMPS, args...) = iterate(mps.tensors, args...)
Base.eltype(::Type{<:PeriodicMPS{A}}) where {A} = A
scalartype(mps::PeriodicMPS) = scalartype(mps[1])
Base.getindex(mps::PeriodicMPS, pos::HalfInt) = left_virtualspace(mps[pos.n])

virtualspace(mps::PeriodicMPS{<:GenericMPSTensor}) = left_virtualspace(mps[1])
function virtualspaces(mps::PeriodicMPS{<:GenericMPSTensor})
    spaces = PeriodicVector([left_virtualspace(mps[i]) for i in 1:length(mps)])
    return HalfPeriodicVector(spaces)
end

physicalspace(mps::PeriodicMPS{<:GenericMPSTensor}) = physicalspace(mps[1])
physicalspaces(mps::PeriodicMPS{<:GenericMPSTensor}) = PeriodicVector([physicalspace(t) for t in mps])

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
function random_mps(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    tensors = [randn(elt, ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps(elt::Type{<:Number}, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [rand(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps(elt::Type{<:Number}, P::PeriodicVector{<:ElementarySpace}, V::HalfPeriodicVector{<:ElementarySpace})
    N = length(P)
    @assert length(V) == N "physical and virtual spaces must have the same length"
    tensors = [rand(elt, V[i - 1//2] ⊗ P[i] ← V[i + 1//2]) for i in 1:N]
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

function one_mps(elt::Type{<:Number}, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [ones(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function one_mps(elt::Type{<:Number}, P::PeriodicVector{<:ElementarySpace}, V::HalfPeriodicVector{<:ElementarySpace})
    N = length(P)
    @assert length(V) == N "physical and virtual spaces must have the same length"
    tensors = [ones(elt, V[i - 1//2] ⊗ P[i] ← V[i + 1//2]) for i in 1:N]
    return PeriodicMPS(tensors)
end

"""
    random_mps2(elt, N, d, D) -> PeriodicMPS

Random two-site periodic MPS with `N` sites: each site carries two physical
legs `ℂ^d` (loop order: site a, site b) and virtual space `ℂ^D`. The tensors
live in `ℂ^D ⊗ ℂ^d ⊗ ℂ^d ← ℂ^D`, as produced by `pair_contraction`.

    random_mps2(elt, N, P, V) -> PeriodicMPS

Same with graded spaces `P` (physical) and `V` (virtual), e.g. Z2Irrep.
"""
function random_mps2(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    tensors = [randn(elt, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps2(elt::Type{<:Number}, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [rand(elt, V ⊗ P ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps2(elt::Type{<:Number}, P::PeriodicVector{<:ProductSpace}, V::HalfPeriodicVector{<:ElementarySpace})
    N = length(P)
    @assert length(V) == N "physical and virtual spaces must have the same length"
    tensors = [rand(elt, V[i - 1//2] ⊗ P[i] ← V[i + 1//2]) for i in 1:N]
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

function one_mps2(elt::Type{<:Number}, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [ones(elt, V ⊗ P ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function one_mps2(elt::Type{<:Number}, P::PeriodicVector{<:ProductSpace}, V::HalfPeriodicVector{<:ElementarySpace})
    N = length(P)
    @assert length(V) == N "physical and virtual spaces must have the same length"
    tensors = [ones(elt, V[i - 1//2] ⊗ P[i] ← V[i + 1//2]) for i in 1:N]
    return PeriodicMPS(tensors)
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
    return PeriodicMPS(tensors)
end

"""
    lift(mps::PeriodicMPS, ::Val{Inf}) -> InfiniteMPS

Lift to an infinite MPS (MPSKit.InfiniteMPS) by repeating the unit cell.
"""
function lift(mps::PeriodicMPS, ::Val{Inf})
    return MPSKit.InfiniteMPS(collect(mps))
end
