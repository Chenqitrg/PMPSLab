# ------------------------------------------------------------------------------
# HalfInt — half-integer bond index
# ------------------------------------------------------------------------------

struct HalfInt
    n::Int  # represents n + 1/2, i.e. the bond between site n and site n+1
end
Base.convert(::Type{HalfInt}, n::Int) = HalfInt(n)
function HalfInt(x::Rational{Int})
    denominator(x) == 2 || throw(ArgumentError("HalfInt: expected a half-integer n + 1/2 (denominator 2), got $x"))
    # numerator is odd (lowest terms), so n = (num - 1) / 2 exactly;
    # div truncates toward zero, which would give 0 for -1//2.
    return HalfInt(div(numerator(x) - 1, 2))
end
Base.hash(h::HalfInt, h0::UInt) = hash(h.n, h0)
Base.:(==)(a::HalfInt, b::HalfInt) = a.n == b.n
Base.:+(h::HalfInt, n::Int) = HalfInt(h.n + n)
Base.:-(h::HalfInt, n::Int) = HalfInt(h.n - n)

# ------------------------------------------------------------------------------
# HalfPeriodicVector — N bond elements, indexed by HalfInt
# ------------------------------------------------------------------------------

struct HalfPeriodicVector{A} <: AbstractVector{A}
    elements::PeriodicVector{A}
end
Base.size(v::HalfPeriodicVector) = size(parent(v))
Base.parent(v::HalfPeriodicVector) = v.elements
Base.IndexStyle(::Type{<:HalfPeriodicVector}) = IndexLinear()
Base.getindex(v::HalfPeriodicVector, pos::HalfInt) = v.elements[pos.n]
Base.getindex(v::HalfPeriodicVector, i::Int) = v.elements[i]
Base.getindex(v::HalfPeriodicVector, x::Rational{Int}) = v[HalfInt(x)]

HalfPeriodicVector(v::AbstractVector) = HalfPeriodicVector(PeriodicVector(v))


# ------------------------------------------------------------------------------
# PeriodicMPS — periodic boundary MPS
# ------------------------------------------------------------------------------

struct PeriodicMPS{A <: GenericMPSTensor} <: MPSKit.AbstractMPS
    tensors::PeriodicVector{A}
end
function PeriodicMPS(tensors::AbstractVector{<:GenericMPSTensor})
    isempty(tensors) && throw(ArgumentError("a periodic MPS must contain at least one tensor"))
    mps = PeriodicMPS(PeriodicVector(tensors))
    assertconsistency(mps)
    return mps
end

"""
    assertconsistency(mps::PeriodicMPS)

Check that the virtual spaces of adjacent site tensors are compatible around
the period. Throws `TensorKit.SpaceMismatch` on failure.
"""
function assertconsistency(mps::PeriodicMPS)
    for i in eachindex(mps)
        L = left_virtualspace(mps[i])
        R = right_virtualspace(mps[i - 1])
        L == R || throw(TensorKit.SpaceMismatch(
            "left_virtualspace at site $i ($L) ≠ right_virtualspace at site $(i-1) ($R)"
        ))
    end
    return nothing
end

# -- container interface ---------------------------------------------------------

Base.size(mps::PeriodicMPS, args...) = size(mps.tensors, args...)
Base.length(mps::PeriodicMPS) = length(mps.tensors)
Base.firstindex(mps::PeriodicMPS) = firstindex(mps.tensors)
Base.lastindex(mps::PeriodicMPS) = lastindex(mps.tensors)
Base.eachindex(mps::PeriodicMPS) = eachindex(mps.tensors)
Base.eachindex(l::IndexStyle, mps::PeriodicMPS) = eachindex(l, mps.tensors)
eachsite(mps::PeriodicMPS) = PeriodicVector(eachindex(mps))
Base.checkbounds(::Type{Bool}, mps::PeriodicMPS, i::Integer) = true

# -- indexing & iteration --------------------------------------------------------

Base.getindex(mps::PeriodicMPS, i::Int) = mps.tensors[i]
Base.setindex!(mps::PeriodicMPS, v, i::Int) = (mps.tensors[i] = v)
Base.iterate(mps::PeriodicMPS, args...) = iterate(mps.tensors, args...)

# -- type traits -----------------------------------------------------------------

Base.eltype(::Type{<:PeriodicMPS{A}}) where {A} = A
Base.eltype(mps::PeriodicMPS) = eltype(typeof(mps))
Base.isfinite(::Type{<:PeriodicMPS}) = true
struct PeriodicChainStyle <: MPSKit.GeometryStyle end
MPSKit.GeometryStyle(::Type{<:PeriodicMPS}) = PeriodicChainStyle()
MPSKit.site_type(::Type{<:PeriodicMPS{A}}) where {A} = A
scalartype(mps::PeriodicMPS) = scalartype(first(mps))

# -- copy & transform ------------------------------------------------------------

Base.parent(mps::PeriodicMPS) = mps.tensors
Base.copy(mps::PeriodicMPS) = PeriodicMPS(copy(parent(mps)))
function Base.copyto!(dst::PeriodicMPS, src::PeriodicMPS)
    copyto!(parent(dst), parent(src))
    return dst
end
Base.repeat(mps::PeriodicMPS, i::Int) = PeriodicMPS(repeat(mps.tensors, i))
Base.circshift(mps::PeriodicMPS, n) = PeriodicMPS(circshift(mps.tensors, n))
Base.similar(mps::PeriodicMPS{A}) where {A} = PeriodicMPS(similar(mps.tensors))

Base.getindex(mps::PeriodicMPS, pos::HalfInt) = right_virtualspace(mps[pos.n])

virtualspace(mps::PeriodicMPS{<:GenericMPSTensor}) = left_virtualspace(first(mps))
virtualspaces(mps::PeriodicMPS{<:GenericMPSTensor}) =
    HalfPeriodicVector(map(right_virtualspace, mps))

physicalspace(mps::PeriodicMPS{<:GenericMPSTensor}) = physicalspace(first(mps))
physicalspaces(mps::PeriodicMPS{<:GenericMPSTensor}) =
    PeriodicVector(map(physicalspace, mps))

function _check_matching_periods(P, V)
    length(P) == length(V) || throw(DimensionMismatch(
        "physical and virtual spaces must have the same length: " *
        "got $(length(P)) and $(length(V))"
    ))
    return eachindex(P)
end

# ------------------------------------------------------------------------------
# Constructors — random and uniform (all-ones) periodic MPS
# (mirrors MPSKit's `FiniteMPS(f, elt, L, d, D)` / `FiniteMPS(ones, …)`.)
# ------------------------------------------------------------------------------

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

function random_mps(elt::Type{<:Number}, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [rand(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps(elt::Type{<:Number}, P::PeriodicVector{<:ElementarySpace}, V::HalfPeriodicVector{<:ElementarySpace})
    tensors = [rand(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
               for i in _check_matching_periods(P, V)]
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
    tensors = [ones(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
               for i in _check_matching_periods(P, V)]
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

function random_mps2(elt::Type{<:Number}, N::Int, P::ElementarySpace, V::ElementarySpace)
    tensors = [rand(elt, V ⊗ P ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function random_mps2(elt::Type{<:Number}, P::PeriodicVector{<:ProductSpace}, V::HalfPeriodicVector{<:ElementarySpace})
    tensors = [rand(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
               for i in _check_matching_periods(P, V)]
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
    tensors = [ones(elt, V[HalfInt(i - 1)] ⊗ P[i] ← V[HalfInt(i)])
               for i in _check_matching_periods(P, V)]
    return PeriodicMPS(tensors)
end

# ------------------------------------------------------------------------------
# Corner-double-line test states
# ------------------------------------------------------------------------------

"""
    random_cdl(Vdl, Vdr, Vlu, Vld, Vru, Vrd)
    random_cdl(elt, Vdl, Vdr, Vlu, Vld, Vru, Vrd)

Construct a local corner-double-line tensor from three independent random maps

```
Vdl ← Vld,    Vrd ← Vdr,    Vlu ← Vru.
```

The open legs are bent into the layout

```
Vlu ⊗ Vld' ⊗ Vdl ⊗ Vdr' ← Vru ⊗ Vrd'.
```
"""
random_cdl(
    Vdl::ElementarySpace, Vdr::ElementarySpace,
    Vlu::ElementarySpace, Vld::ElementarySpace,
    Vru::ElementarySpace, Vrd::ElementarySpace,
) = random_cdl(ComplexF64, Vdl, Vdr, Vlu, Vld, Vru, Vrd)

function random_cdl(
        elt::Type{<:Number},
        Vdl::ElementarySpace, Vdr::ElementarySpace,
        Vlu::ElementarySpace, Vld::ElementarySpace,
        Vru::ElementarySpace, Vrd::ElementarySpace
    )
    down = rand(elt, Vdl ← Vld)
    lower_right = rand(elt, Vrd ← Vdr)
    upper = rand(elt, Vlu ← Vru)
    @plansor tensor[-1 -2 -3 -4; -5 -6] :=
        upper[-1; -5] * down[-3; -2] * lower_right[-6; -4]
    return tensor
end

"""
    random_cdl_mps(elt, N, χ) -> PeriodicMPS
    random_cdl_mps(elt, N, V) -> PeriodicMPS

Construct an `N`-site periodic CDL MPS. Each local tensor is built with
`random_cdl`; the two virtual double lines are fused into a bond space of
dimension `dim(V)^2`. TensorKit unitary fusion maps are inserted on the two
sides of every bond as mutually inverse pairs, so they do not change the
contracted periodic state.
"""
random_cdl_mps(elt::Type{<:Number}, N::Int, χ::Int) =
    random_cdl_mps(elt, N, ℂ^χ)

function random_cdl_mps(
        elt::Type{<:Number}, N::Int, V::ElementarySpace
    )
    N > 0 || throw(ArgumentError("N must be positive"))

    virtual = fuse(V ⊗ V')
    fusion = unitary(elt, V ⊗ V', virtual)
    tensors = map(1:N) do _
        tensor = random_cdl(elt, V, V, V, V, V, V)
        @plansor mps_tensor[-1 -2 -3; -4] :=
            fusion'[-1; 1 2] * tensor[1 2 -2 -3; 3 4] * fusion[3 4; -4]
        return mps_tensor
    end
    return PeriodicMPS(tensors)
end

# ------------------------------------------------------------------------------
# lift — replicate unit cell (finite → larger finite, or ∞ → infinite)
# ------------------------------------------------------------------------------

"""
    lift(mps::PeriodicMPS, fold::Int) -> PeriodicMPS

Replicate the unit cell `fold` times by concatenation.
Corresponds to lifting a periodic cell to a finite cover.
"""
lift(mps::PeriodicMPS, fold::Int) = repeat(mps, fold)

"""
    lift(mps::PeriodicMPS, ::Val{Inf}) -> InfiniteMPS

Lift to an infinite MPS (MPSKit.InfiniteMPS) by repeating the unit cell.
"""
lift(mps::PeriodicMPS, ::Val{Inf}) = MPSKit.InfiniteMPS(collect(mps))
