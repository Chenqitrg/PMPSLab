# Random and uniform periodic MPS constructors.

function _check_matching_periods(P, V)
    length(P) == length(V) || throw(DimensionMismatch(
        "physical and virtual spaces must have the same length: " *
        "got $(length(P)) and $(length(V))"
    ))
    isempty(P) && throw(ArgumentError("a periodic MPS must contain at least one tensor"))
    Base.require_one_based_indexing(P, V)
    return eachindex(P)
end

"""
    rand([rng], PeriodicMPS, elt, N, d, D)
    rand([rng], PeriodicMPS, elt, N, physical_dimensions, D)
    rand([rng], PeriodicMPS, elt, N, P, V)
    rand([rng], PeriodicMPS, elt, P, V)

Generate a periodic MPS. An integer `d` creates one physical leg, while a tuple
such as `(d₁, d₂)` creates multiple physical legs. `P` may likewise be an
elementary or product space. The final form accepts periodic physical spaces
`P` and virtual spaces `V`. Both may be ordinary vectors or `PeriodicVector`s.
As in MPSKit, `V[i]` is the right virtual space of site `i`; the left space
is `V[i-1]`, wrapping periodically.
"""
function Random.rand(
        rng::Random.AbstractRNG,
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        N::Integer,
        d::Integer,
        D::Integer,
    )
    return Random.rand(rng, PeriodicMPS, elt, N, (d,), D)
end

function Random.rand(
        rng::Random.AbstractRNG,
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        N::Integer,
        physical_dimensions::Tuple{Vararg{Integer}},
        D::Integer,
    )
    isempty(physical_dimensions) && throw(ArgumentError(
        "at least one physical dimension is required"
    ))
    physical_space = reduce(⊗, ℂ^d for d in physical_dimensions)
    return Random.rand(rng, PeriodicMPS, elt, N, physical_space, ℂ^D)
end

function Random.rand(
        rng::Random.AbstractRNG,
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        N::Integer,
        P::Union{ElementarySpace,ProductSpace},
        V::ElementarySpace,
    )
    tensors = [Random.rand(rng, elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function Random.rand(
        rng::Random.AbstractRNG,
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        P::AbstractVector{<:Union{ElementarySpace,ProductSpace}},
        V::AbstractVector{<:ElementarySpace},
    )
    tensors = [
        Random.rand(
            rng,
            elt,
            V[mod1(i - 1, length(V))] ⊗ P[i] ← V[i],
        )
        for i in _check_matching_periods(P, V)
    ]
    return PeriodicMPS(tensors)
end

function Random.rand(
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        N::Integer,
        d::Integer,
        D::Integer,
    )
    return Random.rand(Random.default_rng(), PeriodicMPS, elt, N, d, D)
end

function Random.rand(
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        N::Integer,
        physical_dimensions::Tuple{Vararg{Integer}},
        D::Integer,
    )
    return Random.rand(
        Random.default_rng(),
        PeriodicMPS,
        elt,
        N,
        physical_dimensions,
        D,
    )
end

function Random.rand(
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        N::Integer,
        P::Union{ElementarySpace,ProductSpace},
        V::ElementarySpace,
    )
    return Random.rand(Random.default_rng(), PeriodicMPS, elt, N, P, V)
end

function Random.rand(
        ::Type{PeriodicMPS},
        elt::Type{<:Number},
        P::AbstractVector{<:Union{ElementarySpace,ProductSpace}},
        V::AbstractVector{<:ElementarySpace},
    )
    return Random.rand(Random.default_rng(), PeriodicMPS, elt, P, V)
end

"""
    one_mps(elt, N, d, D) -> PeriodicMPS
    one_mps(elt, N, physical_dimensions, D) -> PeriodicMPS

Uniform periodic MPS whose tensors are all-ones matrices: physical space
`ℂ^d`, virtual space `ℂ^D`. A tuple such as `(d₁, d₂)` creates multiple
physical legs.

    one_mps(elt, N, P, V) -> PeriodicMPS

Same with an elementary or product physical space `P` and virtual space `V`.
Vectors of spaces are also accepted; `V[i]` specifies the right virtual space
of site `i`, as for `rand`.
"""
function one_mps(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    return one_mps(elt, N, (d,), D)
end

function one_mps(
        elt::Type{<:Number},
        N::Int,
        physical_dimensions::Tuple{Vararg{Integer}},
        D::Int,
    )
    isempty(physical_dimensions) && throw(ArgumentError(
        "at least one physical dimension is required"
    ))
    physical_space = reduce(⊗, ℂ^d for d in physical_dimensions)
    return one_mps(elt, N, physical_space, ℂ^D)
end

function one_mps(
        elt::Type{<:Number},
        N::Int,
        P::Union{ElementarySpace,ProductSpace},
        V::ElementarySpace,
    )
    tensors = [ones(elt, V ⊗ P ← V) for _ in 1:N]
    return PeriodicMPS(tensors)
end

function one_mps(
        elt::Type{<:Number},
        P::AbstractVector{<:Union{ElementarySpace,ProductSpace}},
        V::AbstractVector{<:ElementarySpace},
    )
    tensors = [
        ones(
            elt,
            V[mod1(i - 1, length(V))] ⊗ P[i] ← V[i],
        )
        for i in _check_matching_periods(P, V)
    ]
    return PeriodicMPS(tensors)
end

# Compatibility wrappers for the former two-physical-leg-specific constructor.
function one_mps2(elt::Type{<:Number}, N::Int, d::Int, D::Int)
    Base.depwarn("use one_mps(elt, N, (d, d), D)", :one_mps2)
    return one_mps(elt, N, (d, d), D)
end

function one_mps2(
        elt::Type{<:Number},
        N::Int,
        P::ElementarySpace,
        V::ElementarySpace,
    )
    Base.depwarn("use one_mps(elt, N, P ⊗ P, V)", :one_mps2)
    return one_mps(elt, N, P ⊗ P, V)
end


function one_mps2(
        elt::Type{<:Number},
        P::AbstractVector{<:ProductSpace},
        V::AbstractVector{<:ElementarySpace},
    )
    Base.depwarn("use one_mps(elt, P, V)", :one_mps2)
    return one_mps(elt, P, V)
end
