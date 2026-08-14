# Random and uniform periodic MPS constructors.

function _check_matching_periods(P, V)
    length(P) == length(V) || throw(DimensionMismatch(
        "physical and virtual spaces must have the same length: " *
        "got $(length(P)) and $(length(V))"
    ))
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
`P` and half-periodic virtual spaces `V`.
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
        P::PeriodicVector{<:Union{ElementarySpace,ProductSpace}},
        V::HalfPeriodicVector{<:ElementarySpace},
    )
    tensors = [
        Random.rand(
            rng,
            elt,
            V[half(2i - 1)] ⊗ P[i] ← V[half(2i + 1)],
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
        P::PeriodicVector{<:Union{ElementarySpace,ProductSpace}},
        V::HalfPeriodicVector{<:ElementarySpace},
    )
    return Random.rand(Random.default_rng(), PeriodicMPS, elt, P, V)
end

"""
    one_mps(elt, N, d, D) -> PeriodicMPS

Uniform periodic MPS whose tensors are all-ones matrices: physical space
`ℂ^d`, virtual space `ℂ^D`.

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
        ones(
            elt,
            V[half(2i - 1)] ⊗ P[i] ← V[half(2i + 1)],
        )
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
        ones(
            elt,
            V[half(2i - 1)] ⊗ P[i] ← V[half(2i + 1)],
        )
        for i in _check_matching_periods(P, V)
    ]
    return PeriodicMPS(tensors)
end
