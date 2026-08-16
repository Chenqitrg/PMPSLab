# Collection of transfer matrices over one period.
struct PeriodicTransferMatrix{T<:AbstractTransferMatrix} <:
       AbstractPeriodicProduct{T}
    matrices::PeriodicVector{T}
end

Base.length(v::PeriodicTransferMatrix) = length(v.matrices)
Base.parent(v::PeriodicTransferMatrix) = v.matrices
Base.eachindex(v::PeriodicTransferMatrix) = eachindex(parent(v))
Base.iterate(v::PeriodicTransferMatrix, state::S...) where {S} =
    iterate(parent(v), state...)
Base.getindex(v::PeriodicTransferMatrix, i::Int) = v.matrices[i]
function Base.setindex!(
        v::PeriodicTransferMatrix{T}, transfer::T, i::Int
    ) where {T<:AbstractTransferMatrix}
    return v.matrices[i] = transfer
end

function _transfer_period_length(
        bra::PeriodicMPS{<:AbstractTensorMap{E,S,M,1}},
        ket::PeriodicMPS{<:AbstractTensorMap{E,S,M,1}},
    ) where {E,S,M}
    length(bra) == length(ket) || throw(DimensionMismatch(
        "bra has $(length(bra)) sites but ket has $(length(ket)) sites"
    ))
    return length(bra)
end

function _transfer_period_length(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
    ) where {E,S}
    length(approximation) == 2 * length(target) || throw(DimensionMismatch(
        "the target has $(length(target)) two-physical-leg tensors but the " *
        "approximation has $(length(approximation)) single-physical-leg " *
        "tensors (expected $(2 * length(target)))"
    ))
    return length(target)
end

_transfer_period_length(
        approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
    ) where {E,S} = _transfer_period_length(target, approximation)

"""
    transfer_at(mps::PeriodicMPS, site::Int)
    transfer_at(mps::PeriodicMPS, site::Int, trunc)
    transfer_at(bra::PeriodicMPS, ket::PeriodicMPS, site::Int)
    transfer_at(bra::PeriodicMPS, ket::PeriodicMPS, site::Int, trunc)

Construct the transfer matrix at `site`. A mixed target/approximation pair is
contracted over the two corresponding single-physical-leg approximation
tensors. Supplying `trunc` returns a [`TruncatedTransferMatrix`](@ref);
otherwise the result is dense.
"""
transfer_at(mps::PeriodicMPS, site::Int) = transfer_at(mps, mps, site)

transfer_at(bra::PeriodicMPS, ket::PeriodicMPS, site::Int) =
    DenseTransferMatrix(bra[site], ket[site])

transfer_at(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        site::Int,
    ) where {E,S} = DenseTransferMatrix(
        target[site], approximation[2site - 1], approximation[2site]
    )

transfer_at(
        approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        site::Int,
    ) where {E,S} = DenseTransferMatrix(
        approximation[2site - 1], approximation[2site], target[site]
    )

transfer_at(
        mps::PeriodicMPS,
        site::Int,
        trunc::TruncationStrategy,
    ) = transfer_at(mps, mps, site, trunc)

transfer_at(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        site::Int,
        trunc::TruncationStrategy,
    ) = TruncatedTransferMatrix(bra[site], ket[site], trunc)

transfer_at(
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        site::Int,
        trunc::TruncationStrategy,
    ) where {E,S} = TruncatedTransferMatrix(
        target[site], approximation[2site - 1], approximation[2site], trunc
    )

transfer_at(
        approximation::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        target::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        site::Int,
        trunc::TruncationStrategy,
    ) where {E,S} = TruncatedTransferMatrix(
        approximation[2site - 1], approximation[2site], target[site], trunc
    )

function PeriodicTransferMatrix(bra::PeriodicMPS, ket::PeriodicMPS)
    N = _transfer_period_length(bra, ket)
    matrices = [transfer_at(bra, ket, site) for site in 1:N]
    return PeriodicTransferMatrix(PeriodicVector(matrices))
end

PeriodicTransferMatrix(mps::PeriodicMPS) = PeriodicTransferMatrix(mps, mps)

function PeriodicTransferMatrix(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    matrices = [
        transfer_at(bra, ket, site, trunc) for site in 1:N
    ]
    return PeriodicTransferMatrix(PeriodicVector(matrices))
end

PeriodicTransferMatrix(mps::PeriodicMPS, trunc::TruncationStrategy) =
    PeriodicTransferMatrix(mps, mps, trunc)

"""
    build_cache(bra::PeriodicMPS, ket::PeriodicMPS)
    build_cache(bra::PeriodicMPS, ket::PeriodicMPS, trunc)

Build the cache directly from generated transfer elements. No
`PeriodicTransferMatrix` or separate period-sized element collection is created.
"""
function build_cache(bra::PeriodicMPS, ket::PeriodicMPS)
    N = _transfer_period_length(bra, ket)
    product_element_at(site::Int) = transfer_at(bra, ket, site)
    return build_cache(N, product_element_at)
end

build_cache(mps::PeriodicMPS) = build_cache(mps, mps)

function build_cache(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    product_element_at(site::Int) = transfer_at(bra, ket, site, trunc)
    return build_cache(N, product_element_at)
end

build_cache(mps::PeriodicMPS, trunc::TruncationStrategy) =
    build_cache(mps, mps, trunc)

"""
    rebuild_left!(cache, bra::PeriodicMPS, ket::PeriodicMPS)
    rebuild_right!(cache, bra::PeriodicMPS, ket::PeriodicMPS)
    rebuild_left!(cache, bra::PeriodicMPS, ket::PeriodicMPS, trunc)
    rebuild_right!(cache, bra::PeriodicMPS, ket::PeriodicMPS, trunc)
    rebuild!(cache, bra::PeriodicMPS, ket::PeriodicMPS)
    rebuild!(cache, bra::PeriodicMPS, ket::PeriodicMPS, trunc)

Rebuild one or both cache directions directly from generated transfer elements
without materializing a [`PeriodicTransferMatrix`](@ref).
"""
function rebuild_left!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site)
    return rebuild_left!(cache, product_element_at)
end

function rebuild_right!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site)
    return rebuild_right!(cache, product_element_at)
end

function rebuild_left!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site, trunc)
    return rebuild_left!(cache, product_element_at)
end

function rebuild_right!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site, trunc)
    return rebuild_right!(cache, product_element_at)
end

function rebuild!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
    )
    rebuild_left!(cache, bra, ket)
    rebuild_right!(cache, bra, ket)
    return cache
end

function rebuild!(
        cache::PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::TruncationStrategy,
    )
    rebuild_left!(cache, bra, ket, trunc)
    rebuild_right!(cache, bra, ket, trunc)
    return cache
end

# Full-period product E₁ * E₂ * ... * E_N
Base.:*(ptm::PeriodicTransferMatrix) = range_product(ptm, eachindex(ptm))

expectation(ptm::PeriodicTransferMatrix) = tr(dense(*(ptm)))

"""
    expectation(bra::PeriodicMPS, ket::PeriodicMPS)
    expectation(bra::PeriodicMPS, ket::PeriodicMPS, trunc)

Contract transfer elements one site at a time without materializing a
[`PeriodicTransferMatrix`](@ref). Each site transfer is constructed exactly once.
"""
function expectation(bra::PeriodicMPS, ket::PeriodicMPS)
    N = _transfer_period_length(bra, ket)
    product = transfer_at(bra, ket, 1)
    for site in 2:N
        product = product * transfer_at(bra, ket, site)
    end
    return tr(dense(product))
end

expectation(mps::PeriodicMPS) = expectation(mps, mps)

function expectation(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    product = transfer_at(bra, ket, 1, trunc)
    for site in 2:N
        product = product * transfer_at(bra, ket, site, trunc)
    end
    return tr(dense(product))
end

expectation(mps::PeriodicMPS, trunc::TruncationStrategy) =
    expectation(mps, mps, trunc)

TensorKit.dot(bra::PeriodicMPS, ket::PeriodicMPS) =
    expectation(bra, ket)
