# Collection of transfer matrices over one period.
struct PeriodicTransferMatrix{T<:MPSKit.AbstractTransferMatrix} <:
       PeriodicProductsLab.AbstractPeriodicProduct{T}
    matrices::MPSKit.PeriodicVector{T}
end

Base.length(v::PeriodicTransferMatrix) = length(v.matrices)
Base.parent(v::PeriodicTransferMatrix) = v.matrices
Base.eachindex(v::PeriodicTransferMatrix) = eachindex(parent(v))
Base.iterate(v::PeriodicTransferMatrix, state::S...) where {S} =
    iterate(parent(v), state...)
Base.getindex(v::PeriodicTransferMatrix, i::Int) = v.matrices[i]
function Base.setindex!(
        v::PeriodicTransferMatrix{T}, transfer::T, i::Int
    ) where {T<:MPSKit.AbstractTransferMatrix}
    return v.matrices[i] = transfer
end

function _transfer_period_length(
        bra::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,M,1}},
        ket::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,M,1}},
    ) where {E,S,M}
    length(bra) == length(ket) || throw(DimensionMismatch(
        "bra has $(length(bra)) sites but ket has $(length(ket)) sites"
    ))
    return length(bra)
end

function _transfer_period_length(
        target::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,3,1}},
        approximation::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,2,1}},
    ) where {E,S}
    length(approximation) == 2 * length(target) || throw(DimensionMismatch(
        "the target has $(length(target)) two-physical-leg tensors but the " *
        "approximation has $(length(approximation)) single-physical-leg " *
        "tensors (expected $(2 * length(target)))"
    ))
    return length(target)
end

_transfer_period_length(
        approximation::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,2,1}},
        target::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,3,1}},
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
        target::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,3,1}},
        approximation::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,2,1}},
        site::Int,
    ) where {E,S} = DenseTransferMatrix(
        target[site], approximation[2site - 1], approximation[2site]
    )

transfer_at(
        approximation::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,2,1}},
        target::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,3,1}},
        site::Int,
    ) where {E,S} = DenseTransferMatrix(
        approximation[2site - 1], approximation[2site], target[site]
    )

transfer_at(
        mps::PeriodicMPS,
        site::Int,
        trunc::MPSKit.TruncationStrategy,
    ) = transfer_at(mps, mps, site, trunc)

transfer_at(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        site::Int,
        trunc::MPSKit.TruncationStrategy,
    ) = TruncatedTransferMatrix(bra[site], ket[site], trunc)

transfer_at(
        target::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,3,1}},
        approximation::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,2,1}},
        site::Int,
        trunc::MPSKit.TruncationStrategy,
    ) where {E,S} = TruncatedTransferMatrix(
        target[site], approximation[2site - 1], approximation[2site], trunc
    )

transfer_at(
        approximation::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,2,1}},
        target::PeriodicMPS{<:TensorKit.AbstractTensorMap{E,S,3,1}},
        site::Int,
        trunc::MPSKit.TruncationStrategy,
    ) where {E,S} = TruncatedTransferMatrix(
        approximation[2site - 1], approximation[2site], target[site], trunc
    )

function PeriodicTransferMatrix(bra::PeriodicMPS, ket::PeriodicMPS)
    N = _transfer_period_length(bra, ket)
    matrices = [transfer_at(bra, ket, site) for site in 1:N]
    return PeriodicTransferMatrix(MPSKit.PeriodicVector(matrices))
end

PeriodicTransferMatrix(mps::PeriodicMPS) = PeriodicTransferMatrix(mps, mps)

function PeriodicTransferMatrix(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::MPSKit.TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    matrices = [
        transfer_at(bra, ket, site, trunc) for site in 1:N
    ]
    return PeriodicTransferMatrix(MPSKit.PeriodicVector(matrices))
end

PeriodicTransferMatrix(mps::PeriodicMPS, trunc::MPSKit.TruncationStrategy) =
    PeriodicTransferMatrix(mps, mps, trunc)

"""
    build_transfer_cache(bra::PeriodicMPS, ket::PeriodicMPS)
    build_transfer_cache(bra::PeriodicMPS, ket::PeriodicMPS, trunc)

Build a transfer-product cache from periodic MPS inputs. This PMPS-specific
adapter delegates generated transfer factors to `PeriodicProductsLab.build_cache`. No
`PeriodicTransferMatrix` or separate period-sized element collection is created.
"""
function build_transfer_cache(bra::PeriodicMPS, ket::PeriodicMPS)
    N = _transfer_period_length(bra, ket)
    product_element_at(site::Int) = transfer_at(bra, ket, site)
    return PeriodicProductsLab.build_cache(N, product_element_at)
end

build_transfer_cache(mps::PeriodicMPS) = build_transfer_cache(mps, mps)

function build_transfer_cache(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::MPSKit.TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    product_element_at(site::Int) = transfer_at(bra, ket, site, trunc)
    return PeriodicProductsLab.build_cache(N, product_element_at)
end

build_transfer_cache(mps::PeriodicMPS, trunc::MPSKit.TruncationStrategy) =
    build_transfer_cache(mps, mps, trunc)

"""
    rebuild_left_transfer_cache!(cache, bra::PeriodicMPS, ket::PeriodicMPS)
    rebuild_right_transfer_cache!(cache, bra::PeriodicMPS, ket::PeriodicMPS)
    rebuild_left_transfer_cache!(cache, bra::PeriodicMPS, ket::PeriodicMPS, trunc)
    rebuild_right_transfer_cache!(cache, bra::PeriodicMPS, ket::PeriodicMPS, trunc)
    rebuild_transfer_cache!(cache, bra::PeriodicMPS, ket::PeriodicMPS)
    rebuild_transfer_cache!(cache, bra::PeriodicMPS, ket::PeriodicMPS, trunc)

Repair invalid transfer products from current periodic MPS inputs. These
PMPS-specific adapters repair one or both cache directions using generated
transfer elements, reusing valid prefixes and suffixes without materializing a
[`PeriodicTransferMatrix`](@ref). Report every changed transfer site with
`invalidate!` or `advance!` first. If the transfer policy changes, all affected
sites must also be invalidated, or a new cache must be built.
"""
function rebuild_left_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site)
    return PeriodicProductsLab.rebuild_left!(cache, product_element_at)
end

function rebuild_right_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site)
    return PeriodicProductsLab.rebuild_right!(cache, product_element_at)
end

function rebuild_left_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::MPSKit.TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site, trunc)
    return PeriodicProductsLab.rebuild_left!(cache, product_element_at)
end

function rebuild_right_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::MPSKit.TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    N == length(cache) || throw(DimensionMismatch(
        "cache has $(length(cache)) sites but transfer product has $N"
    ))
    product_element_at(site::Int) = transfer_at(bra, ket, site, trunc)
    return PeriodicProductsLab.rebuild_right!(cache, product_element_at)
end

function rebuild_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
    )
    rebuild_left_transfer_cache!(cache, bra, ket)
    rebuild_right_transfer_cache!(cache, bra, ket)
    return cache
end

function rebuild_transfer_cache!(
        cache::PeriodicProductsLab.PeriodicCache,
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::MPSKit.TruncationStrategy,
    )
    rebuild_left_transfer_cache!(cache, bra, ket, trunc)
    rebuild_right_transfer_cache!(cache, bra, ket, trunc)
    return cache
end

# Full-period product E₁ * E₂ * ... * E_N
Base.:*(ptm::PeriodicTransferMatrix) = PeriodicProductsLab.range_product(ptm, eachindex(ptm))

expectation(ptm::PeriodicTransferMatrix) = LinearAlgebra.tr(TensorKit.TensorMap(*(ptm)))

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
    return LinearAlgebra.tr(TensorKit.TensorMap(product))
end

expectation(mps::PeriodicMPS) = expectation(mps, mps)

function expectation(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        trunc::MPSKit.TruncationStrategy,
    )
    N = _transfer_period_length(bra, ket)
    product = transfer_at(bra, ket, 1, trunc)
    for site in 2:N
        product = product * transfer_at(bra, ket, site, trunc)
    end
    return LinearAlgebra.tr(TensorKit.TensorMap(product))
end

expectation(mps::PeriodicMPS, trunc::MPSKit.TruncationStrategy) =
    expectation(mps, mps, trunc)

TensorKit.dot(bra::PeriodicMPS, ket::PeriodicMPS) =
    expectation(bra, ket)
