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
        mpsA::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        mpsB::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
    ) where {E,S}
    length(mpsB) == 2 * length(mpsA) || throw(DimensionMismatch(
        "two-site mpsA has $(length(mpsA)) sites but one-site mpsB has " *
        "$(length(mpsB)) (expected $(2 * length(mpsA)))"
    ))
    return length(mpsA)
end

_transfer_period_length(
        mpsB::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        mpsA::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
    ) where {E,S} = _transfer_period_length(mpsA, mpsB)

"""
    transfer_at(bra::PeriodicMPS, ket::PeriodicMPS, site::Int)
    transfer_at(bra::PeriodicMPS, ket::PeriodicMPS, site::Int, trunc)

Construct the transfer matrix at `site`. Mixed two-site/one-site MPS pairs are
contracted over the corresponding pair of one-site tensors. Supplying `trunc`
returns a [`TruncatedTransferMatrix`](@ref); otherwise the result is dense.
"""
transfer_at(bra::PeriodicMPS, ket::PeriodicMPS, site::Int) =
    DenseTransferMatrix(bra[site], ket[site])

transfer_at(
        mpsA::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        mpsB::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        site::Int,
    ) where {E,S} = DenseTransferMatrix(
        mpsA[site], mpsB[2site - 1], mpsB[2site]
    )

transfer_at(
        mpsB::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        mpsA::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        site::Int,
    ) where {E,S} = DenseTransferMatrix(
        mpsB[2site - 1], mpsB[2site], mpsA[site]
    )

transfer_at(
        bra::PeriodicMPS,
        ket::PeriodicMPS,
        site::Int,
        trunc::TruncationStrategy,
    ) = TruncatedTransferMatrix(bra[site], ket[site], trunc)

transfer_at(
        mpsA::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        mpsB::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        site::Int,
        trunc::TruncationStrategy,
    ) where {E,S} = TruncatedTransferMatrix(
        mpsA[site], mpsB[2site - 1], mpsB[2site], trunc
    )

transfer_at(
        mpsB::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
        mpsA::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        site::Int,
        trunc::TruncationStrategy,
    ) where {E,S} = TruncatedTransferMatrix(
        mpsB[2site - 1], mpsB[2site], mpsA[site], trunc
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

# Full-period product E₁ * E₂ * ... * E_N
Base.:*(ptm::PeriodicTransferMatrix) = range_product(ptm, eachindex(ptm))

expectation(ptm::PeriodicTransferMatrix) = tr(dense(*(ptm)))

TensorKit.dot(bra::PeriodicMPS, ket::PeriodicMPS) =
    expectation(PeriodicTransferMatrix(bra, ket))
