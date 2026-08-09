# Collection of transfer matrices over one period.
struct PeriodicTransferMatrix{T<:AbstractTransferMatrix} <: AbstractPeriodicProduct
    matrices::PeriodicVector{T}
end

Base.length(v::PeriodicTransferMatrix) = length(v.matrices)
Base.eltype(::Type{<:PeriodicTransferMatrix{T}}) where {T} = T
Base.parent(v::PeriodicTransferMatrix) = v.matrices
Base.eachindex(v::PeriodicTransferMatrix) = eachindex(parent(v))
Base.iterate(v::PeriodicTransferMatrix, state...) = iterate(parent(v), state...)
Base.getindex(v::PeriodicTransferMatrix, i::Int) = v.matrices[i]
function Base.setindex!(
        v::PeriodicTransferMatrix{T}, transfer::T, i::Int
    ) where {T<:AbstractTransferMatrix}
    return v.matrices[i] = transfer
end

# Construct from a PeriodicMPS
function PeriodicTransferMatrix(bra::PeriodicMPS{<:AbstractTensorMap{E, S, M, 1}}, ket::PeriodicMPS{<:AbstractTensorMap{E, S, M, 1}}) where {E, S, M}
    length(bra) == length(ket) || throw(DimensionMismatch(
        "bra has $(length(bra)) sites but ket has $(length(ket)) sites"
    ))
    tmats = [
        DenseTransferMatrix(bratensor, kettensor)
        for (bratensor, kettensor) in zip(bra, ket)
    ]
    return PeriodicTransferMatrix(PeriodicVector(tmats))
end

function PeriodicTransferMatrix(mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}) where {E, S}
    length(mpsB) == 2 * length(mpsA) ||
        throw(DimensionMismatch("two-site mpsA has $(length(mpsA)) sites but one-site mpsB has $(length(mpsB)) (expected $(2 * length(mpsA)))"))
    tmats = [
        DenseTransferMatrix(mpsA[i], mpsB[2i - 1], mpsB[2i])
        for i in eachindex(mpsA)
    ]
    return PeriodicTransferMatrix(PeriodicVector(tmats))
end

function PeriodicTransferMatrix(mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}, mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}) where {E, S}
    length(mpsB) == 2 * length(mpsA) ||
        throw(DimensionMismatch("two-site mpsA has $(length(mpsA)) sites but one-site mpsB has $(length(mpsB)) (expected $(2 * length(mpsA)))"))
    tmats = [
        DenseTransferMatrix(mpsB[2i - 1], mpsB[2i], mpsA[i])
        for i in eachindex(mpsA)
    ]
    return PeriodicTransferMatrix(PeriodicVector(tmats))
end

PeriodicTransferMatrix(mps::PeriodicMPS) = PeriodicTransferMatrix(mps, mps)

edge_identity(ptm::PeriodicTransferMatrix) =
    DenseTransferMatrix(id(scalartype(ptm[1]), codomain(ptm[1])))

# Full-period product E₁ * E₂ * ... * E_N
Base.:*(ptm::PeriodicTransferMatrix) = range_product(ptm, eachindex(ptm))

expectation(ptm::PeriodicTransferMatrix) = tr(dense(*(ptm)))

TensorKit.dot(bra::PeriodicMPS, ket::PeriodicMPS) =
    expectation(PeriodicTransferMatrix(bra, ket))
