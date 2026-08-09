# ------------------------------------------------------------------------------
# Dense transfer matrix — eagerly contracted, unlike MPSKit's lazy version
# ------------------------------------------------------------------------------

"""
    DenseTransferMatrix

An eagerly contracted transfer matrix backed by a TensorKit tensor map. This
is the exact dense representation in the MPSKit [`AbstractTransferMatrix`](@ref)
hierarchy.
"""
struct DenseTransferMatrix{T<:AbstractTensorMap} <: AbstractTransferMatrix
    data::T
end

dense(transfer::DenseTransferMatrix) = transfer.data
scalartype(transfer::DenseTransferMatrix) = scalartype(dense(transfer))
TensorKit.codomain(transfer::DenseTransferMatrix) = codomain(dense(transfer))
TensorKit.domain(transfer::DenseTransferMatrix) = domain(dense(transfer))

Base.copy(transfer::DenseTransferMatrix) =
    DenseTransferMatrix(copy(dense(transfer)))
function Base.copy!(destination::DenseTransferMatrix, source::DenseTransferMatrix)
    copy!(dense(destination), dense(source))
    return destination
end

function Base.:*(left::DenseTransferMatrix, right::DenseTransferMatrix)
    return DenseTransferMatrix(dense(left) * dense(right))
end

function (transfer::DenseTransferMatrix)(
        vector::AbstractTensorMap{<:Any,<:Any,1,1},
    )
    data = dense(transfer)
    @plansor result[-1; -2] := data[-1 -2; 1 2] * vector[1; 2]
    return result
end

function LinearAlgebra.mul!(
        destination::DenseTransferMatrix,
        left::DenseTransferMatrix,
        right::DenseTransferMatrix,
    )
    mul!(dense(destination), dense(left), dense(right))
    return destination
end

function Base.isapprox(
        left::DenseTransferMatrix,
        right::DenseTransferMatrix;
        kwargs...,
    )
    return isapprox(dense(left), dense(right); kwargs...)
end
function Base.isapprox(
        left::DenseTransferMatrix,
        right::AbstractTensorMap;
        kwargs...,
    )
    return isapprox(dense(left), right; kwargs...)
end
function Base.isapprox(
        left::AbstractTensorMap,
        right::DenseTransferMatrix;
        kwargs...,
    )
    return isapprox(left, dense(right); kwargs...)
end

# Single-site transfer matrix: traces out physical legs of one site tensor.
# T: L ⊗ P₁ ⊗ P₂ ← R  →  E: (L ⊗ dual(L)) ← (R ⊗ dual(R))
function site_transfer_matrix(bra::AbstractTensorMap{E,S,3,1}, ket::AbstractTensorMap{E,S,3,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := bra'[-4; -2 1 2] * ket[-1 1 2; -3]
    return 𝔼
end

site_transfer_matrix(ket::AbstractTensorMap{E,S,3,1}) where {E,S} =
    site_transfer_matrix(ket, ket)

function site_transfer_matrix(bra::AbstractTensorMap{E,S,2,1}, ket::AbstractTensorMap{E,S,2,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := bra'[-4; -2 1] * ket[-1 1; -3]
    return 𝔼
end

site_transfer_matrix(ket::AbstractTensorMap{E,S,2,1}) where {E,S} =
    site_transfer_matrix(ket, ket)

function site_transfer_matrix(bra::AbstractTensorMap{E,S,3,1}, ketL::AbstractTensorMap{E,S,2,1}, ketR::AbstractTensorMap{E,S,2,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := ketL[-1 l; m] * ketR[m r; -3] * bra'[-4; -2 l r]
    return 𝔼
end

function site_transfer_matrix(braL::AbstractTensorMap{E,S,2,1}, braR::AbstractTensorMap{E,S,2,1}, ket::AbstractTensorMap{E,S,3,1}) where {E,S}
    bL, bR = braL', braR'
    @plansor 𝔼[-1 -2; -3 -4] := bL[1; -2 2] * bR[-4; 1 3] * ket[-1 2 3; -3]
    return 𝔼
end

DenseTransferMatrix(ket::AbstractTensorMap{<:Any,<:Any,2,1}) =
    DenseTransferMatrix(site_transfer_matrix(ket))
DenseTransferMatrix(ket::AbstractTensorMap{<:Any,<:Any,3,1}) =
    DenseTransferMatrix(site_transfer_matrix(ket))
DenseTransferMatrix(
        bra::AbstractTensorMap{<:Any,<:Any,M,1},
        ket::AbstractTensorMap{<:Any,<:Any,M,1},
    ) where {M} = DenseTransferMatrix(site_transfer_matrix(bra, ket))
DenseTransferMatrix(
        bra::AbstractTensorMap{<:Any,<:Any,3,1},
        ket_left::AbstractTensorMap{<:Any,<:Any,2,1},
        ket_right::AbstractTensorMap{<:Any,<:Any,2,1},
    ) = DenseTransferMatrix(site_transfer_matrix(bra, ket_left, ket_right))
DenseTransferMatrix(
        bra_left::AbstractTensorMap{<:Any,<:Any,2,1},
        bra_right::AbstractTensorMap{<:Any,<:Any,2,1},
        ket::AbstractTensorMap{<:Any,<:Any,3,1},
    ) = DenseTransferMatrix(site_transfer_matrix(bra_left, bra_right, ket))
