# ------------------------------------------------------------------------------
# Dense transfer matrix — eagerly contracted, unlike MPSKit's lazy version
# ------------------------------------------------------------------------------

"""
    DenseTransferMatrix

An eagerly contracted transfer matrix backed by a TensorKit tensor map. This
is the exact dense representation in the MPSKit [`AbstractTransferMatrix`](@ref)
hierarchy.
"""
struct DenseTransferMatrix{T<:TensorKit.AbstractTensorMap} <: MPSKit.AbstractTransferMatrix
    data::T
end

TensorKit.TensorMap(transfer::DenseTransferMatrix) = transfer.data
TensorKit.scalartype(transfer::DenseTransferMatrix) = TensorKit.scalartype(TensorKit.TensorMap(transfer))
TensorKit.codomain(transfer::DenseTransferMatrix) = TensorKit.codomain(TensorKit.TensorMap(transfer))
TensorKit.domain(transfer::DenseTransferMatrix) = TensorKit.domain(TensorKit.TensorMap(transfer))

Base.copy(transfer::DenseTransferMatrix) =
    DenseTransferMatrix(copy(TensorKit.TensorMap(transfer)))
function Base.copy!(destination::DenseTransferMatrix, source::DenseTransferMatrix)
    copy!(TensorKit.TensorMap(destination), TensorKit.TensorMap(source))
    return destination
end

function Base.:*(left::DenseTransferMatrix, right::DenseTransferMatrix)
    return DenseTransferMatrix(TensorKit.TensorMap(left) * TensorKit.TensorMap(right))
end

function (transfer::DenseTransferMatrix)(
        vector::TensorKit.AbstractTensorMap{<:Any,<:Any,1,1},
    )
    data = TensorKit.TensorMap(transfer)
    @plansor result[-1; -2] := data[-1 -2; 1 2] * vector[1; 2]
    return result
end

function LinearAlgebra.mul!(
        destination::DenseTransferMatrix,
        left::DenseTransferMatrix,
        right::DenseTransferMatrix,
    )
    LinearAlgebra.mul!(TensorKit.TensorMap(destination), TensorKit.TensorMap(left), TensorKit.TensorMap(right))
    return destination
end

function Base.isapprox(
        left::DenseTransferMatrix,
        right::DenseTransferMatrix;
        kwargs...,
    )
    return isapprox(TensorKit.TensorMap(left), TensorKit.TensorMap(right); kwargs...)
end
function Base.isapprox(
        left::DenseTransferMatrix,
        right::TensorKit.AbstractTensorMap;
        kwargs...,
    )
    return isapprox(TensorKit.TensorMap(left), right; kwargs...)
end
function Base.isapprox(
        left::TensorKit.AbstractTensorMap,
        right::DenseTransferMatrix;
        kwargs...,
    )
    return isapprox(left, TensorKit.TensorMap(right); kwargs...)
end

# Construct a dense transfer by tracing the physical legs of site tensors.
# T: L ⊗ P₁ ⊗ P₂ ← R  →  E: (L ⊗ dual(L)) ← (R ⊗ dual(R))
function DenseTransferMatrix(
        bra::TensorKit.AbstractTensorMap{E,S,3,1},
        ket::TensorKit.AbstractTensorMap{E,S,3,1},
    ) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := bra'[-4; -2 1 2] * ket[-1 1 2; -3]
    return DenseTransferMatrix(𝔼)
end

DenseTransferMatrix(ket::TensorKit.AbstractTensorMap{E,S,3,1}) where {E,S} =
    DenseTransferMatrix(ket, ket)

function DenseTransferMatrix(
        bra::TensorKit.AbstractTensorMap{E,S,2,1},
        ket::TensorKit.AbstractTensorMap{E,S,2,1},
    ) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := bra'[-4; -2 1] * ket[-1 1; -3]
    return DenseTransferMatrix(𝔼)
end

DenseTransferMatrix(ket::TensorKit.AbstractTensorMap{E,S,2,1}) where {E,S} =
    DenseTransferMatrix(ket, ket)

function DenseTransferMatrix(
        bra::TensorKit.AbstractTensorMap{E,S,3,1},
        ket_left::TensorKit.AbstractTensorMap{E,S,2,1},
        ket_right::TensorKit.AbstractTensorMap{E,S,2,1},
    ) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := ket_left[-1 l; m] * ket_right[m r; -3] * bra'[-4; -2 l r]
    return DenseTransferMatrix(𝔼)
end

function DenseTransferMatrix(
        bra_left::TensorKit.AbstractTensorMap{E,S,2,1},
        bra_right::TensorKit.AbstractTensorMap{E,S,2,1},
        ket::TensorKit.AbstractTensorMap{E,S,3,1},
    ) where {E,S}
    bL, bR = bra_left', bra_right'
    @plansor 𝔼[-1 -2; -3 -4] := bL[1; -2 2] * bR[-4; 1 3] * ket[-1 2 3; -3]
    return DenseTransferMatrix(𝔼)
end
