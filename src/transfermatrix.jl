# ------------------------------------------------------------------------------
# Dense transfer matrix — eagerly contracted, unlike MPSKit's lazy version
# ------------------------------------------------------------------------------

# Single-site transfer matrix: traces out physical legs of one site tensor.
# T: L ⊗ P₁ ⊗ P₂ ← R  →  E: (L ⊗ dual(L)) ← (R ⊗ dual(R))
function site_transfer_matrix(bra::AbstractTensorMap{E,S,3,1}, ket::AbstractTensorMap{E,S,3,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := bra'[-4; -2 1 2] * ket[-1 1 2; -3]
    return 𝔼
end

function site_transfer_matrix(ket::AbstractTensorMap{E,S,3,1}) where {E,S}
    return site_transfer_matrix(ket, ket)
end

function site_transfer_matrix(bra::AbstractTensorMap{E,S,2,1}, ket::AbstractTensorMap{E,S,2,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := bra'[-4; -2 1] * ket[-1 1; -3]
    return 𝔼
end

function site_transfer_matrix(ket::AbstractTensorMap{E,S,2,1}) where {E,S}
    return site_transfer_matrix(ket, ket)
end

function site_transfer_matrix(bra::AbstractTensorMap{E,S,3,1}, ketL::AbstractTensorMap{E,S,2,1}, ketR::AbstractTensorMap{E,S,2,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := ketL[-1 l; m] * ketR[m r; -3] * bra'[-4; -2 l r]
    return 𝔼
end

function site_transfer_matrix(braL::AbstractTensorMap{E,S,2,1}, braR::AbstractTensorMap{E,S,2,1}, ket::AbstractTensorMap{E,S,3,1}) where {E,S}
    bL, bR = braL', braR'
    @plansor 𝔼[-1 -2; -3 -4] := bL[1; -2 2] * bR[-4; 1 3] * ket[-1 2 3; -3]
    return 𝔼
end

# Collection of transfer matrices wrapping a full loop.
struct LoopTransferMatrix{T <: AbstractTensorMap} <: LoopMonoid
    matrices::PeriodicVector{T}
end

Base.length(v::LoopTransferMatrix) = length(v.matrices)
Base.eltype(::Type{<:LoopTransferMatrix{T}}) where {T} = T
Base.getindex(v::LoopTransferMatrix, i::Int) = v.matrices[i]
Base.setindex!(v::LoopTransferMatrix, T::AbstractTensorMap, i::Int) = (v.matrices[i] = T)

# Construct from a PeriodicMPS
function LoopTransferMatrix(bra::PeriodicMPS{<:AbstractTensorMap{E, S, M, 1}}, ket::PeriodicMPS{<:AbstractTensorMap{E, S, M, 1}}) where {E, S, M}
    tmats = [site_transfer_matrix(bratensor, kettensor) for (bratensor, kettensor) in zip(bra, ket)]
    return LoopTransferMatrix(PeriodicVector(tmats))
end

function LoopTransferMatrix(mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}) where {E, S}
    length(mpsB) == 2 * length(mpsA) ||
        throw(DimensionMismatch("two-site mpsA has $(length(mpsA)) sites but one-site mpsB has $(length(mpsB)) (expected $(2 * length(mpsA)))"))
    tmats = [site_transfer_matrix(mpsA[i], mpsB[2i-1], mpsB[2i])
             for i in 1:length(mpsA)]
    return LoopTransferMatrix(PeriodicVector(tmats))
end

function LoopTransferMatrix(mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}, mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}) where {E, S}
    length(mpsB) == 2 * length(mpsA) ||
        throw(DimensionMismatch("two-site mpsA has $(length(mpsA)) sites but one-site mpsB has $(length(mpsB)) (expected $(2 * length(mpsA)))"))
    tmats = [site_transfer_matrix(mpsB[2i-1], mpsB[2i], mpsA[i])
             for i in 1:length(mpsA)]
    return LoopTransferMatrix(PeriodicVector(tmats))
end

LoopTransferMatrix(mps::PeriodicMPS) = LoopTransferMatrix(mps, mps)

edge_unit(ltm::LoopTransferMatrix) = id(scalartype(ltm[1]), codomain(ltm[1]))

# Full loop product E₁ * E₂ * ... * E_N
function Base.:*(ltm::LoopTransferMatrix)
    N = length(ltm.matrices)
    result = ltm.matrices[1]
    for i in 2:N
        result = result * ltm.matrices[i]
    end
    return result
end

function expectation(ltm::LoopTransferMatrix)
    return tr(*(ltm))
end

function diff_norm(mps1::PeriodicMPS, mps2::PeriodicMPS)
    transfer1 = LoopTransferMatrix(mps1)
    transfer2 = LoopTransferMatrix(mps2)
    transfer12 = LoopTransferMatrix(mps1, mps2)
    return expectation(transfer1) + expectation(transfer2) - 2 * real(expectation(transfer12))
end
