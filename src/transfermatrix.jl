# ------------------------------------------------------------------------------
# Dense transfer matrix — eagerly contracted, unlike MPSKit's lazy version
# ------------------------------------------------------------------------------

# Single-site transfer matrix: traces out physical legs of one site tensor.
# T: L ⊗ P₁ ⊗ P₂ ← R  →  E: (L ⊗ dual(L)) ← (R ⊗ dual(R))
function site_transfer_matrix(T::AbstractTensorMap{E,S,3,1}, P::AbstractTensorMap{E,S,3,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := T[-1 1 2; -3] * P'[-4; -2 1 2]
    return 𝔼
end

function site_transfer_matrix(T::AbstractTensorMap{E,S,3,1}) where {E,S}
    return site_transfer_matrix(T, T)
end

function site_transfer_matrix(T::AbstractTensorMap{E,S,2,1}, P::AbstractTensorMap{E,S,2,1}) where {E,S}
    @plansor 𝔼[-1 -2; -3 -4] := T[-1 1; -3] * P'[-4; -2 1]
    return 𝔼
end

function site_transfer_matrix(T::AbstractTensorMap{E,S,2,1}) where {E,S}
    return site_transfer_matrix(T, T)
end

# Collection of transfer matrices wrapping a full loop.
struct LoopTransferMatrix{T <: AbstractTensorMap}
    matrices::PeriodicVector{T}
end

# Construct from a PeriodicMPS
function LoopTransferMatrix(mps1::PeriodicMPS, mps2::PeriodicMPS)
    tmats = [site_transfer_matrix(T1, T2) for (T1, T2) in zip(mps1, mps2)]
    return LoopTransferMatrix(PeriodicVector(tmats))
end

LoopTransferMatrix(mps::PeriodicMPS) = LoopTransferMatrix(mps, mps)

# Full loop product E₁ * E₂ * ... * E_N
function Base.:*(ltm::LoopTransferMatrix)
    N = length(ltm.matrices)
    result = ltm.matrices[1]
    for i in 2:N
        result = result * ltm.matrices[i]
    end
    return result
end

# Apply the full loop transfer matrix to a vector
function Base.:*(ltm::LoopTransferMatrix, vec::AbstractTensorMap)
    return *(ltm) * vec
end

# Product over a range [i, j] with periodic wrap
function product_range(ltm::LoopTransferMatrix, i::Int, j::Int)
    N = length(ltm.matrices)
    result = ltm.matrices[i]
    k = i
    while k != j
        k = mod1(k + 1, N)
        result = result * ltm.matrices[k]
    end
    return result
end
