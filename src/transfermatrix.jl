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
struct LoopTransferMatrix{T <: AbstractTensorMap} <: LoopDenseCP
    matrices::PeriodicVector{T}
end

Base.length(v::LoopTransferMatrix) = length(v.matrices)
Base.eltype(::Type{<:LoopTransferMatrix{T}}) where {T} = T
Base.getindex(v::LoopTransferMatrix, i::Int) = v.matrices[i]
Base.setindex!(v::LoopTransferMatrix, T::AbstractTensorMap, i::Int) = (v.matrices[i] = T)

# Construct from a PeriodicMPS
function LoopTransferMatrix(bra::PeriodicMPS{<:AbstractTensorMap{E, S, M, 1}}, ket::PeriodicMPS{<:AbstractTensorMap{E, S, M, 1}}) where {E, S, M}
    length(bra) == length(ket) || throw(DimensionMismatch(
        "bra has $(length(bra)) sites but ket has $(length(ket)) sites"
    ))
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

TensorKit.dot(bra::PeriodicMPS, ket::PeriodicMPS) = expectation(LoopTransferMatrix(bra, ket))

function _nonnegative_squared(value::Real, scale::Real, quantity::AbstractString)
    value >= 0 && return value
    roundoff = 100 * eps(float(one(value))) * max(abs(scale), one(scale))
    value >= -roundoff && return zero(value)
    throw(DomainError(value, "$quantity is negative beyond roundoff tolerance"))
end

function squared_norm(mps::PeriodicMPS)
    value = real(dot(mps, mps))
    return _nonnegative_squared(value, value, "squared MPS norm")
end
function TensorKit.norm(mps::PeriodicMPS)
    return sqrt(squared_norm(mps))
end

function _squared_distance_data(
        mps1::PeriodicMPS,
        mps2::PeriodicMPS
    )
    n1 = squared_norm(mps1)
    n2 = squared_norm(mps2)
    distance2 = n1 + n2 - 2 * real(dot(mps1, mps2))
    scale = max(abs(n1), abs(n2))
    return _nonnegative_squared(distance2, scale, "squared MPS distance"), n1, n2
end

squared_distance(mps1::PeriodicMPS, mps2::PeriodicMPS) =
    first(_squared_distance_data(mps1, mps2))

# Compatibility name retained for callers that already treated `diff_norm` as
# a squared distance.
diff_norm(mps1::PeriodicMPS, mps2::PeriodicMPS) = squared_distance(mps1, mps2)

function Base.isapprox(
        mps1::PeriodicMPS,
        mps2::PeriodicMPS;
        atol::Real = 0,
        rtol::Real = Base.rtoldefault(scalartype(mps1), scalartype(mps2), atol)
    )
    distance2, n1, n2 = _squared_distance_data(mps1, mps2)
    tolerance = max(atol, rtol * max(sqrt(abs(n1)), sqrt(abs(n2))))
    return distance2 <= tolerance^2
end

function LinearAlgebra.normalize(mps::PeriodicMPS)
    n = norm(mps)
    iszero(n) && throw(ArgumentError("cannot normalize a zero periodic MPS"))
    result = copy(mps)
    result[1] = result[1] / n
    return result
end
