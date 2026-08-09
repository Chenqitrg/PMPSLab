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
    scale2 = max(abs(n1), abs(n2))
    tolerance = max(atol, rtol * sqrt(scale2))

    # `distance2` subtracts three independently contracted quantities of size
    # `scale2`. At the default `rtol`, squaring the requested tolerance reaches
    # machine precision, so contraction roundoff can otherwise decide the
    # result solely by its sign. Use the same roundoff budget in both
    # directions; unlike `_nonnegative_squared`, keep the reported distance
    # itself unchanged when it is small and positive.
    roundoff2 = 100 * eps(float(one(distance2))) * scale2
    return distance2 <= max(tolerance^2, roundoff2)
end

function LinearAlgebra.normalize(mps::PeriodicMPS)
    n = norm(mps)
    iszero(n) && throw(ArgumentError("cannot normalize a zero periodic MPS"))
    result = copy(mps)
    result[1] = result[1] / n
    return result
end
