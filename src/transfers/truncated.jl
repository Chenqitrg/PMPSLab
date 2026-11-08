# ------------------------------------------------------------------------------
# Truncated transfer matrix — SVD factors retained without dense reconstruction
# ------------------------------------------------------------------------------

"""
    TruncatedTransferMatrix

An approximate transfer matrix stored as the truncated singular-value
factorization `u * s * vᴴ`. The two outer factors are rank-three tensor maps;
the transfer's original left and right double bonds are never fused into a
dense rank-four tensor unless explicitly requested with
`TensorKit.TensorMap(transfer)`.

`trunc` is retained so that multiplying two truncated transfers can choose a
combined truncation policy in a later step.
"""
struct TruncatedTransferMatrix{
        U<:TensorKit.AbstractTensorMap{<:Any,<:Any,2,1},
        S<:TensorKit.AbstractTensorMap{<:Any,<:Any,1,1},
        V<:TensorKit.AbstractTensorMap{<:Any,<:Any,1,2},
        T<:MPSKit.TruncationStrategy,
    } <: MPSKit.AbstractTransferMatrix
    u::U
    s::S
    vᴴ::V
    trunc::T
end

function TruncatedTransferMatrix(
        transfer::DenseTransferMatrix,
        trunc::MPSKit.TruncationStrategy,
    )
    u, s, vᴴ, _ = TensorKit.svd_trunc(TensorKit.TensorMap(transfer); trunc)
    return TruncatedTransferMatrix(u, s, vᴴ, trunc)
end

TruncatedTransferMatrix(
    ket::TensorKit.AbstractTensorMap{<:Any,<:Any,2,1}, trunc::MPSKit.TruncationStrategy
) = TruncatedTransferMatrix(DenseTransferMatrix(ket), trunc)
TruncatedTransferMatrix(
    ket::TensorKit.AbstractTensorMap{<:Any,<:Any,3,1}, trunc::MPSKit.TruncationStrategy
) = TruncatedTransferMatrix(DenseTransferMatrix(ket), trunc)
TruncatedTransferMatrix(
    bra::TensorKit.AbstractTensorMap{<:Any,<:Any,M,1},
    ket::TensorKit.AbstractTensorMap{<:Any,<:Any,M,1},
    trunc::MPSKit.TruncationStrategy,
) where {M} = TruncatedTransferMatrix(DenseTransferMatrix(bra, ket), trunc)
TruncatedTransferMatrix(
    bra::TensorKit.AbstractTensorMap{<:Any,<:Any,3,1},
    ket_left::TensorKit.AbstractTensorMap{<:Any,<:Any,2,1},
    ket_right::TensorKit.AbstractTensorMap{<:Any,<:Any,2,1},
    trunc::MPSKit.TruncationStrategy,
) = TruncatedTransferMatrix(
    DenseTransferMatrix(bra, ket_left, ket_right), trunc
)
TruncatedTransferMatrix(
    bra_left::TensorKit.AbstractTensorMap{<:Any,<:Any,2,1},
    bra_right::TensorKit.AbstractTensorMap{<:Any,<:Any,2,1},
    ket::TensorKit.AbstractTensorMap{<:Any,<:Any,3,1},
    trunc::MPSKit.TruncationStrategy,
) = TruncatedTransferMatrix(
    DenseTransferMatrix(bra_left, bra_right, ket), trunc
)

TensorKit.scalartype(transfer::TruncatedTransferMatrix) = TensorKit.scalartype(transfer.u)
TensorKit.codomain(transfer::TruncatedTransferMatrix) = TensorKit.codomain(transfer.u)
TensorKit.domain(transfer::TruncatedTransferMatrix) = TensorKit.domain(transfer.vᴴ)
TensorKit.TensorMap(transfer::TruncatedTransferMatrix) =
    transfer.u * transfer.s * transfer.vᴴ

Base.copy(transfer::TruncatedTransferMatrix) = TruncatedTransferMatrix(
    copy(transfer.u), copy(transfer.s), copy(transfer.vᴴ), transfer.trunc
)

function _can_reuse(
        destination::TruncatedTransferMatrix,
        s::TensorKit.AbstractTensorMap{<:Any,<:Any,1,1},
    )
    return TensorKit.space(destination.s, 1) == TensorKit.space(s, 1)
end

function Base.copy!(
        destination::TruncatedTransferMatrix,
        source::TruncatedTransferMatrix,
    )
    if destination.trunc == source.trunc && _can_reuse(destination, source.s)
        copy!(destination.u, source.u)
        copy!(destination.s, source.s)
        copy!(destination.vᴴ, source.vᴴ)
        return destination
    end
    return copy(source)
end

DenseTransferMatrix(transfer::TruncatedTransferMatrix) =
    DenseTransferMatrix(TensorKit.TensorMap(transfer))

function Base.:*(
        left::TruncatedTransferMatrix,
        right::TruncatedTransferMatrix,
    )
    trunc = combine_truncation(left.trunc, right.trunc)

    # Contract only through the retained SVD spaces. In particular, do not
    # materialize either rank-four transfer matrix before truncating again.
    middle = left.s * left.vᴴ * right.u * right.s
    u_middle, s, vᴴ_middle, _ = TensorKit.svd_trunc(middle; trunc)

    u = left.u * u_middle
    vᴴ = vᴴ_middle * right.vᴴ
    return TruncatedTransferMatrix(u, s, vᴴ, trunc)
end

"""
    mul!(destination, left, right)

Multiply two truncated transfers, reusing `destination` when the newly
truncated SVD space matches its existing middle space. The cache invariant
already fixes the outer bond spaces and truncation policy for a slot. If the
middle space does not match, return a newly allocated transfer. Callers must
always use the return value, and `destination` must not alias either input.
"""
function LinearAlgebra.mul!(
        destination::TruncatedTransferMatrix,
        left::TruncatedTransferMatrix,
        right::TruncatedTransferMatrix,
    )
    trunc = combine_truncation(left.trunc, right.trunc)
    middle = left.s * left.vᴴ * right.u * right.s
    u_middle, s, vᴴ_middle, _ = TensorKit.svd_trunc(middle; trunc)

    if _can_reuse(destination, s)
        LinearAlgebra.mul!(destination.u, left.u, u_middle)
        copy!(destination.s, s)
        LinearAlgebra.mul!(destination.vᴴ, vᴴ_middle, right.vᴴ)
        return destination
    end

    u = left.u * u_middle
    vᴴ = vᴴ_middle * right.vᴴ
    return TruncatedTransferMatrix(u, s, vᴴ, trunc)
end

# Native conversion shares the dense map and materializes a factored transfer.
Base.convert(::Type{TensorKit.TensorMap}, transfer::Union{DenseTransferMatrix,TruncatedTransferMatrix}) =
    TensorKit.TensorMap(transfer)

"""Materialize a local transfer as a TensorMap."""
dense(transfer::Union{DenseTransferMatrix,TruncatedTransferMatrix}) = TensorKit.TensorMap(transfer)
