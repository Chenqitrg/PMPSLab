# ------------------------------------------------------------------------------
# Truncated transfer matrix — SVD factors retained without dense reconstruction
# ------------------------------------------------------------------------------

"""
    TruncatedTransferMatrix

An approximate transfer matrix stored as the truncated singular-value
factorization `u * s * vᴴ`. The two outer factors are rank-three tensor maps;
the transfer's original left and right double bonds are never fused into a
dense rank-four tensor unless explicitly requested with
[`dense`](@ref).

`trunc` is retained so that multiplying two truncated transfers can choose a
combined truncation policy in a later step.
"""
struct TruncatedTransferMatrix{
        U<:AbstractTensorMap{<:Any,<:Any,2,1},
        S<:AbstractTensorMap{<:Any,<:Any,1,1},
        V<:AbstractTensorMap{<:Any,<:Any,1,2},
        T<:TruncationStrategy,
    } <: AbstractTransferMatrix
    u::U
    s::S
    vᴴ::V
    trunc::T
end

function TruncatedTransferMatrix(
        transfer::DenseTransferMatrix,
        trunc::TruncationStrategy,
    )
    u, s, vᴴ, _ = svd_trunc(dense(transfer); trunc)
    return TruncatedTransferMatrix(u, s, vᴴ, trunc)
end

TruncatedTransferMatrix(
    ket::AbstractTensorMap{<:Any,<:Any,2,1}, trunc::TruncationStrategy
) = TruncatedTransferMatrix(DenseTransferMatrix(ket), trunc)
TruncatedTransferMatrix(
    ket::AbstractTensorMap{<:Any,<:Any,3,1}, trunc::TruncationStrategy
) = TruncatedTransferMatrix(DenseTransferMatrix(ket), trunc)
TruncatedTransferMatrix(
    bra::AbstractTensorMap{<:Any,<:Any,M,1},
    ket::AbstractTensorMap{<:Any,<:Any,M,1},
    trunc::TruncationStrategy,
) where {M} = TruncatedTransferMatrix(DenseTransferMatrix(bra, ket), trunc)
TruncatedTransferMatrix(
    bra::AbstractTensorMap{<:Any,<:Any,3,1},
    ket_left::AbstractTensorMap{<:Any,<:Any,2,1},
    ket_right::AbstractTensorMap{<:Any,<:Any,2,1},
    trunc::TruncationStrategy,
) = TruncatedTransferMatrix(
    DenseTransferMatrix(bra, ket_left, ket_right), trunc
)
TruncatedTransferMatrix(
    bra_left::AbstractTensorMap{<:Any,<:Any,2,1},
    bra_right::AbstractTensorMap{<:Any,<:Any,2,1},
    ket::AbstractTensorMap{<:Any,<:Any,3,1},
    trunc::TruncationStrategy,
) = TruncatedTransferMatrix(
    DenseTransferMatrix(bra_left, bra_right, ket), trunc
)

scalartype(transfer::TruncatedTransferMatrix) = scalartype(transfer.u)
TensorKit.codomain(transfer::TruncatedTransferMatrix) = codomain(transfer.u)
TensorKit.domain(transfer::TruncatedTransferMatrix) = domain(transfer.vᴴ)
dense(transfer::TruncatedTransferMatrix) =
    transfer.u * transfer.s * transfer.vᴴ

Base.copy(transfer::TruncatedTransferMatrix) = TruncatedTransferMatrix(
    copy(transfer.u), copy(transfer.s), copy(transfer.vᴴ), transfer.trunc
)

function _can_reuse(
        destination::TruncatedTransferMatrix,
        s::AbstractTensorMap{<:Any,<:Any,1,1},
    )
    return space(destination.s, 1) == space(s, 1)
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
    DenseTransferMatrix(dense(transfer))

function Base.:*(
        left::TruncatedTransferMatrix,
        right::TruncatedTransferMatrix,
    )
    trunc = combine_truncation(left.trunc, right.trunc)

    # Contract only through the retained SVD spaces. In particular, do not
    # materialize either rank-four transfer matrix before truncating again.
    middle = left.s * left.vᴴ * right.u * right.s
    u_middle, s, vᴴ_middle, _ = svd_trunc(middle; trunc)

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
    u_middle, s, vᴴ_middle, _ = svd_trunc(middle; trunc)

    if _can_reuse(destination, s)
        mul!(destination.u, left.u, u_middle)
        copy!(destination.s, s)
        mul!(destination.vᴴ, vᴴ_middle, right.vᴴ)
        return destination
    end

    u = left.u * u_middle
    vᴴ = vᴴ_middle * right.vᴴ
    return TruncatedTransferMatrix(u, s, vᴴ, trunc)
end
