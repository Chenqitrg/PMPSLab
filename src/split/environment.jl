# Split-specific consumers for dense and factored transfer environments.

function _split_norm_action(
        x::AbstractTensorMap,
        transfer::DenseTransferMatrix,
    )
    norm_environment = transpose(
        dense(transfer), ((3, 1), (4, 2))
    )
    return x * norm_environment
end

function _split_norm_action(
        x::AbstractTensorMap,
        transfer::TruncatedTransferMatrix,
    )
    @plansor opt = true result[-1; -2 -3] := x[-1; 1 2] *
        transfer.vᴴ[4; 1 -2] * transfer.s[5; 4] * transfer.u[2 -3; 5]
    return result
end

function _split_rhs(
        transfer::DenseTransferMatrix,
        A::AbstractTensorMap,
        other_Bᴴ::AbstractTensorMap,
        site::Int,
    )
    environment = dense(transfer)
    if isodd(site)
        @plansor W[-1; -2 -3] := environment[1 2; 3 -2] *
            A[3 -1 4; 1] * other_Bᴴ[2; -3 4]
    else
        @plansor W[-1; -2 -3] := environment[1 -3; 2 4] *
            A[2 3 -1; 1] * other_Bᴴ[-2; 4 3]
    end
    return W
end

function _split_rhs(
        transfer::TruncatedTransferMatrix,
        A::AbstractTensorMap,
        other_Bᴴ::AbstractTensorMap,
        site::Int,
    )
    if isodd(site)
        @plansor opt = true W[-1; -2 -3] := transfer.u[1 2; 5] *
            transfer.s[5; 6] * transfer.vᴴ[6; 3 -2] *
            A[3 -1 4; 1] * other_Bᴴ[2; -3 4]
    else
        @plansor opt = true W[-1; -2 -3] := transfer.u[1 -3; 5] *
            transfer.s[5; 6] * transfer.vᴴ[6; 2 4] *
            A[2 3 -1; 1] * other_Bᴴ[-2; 4 3]
    end
    return W
end

function _solve_split_norm(
        solve::AbstractSolve,
        x0::AbstractTensorMap,
        transfer::AbstractTransferMatrix,
        W::AbstractTensorMap,
    )
    norm_environment = transpose(
        dense(transfer), ((3, 1), (4, 2))
    )
    return rdiv(solve, x0, norm_environment, W)
end

function _solve_split_norm(
        solve::KrylovSolve,
        x0::AbstractTensorMap,
        transfer::TruncatedTransferMatrix,
        W::AbstractTensorMap,
    )
    action = x -> _split_norm_action(x, transfer)
    return rdiv(solve, x0, action, W)
end
