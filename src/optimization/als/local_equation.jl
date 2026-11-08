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
        target::AbstractTensorMap,
        paired_approximationᴴ::AbstractTensorMap,
        site::Int,
    )
    environment = dense(transfer)
    if isodd(site)
        @plansor W[-1; -2 -3] := environment[1 2; 3 -2] *
            target[3 -1 4; 1] * paired_approximationᴴ[2; -3 4]
    else
        @plansor W[-1; -2 -3] := environment[1 -3; 2 4] *
            target[2 3 -1; 1] * paired_approximationᴴ[-2; 4 3]
    end
    return W
end

function _split_rhs(
        transfer::TruncatedTransferMatrix,
        target::AbstractTensorMap,
        paired_approximationᴴ::AbstractTensorMap,
        site::Int,
    )
    if isodd(site)
        @plansor opt = true W[-1; -2 -3] := transfer.u[1 2; 5] *
            transfer.s[5; 6] * transfer.vᴴ[6; 3 -2] *
            target[3 -1 4; 1] * paired_approximationᴴ[2; -3 4]
    else
        @plansor opt = true W[-1; -2 -3] := transfer.u[1 -3; 5] *
            transfer.s[5; 6] * transfer.vᴴ[6; 2 4] *
            target[2 3 -1; 1] * paired_approximationᴴ[-2; 4 3]
    end
    return W
end

function _solve_split_norm(
        solver::TensorLinearSolveLab.AbstractSolve,
        x0::AbstractTensorMap,
        transfer::MPSKit.AbstractTransferMatrix,
        W::AbstractTensorMap;
        scale::Real=1,
        shift::Real=0,
    )
    norm_environment = transpose(
        dense(transfer), ((3, 1), (4, 2))
    )
    if scale != 1
        norm_environment = scale * norm_environment
    end
    if !iszero(shift)
        norm_environment += shift * id(scalartype(norm_environment), domain(norm_environment))
    end
    return TensorLinearSolveLab.rdiv(solver, x0, norm_environment, W)
end

function _solve_split_norm(
        solver::TensorLinearSolveLab.KrylovSolve,
        x0::AbstractTensorMap,
        transfer::TruncatedTransferMatrix,
        W::AbstractTensorMap;
        scale::Real=1,
        shift::Real=0,
    )
    if isone(scale) && iszero(shift)
        return TensorLinearSolveLab.rdiv(
            solver, x0, x -> _split_norm_action(x, transfer), W,
        )
    end
    action = x -> scale * _split_norm_action(x, transfer) + shift * x
    return TensorLinearSolveLab.rdiv(solver, x0, action, W)
end
