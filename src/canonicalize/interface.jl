# ------------------------------------------------------------------------------
# Canonicalization interface
# ------------------------------------------------------------------------------

# Absorb projectors into 2-site MPS tensor: T_i = PL_i * T_i * PR_i
function _absorb_projectors!(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, PL_vec::PeriodicVector{<:TensorMap{E,S,1,1}}, PR_vec::PeriodicVector{<:TensorMap{E,S,1,1}}) where {E,S}
    N = length(mps)
    for i in 1:N
        @plansor mps[i][-1 -2 -3; -4] := PL_vec[i][-1; 1] * mps[i][1 -2 -3; 4] * PR_vec[i][4; -4]
    end
end

# Absorb projectors into 1-site MPS tensor
function _absorb_projectors!(mps::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}}, PL_vec::PeriodicVector{<:TensorMap{E,S,1,1}}, PR_vec::PeriodicVector{<:TensorMap{E,S,1,1}}) where {E,S}
    N = length(mps)
    for i in 1:N
        @plansor mps[i][-1 -2; -3] := PL_vec[i][-1; 1] * mps[i][1 -2; 2] * PR_vec[i][2; -3]
    end
end

function canonicalize!(mps::PeriodicMPS, alg::ProjectorCanonical)
    t_start = time()
    log = ProjectorLog()
    PR_vec, PL_vec = find_projectors(mps, alg, log)
    _absorb_projectors!(mps, PL_vec, PR_vec)
    log.sweep_times = [time() - t_start]
    log.converged = log.iterations < alg.maxiter
    return log
end
