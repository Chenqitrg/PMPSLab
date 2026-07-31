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

"""
    canonicalize!(mps::PeriodicMPS, alg::ProjectorCanonical;
                  log = ProjectorLog(), observer = nothing)

Absorb the transfer-matrix fixed-point environments into `mps` by iteratively
computing left/right projectors at every bond and folding them into the site
tensors. Modifies `mps` in place and returns a [`ProjectorLog`](@ref).

# Keyword arguments
- `log::ProjectorLog`: external log for live progress polling.
  Pass a pre-allocated `ProjectorLog` and read `log.iterations`,
  `log.error_history`, or `log.converged` from another task.
- `observer`: callable `f(log::ProjectorLog)` fired after each bond's
  projector computation and once at completion. Use for progress bars,
  logging, or early termination.

# Examples
# === Basic usage ===
log = canonicalize!(mps, alg)

# === Live polling from another task ===
# @async (concurrent task) polls the log while canonicalize! runs
log = ProjectorLog()
@async while !log.converged
    println("iter: \$(log.iterations), bonds: \$(length(log.error_history))")
    sleep(0.1)
end
canonicalize!(mps, alg; log = log)

# === Observer callback (alternative to polling) ===
canonicalize!(mps, alg; observer = log -> @info "bond done" log.iterations)
"""
function canonicalize!(mps::PeriodicMPS, alg::ProjectorCanonical;
                       log::ProjectorLog = ProjectorLog(),
                       observer = nothing)
    t_start = time()
    PR_vec, PL_vec = find_projectors(mps, alg, log, observer)
    _absorb_projectors!(mps, PL_vec, PR_vec)
    log.sweep_times = [time() - t_start]
    log.converged = log.iterations < alg.maxiter
    observer === nothing || observer(log)
    return log
end
