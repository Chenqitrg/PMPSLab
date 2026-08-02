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
    gaugefix!(mps::PeriodicMPS, alg::ProjectorCanonical;
                  log = ProjectorLog(), observer = nothing)

Absorb the transfer-matrix fixed-point environments into `mps` by iteratively
computing left/right projectors at every bond and folding them into the site
tensors. Modifies `mps` in place and returns a [`ProjectorLog`](@ref).

# Keyword arguments
- `log::ProjectorLog`: external log for live progress polling.
  Pass a pre-allocated `ProjectorLog` and read `log.sweeps`,
  `log.converged` from another task.
- `observer`: callable `f(logL::PowerIterLog, logR::PowerIterLog)` fired after
  each bond's L/R computation. Use for progress bars or live monitoring.

# Examples
# === Basic usage ===
log = gaugefix!(mps, alg)

# === Live polling from another task ===
log = ProjectorLog()
@async while !log.converged
    println("sweeps: \$(length(log.sweeps))")
    sleep(0.1)
end
gaugefix!(mps, alg; log = log)

# === Observer callback ===
gaugefix!(mps, alg; observer = (logL, logR) -> @info "bond done" logL.pos)
"""
function gaugefix!(mps::PeriodicMPS, alg::ProjectorCanonical;
                       log::ProjectorLog = ProjectorLog(),
                       observer = nothing)
    PR_vec, PL_vec, sweep_log = find_projectors(mps, alg; observer)
    _absorb_projectors!(mps, PL_vec, PR_vec)
    push!(log.sweeps, sweep_log)
    log.converged = all(l -> l.converged, sweep_log.site_logs)
    return log
end
