# ------------------------------------------------------------------------------
# Projector-based canonicalization
# ------------------------------------------------------------------------------

abstract type AbstractLoopCanonical <: Algorithm end

@kwdef struct ProjectorCanonical <: AbstractLoopCanonical
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trscheme::TruncationStrategy = TruncationStrategy()
end

# ------------------------------------------------------------------------------
# QR step: apply one transfer matrix and re-orthogonalize
# ------------------------------------------------------------------------------

function QR_L(L::AbstractTensorMap{E,S,1,1}, T::AbstractTensorMap{E,S,2,1}) where {E,S}
    @plansor LT[-2 -3; -1] := L[-2; 2] * T[2 -3; -1]
    new_T, new_L = left_orth(LT)
    return new_L
end

function QR_L(L::AbstractTensorMap{E,S,1,1}, T::AbstractTensorMap{E,S,3,1}) where {E,S}
    @plansor LT[-2 -3 -4; -1] := L[-2; 2] * T[2 -3 -4; -1]
    new_T, new_L = left_orth(LT)
    return new_L
end

function QR_R(R::AbstractTensorMap{E,S,1,1}, T::AbstractTensorMap{E,S,2,1}) where {E,S}
    TR_folded = transpose(T * R, ((1,), (3, 2)))
    new_R, new_T = right_orth(TR_folded)
    return new_R
end

function QR_R(R::AbstractTensorMap{E,S,1,1}, T::AbstractTensorMap{E,S,3,1}) where {E,S}
    TR_folded = transpose(T * R, ((1,), (4, 3, 2)))
    new_R, new_T = right_orth(TR_folded)
    return new_R
end

# ------------------------------------------------------------------------------
# Power iteration: find dominant left/right environments
# ------------------------------------------------------------------------------

function find_L(pos::HalfInt, mps::PeriodicMPS{<:AbstractTensorMap{E,S}}, alg::ProjectorCanonical) where {E,S}
    L = id(E, mps[pos])
    steps = 0
    err = Inf
    N = length(mps)
    pos_errors = Float64[]
    while steps < alg.maxiter && abs(err) > alg.tol
        new_L = copy(L)
        for i in (pos.n + 1):(pos.n + N)
            new_L = QR_L(new_L, mps[i])
        end
        new_L = new_L / norm(new_L, Inf)
        if space(new_L) == space(L)
            err = norm(new_L - L)
        end
        push!(pos_errors, err)
        L = new_L
        steps += 1
    end
    converged = steps < alg.maxiter
    return L, PowerIterLog(pos, pos_errors, steps, converged)
end

function find_R(pos::HalfInt, mps::PeriodicMPS{<:AbstractTensorMap{E,S}}, alg::ProjectorCanonical) where {E,S}
    R = id(E, mps[pos])
    steps = 0
    err = Inf
    N = length(mps)
    pos_errors = Float64[]
    while steps < alg.maxiter && abs(err) > alg.tol
        new_R = copy(R)
        for i in (pos.n):-1:(pos.n - N + 1)
            new_R = QR_R(new_R, mps[i])
        end
        new_R = new_R / norm(new_R, Inf)
        if space(new_R) == space(R)
            err = norm(new_R - R)
        end
        push!(pos_errors, err)
        R = new_R
        steps += 1
    end
    converged = steps < alg.maxiter
    return R, PowerIterLog(pos, pos_errors, steps, converged)
end

# ------------------------------------------------------------------------------
# Projector decomposition
# ------------------------------------------------------------------------------

"""
    P_decomp(R::TensorMap, L::TensorMap, trunc::TruncationStrategy) -> (PR, PL)

Decompose `L * R` via truncated SVD into right and left projectors satisfying
`PR * PL ≈ I`. Small singular values are suppressed via [`pseudopow`](@ref).

Used in entanglement filtering: `PL` and `PR` are inserted on the left and right side
of a bond tensor to absorb the transfer matrix fixed-point environment.
"""
function P_decomp(R::AbstractTensorMap{E,A,1,1}, L::AbstractTensorMap{E,A,1,1}, trunc::TruncationStrategy) where {E,A}
    U, S, Vh, _ = svd_trunc(L * R; trunc=trunc)
    re_sq = pseudopow(S, -0.5)
    PR = R * Vh' * re_sq       # Vh = Vᴴ, so adjoint(V) = Vh
    PL = re_sq * U' * L
    return PR, PL
end

# ------------------------------------------------------------------------------
# Full projector computation at every site
# ------------------------------------------------------------------------------

function find_projectors(mps::PeriodicMPS, alg::ProjectorCanonical; observer = nothing)
    t_start = time()
    N = length(mps)
    L1, logL1 = find_L(HalfInt(1), mps, alg)
    R1, logR1 = find_R(HalfInt(1), mps, alg)
    PR0, PL0 = P_decomp(R1, L1, alg.trscheme)
    PR_vec = Vector{typeof(PR0)}(undef, N)
    PL_vec = Vector{typeof(PL0)}(undef, N)
    PR = PeriodicVector(PR_vec)
    PL = PeriodicVector(PL_vec)
    PR[1], PL[2] = PR0, PL0
    site_logs = PowerIterLog[logL1, logR1]
    observer === nothing || observer(logL1, logR1)
    for i in 2:N
        L, logL = find_L(HalfInt(i), mps, alg)
        R, logR = find_R(HalfInt(i), mps, alg)
        PR[i], PL[i + 1] = P_decomp(R, L, alg.trscheme)
        push!(site_logs, logL, logR)
        observer === nothing || observer(logL, logR)
    end
    sweep_log = SweepLog(site_logs, time() - t_start)
    return PR, PL, sweep_log
end
