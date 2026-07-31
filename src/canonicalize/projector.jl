# Projector-based canonicalization: TNRKit's find_L/find_R/P_decomp approach.
# Power-iterate around the loop with rank-3 QR orthogonalization,
# then absorb projectors into site tensors.

# Pseudo-power: suppresses small singular values to avoid 1/0 blowup
function pseudopow(t::DiagonalTensorMap, a::Real; tol = eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (c, b) in blocks(t′)
        @inbounds for i in 1:size(b, 1)
            b.diag[i] = abs(b.diag[i]) < tol ? b.diag[i] : b.diag[i]^a
        end
    end
    return t′
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

function find_L(pos::HalfInt, mps::PeriodicMPS{<:AbstractTensorMap{E,S}}, alg::ProjectorCanonical, log::ProjectorLog) where {E,S}
    L = id(E, right_virtualspace(mps[pos.n]))
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
    log.error_history[pos] = pos_errors
    log.iterations = max(log.iterations, steps)
    return L
end

function find_R(pos::HalfInt, mps::PeriodicMPS{<:AbstractTensorMap{E,S}}, alg::ProjectorCanonical, log::ProjectorLog) where {E,S}
    R = id(E, right_virtualspace(mps[pos.n]))
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
    log.error_history[pos] = pos_errors
    log.iterations = max(log.iterations, steps)
    return R
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

function find_projectors(mps::PeriodicMPS, alg::ProjectorCanonical,
                         log::ProjectorLog, observer = nothing)
    N = length(mps)
    L1 = find_L(HalfInt(1), mps, alg, log)
    R1 = find_R(HalfInt(1), mps, alg, log)
    PR0, PL0 = P_decomp(R1, L1, alg.trscheme)
    PR_vec = Vector{typeof(PR0)}(undef, N)
    PL_vec = Vector{typeof(PL0)}(undef, N)
    PR = PeriodicVector(PR_vec)
    PL = PeriodicVector(PL_vec)
    PR[1], PL[2] = PR0, PL0
    observer === nothing || observer(log)
    for i in 2:N
        L = find_L(HalfInt(i), mps, alg, log)
        R = find_R(HalfInt(i), mps, alg, log)
        PR[i], PL[i + 1] = P_decomp(R, L, alg.trscheme)
        observer === nothing || observer(log)
    end
    return PR, PL
end
