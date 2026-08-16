# ------------------------------------------------------------------------------
# Linear solvers for per-site update: solve N * ψ = W
#
# N = tN(left, right) is a Gram matrix — Hermitian positive semi-definite.
# Three direct solvers dispatched by type, plus a Krylov wrapper.
#
# Block-wise implementation follows TNRKit/src/utility/hermitian_tensor.jl
# ------------------------------------------------------------------------------

# Pseudo-power: suppresses small singular values to avoid 1/0 blowup
function pseudopow(t::DiagonalTensorMap, a::Real; tol = eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (_, b) in blocks(t′)
        @inbounds for i in 1:size(b, 1)
            b.diag[i] = abs(b.diag[i]) < tol ? b.diag[i] : b.diag[i]^a
        end
    end
    return t′
end

abstract type DenseSolve <: AbstractSolve end

struct LUSolve <: DenseSolve end
struct LDLTSolve <: DenseSolve end
struct CholeskySolve <: DenseSolve end

"""
    KrylovSolve(; algorithm)

Use the supplied KrylovKit linear-solve algorithm for local split updates.
"""
@kwdef struct KrylovSolve{A} <: AbstractSolve
    algorithm::A   # e.g. GMRES(; maxiter=250, krylovdim=128, tol=1.0e-10)
end

# ---------------------------------------------------------------------------
# Block-wise direct solvers
# ---------------------------------------------------------------------------

function _block_ldiv(f, A::AbstractTensorMap, b::AbstractTensorMap)
    codomain(A) == codomain(b) ||
        throw(SpaceMismatch("non-matching codomains"))
    T = float(promote_type(scalartype(A), scalartype(b)))
    x = similar(b, T, domain(A) ← domain(b))
    for (c, block_b) in blocks(x)
        copyto!(block_b, f(block(A, c), block(b, c)))
    end
    return x
end

ldiv(::LUSolve, A::AbstractTensorMap, b::AbstractTensorMap) = A \ b
ldiv(::LUSolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap) = x + A \ (b - A * x)

function ldiv(::LDLTSolve, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U)
    return _block_ldiv(A, b) do M, R
        Hermitian(M, uplo) \ R
    end
end

ldiv(::LDLTSolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U) = x + ldiv(LDLTSolve(), A, b - A * x; uplo=uplo)

function ldiv(::CholeskySolve, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U)
    return _block_ldiv(A, b) do M, R
        cholesky(Hermitian(M, uplo)) \ R
    end
end

ldiv(::CholeskySolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U) = x + ldiv(CholeskySolve(), A, b - A * x; uplo=uplo)

# ---------------------------------------------------------------------------
# Krylov solver — delegates to KrylovKit.linsolve
# ---------------------------------------------------------------------------

function ldiv(solver::KrylovSolve, x0::AbstractTensorMap, A::AbstractTensorMap,
              b::AbstractTensorMap)
    x, info = linsolve(x -> A * x, b, x0, solver.algorithm)
    iszero(info.converged) && @warn "Krylov linsolve did not converge."
    return x
end

# ---------------------------------------------------------------------------
# Right-division: solve x * A = b  →  x = b / A
# ---------------------------------------------------------------------------

function _block_rdiv(f, A::AbstractTensorMap, b::AbstractTensorMap)
    domain(A) == domain(b) ||
        throw(SpaceMismatch("non-matching domains"))
    T = float(promote_type(scalartype(A), scalartype(b)))
    x = similar(b, T, codomain(b) ← codomain(A))
    for (c, block_x) in blocks(x)
        copyto!(block_x, f(block(b, c), block(A, c)))
    end
    return x
end

rdiv(::LUSolve, A::AbstractTensorMap, b::AbstractTensorMap) = b / A
rdiv(::LUSolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap) =
    x + (b - x * A) / A

function rdiv(::LDLTSolve, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U)
    return _block_rdiv(A, b) do R, M
        R / Hermitian(M, uplo)
    end
end

rdiv(::LDLTSolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U) =
    x + rdiv(LDLTSolve(), A, b - x * A; uplo=uplo)

function rdiv(::CholeskySolve, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U)
    return _block_rdiv(A, b) do R, M
        R / cholesky(Hermitian(M, uplo))
    end
end

rdiv(::CholeskySolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U) =
    x + rdiv(CholeskySolve(), A, b - x * A; uplo=uplo)

function rdiv(
        solver::KrylovSolve,
        x0::AbstractTensorMap,
        action::Function,
        b::AbstractTensorMap,
    )
    x, info = linsolve(action, b, x0, solver.algorithm)
    iszero(info.converged) && @warn "Krylov linsolve did not converge."
    return x
end

function rdiv(
        solver::KrylovSolve,
        x0::AbstractTensorMap,
        A::AbstractTensorMap,
        b::AbstractTensorMap,
    )
    return rdiv(solver, x0, x -> x * A, b)
end
