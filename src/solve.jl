# ------------------------------------------------------------------------------
# Linear solvers for per-site update: solve N * ψ = W
#
# N = tN(left, right) is a Gram matrix — Hermitian positive semi-definite.
# Three direct solvers dispatched by type, plus a Krylov wrapper.
#
# Block-wise implementation follows TNRKit/src/utility/hermitian_tensor.jl
# ------------------------------------------------------------------------------

abstract type DenseSolve <: AbstractSolve end

struct LUSolve <: DenseSolve end
struct LDLTSolve <: DenseSolve end
struct CholeskySolve <: DenseSolve end

struct KrylovSolve{A} <: AbstractSolve
    alg::A   # e.g. GMRES(; maxiter=250, krylovdim=128, tol=1.0e-10)
end

# ---------------------------------------------------------------------------
# Block-wise direct solvers
# ---------------------------------------------------------------------------

function _block_solve(f, A::AbstractTensorMap, b::AbstractTensorMap)
    codomain(A) == codomain(b) ||
        throw(SpaceMismatch("non-matching codomains"))
    T = float(promote_type(scalartype(A), scalartype(b)))
    x = similar(b, T, domain(A) ← domain(b))
    for (c, block_b) in blocks(x)
        copy!(block_b, f(block(A, c), block(b, c)))
    end
    return x
end

solve(::LUSolve, A::AbstractTensorMap, b::AbstractTensorMap) = A \ b
solve(::LUSolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap) = x + A \ (b - A * x)

function solve(::LDLTSolve, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U)
    return _block_solve(A, b) do M, R
        Hermitian(M, uplo) \ R
    end
end

solve(::LDLTSolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U) = x + solve(LDLTSolve(), A, b - A * x; uplo=uplo)

function solve(::CholeskySolve, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U)
    return _block_solve(A, b) do M, R
        cholesky(Hermitian(M, uplo)) \ R
    end
end

solve(::CholeskySolve, x::AbstractTensorMap, A::AbstractTensorMap, b::AbstractTensorMap; uplo::Symbol = :U) = x + solve(CholeskySolve(), A, b - A * x; uplo=uplo)

# ---------------------------------------------------------------------------
# Krylov solver — delegates to KrylovKit.linsolve
# ---------------------------------------------------------------------------

function solve(sol::KrylovSolve, A::AbstractTensorMap, b::AbstractTensorMap,
               x0::AbstractTensorMap)
    x, info = linsolve(x -> A * x, b, x0, sol.alg)
    info.converged == 0 && @warn "Krylov linsolve did not converge."
    return x
end
