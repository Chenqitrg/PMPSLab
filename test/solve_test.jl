# Tests for src/solve.jl — block-wise linear solvers
# Each solver is tested against the naive backslash on a dense matrix.

using Test
using PeriodicMPS: LUSolve, LDLTSolve, CholeskySolve, KrylovSolve, solve
using TensorKit
using LinearAlgebra: Hermitian, I
using KrylovKit: GMRES

# Build a Hermitian positive-definite Gram matrix N = tN(left, right) in two
# ways: (a) trivial bond D=1 (blocks are 1×1), (b) graded space with D>1.
@testset "LUSolve" begin
    # Dense: random PD matrix
    M = randn(ComplexF64, 4, 4)
    A = TensorMap(M + M', ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)   # Hermitian PSD
    b = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^1)

    x = solve(LUSolve(), A, b)
    @test x ≈ A \ b
end

@testset "LDLTSolve" begin
    M = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
    A = M + M'
    b = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^1)

    x_lu = A \ b
    x_ldlt = solve(LDLTSolve(), A, b)
    @test x_ldlt ≈ x_lu rtol=1e-10
end

@testset "CholeskySolve" begin
    M = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2)
    A = M * M' + id(ℂ^2 ⊗ ℂ^2)   # strictly PD
    b = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^1)

    x_lu = A \ b
    x_chol = solve(CholeskySolve(), A, b)
    @test x_chol ≈ x_lu rtol=1e-10
end

@testset "KrylovSolve" begin
    M = randn(Float64, ℝ^2 ⊗ ℝ^2, ℝ^2 ⊗ ℝ^2)
    A = M * M' + id(ℝ^2 ⊗ ℝ^2)
    b = randn(Float64, ℝ^2 ⊗ ℝ^2, ℝ^1)
    x0 = zero(b)

    x_lu = A \ b
    x_kry = solve(KrylovSolve(GMRES(; maxiter=100, tol=1e-12)), A, b, x0)
    @test x_kry ≈ x_lu rtol=1e-10
end

@testset "Graded space (Z2Irrep)" begin
    V = Z2Space(0 => 2, 1 => 1)
    R = Z2Space(0 => 1)                   # trivial domain
    A = rand(ComplexF64, V ← V)
    A = (A + A') / 2 + one(A)             # Hermitian, diagonally shifted → PD
    b = rand(ComplexF64, V ← R)

    x_lu = A \ b
    @test solve(LUSolve(), A, b) ≈ x_lu
    @test solve(LDLTSolve(), A, b) ≈ x_lu rtol=1e-10
end

@testset "solve! via __solve" begin
    # Smoke test: the _solve interface dispatches correctly
    A = TensorMap(randn(ComplexF64, 3, 3) |> M -> M * M' + I,
                  ℂ^1 ⊗ ℂ^3 ← ℂ^1 ⊗ ℂ^3)
    b = TensorMap(randn(ComplexF64, 3, 1), ℂ^1 ⊗ ℂ^3, ℂ^1)

    for solver in (LUSolve(), LDLTSolve(), CholeskySolve())
        x = solve(solver, A, b)
        @test x ≈ A \ b rtol=1e-10
    end
end