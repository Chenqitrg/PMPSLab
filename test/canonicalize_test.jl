using Test
using TensorRing
using TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace
import MPSKit

@testset "HalfInt" begin
    h = HalfInt(3)
    @test h.n == 3
    @test (h + 2).n == 5
    @test (h - 1).n == 2
end

@testset "assertconsistency — valid and invalid" begin
    D = 3
    tvec = [TensorMap(randn(ComplexF64, D*2, D), ℂ^D ⊗ ℂ^2, ℂ^D) for _ in 1:4]
    mps = PeriodicMPS(PeriodicVector(tvec))
    @test assertconsistency(mps) === nothing

    # Mismatched: change one site's right virtual space
    tvec_bad = copy(tvec)
    tvec_bad[2] = TensorMap(randn(ComplexF64, D*2, D+1), ℂ^D ⊗ ℂ^2, ℂ^(D+1))
    mps_bad = PeriodicMPS(PeriodicVector(tvec_bad))
    @test_throws TensorKit.SpaceMismatch assertconsistency(mps_bad)
end

@testset "assertconsistency — Z2Irrep" begin
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => 2)
    tvec = [rand(ComplexF64, V ⊗ P ← V) for _ in 1:4]
    mps = PeriodicMPS(PeriodicVector(tvec))
    @test assertconsistency(mps) === nothing
    tvec_bad = copy(tvec)
    tvec_bad[2] = rand(ComplexF64, V ⊗ P ← Z2Space(0 => 3, 1 => 3))
    mps_bad = PeriodicMPS(PeriodicVector(tvec_bad))
    @test_throws TensorKit.SpaceMismatch assertconsistency(mps_bad)
end

@testset "QR_L — 2-site" begin
    Ld, Pd, Rd = 3, 2, 4
    T = TensorMap(randn(ComplexF64, Ld * Pd, Rd), ℂ^Ld ⊗ ℂ^Pd, ℂ^Rd)
    L = TensorMap(randn(ComplexF64, Ld, Ld), ℂ^Ld, ℂ^Ld)
    new_L = QR_L(L, T)
    @test numout(new_L) == 1
    @test numin(new_L) == 1
    @test dim(domain(new_L)) == Rd
end

@testset "QR_L — 3-site" begin
    Ld, Pd1, Pd2, Rd = 3, 2, 2, 4
    T3 = TensorMap(randn(ComplexF64, Ld * Pd1 * Pd2, Rd), ℂ^Ld ⊗ ℂ^Pd1 ⊗ ℂ^Pd2, ℂ^Rd)
    L = TensorMap(randn(ComplexF64, Ld, Ld), ℂ^Ld, ℂ^Ld)
    new_L = QR_L(L, T3)
    @test numout(new_L) == 1
    @test numin(new_L) == 1
end

@testset "QR_L — 3-site Z2Irrep" begin
    V  = Z2Space(0 => 3, 1 => 2)
    P1 = Z2Space(0 => 2)
    P2 = Z2Space(0 => 1, 1 => 1)
    T3 = rand(ComplexF64, V ⊗ P1 ⊗ P2 ← V)
    L  = rand(ComplexF64, V ← V)
    new_L = QR_L(L, T3)
    @test numout(new_L) == 1 && numin(new_L) == 1
end

@testset "QR_R — 2-site" begin
    Ld, Pd, Rd = 3, 2, 4
    T = TensorMap(randn(ComplexF64, Ld * Pd, Rd), ℂ^Ld ⊗ ℂ^Pd, ℂ^Rd)
    R = TensorMap(randn(ComplexF64, Rd, Rd), ℂ^Rd, ℂ^Rd)
    new_R = QR_R(R, T)
    @test numout(new_R) == 1
    @test numin(new_R) == 1
end

@testset "QR_R — 3-site Z2Irrep" begin
    V  = Z2Space(0 => 3, 1 => 2)
    P1 = Z2Space(0 => 2)
    P2 = Z2Space(0 => 1, 1 => 1)
    T3 = rand(ComplexF64, V ⊗ P1 ⊗ P2 ← V)
    R  = rand(ComplexF64, V ← V)
    new_R = QR_R(R, T3)
    @test numout(new_R) == 1 && numin(new_R) == 1
end

@testset "P_decomp — PR * PL ≈ I (ComplexF64)" begin
    D = 4
    L = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    R = TensorMap(randn(ComplexF64, D, D), ℂ^D, ℂ^D)
    PR, PL = P_decomp(R, L, MPSKit.TruncationStrategy())
    @plansor recon[-1; -2] := PR[-1; 1] * PL[1; -2]
    @test norm(recon - id(domain(R))) < 1e-10
end

@testset "P_decomp — PR * PL ≈ I (Float64)" begin
    D = 4
    L = TensorMap(randn(Float64, D, D), ℝ^D, ℝ^D)
    R = TensorMap(randn(Float64, D, D), ℝ^D, ℝ^D)
    PR, PL = P_decomp(R, L, MPSKit.TruncationStrategy())
    @plansor recon[-1; -2] := PR[-1; 1] * PL[1; -2]
    @test norm(recon - id(domain(R))) < 1e-10
end

@testset "P_decomp — PR * PL ≈ I (Z2Irrep)" begin
    V = Z2Space(0 => 2, 1 => 2)
    L = rand(ComplexF64, V ← V)
    R = rand(ComplexF64, V ← V)
    PR, PL = P_decomp(R, L, MPSKit.TruncationStrategy())
    @plansor recon[-1; -2] := PR[-1; 1] * PL[1; -2]
    @test norm(recon - id(domain(R))) < 1e-10
end

@testset "QR_L / QR_R — Z2Irrep" begin
    V_left  = Z2Space(0 => 3, 1 => 2)
    V_phys  = Z2Space(0 => 2)
    V_right = Z2Space(0 => 3, 1 => 2)
    T     = rand(ComplexF64, V_left ⊗ V_phys ← V_right)
    L_env = rand(ComplexF64, V_left ← V_left)
    R_env = rand(ComplexF64, V_right ← V_right)
    new_L = QR_L(L_env, T)
    new_R = QR_R(R_env, T)
    @test numout(new_L) == 1 && numin(new_L) == 1
    @test numout(new_R) == 1 && numin(new_R) == 1
end

@testset "find_L / find_R convergence" begin
    N, d, D = 4, 2, 3
    tvec = [TensorMap(randn(ComplexF64, D * d, D), ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    mps = PeriodicMPS(PeriodicVector(tvec))

    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())

    pos = HalfInt(2)
    log = ProjectorLog()
    L = find_L(pos, mps, alg, log)
    @test numout(L) == 1
    @test numin(L) == 1

    R = find_R(pos, mps, alg, log)
    @test numout(R) == 1
    @test numin(R) == 1
end

@testset "find_L / find_R convergence — Z2Irrep" begin
    N, d = 4, 2
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => d)
    tvec = [rand(ComplexF64, V ⊗ P ← V) for _ in 1:N]
    mps = PeriodicMPS(PeriodicVector(tvec))

    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())

    pos = HalfInt(2)
    log = ProjectorLog()
    L = find_L(pos, mps, alg, log)
    R = find_R(pos, mps, alg, log)
    @test numout(L) == 1 && numin(L) == 1
    @test numout(R) == 1 && numin(R) == 1
end

@testset "canonicalize! — integration" begin
    N, d, D = 4, 2, 3
    tvec = [TensorMap(randn(ComplexF64, D * d, D), ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    mps = PeriodicMPS(PeriodicVector(copy(tvec)))
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    log = canonicalize!(mps, alg)
    @test log.converged
    @test length(log.sweep_times) == 1
    # After canonicalization, consistency should still hold
    @test assertconsistency(mps) === nothing
    # Tensors should have been modified
    for i in 1:N
        @test !(mps[i] ≈ tvec[i])
    end
end

@testset "canonicalize! — Z2Irrep" begin
    N = 4
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => 2)
    tvec = [rand(ComplexF64, V ⊗ P ← V) for _ in 1:N]
    mps = PeriodicMPS(PeriodicVector(copy(tvec)))
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    log = canonicalize!(mps, alg)
    @test log.converged
    @test assertconsistency(mps) === nothing
    for i in 1:N
        @test !(mps[i] ≈ tvec[i])
    end
end

@testset "canonicalize! — self-consistency" begin
    N, d, D = 4, 2, 2
    tvec = [TensorMap(randn(ComplexF64, D * d, D), ℂ^D ⊗ ℂ^d, ℂ^D) for _ in 1:N]
    mps = PeriodicMPS(PeriodicVector(copy(tvec)))
    dense_before = dense(mps)
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    canonicalize!(mps, alg)
    dense_after = dense(mps)
    @test norm(dense_before - dense_after) < 1e-10
end

