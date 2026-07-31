using Test
using PeriodicMPS:
    PeriodicMPS,
    HalfInt,
    assertconsistency,
    canonicalize!,
    dense,
    pair_contraction,
    random_mps,
    one_mps,
    ProjectorCanonical,
    ProjectorLog,
    PowerIterLog,
    SweepLog,
    QR_L,
    QR_R,
    P_decomp,
    find_L,
    find_R
using TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace
import MPSKit

@testset "HalfInt" begin
    h = HalfInt(3)
    @test h.n == 3
    @test (h + 2).n == 5
    @test (h - 1).n == 2
end

@testset "random_mps / one_mps" begin
    mps = random_mps(ComplexF64, 4, 2, 3)
    @test length(mps) == 4
    @test assertconsistency(mps) === nothing
    @test scalartype(mps[1]) == ComplexF64
    @test codomain(mps[1])[1] == ℂ^3           # virtual ℂ^D
    @test codomain(mps[1])[2] == ℂ^2           # physical ℂ^d
    @test domain(mps[1])[1] == ℂ^3

    # deterministic: one_mps is the uniform all-ones MPS
    mps_a = one_mps(ComplexF64, 3, 2, 2)
    mps_b = one_mps(ComplexF64, 3, 2, 2)
    @test all(mps_a[i] ≈ mps_b[i] for i in 1:3)
    @test assertconsistency(one_mps(Float64, 4, 2, 2)) === nothing
    @test scalartype(one_mps(Float64, 4, 2, 2)[1]) == Float64

    # graded spaces
    V = Z2Space(0 => 2, 1 => 2)
    P = Z2Space(0 => 2)
    mps_g = random_mps(ComplexF64, 3, P, V)
    @test length(mps_g) == 3
    @test assertconsistency(mps_g) === nothing
    @test assertconsistency(one_mps(ComplexF64, 3, P, V)) === nothing
end

@testset "assertconsistency — valid and invalid" begin
    D = 3
    mps = random_mps(ComplexF64, 4, 2, D)
    tvec = [copy(mps[i]) for i in 1:4]
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
    mps = random_mps(ComplexF64, 4, P, V)
    tvec = [copy(mps[i]) for i in 1:4]
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
    mps = random_mps(ComplexF64, N, d, D)

    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())

    pos = HalfInt(2)
    L, logL = find_L(pos, mps, alg)
    @test numout(L) == 1
    @test numin(L) == 1
    @test logL isa PowerIterLog

    R, logR = find_R(pos, mps, alg)
    @test numout(R) == 1
    @test numin(R) == 1
    @test logR isa PowerIterLog
end

@testset "find_L / find_R convergence — Z2Irrep" begin
    N, d = 4, 2
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => d)
    mps = random_mps(ComplexF64, N, P, V)

    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())

    pos = HalfInt(2)
    L, _ = find_L(pos, mps, alg)
    R, _ = find_R(pos, mps, alg)
    @test numout(L) == 1 && numin(L) == 1
    @test numout(R) == 1 && numin(R) == 1
end

@testset "canonicalize! — integration" begin
    N, d, D = 4, 2, 3
    mps = random_mps(ComplexF64, N, d, D)
    tvec = [copy(mps[i]) for i in 1:N]
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    log = canonicalize!(mps, alg)
    @test log.converged
    @test length(log.sweeps) == 1
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
    mps = random_mps(ComplexF64, N, P, V)
    tvec = [copy(mps[i]) for i in 1:N]
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
    mps = random_mps(ComplexF64, N, d, D)
    dense_before = dense(mps)
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    canonicalize!(mps, alg)
    dense_after = dense(mps)
    @test norm(dense_before - dense_after) < 1e-10
end

# ==============================================================================
# dense — comprehensive tests
# ==============================================================================

@testset "dense — OneSiteMPS" begin
    @testset "N=1 (partial trace)" begin
        D, d = 2, 3
        mps = random_mps(ComplexF64, 1, d, D)
        T = mps[1]
        result = dense(mps)
        @test codomain(result)[1] == ℂ^d
        @plansor manual[-1; ()] := T[1 -1; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "N=2 vs manual @plansor" begin
        D, d = 2, 2
        mps = random_mps(ComplexF64, 2, d, D)
        result = dense(mps)
        @plansor manual[-1 -2; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "N=3 vs manual @plansor" begin
        D, d = 2, 2
        mps = random_mps(ComplexF64, 3, d, D)
        result = dense(mps)
        @plansor manual[-1 -2 -3; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 3] *
                                        mps[3][3 -3; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "Float64 element type" begin
        D, d = 2, 2
        mps = random_mps(Float64, 2, d, D)
        result = dense(mps)
        @plansor manual[-1 -2; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "Z2Irrep N=3 vs manual @plansor" begin
        V = Z2Space(0 => 2, 1 => 2)
        P = Z2Space(0 => 2)
        mps = random_mps(ComplexF64, 3, P, V)
        result = dense(mps)
        @plansor manual[-1 -2 -3; ()] := mps[1][1 -1; 2] * mps[2][2 -2; 3] *
                                        mps[3][3 -3; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "D=1 trivial bond" begin
        d = 3
        mps = random_mps(ComplexF64, 3, d, 1)
        result = dense(mps)
        @test codomain(result) == (ℂ^d ⊗ ℂ^d ⊗ ℂ^d)
    end
end

# ==============================================================================
# pair_contraction — adjacent-pair coarse graining
# ==============================================================================

@testset "pair_contraction" begin
    @testset "structure — N=4 gives N/2 two-site tensors" begin
        D, d, N = 2, 2, 4
        mps = random_mps(ComplexF64, N, d, D)
        paired = pair_contraction(mps)

        @test length(paired) == N ÷ 2
        for T in paired
            @test length(codomain(T)) == 3          # L_{2k-1} ⊗ P_{2k-1} ⊗ P_{2k}
            @test codomain(T)[1] == ℂ^D             # left virtual leg survives
            @test codomain(T)[2] == ℂ^d             # P_{2k-1}
            @test codomain(T)[3] == ℂ^d             # P_{2k}
            @test length(domain(T)) == 1
            @test domain(T)[1] == ℂ^D               # right virtual leg
        end
    end

    @testset "pair matches manual @plansor" begin
        D, d = 2, 2
        mps = random_mps(ComplexF64, 2, d, D)
        paired = pair_contraction(mps)

        @test length(paired) == 1
        @plansor manual[-1 -2 -3; -4] := mps[1][-1 -2; 1] * mps[2][1 -3; -4]
        @test paired[1] ≈ manual
    end

    @testset "dense invariant — N ∈ {2, 4, 6}" begin
        D, d = 2, 2
        for N in [2, 4, 6]
            mps = random_mps(ComplexF64, N, d, D)
            # Same physical leg order as dense(mps) — the loop product is preserved.
            @test dense(pair_contraction(mps)) ≈ dense(mps)
        end
    end

    @testset "odd number of sites throws" begin
        D, d = 2, 2
        mps = random_mps(ComplexF64, 3, d, D)
        @test_throws ArgumentError pair_contraction(mps)
    end

    @testset "Z2Irrep — dense invariant and structure" begin
        N = 4
        V = Z2Space(0 => 2, 1 => 2)
        P = Z2Space(0 => 2)
        mps = random_mps(ComplexF64, N, P, V)
        paired = pair_contraction(mps)

        @test length(paired) == 2
        @test codomain(paired[1]) == V ⊗ P ⊗ P
        @test dense(paired) ≈ dense(mps)
    end

    @testset "Float64 element type" begin
        D, d, N = 2, 2, 4
        mps = random_mps(Float64, N, d, D)
        paired = pair_contraction(mps)
        @test scalartype(paired[1]) == Float64
        @test dense(paired) ≈ dense(mps)
    end
end

@testset "dense — TwoSiteMPS" begin
    @testset "N=1 (partial trace)" begin
        D, d = 2, 2
        T = TensorMap(randn(ComplexF64, D * d * d, D), ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
        mps = PeriodicMPS(PeriodicVector([T]))
        result = dense(mps)
        @test length(codomain(result)) == 2
        @plansor manual[-1 -2; ()] := T[1 -1 -2; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "N=2 vs manual @plansor" begin
        D, d = 2, 2
        tvec = [TensorMap(randn(ComplexF64, D * d * d, D), ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D) for _ in 1:2]
        mps = PeriodicMPS(PeriodicVector(tvec))
        result = dense(mps)
        # dense leg order: P_{1,a}, P_{1,b}, P_{2,a}, P_{2,b}
        @plansor manual[-1 -2 -3 -4; ()] := tvec[1][circ -1 -2; mid] *
                                            tvec[2][mid -3 -4; circ]
        @test norm(result - manual) < 1e-12
    end

    @testset "Z2Irrep N=2 vs manual @plansor" begin
        V = Z2Space(0 => 3, 1 => 2)
        P = Z2Space(0 => 2)
        tvec = [rand(ComplexF64, V ⊗ P ⊗ P ← V) for _ in 1:2]
        mps = PeriodicMPS(PeriodicVector(tvec))
        result = dense(mps)
        @plansor manual[-1 -2 -3 -4; ()] := tvec[1][circ -1 -2; mid] *
                                            tvec[2][mid -3 -4; circ]
        @test norm(result - manual) < 1e-12
    end
end

