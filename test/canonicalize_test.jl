using Test
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    HalfPeriodicVector,
    assertconsistency,
    gaugefix!,
    dense,
    pair_contraction,
    random_mps,
    one_mps,
    random_mps2,
    one_mps2,
    ProjectorCanonical,
    P_decomp,
    find_projectors
using TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace
import MPSKit

@testset "HalfInt" begin
    h = HalfInt(3)
    @test h.n == 3
    @test (h + 2).n == 5
    @test (h - 1).n == 2
end

@testset "MPSKit gauges — normalized on nonuniform virtual bonds" begin
    P = PeriodicVector(fill(ℂ^2, 4))
    V = HalfPeriodicVector(PeriodicVector([ℂ^2, ℂ^3, ℂ^4, ℂ^3]))
    mps = random_mps(ComplexF64, P, V)
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100)

    L, R = PMPSLab._mpskit_gauges(mps, alg)

    @test L isa HalfPeriodicVector
    @test R isa HalfPeriodicVector
    for i in eachindex(mps)
        pos = HalfInt(i)
        bond = right_virtualspace(mps[i])
        @test codomain(L[pos])[1] == bond
        @test domain(L[pos])[1] == bond
        @test codomain(R[pos])[1] == bond
        @test domain(R[pos])[1] == bond
        @test norm(L[pos]) ≈ 1
        @test norm(R[pos]) ≈ 1
    end
end

@testset "MPSKit gauge adapter — QR/LQ iteration only" begin
    alg = ProjectorCanonical()
    @test !hasproperty(alg, :alg_eigsolve)
    @test !hasproperty(alg, :eig_miniter)
    @test PMPSLab._left_algorithm(alg).eig_miniter == typemax(Int)
    @test PMPSLab._right_algorithm(alg).eig_miniter == typemax(Int)
end

@testset "find_projectors — without input copies" begin
    mps = random_mps(ComplexF64, 4, 2, 3)
    tensor_objects = [mps[i] for i in 1:length(mps)]
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100)
    PR, PL = find_projectors(mps, alg)

    # Input tensors are passed by reference without copying or mutation.
    @test all(mps[i] === tensor_objects[i] for i in 1:length(mps))
    for bond in 1:length(mps)
        pos = HalfInt(bond)
        @test PR[pos] * PL[pos] ≈ id(domain(PR[pos])) atol=1e-10
    end
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

@testset "random_mps2 / one_mps2 — two-site tensors" begin
    mps = random_mps2(ComplexF64, 4, 2, 3)
    @test length(mps) == 4
    @test assertconsistency(mps) === nothing
    @test scalartype(mps[1]) == ComplexF64
    @test numout(mps[1]) == 3
    @test codomain(mps[1])[1] == ℂ^3           # virtual ℂ^D
    @test codomain(mps[1])[2] == ℂ^2           # physical ℂ^d (site a)
    @test codomain(mps[1])[3] == ℂ^2           # physical ℂ^d (site b)
    @test domain(mps[1])[1] == ℂ^3

    # deterministic: one_mps2 is the uniform all-ones two-site MPS
    mps_a = one_mps2(ComplexF64, 3, 2, 2)
    mps_b = one_mps2(ComplexF64, 3, 2, 2)
    @test all(mps_a[i] ≈ mps_b[i] for i in 1:3)
    @test assertconsistency(one_mps2(Float64, 4, 2, 2)) === nothing
    @test scalartype(one_mps2(Float64, 4, 2, 2)[1]) == Float64

    # graded spaces
    V = Z2Space(0 => 2, 1 => 2)
    P = Z2Space(0 => 2)
    mps_g = random_mps2(ComplexF64, 3, P, V)
    @test length(mps_g) == 3
    @test assertconsistency(mps_g) === nothing
    @test assertconsistency(one_mps2(ComplexF64, 3, P, V)) === nothing
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

@testset "rank-deficient projectors compress the unsupported virtual subspace" begin
    mps = one_mps(ComplexF64, 4, 2, 3)
    dense_before = dense(mps)
    PR, PL = find_projectors(mps, ProjectorCanonical(; tol=1e-12, maxiter=100))

    for bond in 1:length(mps)
        pos = HalfInt(bond)
        support_projector = PR[pos] * PL[pos]
        @test support_projector * support_projector ≈ support_projector atol=1e-10
        @test !(support_projector ≈ id(domain(PR[pos])))
    end

    gaugefix!(mps, ProjectorCanonical(; tol=1e-12, maxiter=100))
    @test dense(mps) ≈ dense_before atol=1e-10
end

@testset "gaugefix! — integration" begin
    N, d, D = 4, 2, 3
    mps = random_mps(ComplexF64, N, d, D)
    tvec = [copy(mps[i]) for i in 1:N]
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    @test gaugefix!(mps, alg) === mps
    # After canonicalization, consistency should still hold
    @test assertconsistency(mps) === nothing
    # Tensors should have been modified
    for i in 1:N
        @test !(mps[i] ≈ tvec[i])
    end
end

@testset "gaugefix! — Z2Irrep" begin
    N = 4
    V = Z2Space(0 => 3, 1 => 2)
    P = Z2Space(0 => 2)
    mps = random_mps(ComplexF64, N, P, V)
    tvec = [copy(mps[i]) for i in 1:N]
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    @test gaugefix!(mps, alg) === mps
    @test assertconsistency(mps) === nothing
    for i in 1:N
        @test !(mps[i] ≈ tvec[i])
    end
end

@testset "gaugefix! — self-consistency" begin
    N, d, D = 4, 2, 2
    mps = random_mps(ComplexF64, N, d, D)
    dense_before = dense(mps)
    alg = ProjectorCanonical(; tol=1e-12, maxiter=100, trscheme=MPSKit.TruncationStrategy())
    gaugefix!(mps, alg)
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
        mps = random_mps2(ComplexF64, 1, d, D)
        T = mps[1]
        result = dense(mps)
        @test length(codomain(result)) == 2
        @plansor manual[-1 -2; ()] := T[1 -1 -2; 1]
        @test norm(result - manual) < 1e-12
    end

    @testset "N=2 vs manual @plansor" begin
        D, d = 2, 2
        mps = random_mps2(ComplexF64, 2, d, D)
        result = dense(mps)
        # dense leg order: P_{1,a}, P_{1,b}, P_{2,a}, P_{2,b}
        @plansor manual[-1 -2 -3 -4; ()] := mps[1][circ -1 -2; mid] *
                                            mps[2][mid -3 -4; circ]
        @test norm(result - manual) < 1e-12
    end

    @testset "Z2Irrep N=2 vs manual @plansor" begin
        V = Z2Space(0 => 3, 1 => 2)
        P = Z2Space(0 => 2)
        mps = random_mps2(ComplexF64, 2, P, V)
        result = dense(mps)
        @plansor manual[-1 -2 -3 -4; ()] := mps[1][circ -1 -2; mid] *
                                            mps[2][mid -3 -4; circ]
        @test norm(result - manual) < 1e-12
    end
end
