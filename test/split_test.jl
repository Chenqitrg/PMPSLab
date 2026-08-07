# Tests for src/split/ — 2-site → 1-site splitting: ansatz, SplitContext,
# the per-site update rule and the full split run.

using Test
using PMPSLab:
    PeriodicMPS,
    HalfInt,
    random_mps,
    one_mps,
    random_mps2,
    pair_contraction,
    dense,
    assertconsistency,
    scalartype,
    SplitContext,
    split,
    update!,
    ZigzagSweep,
    CircularSweep,
    SweepState,
    LoopTransferMatrix,
    LUSolve,
    CholeskySolve
import PMPSLab: LoopSplit, RandomAnsatz, OneAnsatz, SVDAnsatz, OneLoopAnsatz,
                initial_approximation, split_site, diff_norm,
                site, this_B, other_B, this_A, pa_from_pb, expectation,
                ProjectorCanonical, gaugefix!
using TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace,
              physicalspace, TruncationStrategy

# ==============================================================================
# Ansätze
# ==============================================================================

@testset "RandomAnsatz / OneAnsatz — construction and initial approximation" begin
    N = 2                      # 2-site MPS with N sites → 2N one-site tensors
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = random_mps2(ComplexF64, N, P, V)

    for ansatz in (RandomAnsatz(V, 2N), OneAnsatz(V, 2N))
        @test length(ansatz.virtual_space) == 2N
        @test ansatz.virtual_space[HalfInt(1)] == V

        mpsB = initial_approximation(mpsA, ansatz)
        @test length(mpsB) == 2N
        @test assertconsistency(mpsB) === nothing
        @test scalartype(mpsB[1]) == ComplexF64
        # every bond carries the ansatz virtual space
        for i in 1:2N
            @test left_virtualspace(mpsB[i]) == V
        end
        # physical spaces are inherited from the two-site tensors, flattened
        @test physicalspace(mpsB[1]) == P
        @test physicalspace(mpsB[2]) == P
    end

    # length mismatch is rejected
    @test_throws AssertionError initial_approximation(mpsA, RandomAnsatz(V, 2N + 1))
end

@testset "split_site — SVD split of one 2-site tensor" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    trunc = TruncationStrategy()
    B1, B2 = split_site(T, trunc)

    # (L ⊗ P₁ ← χ) and (χ ⊗ P₂ ← R)
    @test length(codomain(B1)) == 2
    @test codomain(B1)[1] == ℂ^D
    @test codomain(B1)[2] == ℂ^d
    @test length(domain(B1)) == 1
    @test length(codomain(B2)) == 2
    @test codomain(B2)[1] == domain(B1)[1]      # shared bond χ
    @test codomain(B2)[2] == ℂ^d
    @test domain(B2)[1] == ℂ^D

    # the pair reconstructs the two-site tensor
    @plansor recon[-1 -2 -3; -4] := B1[-1 -2; 1] * B2[1 -3; -4]
    @test recon ≈ T rtol=1e-12
end

@testset "SVDAnsatz — initial approximation preserves the loop density" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    mpsB = initial_approximation(A, SVDAnsatz(TruncationStrategy()))

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    # exact for factorizable A (the SVD rank is not truncated)
    @test dense(mpsB) ≈ dense(A) rtol=1e-10
end

# ==============================================================================
# SplitContext
# ==============================================================================

@testset "SplitContext — construction and accessors" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)

    @test site(ctx) == 1
    @test this_B(ctx) == mpsB[1]
    @test other_B(ctx) == mpsB[2]           # odd site: the right neighbour
    @test this_A(ctx) == mpsA[1]
    @test pa_from_pb(1) == 1 && pa_from_pb(2) == 1 && pa_from_pb(3) == 2

    # the caches sit at cut 1, the A-pair expectation is precomputed
    @test ctx.cacheBB.site == 1
    @test ctx.cacheAB.site == 1
    @test ctx.expectationAA ≈ expectation(ctx.braketAA)

    # mismatched site counts are rejected
    @test_throws DimensionMismatch SplitContext(mpsA, random_mps(ComplexF64, 3, d, D))
end

@testset "update! — site mismatch throws" begin
    mpsB = random_mps(ComplexF64, 4, 2, 2)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, 2, 2))
    ctx = SplitContext(mpsA, mpsB)
    @test_throws ArgumentError update!(ctx, CholeskySolve(), SweepState(2, 1, 0))
end

@testset "update! — one forward step slides both caches" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)
    old1 = copy(mpsB[1])

    # step 1: update site 1 (odd) — only the BB cache slides
    update!(ctx, CholeskySolve(), SweepState(1, 1, 0))
    @test site(ctx) == 2
    @test !(ctx.mpsB[1] ≈ old1)
    @test ctx.cacheBB.site == 2
    @test ctx.cacheAB.site == 1          # AB cache only moves at even sites

    # step 2: update site 2 (even) — the AB cache slides too
    update!(ctx, CholeskySolve(), SweepState(2, 1, 0))
    @test site(ctx) == 3
    @test ctx.cacheBB.site == 3
    @test ctx.cacheAB.site == 2

    # the new tensor keeps the site's space
    @test space(ctx.mpsB[2]) == space(mpsB[2])
end

@testset "update! — one step recovers exact B from perturbed guess" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)

    # odd site: perturb site 1, others exact → one update recovers B0[1]
    B_init = deepcopy(B0)
    B_init[1] = rand(ComplexF64, space(B0[1]))
    ctx = SplitContext(A, B_init)
    update!(ctx, CholeskySolve(), SweepState(1, 1, 0))
    @test ctx.mpsB[1] ≈ B0[1] rtol=1e-10
end

# ==============================================================================
# split — full run
# ==============================================================================

@testset "split — SVDAnsatz, circular sweep, exact solve" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    alg = LoopSplit(
        SVDAnsatz(TruncationStrategy()),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
    )
    B1, log = split(A, alg)
    
    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
    # the loop state is preserved: dense(B) ≈ dense(A)
    @test dense(B1) ≈ dense(A) rtol=1e-6
end

@testset "split — exact B0 stays unchanged after full sweep" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    ctx = SplitContext(A, deepcopy(B0))
    for st in CircularSweep(4; maximal_sweep=1)
        update!(ctx, CholeskySolve(), st)
    end
    @test diff_norm(ctx.mpsB, B0) ≈ 0 atol=1e-10
end

@testset "split — RandomAnsatz + Z2 spaces, zigzag sweep" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = random_mps2(ComplexF64, 2, P, V)
    alg = LoopSplit(
        RandomAnsatz(V, 4),
        ZigzagSweep(4; maximal_sweep=1),
        LUSolve(),
    )
    B1, log = split(mpsA, alg)

    @test length(B1) == 4
    @test assertconsistency(B1) === nothing
    @test scalartype(B1[1]) == ComplexF64
end

@testset "OneLoopAnsatz — SVD pre-split + projector gauge fixing" begin
    D, d = 2, 2
    B0 = random_mps(ComplexF64, 4, d, D)
    A = pair_contraction(B0)
    ansatz = OneLoopAnsatz(SVDAnsatz(TruncationStrategy()), ProjectorCanonical())
    mpsB = initial_approximation(A, ansatz)

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test scalartype(mpsB[1]) == ComplexF64
    # SVD is exact for factorizable A, and gaugefix! doesn't change the state.
    @test dense(mpsB) ≈ dense(A) rtol=1e-10
end

@testset "OneLoopAnsatz — Z2-graded spaces" begin
    V = Z2Space(0 => 2, 1 => 1)
    P = Z2Space(0 => 2)
    mpsA = random_mps2(ComplexF64, 2, P, V)
    ansatz = OneLoopAnsatz(SVDAnsatz(TruncationStrategy()), ProjectorCanonical())
    mpsB = initial_approximation(mpsA, ansatz)

    @test length(mpsB) == 4
    @test assertconsistency(mpsB) === nothing
    @test scalartype(mpsB[1]) == ComplexF64
end
