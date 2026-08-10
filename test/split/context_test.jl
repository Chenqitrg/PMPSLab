@testset "SplitContext — construction and accessors" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)

    @test this_B(ctx, 1) == mpsB[1]
    @test other_B(ctx, 1) == mpsB[2]
    @test other_B(ctx, 2) == mpsB[1]
    @test this_A(ctx, 1) == mpsA[1]
    @test pa_from_pb(1) == 1 && pa_from_pb(2) == 1 && pa_from_pb(3) == 2

    @test !hasproperty(ctx.cacheBB, :site)
    @test !hasproperty(ctx.cacheAB, :site)
    @test length(ctx.cacheBB.left) == length(mpsB) - 1
    @test length(ctx.cacheBB.right) == length(mpsB) - 1
    @test !hasproperty(ctx, :braketAA)
    @test !hasproperty(ctx, :braketBB)
    @test !hasproperty(ctx, :braketAB)
    @test ctx.expectationAA ≈ expectation(PeriodicTransferMatrix(mpsA, mpsA))

    @test_throws DimensionMismatch SplitContext(mpsA, random_mps(ComplexF64, 3, d, D))
end

@testset "update! — Krylov solver follows the split protocol" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)
    mpsB = initial_approximation(mpsA, SVDAnsatz())
    ctx = SplitContext(mpsA, mpsB)
    solver = KrylovSolve(GMRES(; maxiter=100, tol=1e-12))

    _, cost = update!(ctx, solver, SweepState{Forward}(1, 0))
    @test isfinite(cost)
    @test ctx.cacheBB.left[1] ≈ DenseTransferMatrix(ctx.mpsB[1])
end

@testset "update! — SweepState is the site source" begin
    mpsB = random_mps(ComplexF64, 4, 2, 2)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, 2, 2))
    ctx = SplitContext(mpsA, mpsB)
    _, cost = update!(ctx, CholeskySolve(), SweepState{Forward}(2, 0))
    @test isfinite(cost)
    @test ctx.cacheBB.left[2] ≈
          DenseTransferMatrix(ctx.mpsB[1]) *
          DenseTransferMatrix(ctx.mpsB[2])
    @test_throws BoundsError update!(
        ctx, CholeskySolve(), SweepState{Forward}(0, 0)
    )
end

@testset "update! — one forward step refreshes both caches" begin
    D, d = 2, 2
    mpsB = random_mps(ComplexF64, 4, d, D)
    mpsA = pair_contraction(random_mps(ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)
    old1 = copy(mpsB[1])
    ab_left_slot = ctx.cacheAB.left[1]
    old_ab_left = copy(ab_left_slot)

    update!(ctx, CholeskySolve(), SweepState{Forward}(1, 0))
    @test !(ctx.mpsB[1] ≈ old1)
    @test ctx.cacheBB.left[1] ≈ DenseTransferMatrix(ctx.mpsB[1])
    @test ctx.cacheAB.left[1] === ab_left_slot
    @test ctx.cacheAB.left[1] ≈ old_ab_left

    update!(ctx, CholeskySolve(), SweepState{Forward}(2, 0))
    @test ctx.cacheBB.left[2] ≈
          DenseTransferMatrix(ctx.mpsB[1]) *
          DenseTransferMatrix(ctx.mpsB[2])
    @test ctx.cacheAB.left[1] === ab_left_slot
    @test ctx.cacheAB.left[1] ≈ DenseTransferMatrix(
        ctx.mpsB[1], ctx.mpsB[2], ctx.mpsA[1]
    )
    @test space(ctx.mpsB[2]) == space(mpsB[2])
end

@testset "update! — one step recovers exact B from perturbed guess" begin
    B0 = random_mps(ComplexF64, 4, 2, 2)
    A = pair_contraction(B0)
    B_init = deepcopy(B0)
    B_init[1] = rand(ComplexF64, space(B0[1]))
    ctx = SplitContext(A, B_init)
    update!(ctx, CholeskySolve(), SweepState{Forward}(1, 0))
    @test ctx.mpsB[1] ≈ B0[1] rtol=1e-10
end
