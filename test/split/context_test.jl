@testset "SplitContext — construction and accessors" begin
    D, d = 2, 2
    mpsB = rand(PeriodicMPS, ComplexF64, 4, d, D)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)

    @test current_approximation(ctx, 1) == mpsB[1]
    @test paired_approximation(ctx, 1) == mpsB[2]
    @test paired_approximation(ctx, 2) == mpsB[1]
    @test target_tensor(ctx, 1) == mpsA[1]
    @test target_site_for(1) == 1 && target_site_for(2) == 1 && target_site_for(3) == 2

    @test !hasproperty(ctx.normal_cache, :site)
    @test !hasproperty(ctx.mixed_cache, :site)
    @test length(ctx.normal_cache.left) == length(mpsB)
    @test length(ctx.normal_cache.right) == length(mpsB)
    @test !hasproperty(ctx, :braketAA)
    @test !hasproperty(ctx, :braketBB)
    @test !hasproperty(ctx, :braketAB)
    @test !hasproperty(ctx, :environments)
    @test fieldnames(typeof(ctx)) == (
        :target,
        :approximation,
        :normal_cache,
        :mixed_cache,
        :target_squared_norm,
    )
    @test ctx.target_squared_norm ≈ expectation(PeriodicTransferMatrix(mpsA, mpsA))

    @test_throws DimensionMismatch SplitContext(mpsA, rand(PeriodicMPS, ComplexF64, 3, d, D))
end

@testset "update! — Krylov solver follows the split protocol" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    mpsB = initial_approximation(mpsA, SVDAnsatz())
    ctx = SplitContext(mpsA, mpsB)
    solver = KrylovSolve(; algorithm=GMRES(; maxiter=100, tol=1e-12))

    _, cost = update!(ctx, solver, SweepState{Forward}(1, 0))
    @test isfinite(cost)
    @test ctx.normal_cache.left[2] ≈ DenseTransferMatrix(ctx.approximation[1])
end

@testset "update! — SweepState is the site source" begin
    mpsB = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    ctx = SplitContext(mpsA, mpsB)
    _, cost = update!(ctx, CholeskySolve(), SweepState{Forward}(2, 0))
    @test isfinite(cost)
    @test ctx.normal_cache.left[3] ≈
          DenseTransferMatrix(ctx.approximation[1]) *
          DenseTransferMatrix(ctx.approximation[2])
    @test_throws BoundsError update!(
        ctx, CholeskySolve(), SweepState{Forward}(0, 0)
    )
end

@testset "update! — incomplete pairs invalidate without contraction" begin
    D, d = 2, 2
    mpsB = rand(PeriodicMPS, ComplexF64, 4, d, D)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, d, D))
    ctx = SplitContext(mpsA, mpsB)
    old1 = copy(mpsB[1])
    ab_left_slot = ctx.mixed_cache.left[2]
    old_ab_left = copy(ab_left_slot)

    update!(ctx, CholeskySolve(), SweepState{Forward}(1, 0))
    @test !(ctx.approximation[1] ≈ old1)
    @test ctx.normal_cache.left[2] ≈ DenseTransferMatrix(ctx.approximation[1])
    @test ctx.mixed_cache.left[2] === ab_left_slot
    @test ctx.mixed_cache.left[2] ≈ old_ab_left
    @test ctx.mixed_cache.left_valid_through == 0
    @test_nowarn environment(ctx.mixed_cache, 1)
    @test_throws StaleCacheError environment(ctx.mixed_cache, 2)

    update!(ctx, CholeskySolve(), SweepState{Forward}(2, 0))
    @test ctx.normal_cache.left[3] ≈
          DenseTransferMatrix(ctx.approximation[1]) *
          DenseTransferMatrix(ctx.approximation[2])
    @test ctx.mixed_cache.left[2] === ab_left_slot
    @test ctx.mixed_cache.left[2] ≈ DenseTransferMatrix(
        ctx.approximation[1], ctx.approximation[2], ctx.target[1]
    )
    @test space(ctx.approximation[2]) == space(mpsB[2])
end

@testset "update! — out-of-order sites fail before using stale caches" begin
    mpsB = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    ctx = SplitContext(mpsA, mpsB)

    update!(ctx, CholeskySolve(), SweepState{Forward}(1, 0))
    old3 = copy(ctx.approximation[3])
    @test_throws StaleCacheError update!(
        ctx, CholeskySolve(), SweepState{Forward}(3, 0)
    )
    @test ctx.approximation[3] == old3

    # Completing the original pair performs the one mixed-transfer contraction
    # required by the ordinary sweep and restores the forward validity frontier.
    update!(ctx, CholeskySolve(), SweepState{Forward}(2, 0))
    @test_nowarn environment(ctx.mixed_cache, 2)

    @test fieldnames(SweepState{Forward}) == (:site, :completed_sweeps)
end

@testset "update! — one step recovers exact B from perturbed guess" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    A = contract_pairs(B0)
    B_init = deepcopy(B0)
    B_init[1] = rand(ComplexF64, space(B0[1]))
    ctx = SplitContext(A, B_init)
    update!(ctx, CholeskySolve(), SweepState{Forward}(1, 0))
    @test ctx.approximation[1] ≈ B0[1] rtol=1e-10
end
