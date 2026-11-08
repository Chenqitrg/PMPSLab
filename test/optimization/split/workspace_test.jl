using PMPSLab: DenseTransferMatrix, SVDInitialization
@testset "SplitState — numerical workspace" begin
    D, d = 2, 2
    mpsB = rand(PeriodicMPS, ComplexF64, 4, d, D)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, d, D))
    ctx = test_split_state(mpsA, mpsB)

    @test ctx.approximation[1] == mpsB[1]
    @test paired_approximation(ctx, 1) == mpsB[2]
    @test paired_approximation(ctx, 2) == mpsB[1]
    @test ctx.target[target_site_for(1)] == mpsA[1]
    @test target_site_for(1) == 1 && target_site_for(2) == 1 && target_site_for(3) == 2

    @test !hasproperty(ctx.normal_cache, :site)
    @test !hasproperty(ctx.mixed_cache, :site)
    @test length(ctx.normal_cache.left) == length(mpsB)
    @test length(ctx.normal_cache.right) == length(mpsB)
    @test !hasproperty(ctx, :braketAA)
    @test !hasproperty(ctx, :braketBB)
    @test !hasproperty(ctx, :braketAB)
    @test !hasproperty(ctx, :environments)
    @test ctx.target_squared_norm ≈ expectation(PeriodicTransferMatrix(mpsA, mpsA))

    @test_throws DimensionMismatch test_split_state(mpsA, rand(PeriodicMPS, ComplexF64, 3, d, D))
end

@testset "update! — Krylov solver follows the split protocol" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    mpsB = initial_approximation(mpsA, SVDInitialization())
    solver = KrylovSolve(; algorithm=GMRES(; maxiter=100, tol=1e-12))
    ctx = test_split_state(mpsA, mpsB; solver)

    _, cost = update!(ctx, SweepState{Forward}(1, 0))
    @test isfinite(cost)
    @test ctx.normal_cache.left[2] ≈ DenseTransferMatrix(ctx.approximation[1])
end

@testset "update! — SweepState is the site source" begin
    mpsB = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    ctx = test_split_state(mpsA, mpsB)
    _, cost = update!(ctx, SweepState{Forward}(2, 0))
    @test isfinite(cost)
    @test ctx.normal_cache.left[3] ≈
          DenseTransferMatrix(ctx.approximation[1]) *
          DenseTransferMatrix(ctx.approximation[2])
    @test_throws BoundsError update!(
        ctx, SweepState{Forward}(0, 0)
    )
end

@testset "update! — incomplete pairs invalidate without contraction" begin
    D, d = 2, 2
    mpsB = rand(PeriodicMPS, ComplexF64, 4, d, D)
    mpsA = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, d, D))
    ctx = test_split_state(mpsA, mpsB)
    old1 = copy(mpsB[1])
    ab_left_slot = ctx.mixed_cache.left[2]
    old_ab_left = copy(ab_left_slot)

    update!(ctx, SweepState{Forward}(1, 0))
    @test !(ctx.approximation[1] ≈ old1)
    @test ctx.normal_cache.left[2] ≈ DenseTransferMatrix(ctx.approximation[1])
    @test ctx.mixed_cache.left[2] === ab_left_slot
    @test ctx.mixed_cache.left[2] ≈ old_ab_left
    @test ctx.mixed_cache.left_valid_through == 0
    @test_nowarn environment(ctx.mixed_cache, 1)
    @test_throws StaleCacheError environment(ctx.mixed_cache, 2)

    update!(ctx, SweepState{Forward}(2, 0))
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
    ctx = test_split_state(mpsA, mpsB)

    update!(ctx, SweepState{Forward}(1, 0))
    old3 = copy(ctx.approximation[3])
    @test_throws StaleCacheError update!(
        ctx, SweepState{Forward}(3, 0)
    )
    @test ctx.approximation[3] == old3

    # Completing the original pair performs the one mixed-transfer contraction
    # required by the ordinary sweep and restores the forward validity frontier.
    update!(ctx, SweepState{Forward}(2, 0))
    @test_nowarn environment(ctx.mixed_cache, 2)

    @test fieldnames(SweepState{Forward}) == (:site, :completed_sweeps)
end

@testset "update! — one step recovers exact B from perturbed guess" begin
    B0 = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    A = contract_pairs(B0)
    B_init = deepcopy(B0)
    B_init[1] = rand(ComplexF64, space(B0[1]))
    ctx = test_split_state(A, B_init)
    update!(ctx, SweepState{Forward}(1, 0))
    @test ctx.approximation[1] ≈ B0[1] rtol=1e-10
end

@testset "local solve leaves tensors and cached environments unchanged" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    approximation = initial_approximation(target, SVDInitialization())
    ctx = test_split_state(target, approximation)
    before = deepcopy(ctx)
    state = SweepState{Forward}(1, 0)
    tensor, cost = PMPSLab._solve_local_update(ctx, state)
    @test isfinite(cost)
    @test all(ctx.approximation[i] == before.approximation[i] for i in eachindex(approximation))
    @test all(ctx.target[i] == before.target[i] for i in eachindex(target))
    for (cache, original) in ((ctx.normal_cache, before.normal_cache), (ctx.mixed_cache, before.mixed_cache))
        @test cache.left_valid_through == original.left_valid_through
        @test cache.right_valid_from == original.right_valid_from
        for site in 1:length(cache)
            @test dense(environment(cache, site)) == dense(environment(original, site))
        end
    end
    committed, committed_cost = update!(ctx, state)
    @test committed ≈ tensor
    @test committed_cost ≈ cost
end
