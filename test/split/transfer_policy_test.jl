@testset "split transfer policies — construction and defaults" begin
    dense_policy = DenseTransferPolicy()
    truncated_policy = TruncatedTransferPolicy(TruncationStrategy(maxrank=2))

    defaults = SplitEnvironmentPolicy()
    @test defaults.norm isa DenseTransferPolicy
    @test defaults.mixed isa DenseTransferPolicy

    shared = SplitEnvironmentPolicy(truncated_policy)
    @test shared.norm === truncated_policy
    @test shared.mixed === truncated_policy

    algorithm = PeriodicSplit(
        SVDAnsatz(), CircularSweep(4; maximal_sweep=1), CholeskySolve();
        environments=SplitEnvironmentPolicy(dense_policy),
    )
    @test algorithm.environments.norm isa DenseTransferPolicy
    @test algorithm.environments.mixed isa DenseTransferPolicy
    @test dense_policy isa AbstractTransferPolicy
end

@testset "SplitContext — norm and mixed transfer policies are independent" begin
    mpsB = random_mps(ComplexF64, 4, 2, 2)
    mpsA = pair_contraction(mpsB)
    dense_policy = DenseTransferPolicy()
    truncated_policy = TruncatedTransferPolicy(TruncationStrategy())

    cases = (
        (SplitEnvironmentPolicy(dense_policy, dense_policy),
         DenseTransferMatrix, DenseTransferMatrix),
        (SplitEnvironmentPolicy(dense_policy, truncated_policy),
         DenseTransferMatrix, TruncatedTransferMatrix),
        (SplitEnvironmentPolicy(truncated_policy, dense_policy),
         TruncatedTransferMatrix, DenseTransferMatrix),
        (SplitEnvironmentPolicy(truncated_policy, truncated_policy),
         TruncatedTransferMatrix, TruncatedTransferMatrix),
    )

    for (environments, norm_type, mixed_type) in cases
        ctx = PMPSLab._build_split_context(mpsA, deepcopy(mpsB), environments)
        @test !hasproperty(ctx, :environments)
        @test environment(ctx.cacheBB, 1) isa norm_type
        @test environment(ctx.cacheAB, 1) isa mixed_type
    end
end

@testset "split transfer policies — supported solver combinations" begin
    Random.seed!(20260810)
    mpsA = random_mps2(ComplexF64, 2, 2, 2)
    mpsB = initial_approximation(mpsA, SVDAnsatz())
    dense_policy = DenseTransferPolicy()
    truncated_policy = TruncatedTransferPolicy(TruncationStrategy())

    # The mixed environment only constructs W, so it may be factored while the
    # dense normal environment is handled by a direct solver.
    direct_environments = SplitEnvironmentPolicy(
        dense_policy, truncated_policy
    )
    direct_ctx = PMPSLab._build_split_context(
        mpsA, deepcopy(mpsB), direct_environments
    )
    _, direct_cost = update!(
        direct_ctx,
        CholeskySolve(),
        direct_environments,
        SweepState{Forward}(1, 0),
    )
    @test isfinite(direct_cost)

    # A factored normal environment supplies an action rather than a materialized
    # matrix and therefore composes with KrylovSolve.
    factored_environments = SplitEnvironmentPolicy(truncated_policy)
    krylov_ctx = PMPSLab._build_split_context(
        mpsA, deepcopy(mpsB), factored_environments
    )
    krylov = KrylovSolve(GMRES(; maxiter=100, tol=1e-12))
    _, krylov_cost = update!(
        krylov_ctx,
        krylov,
        factored_environments,
        SweepState{Forward}(1, 0),
    )
    @test isfinite(krylov_cost)

    # Direct solvers may materialize a factored normal environment locally.
    factored_direct_ctx = PMPSLab._build_split_context(
        mpsA, deepcopy(mpsB), factored_environments
    )
    _, factored_direct_cost = update!(
        factored_direct_ctx,
        CholeskySolve(),
        factored_environments,
        SweepState{Forward}(1, 0),
    )
    @test isfinite(factored_direct_cost)

    direct_algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
        factored_environments,
    )
    @test_nowarn initialize_run(mpsA, direct_algorithm)
end

@testset "split — truncated transfer policy is wired through a complete run" begin
    Random.seed!(20260811)
    mpsA = random_mps2(ComplexF64, 2, 2, 2)
    policy = SplitEnvironmentPolicy(
        TruncatedTransferPolicy(TruncationStrategy())
    )
    algorithm = PeriodicSplit(
        SVDAnsatz(TruncationStrategy(maxrank=2)),
        CircularSweep(4; maximal_sweep=2),
        KrylovSolve(GMRES(; maxiter=100, tol=1e-12)),
        policy,
    )

    run = initialize_run(mpsA, algorithm)
    @test environment(run.context.cacheBB, 1) isa TruncatedTransferMatrix
    @test environment(run.context.cacheAB, 1) isa TruncatedTransferMatrix

    step!(run)
    run = restore_run(snapshot(run), mpsA, algorithm)
    @test environment(run.context.cacheBB, 1) isa TruncatedTransferMatrix
    @test environment(run.context.cacheAB, 1) isa TruncatedTransferMatrix

    while !isfinished(run)
        step!(run)
    end
    mpsB, log = run_result(run)
    @test length(mpsB) == 4
    @test length(log.site_logs) == 8
    @test all(site_log -> isfinite(site_log.error), log.site_logs)
end
