@testset "split transfer policies — construction and defaults" begin
    dense_policy = DenseTransferPolicy()
    truncated_policy = TruncatedTransferPolicy(; trunc=TruncationStrategy(maxrank=2))
    @test TruncatedTransferPolicy().trunc ==
        truncerror(; rtol=1.0e-13)

    defaults = SplitEnvironmentPolicy()
    @test defaults.normal isa DenseTransferPolicy
    @test defaults.mixed isa DenseTransferPolicy

    shared = SplitEnvironmentPolicy(;
        normal=truncated_policy,
        mixed=truncated_policy,
    )
    @test shared.normal === truncated_policy
    @test shared.mixed === truncated_policy

    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=CholeskySolve(),
        environments=SplitEnvironmentPolicy(;
            normal=dense_policy,
            mixed=dense_policy,
        ),
    )
    @test algorithm.environments.normal isa DenseTransferPolicy
    @test algorithm.environments.mixed isa DenseTransferPolicy
    @test dense_policy isa AbstractTransferPolicy
end

@testset "SplitContext — norm and mixed transfer policies are independent" begin
    mpsB = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    mpsA = contract_pairs(mpsB)
    dense_policy = DenseTransferPolicy()
    truncated_policy = TruncatedTransferPolicy(; trunc=TruncationStrategy())

    cases = (
        (SplitEnvironmentPolicy(; normal=dense_policy, mixed=dense_policy),
         DenseTransferMatrix, DenseTransferMatrix),
        (SplitEnvironmentPolicy(; normal=dense_policy, mixed=truncated_policy),
         DenseTransferMatrix, TruncatedTransferMatrix),
        (SplitEnvironmentPolicy(; normal=truncated_policy, mixed=dense_policy),
         TruncatedTransferMatrix, DenseTransferMatrix),
        (SplitEnvironmentPolicy(; normal=truncated_policy, mixed=truncated_policy),
         TruncatedTransferMatrix, TruncatedTransferMatrix),
    )

    for (environments, norm_type, mixed_type) in cases
        ctx = PMPSLab._build_split_context(mpsA, deepcopy(mpsB), environments)
        @test !hasproperty(ctx, :environments)
        @test environment(ctx.normal_cache, 1) isa norm_type
        @test environment(ctx.mixed_cache, 1) isa mixed_type
    end
end

@testset "split transfer policies — supported solver combinations" begin
    Random.seed!(20260810)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    mpsB = initial_approximation(mpsA, SVDAnsatz())
    dense_policy = DenseTransferPolicy()
    truncated_policy = TruncatedTransferPolicy(; trunc=TruncationStrategy())

    # The mixed environment only constructs W, so it may be factored while the
    # dense normal environment is handled by a direct solver.
    direct_environments = SplitEnvironmentPolicy(;
        normal=dense_policy,
        mixed=truncated_policy,
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
    factored_environments = SplitEnvironmentPolicy(;
        normal=truncated_policy,
        mixed=truncated_policy,
    )
    krylov_ctx = PMPSLab._build_split_context(
        mpsA, deepcopy(mpsB), factored_environments
    )
    krylov = KrylovSolve(; algorithm=GMRES(; maxiter=100, tol=1e-12))
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

    direct_algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=CholeskySolve(),
        environments=factored_environments,
    )
    @test_nowarn initialize_run(mpsA, direct_algorithm)
end

@testset "split — truncated transfer policy is wired through a complete run" begin
    Random.seed!(20260811)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    truncated = TruncatedTransferPolicy(; trunc=TruncationStrategy())
    policy = SplitEnvironmentPolicy(;
        normal=truncated,
        mixed=truncated,
    )
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(; trunc=TruncationStrategy(maxrank=2)),
        sweep=CircularSweep(; site_count=4, max_sweeps=2),
        solver=KrylovSolve(; algorithm=GMRES(; maxiter=100, tol=1e-12)),
        environments=policy,
    )

    run = initialize_run(mpsA, algorithm)
    @test environment(run.context.normal_cache, 1) isa TruncatedTransferMatrix
    @test environment(run.context.mixed_cache, 1) isa TruncatedTransferMatrix

    step!(run)
    run = restore_run(snapshot(run), mpsA, algorithm)
    @test environment(run.context.normal_cache, 1) isa TruncatedTransferMatrix
    @test environment(run.context.mixed_cache, 1) isa TruncatedTransferMatrix

    while !isfinished(run)
        step!(run)
    end
    result = run_result(run)
    mpsB = result.mps
    log = result.log
    @test length(mpsB) == 4
    @test length(log.site_logs) == 8
    @test all(site_log -> isfinite(site_log.cost), log.site_logs)
end
