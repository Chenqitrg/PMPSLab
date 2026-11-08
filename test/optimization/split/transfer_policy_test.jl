using PMPSLab: DenseTransferMatrix, TruncatedTransferMatrix, SVDInitialization, SplitEnvironmentPolicy, NuclearNorm
using PMPSLab: AbstractTransferPolicy
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
        initialization=SVDInitialization(),
        sweep=CircularSweep(; max_sweeps=1),
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

@testset "SplitState — norm and mixed transfer policies are independent" begin
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
        ctx = test_split_state(mpsA, deepcopy(mpsB); environments)
        @test !hasproperty(ctx, :environments)
        @test environment(ctx.normal_cache, 1) isa norm_type
        @test environment(ctx.mixed_cache, 1) isa mixed_type
    end
end

@testset "split transfer policies — supported solver combinations" begin
    Random.seed!(20260810)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    mpsB = initial_approximation(mpsA, SVDInitialization())
    dense_policy = DenseTransferPolicy()
    truncated_policy = TruncatedTransferPolicy(; trunc=TruncationStrategy())

    # The mixed environment only constructs W, so it may be factored while the
    # dense normal environment is handled by a direct solver.
    direct_environments = SplitEnvironmentPolicy(;
        normal=dense_policy,
        mixed=truncated_policy,
    )
    direct_ctx = test_split_state(
        mpsA, deepcopy(mpsB); environments=direct_environments
    )
    _, direct_cost = update!(
        direct_ctx,
        SweepState{Forward}(1, 0),
    )
    @test isfinite(direct_cost)

    # A factored normal environment supplies an action rather than a materialized
    # matrix and therefore composes with KrylovSolve.
    factored_environments = SplitEnvironmentPolicy(;
        normal=truncated_policy,
        mixed=truncated_policy,
    )
    krylov = KrylovSolve(; algorithm=GMRES(; maxiter=100, tol=1e-12))
    krylov_ctx = test_split_state(
        mpsA, deepcopy(mpsB); solver=krylov, environments=factored_environments
    )
    _, krylov_cost = update!(
        krylov_ctx,
        SweepState{Forward}(1, 0),
    )
    @test isfinite(krylov_cost)

    # Direct solvers may materialize a factored normal environment locally.
    factored_direct_ctx = test_split_state(
        mpsA, deepcopy(mpsB); environments=factored_environments
    )
    _, factored_direct_cost = update!(
        factored_direct_ctx,
        SweepState{Forward}(1, 0),
    )
    @test isfinite(factored_direct_cost)

    direct_algorithm = ALSSplit(;
        initialization=SVDInitialization(),
        sweep=CircularSweep(; max_sweeps=1),
        solver=CholeskySolve(),
        environments=factored_environments,
    )
    @test_nowarn PMPSLab.initialize_split(mpsA, direct_algorithm)
end

@testset "NNR shifted Krylov solve agrees with a dense solve" begin
    Random.seed!(713)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    initial = initial_approximation(target, SVDInitialization())
    regularization = NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01))
    direct = ALSSplit(; sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve(), regularization)
    factored = TruncatedTransferPolicy(trunc=TruncationStrategy())
    krylov = ALSSplit(; sweep=CircularSweep(max_sweeps=1),
        solver=KrylovSolve(algorithm=GMRES(tol=1e-12, maxiter=200)), regularization,
        environments=SplitEnvironmentPolicy(normal=factored, mixed=factored))
    expected = split(target, initial, direct)
    actual = split(target, initial, krylov)
    @test dense(actual.mps) ≈ dense(expected.mps) rtol=1e-9
    @test actual.progress.local_cost ≈ expected.progress.local_cost atol=1e-10
end
