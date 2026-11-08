using PMPSLab: TruncatedTransferMatrix, SVDInitialization, SplitEnvironmentPolicy
@testset "split — truncated transfer policy is wired through a complete run" begin
    Random.seed!(20260811)
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    truncated = TruncatedTransferPolicy(; trunc=TruncationStrategy())
    policy = SplitEnvironmentPolicy(;
        normal=truncated,
        mixed=truncated,
    )
    algorithm = ALSSplit(;
        initialization=SVDInitialization(; trunc=TruncationStrategy(maxrank=2)),
        sweep=CircularSweep(; max_sweeps=2),
        solver=KrylovSolve(; algorithm=GMRES(; maxiter=100, tol=1e-12)),
        environments=policy,
    )

    run = initialize_split(mpsA, algorithm)
    @test environment(run.normal_cache, 1) isa TruncatedTransferMatrix
    @test environment(run.mixed_cache, 1) isa TruncatedTransferMatrix

    snapshots = [ExperimentLab.copy_state(run)]
    expected = fetch(start_run(PMPSLab.split_steps(run); threaded=false,
        onstep=(_, event) -> event.completed_updates == 1 && push!(snapshots, ExperimentLab.copy_state(run))))
    run = restore_split(snapshots[end], mpsA)
    @test environment(run.normal_cache, 1) isa TruncatedTransferMatrix
    @test environment(run.mixed_cache, 1) isa TruncatedTransferMatrix

    result = fetch(start_run(PMPSLab.split_steps(run); threaded=false))
    @test dense(result.mps) ≈ dense(expected.mps)
    mpsB = result.mps
    progress = result.progress
    @test length(mpsB) == 4
    @test progress.completed_updates == 8
    @test isfinite(progress.local_cost)
end
