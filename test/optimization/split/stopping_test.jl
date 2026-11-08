using PMPSLab: SVDInitialization, CostTolerance, CostConvergence
struct ThrowingStoppingCriterion <: AbstractStoppingCriterion end

function PMPSLab.check_convergence(
        ::ThrowingStoppingCriterion,
        previous, current, sweeps, stable,
    )
    error("stopping criterion failed")
end

function stopped(criterion, costs)
    previous, stable, converged = nothing, 0, false
    for (sweeps, current) in enumerate(costs)
        converged, stable = PMPSLab.check_convergence(criterion, previous, current, sweeps, stable)
        previous = current
    end
    return converged
end

@testset "CostTolerance — absolute cost and relative change" begin
    criterion = CostTolerance(; atol=1e-9, rtol=1e-2, min_sweeps=5)

    @test criterion == deepcopy(criterion)
    @test !stopped(
        criterion,
        ([2e-9, 1.5e-9, 1.1e-9, 5.04e-10]),
    )
    @test stopped(
        criterion,
        ([2e-9, 1.5e-9, 1.1e-9, 5.04e-10, 5e-10]),
    )
    @test !stopped(
        criterion,
        ([2e-9, 1.5e-9, 1.1e-9, 1.015e-9, 1.01e-9]),
    )
    @test !stopped(
        criterion,
        ([2e-9, 1.5e-9, 1.1e-9, 5e-10, 4.9e-10]),
    )
    @test stopped(
        criterion,
        ([2e-9, 1.5e-9, 1.1e-9, 0.0, 0.0]),
    )

    @test_throws ArgumentError CostTolerance(; atol=0, rtol=1e-2)
    @test_throws ArgumentError CostTolerance(; atol=1e-9, rtol=0)
    @test_throws ArgumentError CostTolerance(;
        atol=1e-9,
        rtol=1e-2,
        min_sweeps=1,
    )
end

@testset "CostConvergence — validates configuration" begin
    criterion = CostConvergence(;
        atol=1e-10,
        rtol=1e-6,
        min_sweeps=3,
        consecutive_sweeps=2,
    )

    @test criterion == deepcopy(criterion)
    @test hash(criterion) == hash(deepcopy(criterion))
    @test_throws ArgumentError CostConvergence(; atol=-1)
    @test_throws ArgumentError CostConvergence(; rtol=Inf)
    @test_throws ArgumentError CostConvergence(; min_sweeps=1)
    @test_throws ArgumentError CostConvergence(; consecutive_sweeps=0)
end

@testset "ALSSplit — stopping failures preserve the next cursor" begin
    Random.seed!(20260814)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        initialization=SVDInitialization(),
        sweep=CircularSweep(; max_sweeps=2),
        solver=CholeskySolve(),
        stopping=ThrowingStoppingCriterion(),
    )
    run = PMPSLab.initialize_split(target, algorithm)
    session = start_run(PMPSLab.split_steps(run); threaded=false)
    @test_throws CapturedException fetch(session)
    @test run_status(session) == ExperimentLab.FAILED
    @test run.progress.completed_updates == 4
    @test first(run.pending_iteration) == SweepState{Forward}(1, 1)
    @test_throws ArgumentError resume!(session)

end

@testset "ALSSplit — cost convergence stops at complete sweeps" begin
    Random.seed!(20260811)
    initial = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    target = contract_pairs(initial)

    for (sweep, steps_per_sweep) in (
            (CircularSweep(; max_sweeps=10), 4),
            (ZigzagSweep(; max_sweeps=10), 6),
        )
        algorithm = ALSSplit(;
            initialization=SVDInitialization(),
            sweep,
            solver=CholeskySolve(),
            stopping=CostConvergence(;
                atol=1e-10,
                rtol=0,
                min_sweeps=2,
                consecutive_sweeps=1,
            ),
        )
        result = split(target, algorithm)

        @test result.progress.converged
        @test result.progress.completed_updates == 2steps_per_sweep
        @test result.progress.completed_sweeps == 2
    end
end

@testset "ALSSplit — convergence patience and minimum sweeps" begin
    Random.seed!(20260812)
    target = contract_pairs(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    algorithm = ALSSplit(;
        initialization=SVDInitialization(),
        sweep=CircularSweep(; max_sweeps=10),
        solver=CholeskySolve(),
        stopping=CostConvergence(;
            atol=1e-10,
            rtol=0,
            min_sweeps=4,
            consecutive_sweeps=2,
        ),
    )
    result = split(target, algorithm)

    @test result.progress.converged
    @test result.progress.completed_updates == 16

    capped = ALSSplit(;
        initialization=algorithm.initialization,
        sweep=CircularSweep(; max_sweeps=2),
        solver=algorithm.solver,
        stopping=CostConvergence(;
            atol=1e-10,
            rtol=0,
            min_sweeps=3,
            consecutive_sweeps=1,
        ),
    )
    capped_result = split(target, capped)
    @test !capped_result.progress.converged
    @test capped_result.progress.completed_updates == 8
end

@testset "Incremental convergence resets patience and handles nonfinite errors" begin
    rule = CostConvergence(atol=0.0, rtol=0.0, min_sweeps=2, consecutive_sweeps=2)
    @test !stopped(rule, [1.0, 1.0, 2.0, 2.0])
    @test stopped(rule, [1.0, 1.0, 2.0, 2.0, 2.0])
    @test !stopped(rule, [Inf, Inf, Inf])
    @test !stopped(rule, [NaN, NaN, NaN])
    @test stopped(rule, [0.0, 0.0, 0.0])
end
