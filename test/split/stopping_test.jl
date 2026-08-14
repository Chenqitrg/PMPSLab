struct ThrowingStoppingCriterion <: AbstractStoppingCriterion end

function PMPSLab.should_stop(
        ::ThrowingStoppingCriterion,
        ::AbstractVector{<:PMPSLab.SplitSiteLog},
    )
    error("stopping criterion failed")
end

@testset "CostConvergence — validates configuration" begin
    criterion = CostConvergence(
        atol=1e-10,
        rtol=1e-6,
        min_sweeps=3,
        consecutive_sweeps=2,
    )

    @test criterion == deepcopy(criterion)
    @test hash(criterion) == hash(deepcopy(criterion))
    @test_throws ArgumentError CostConvergence(atol=-1)
    @test_throws ArgumentError CostConvergence(rtol=Inf)
    @test_throws ArgumentError CostConvergence(min_sweeps=1)
    @test_throws ArgumentError CostConvergence(consecutive_sweeps=0)
end

@testset "PeriodicSplit — stopping failures preserve the next cursor" begin
    Random.seed!(20260814)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=2),
        CholeskySolve();
        stopping=ThrowingStoppingCriterion(),
    )
    run = initialize_run(target, algorithm)
    for _ in 1:3
        step!(run)
    end

    @test_throws ErrorException step!(run)
    @test length(run.log.site_logs) == 4
    @test first(run.pending_iteration) == SweepState{Forward}(1, 1)

    event = step!(run)
    @test event.log.state == SweepState{Forward}(1, 1)
    @test getfield.(run.log.site_logs, :state) == collect(
        Iterators.take(algorithm.sweep, 5)
    )
end

@testset "PeriodicSplit — cost convergence stops at complete sweeps" begin
    Random.seed!(20260811)
    initial = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    target = pair_contraction(initial)

    for (sweep, steps_per_sweep) in (
            (CircularSweep(4; maximal_sweep=10), 4),
            (ZigzagSweep(4; maximal_sweep=10), 6),
        )
        algorithm = PeriodicSplit(
            SVDAnsatz(),
            sweep,
            CholeskySolve();
            stopping=CostConvergence(
                atol=1e-10,
                rtol=0,
                min_sweeps=2,
                consecutive_sweeps=1,
            ),
        )
        _, log = split(target, algorithm)

        @test log.converged
        @test length(log.site_logs) == 2steps_per_sweep
        @test last(log.site_logs).state.n == 1
    end
end

@testset "PeriodicSplit — convergence patience and minimum sweeps" begin
    Random.seed!(20260812)
    target = pair_contraction(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=10),
        CholeskySolve();
        stopping=CostConvergence(
            atol=1e-10,
            rtol=0,
            min_sweeps=4,
            consecutive_sweeps=2,
        ),
    )
    _, log = split(target, algorithm)

    @test log.converged
    @test length(log.site_logs) == 16

    capped = PeriodicSplit(
        algorithm.ansatz,
        CircularSweep(4; maximal_sweep=2),
        algorithm.solve;
        stopping=CostConvergence(
            atol=1e-10,
            rtol=0,
            min_sweeps=3,
            consecutive_sweeps=1,
        ),
    )
    _, capped_log = split(target, capped)
    @test !capped_log.converged
    @test length(capped_log.site_logs) == 8
end

@testset "SplitSnapshot — restores cost convergence state" begin
    Random.seed!(20260813)
    target = pair_contraction(rand(PeriodicMPS, ComplexF64, 4, 2, 2))
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=10),
        CholeskySolve();
        stopping=CostConvergence(
            atol=1e-10,
            rtol=0,
            min_sweeps=2,
            consecutive_sweeps=1,
        ),
    )
    run = initialize_run(target, algorithm)
    for _ in 1:4
        step!(run)
    end
    restored = restore_run(snapshot(run), target, algorithm)

    while !isfinished(run)
        step!(run)
    end
    while !isfinished(restored)
        step!(restored)
    end

    expected_result = run_result(run)
    restored_result = run_result(restored)
    expected_mps = expected_result.mps
    expected_log = expected_result.log
    restored_mps = restored_result.mps
    restored_log = restored_result.log
    @test expected_log.converged
    @test restored_log.converged
    @test length(restored_log.site_logs) == 8
    @test dense(restored_mps) ≈ dense(expected_mps) rtol=1e-12

    completed = snapshot(run)
    completed_restore = restore_run(completed, target, algorithm)
    @test isfinished(completed_restore)
    @test run_result(completed_restore).log.converged

    inconsistent_log = SplitLog(
        copy(completed.log.site_logs),
        completed.log.sweep_time,
        false,
    )
    inconsistent = SplitSnapshot(
        completed.mpsB,
        completed.pending_iteration,
        inconsistent_log,
        completed.measurement_results,
        completed.identity,
    )
    @test_throws ArgumentError restore_run(inconsistent, target, algorithm)
end
