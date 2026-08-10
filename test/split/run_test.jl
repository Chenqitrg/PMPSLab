struct DomainlessSweep <: PMPSLab.AbstractSweep end

@testset "SplitRun — one safe step updates exactly one site" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)

    @test run isa SplitRun
    @test run.context isa SplitContext
    @test !hasproperty(run, :target)
    @test !isfinished(run)
    @test first(run.pending_iteration) == SweepState{Forward}(1, 0)
    @test last(run.pending_iteration) == 1
    @test isempty(run.log.site_logs)
    @test_throws ArgumentError run_result(run)

    event = step!(run)
    @test event isa SplitSiteCompleted
    @test event.log.state == SweepState{Forward}(1, 0)
    @test length(run.log.site_logs) == 1
    @test run.context.cacheBB.left[2] ≈
          DenseTransferMatrix(run.context.mpsB[1])
    @test first(run.pending_iteration) == SweepState{Forward}(2, 0)
    @test !isfinished(run)

    while !isfinished(run)
        step!(run)
    end
    mpsB, log = run_result(run)
    @test mpsB === run.context.mpsB
    @test log === run.log
    @test length(log.site_logs) == 4
    @test log.sweep_time == sum(site_log -> site_log.time, log.site_logs)
    @test_throws ArgumentError step!(run)

    blocking_mps, blocking_log = split(mpsA, algorithm)
    @test dense(blocking_mps) ≈ dense(mpsB)
    @test getfield.(blocking_log.site_logs, :error) ≈
          getfield.(log.site_logs, :error)
    @test_throws MethodError split(mpsA, algorithm; callback=nothing)
end

@testset "SplitRun — an empty sweep is complete at initialization" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=0),
        CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)

    @test isfinished(run)
    @test isnothing(run.pending_iteration)
    @test isempty(last(run_result(run)).site_logs)
end

@testset "SplitRun — sweep domain must match approximation" begin
    mpsA = random_mps2(ComplexF64, 2, 2, 1)

    for sweep in (
            CircularSweep(1; maximal_sweep=1),
            CircularSweep(5; maximal_sweep=1),
            ZigzagSweep(3; maximal_sweep=0),
        )
        algorithm = PeriodicSplit(SVDAnsatz(), sweep, CholeskySolve())
        @test_throws DimensionMismatch initialize_run(mpsA, algorithm)
        @test_throws DimensionMismatch split(mpsA, algorithm)
    end

    domainless = PeriodicSplit(
        SVDAnsatz(), DomainlessSweep(), CholeskySolve()
    )
    @test_throws ArgumentError initialize_run(mpsA, domainless)
end
