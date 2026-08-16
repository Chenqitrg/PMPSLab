struct DomainlessSweep <: PMPSLab.AbstractSweep end

@testset "SplitRun — one safe step updates exactly one site" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)

    @test run isa SplitRun
    @test isempty(run.measurement_results)
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
    @test run.context.normal_cache.left[2] ≈
          DenseTransferMatrix(run.context.approximation[1])
    @test first(run.pending_iteration) == SweepState{Forward}(2, 0)
    @test !isfinished(run)

    while !isfinished(run)
        step!(run)
    end
    result = run_result(run)
    @test result isa SplitResult
    @test result.mps === run.context.approximation
    @test result.measurement_results === run.measurement_results
    @test result.log === run.log
    @test isempty(result.measurement_results)
    mpsB = result.mps
    log = result.log
    @test length(log.site_logs) == 4
    @test log.duration == sum(site_log -> site_log.duration, log.site_logs)
    @test_throws ArgumentError step!(run)

    blocking_mps, blocking_log = split(mpsA, algorithm)
    @test dense(blocking_mps) ≈ dense(mpsB)
    @test getfield.(blocking_log.site_logs, :cost) ≈
          getfield.(log.site_logs, :cost)
    @test_throws MethodError split(mpsA, algorithm; callback=nothing)
end

@testset "SplitRun — an empty sweep is complete at initialization" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=4, max_sweeps=0),
        solver=CholeskySolve(),
    )
    run = initialize_run(mpsA, algorithm)

    @test isfinished(run)
    @test isnothing(run.pending_iteration)
    @test isempty(run_result(run).log.site_logs)
end

@testset "SplitRun — sweep domain must match approximation" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)

    for sweep in (
            CircularSweep(; site_count=1, max_sweeps=1),
            CircularSweep(; site_count=5, max_sweeps=1),
            ZigzagSweep(; site_count=3, max_sweeps=0),
        )
        algorithm = ALSSplit(;
            ansatz=SVDAnsatz(),
            sweep,
            solver=CholeskySolve(),
        )
        @test_throws DimensionMismatch initialize_run(mpsA, algorithm)
        @test_throws DimensionMismatch split(mpsA, algorithm)
    end

    domainless = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=DomainlessSweep(),
        solver=CholeskySolve(),
    )
    @test_throws ArgumentError initialize_run(mpsA, domainless)
end
