using PMPSLab: DenseTransferMatrix
struct DomainlessSweep <: LatticeIteratorLab.AbstractSweep end

@testset "SplitState keeps numerical progress, not diagnostic history" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    run = PMPSLab.initialize_split(target, algorithm)
    @test run.progress.completed_updates == 0
    session = start_run(PMPSLab.split_steps(run); threaded=false,
        onstep=(_, event) -> begin
            if event.completed_updates == 1
                @test (event.site, event.sweep, event.direction) == (1, 1, "forward")
                @test run.progress.completed_updates == 1
                @test run.normal_cache.left[2] ≈ DenseTransferMatrix(run.approximation[1])
                @test first(run.pending_iteration) == SweepState{Forward}(2, 0)
            end
        end)
    result = fetch(session)
    @test result.mps === run.approximation
    @test result.progress.completed_updates == 4
    @test result.progress.completed_sweeps == 1
    @test Base.summarysize(result.progress) < 256
    @test run_status(session) == ExperimentLab.COMPLETED
    @test dense(split(target, algorithm).mps) ≈ dense(result.mps)
end

@testset "Empty sweep and invalid sweep domain" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=0), solver=CholeskySolve())
    result = split(target, algorithm)
    @test result.progress.completed_updates == 0
    @test !result.progress.converged
    @test result.progress.local_cost === nothing
    @test_throws ArgumentError PMPSLab.initialize_split(target,
        ALSSplit(sweep=DomainlessSweep(), solver=CholeskySolve()))
end
