using PMPSLab: CostConvergence
@testset "Convergence counters survive restart without retaining history" begin
    Random.seed!(307)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    for sweep in (CircularSweep(max_sweeps=8), ZigzagSweep(max_sweeps=8))
        algorithm = ALSSplit(sweep=sweep, solver=CholeskySolve(),
            stopping=CostConvergence(atol=1e-10, rtol=0, min_sweeps=4, consecutive_sweeps=2))
        run = initialize_split(target, algorithm)
        session = start_run(PMPSLab.split_steps(run); threaded=false,
            onstep=(s, event) -> begin
                if event.sweep_completed && run.progress.completed_sweeps == 2
                    pause!(s)
                end
            end)
        @test timedwait(() -> run_status(session) == ExperimentLab.PAUSED, 20) == :ok
        saved = ExperimentLab.copy_state(run)
        restored = restore_split(saved, target)
        resume!(session)
        expected = fetch(session)
        actual = fetch(start_run(PMPSLab.split_steps(restored); threaded=false))
        @test actual.progress.converged
        @test actual.progress.completed_sweeps == expected.progress.completed_sweeps == 4
        @test restored.stable_comparisons == run.stable_comparisons
        @test dense(actual.mps) ≈ dense(expected.mps)
        finished = restore_split(ExperimentLab.copy_state(restored), target)
        repeated_events = PMPSLab.SplitProgress[]
        result = fetch(start_run(PMPSLab.split_steps(finished); threaded=false,
            onstep=(_, event) -> push!(repeated_events, event)))
        @test isempty(repeated_events)
        @test result.progress.converged
    end
end
