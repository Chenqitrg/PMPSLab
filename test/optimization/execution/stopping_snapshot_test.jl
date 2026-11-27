using PMPSLab: CostConvergence
@testset "Convergence counters survive restart without retaining history" begin
    Random.seed!(307)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    for sweep in (CircularSweep(max_sweeps=8), ZigzagSweep(max_sweeps=8))
        algorithm = ALSSplit(; sweep, solver=CholeskySolve(),
            stopping_criterion=CostConvergence(atol=1e-10, rtol=0, min_sweeps=4, consecutive_sweeps=2))
        problem = PMPSLab.SplitProblem(target)
        run = AI.initialize_state(problem, algorithm)
        while run.progress.completed_sweeps < 2
            @test !AI.is_finished!(problem, algorithm, run)
            AI.increment!(run)
            AI.step!(problem, algorithm, run)
        end
        # Save before checking this sweep's cost, then process that check on restoration.
        saved = PMPSLab.snapshot(run)
        restored = restore_split(saved, target)
        expected = run_split!(run)
        actual = run_split!(restored)
        @test actual.progress.converged
        @test actual.progress.completed_sweeps == expected.progress.completed_sweeps == 4
        @test restored.stopping_criterion_state.stable_comparisons == run.stopping_criterion_state.stable_comparisons
        @test dense(actual.mps) ≈ dense(expected.mps)
        finished = restore_split(PMPSLab.snapshot(restored), target)
        @test AI.is_finished!(problem, algorithm, finished)
        @test run_split!(finished).progress === actual.progress
    end
end
