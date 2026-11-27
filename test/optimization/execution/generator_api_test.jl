using Test, PMPSLab, TensorKit, LatticeIteratorLab, TensorLinearSolveLab
import AlgorithmsInterface as AI

@testset "PMPS implements the native algorithm interface" begin
    @test PMPSLab.AbstractSplit <: AI.Algorithm
    @test PMPSLab.AbstractSplitState <: AI.State
    @test PMPSLab.AbstractStoppingCriterion <: AI.StoppingCriterion
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    problem = PMPSLab.SplitProblem(target)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    state = @inferred AI.initialize_state(problem, algorithm)
    @test problem isa AI.Problem
    @test state.iteration == state.progress.completed_updates == 0
    @test !AI.is_finished!(problem, algorithm, state)

    # A.I. owns the iteration counter; step! performs exactly one numerical update.
    untouched = [copy(state.iterate[i]) for i in 2:4]
    AI.increment!(state)
    @test_logs @test (@inferred AI.step!(problem, algorithm, state)) === state
    @test state.progress.completed_updates == state.iteration == 1
    @test state.progress.site == 1
    @test all(state.iterate[i] == untouched[i - 1] for i in 2:4)
    @test (@inferred AI.finalize_state!(problem, algorithm, state)).mps === state.iterate

    # A checkpoint starts a fresh driver at the next site, without replaying site 1.
    saved = @inferred PMPSLab.snapshot(state)
    continued = @inferred AI.solve(problem, algorithm; snapshot=saved)
    @test continued.progress.completed_updates == 4
    @test dense(continued.mps) ≈ dense(target)
    @test saved.progress.completed_updates == 1
    @test all(saved.approximation[i] !== state.iterate[i] for i in 1:4)

    # solve! resets counters from the supplied iterate; it is not checkpoint resume.
    result = @inferred AI.solve!(problem, algorithm, state)
    @test result.progress.completed_updates == state.iteration == 4
    @test dense(result.mps) ≈ dense(target)
    @test_throws ArgumentError AI.step!(problem, algorithm, state)
end

@testset "Native criteria compose with sweep convergence" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    problem = PMPSLab.SplitProblem(target)
    rule = PMPSLab.CostConvergence(atol=1e-8, min_sweeps=2, consecutive_sweeps=1)
    for sweep in (CircularSweep(max_sweeps=5), ZigzagSweep(max_sweeps=5))
        algorithm = ALSSplit(; sweep, solver=CholeskySolve(),
            stopping_criterion=AI.StopWhenAny(rule, AI.StopAfterIteration(100)))
        state = @inferred AI.initialize_state(problem, algorithm)
        @test state.stopping_criterion_state isa AI.StoppingCriterionState
        while !AI.is_finished!(problem, algorithm, state)
            AI.increment!(state)
            AI.step!(problem, algorithm, state)
            # Saving before the next stopping check must preserve the pending check.
            saved = PMPSLab.snapshot(state)
            restored = @inferred PMPSLab.restore_split(saved, target)
            @test AI.is_finished!(problem, algorithm, restored) == AI.is_finished!(problem, algorithm, state)
        end
        @test state.progress.completed_sweeps == 2
        @test state.progress.completed_updates == (sweep isa CircularSweep ? 8 : 12)
        @test state.progress.converged
        @test AI.indicates_convergence(algorithm, state)
        @test !isempty(AI.get_active_stopping_criteria(algorithm, state))
    end

    # A generic iteration budget may stop mid-sweep and does not imply convergence.
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=5), solver=CholeskySolve(),
        stopping_criterion=AI.StopAfterIteration(2))
    state = AI.initialize_state(problem, algorithm)
    result = AI.solve!(problem, algorithm, state)
    @test result.progress.completed_updates == 2
    @test result.progress.completed_sweeps == 0
    @test !result.progress.converged
    @test !AI.indicates_convergence(algorithm, state)
    restored = PMPSLab.restore_split(PMPSLab.snapshot(state), target)
    @test AI.is_finished!(problem, algorithm, restored)
    @test isnothing(restored.pending_iteration)
end

@testset "Native convergence checkpoints can stop before a sweep boundary" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    problem = PMPSLab.SplitProblem(target)
    # A generic convergence quantity may be satisfied initially or after one site.
    quantity = function remaining(::PMPSLab.SplitProblem, ::ALSSplit, state::PMPSLab.SplitState)::Int
        return -state.iteration
    end
    for steps in (0, 1)
        algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=CholeskySolve(),
            stopping_criterion=AI.StopWhenBelow(quantity, -steps))
        state = @inferred AI.initialize_state(problem, algorithm)
        result = @inferred AI.solve!(problem, algorithm, state)
        @test result.progress.completed_updates == steps
        @test result.progress.completed_sweeps == 0
        @test result.progress.converged
        saved = @inferred PMPSLab.snapshot(state)
        restored = @inferred PMPSLab.restore_split(saved, target)
        @test AI.is_finished!(problem, algorithm, restored)
        @test restored.progress === result.progress
        @test (@inferred AI.solve(problem, algorithm; snapshot=saved)).progress === result.progress
    end
end

@testset "A stopping check evaluates its quantity once" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    problem = PMPSLab.SplitProblem(target)
    calls = Ref(0)
    function remaining_updates(::PMPSLab.SplitProblem, ::ALSSplit, state::PMPSLab.SplitState)::Int
        calls[] += 1
        return 1 - state.iteration
    end
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=CholeskySolve(),
        stopping_criterion=AI.StopWhenBelow(remaining_updates, 0))
    state = AI.initialize_state(problem, algorithm)
    @test !AI.is_finished!(problem, algorithm, state)
    @test calls[] == 1
    AI.increment!(state)
    AI.step!(problem, algorithm, state)
    @test AI.is_finished!(problem, algorithm, state)
    @test calls[] == 2
end
