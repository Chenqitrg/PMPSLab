using PMPSLab: SVDInitialization
using Test, PMPSLab, TensorKit, LatticeIteratorLab, TensorLinearSolveLab, JLD2

@testset "Returning control after a site prevents the next update" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=CholeskySolve())
    problem = PMPSLab.SplitProblem(target)
    state = AI.initialize_state(problem, algorithm)
    untouched = [copy(state.iterate[i]) for i in 3:4]
    for _ in 1:2
        @test !AI.is_finished!(problem, algorithm, state)
        AI.increment!(state)
        AI.step!(problem, algorithm, state)
    end
    @test state.progress.completed_updates == 2
    yield()
    @test state.progress.completed_updates == 2
    @test all(state.iterate[i] == untouched[i - 2] for i in 3:4)
    result = run_split!(state)
    @test result.progress.completed_updates == 8
    @test dense(result.mps) ≈ dense(target)
end

@testset "A JLD2 checkpoint resumes at the next site" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    problem = PMPSLab.SplitProblem(target)
    state = AI.initialize_state(problem, algorithm)
    AI.increment!(state)
    AI.step!(problem, algorithm, state)
    mktempdir(function check_storage(path::String)::Nothing
        file = joinpath(path, "checkpoint.jld2")
        jldsave(file; snapshot=PMPSLab.snapshot(state))
        expected = run_split!(state)
        saved = load(file, "snapshot")
        @test saved.progress.completed_updates == 1
        actual = AI.solve(problem, algorithm; snapshot=saved)
        @test actual.progress.completed_updates == 4
        @test dense(actual.mps) ≈ dense(expected.mps)
        return nothing
    end)
end
