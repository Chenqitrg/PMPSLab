using Test, PMPSLab, LatticeIteratorLab, TensorLinearSolveLab
import ExperimentLab

@testset "The generator is the only split execution interface" begin
    @test PMPSLab.AbstractSplitState <: ExperimentLab.AbstractAlgorithmState
    @test parentmodule(ExperimentLab.algorithm_steps) === ExperimentLab
    @test !isdefined(PMPSLab, :split_steps)
    for name in (:step!, :isfinished, :split_result, :split_progress, :SplitContext, :SplitResult, :SplitUpdate)
        @test !isdefined(PMPSLab, name)
    end
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    state = PMPSLab.initialize_split(target, algorithm)
    @test state isa ExperimentLab.AbstractAlgorithmState
    @test !hasproperty(state.progress, :stable_comparisons)
    @test !hasproperty(state.progress, :last_error)
    @test eltype(ExperimentLab.algorithm_steps(state)) === PMPSLab.SplitProgress
end
