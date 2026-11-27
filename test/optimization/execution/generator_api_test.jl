using Test, PMPSLab, LatticeIteratorLab, TensorLinearSolveLab
import ExperimentLab

@testset "The ExperimentLab generator remains the public execution interface" begin
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

@testset "A numerical step advances one site without logging" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    for sweep in (CircularSweep(max_sweeps=1), ZigzagSweep(max_sweeps=1))
        algorithm = ALSSplit(; sweep, solver=CholeskySolve(), error_measure=:exact)
        state = PMPSLab.initialize_split(target, algorithm)
        untouched = [copy(state.approximation[i]) for i in 2:4]

        # The numerical operation commits one update; its caller owns logging.
        progress = @test_logs @inferred PMPSLab._step!(state)
        @test progress === state.progress
        @test progress.completed_updates == 1
        @test progress.site == 1
        @test progress.completed_sweeps == 0
        @test isnothing(progress.exact_error)
        @test all(state.approximation[i] == untouched[i - 1] for i in 2:4)

        # Switching to the existing driver must continue at site 2, not replay site 1.
        events = collect(ExperimentLab.algorithm_steps(state))
        expected_sites = sweep isa CircularSweep ? [2, 3, 4] : [2, 3, 4, 3, 2]
        @test getfield.(events, :site) == expected_sites
        @test state.progress.completed_updates == length(expected_sites) + 1
        @test state.progress.completed_sweeps == 1
        @test state.progress.exact_error ≈ 0 atol=1e-12
        @test dense(state.approximation) ≈ dense(target)

        # A finished workspace cannot accidentally perform another update.
        @test_throws ArgumentError PMPSLab._step!(state)
    end
end

@testset "PMPS snapshots restore without an ExperimentLab session" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    for regularization in (PMPSLab.NoRegularization(), PMPSLab.NuclearNorm())
        algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=2),
            solver=CholeskySolve(), regularization=regularization)
        state = PMPSLab.initialize_split(target, algorithm)
        saved = @inferred PMPSLab.snapshot(state)
        @test saved.progress.completed_updates == 0

        # Capture a noninitial boundary using only PMPS operations.
        @inferred PMPSLab._step!(state)
        saved = @inferred PMPSLab.snapshot(state)
        legacy = @inferred ExperimentLab.copy_state(state)
        @test saved.progress.completed_updates == 1
        @test legacy.progress === saved.progress
        @test legacy.algorithm == saved.algorithm
        @test dense(legacy.approximation) == dense(saved.approximation)
        before = dense(saved.approximation)

        # Both capture entry points must own their tensors and regularization buffers.
        @test all(saved.approximation[i] !== state.approximation[i] for i in 1:4)
        @test all(legacy.approximation[i] !== state.approximation[i] for i in 1:4)
        if regularization isa PMPSLab.NuclearNorm
            for field in (:auxiliary, :dual)
                @test all(getproperty(saved.regularization, field)[i] !==
                    getproperty(state.regularization, field)[i] for i in 1:4)
            end
        end

        restored = PMPSLab.restore_split(saved, target)
        for run in (state, restored)
            while !isnothing(run.pending_iteration)
                @inferred PMPSLab._step!(run)
            end
        end
        @test state.progress.completed_updates == restored.progress.completed_updates == 8
        @test dense(restored.approximation) ≈ dense(state.approximation)
        @test dense(saved.approximation) == before
        @test saved.progress.completed_updates == 1
    end
end
