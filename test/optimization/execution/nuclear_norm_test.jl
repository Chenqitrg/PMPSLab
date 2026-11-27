using PMPSLab: NuclearNorm, CostConvergence
using Test, Random, LinearAlgebra, TensorKit, PMPSLab
import LatticeIteratorLab, TensorLinearSolveLab

@testset "NNR snapshots preserve detached auxiliary and dual tensors" begin
    Random.seed!(712)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    target[1] *= 7
    for sweep in (LatticeIteratorLab.CircularSweep(max_sweeps=2), LatticeIteratorLab.ZigzagSweep(max_sweeps=2))
        algorithm = ALSSplit(sweep=sweep, solver=TensorLinearSolveLab.CholeskySolve(),
            regularization=NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01)),
            error_measure=:exact)
        # Both sweep orders must infer the NNR workspace and restart payload.
        state = @inferred AI.initialize_state(PMPSLab.SplitProblem(target), algorithm)
        saved = [@inferred PMPSLab.snapshot(state)]
        events = PMPSLab.SplitProgress[]
        expected = run_split!(state; onstep=function capture_buffers(::PMPSLab.SplitState, event::PMPSLab.SplitProgress)::Nothing
                push!(saved, PMPSLab.snapshot(state))
                push!(events, event)
                return nothing
            end)
        @test any(!iszero(norm(t)) for t in state.regularization.dual)
        for snapshot in saved
            @test all(snapshot.regularization.auxiliary[i] !== state.regularization.auxiliary[i] for i in 1:4)
            restored = @inferred PMPSLab.restore_split(snapshot, target)
            recovered_events = PMPSLab.SplitProgress[]
            actual = run_split!(restored; onstep=function collect_events(::PMPSLab.SplitState, event::PMPSLab.SplitProgress)::Nothing
                push!(recovered_events, event)
                return nothing
            end)
            @test dense(actual.mps) ≈ dense(expected.mps) atol=1e-10
            @test getfield.(recovered_events, :penalty) == getfield.(events[snapshot.progress.completed_updates+1:end], :penalty)
            @test all(restored.regularization.dual[i] ≈ state.regularization.dual[i] for i in 1:4)
        end
    end
end

@testset "NNR progress and JLD2 restart use original tensor units" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
        solver=TensorLinearSolveLab.CholeskySolve(), regularization=NuclearNorm())
    state = PMPSLab.initialize_split(target, algorithm)
    mktempdir(function check_storage(path::String)::Nothing
        file = joinpath(path, "checkpoint.jld2")
        events = PMPSLab.SplitProgress[]
        expected = run_split!(state; onstep=function capture_checkpoint(active::PMPSLab.SplitState, progress::PMPSLab.SplitProgress)::Nothing
            push!(events, progress)
            progress.completed_updates == 3 && jldsave(file; snapshot=PMPSLab.snapshot(active))
            return nothing
        end)
        saved = load(file, "snapshot")
        restored = PMPSLab.restore_split(saved, target)
        actual = run_split!(restored)
        @test dense(actual.mps) ≈ dense(expected.mps)
        @test all(event.cost_scale ≈ real(state.target_squared_norm) for event in events)
        @test getfield.(events, :penalty) ≈ [fill(1e-5, 4); fill(8e-6, 4)]
        return nothing
    end)
end

@testset "NNR zero-sweep, graded and converged snapshots remain restartable" begin
    V, P = Z2Space(0=>1, 1=>1), Z2Space(0=>1, 1=>1)
    target = rand(PeriodicMPS, ComplexF64, 2, P ⊗ P, V)
    for max_sweeps in (0, 3)
        algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(; max_sweeps),
            solver=TensorLinearSolveLab.CholeskySolve(), regularization=NuclearNorm(),
            stopping_criterion=CostConvergence(atol=1e6, min_sweeps=2, consecutive_sweeps=1))
        state = PMPSLab.initialize_split(target, algorithm)
        initial = deepcopy(state.iterate)
        expected = run_split!(state)
        restored = PMPSLab.restore_split(PMPSLab.snapshot(state), target)
        actual = run_split!(restored; onstep=function reject_update(::PMPSLab.SplitState, ::PMPSLab.SplitProgress)::Nothing
            error("completed snapshot must not emit another update")
            return nothing
        end)
        @test all(space(actual.mps[i]) == space(initial[i]) for i in eachindex(initial))
        @test dense(actual.mps) ≈ dense(expected.mps)
        if max_sweeps == 0
            @test dense(actual.mps) == dense(initial)
        else
            @test actual.progress.converged
            @test actual.progress.completed_sweeps == 2
        end
    end
end
