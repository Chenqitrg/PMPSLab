using PMPSLab: NuclearNorm, CostConvergence
using Test, Random, LinearAlgebra, TensorKit, PMPSLab, ExperimentLab
import LatticeIteratorLab, TensorLinearSolveLab

@testset "NNR snapshots preserve detached auxiliary and dual tensors" begin
    Random.seed!(712)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    target[1] *= 7
    for sweep in (LatticeIteratorLab.CircularSweep(max_sweeps=2), LatticeIteratorLab.ZigzagSweep(max_sweeps=2))
        algorithm = ALSSplit(sweep=sweep, solver=TensorLinearSolveLab.CholeskySolve(),
            regularization=NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01)),
            error_measure=:exact)
        state = PMPSLab.initialize_split(target, algorithm)
        saved = [ExperimentLab.copy_state(state)]
        events = PMPSLab.SplitProgress[]
        expected = fetch(start_run(ExperimentLab.algorithm_steps(state); threaded=false,
            onstep=(_, event) -> begin
                push!(saved, ExperimentLab.copy_state(state))
                push!(events, event)
            end))
        @test any(!iszero(norm(t)) for t in state.regularization.dual)
        for snapshot in saved
            @test all(snapshot.regularization.auxiliary[i] !== state.regularization.auxiliary[i] for i in 1:4)
            restored = PMPSLab.restore_split(snapshot, target)
            recovered_events = PMPSLab.SplitProgress[]
            actual = fetch(start_run(ExperimentLab.algorithm_steps(restored); threaded=false,
                onstep=(_, event) -> push!(recovered_events, event)))
            @test dense(actual.mps) ≈ dense(expected.mps) atol=1e-10
            @test getfield.(recovered_events, :penalty) == getfield.(events[snapshot.progress.completed_updates+1:end], :penalty)
            @test all(restored.regularization.dual[i] ≈ state.regularization.dual[i] for i in 1:4)
        end
    end
end

@testset "NNR recorded progress and JLD2 restart use original tensor units" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
        solver=TensorLinearSolveLab.CholeskySolve(), regularization=NuclearNorm())
    state = PMPSLab.initialize_split(target, algorithm)
    mktempdir() do path
        store = DataStore(path)
        config = ExperimentLab.record_algorithm!(store, algorithm)
        session = start_run(store, config, ExperimentLab.algorithm_steps(state); threaded=false,
            onstep=(session, progress) -> begin
                if progress.completed_updates == 3
                    ref = record_state!(store, session.execution; step=3)
                    save_snapshot!(store, capture(ref, state))
                end
            end)
        expected = fetch(session)
        snapshot = load_snapshot(store, only(ExperimentLab.records(store, :snapshot)))
        restored = PMPSLab.restore_split(snapshot, target)
        actual = fetch(start_run(ExperimentLab.algorithm_steps(restored); threaded=false))
        @test dense(actual.mps) ≈ dense(expected.mps)
        rows = ExperimentLab.JSON.parse.(readlines(joinpath(path, ExperimentLab.info(store, session.execution)["log_path"])))
        @test all(row["data"]["cost_scale"] ≈ real(state.target_squared_norm) for row in rows)
        @test [row["data"]["penalty"] for row in rows] ≈ [fill(1e-5,4);fill(8e-6,4)]
    end
end

@testset "NNR zero-sweep, graded and converged snapshots remain restartable" begin
    V, P = Z2Space(0=>1, 1=>1), Z2Space(0=>1, 1=>1)
    target = rand(PeriodicMPS, ComplexF64, 2, P ⊗ P, V)
    for max_sweeps in (0, 3)
        algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(; max_sweeps),
            solver=TensorLinearSolveLab.CholeskySolve(), regularization=NuclearNorm(),
            stopping=CostConvergence(atol=1e6, min_sweeps=2, consecutive_sweeps=1))
        state = PMPSLab.initialize_split(target, algorithm)
        initial = deepcopy(state.approximation)
        expected = fetch(start_run(ExperimentLab.algorithm_steps(state); threaded=false))
        restored = PMPSLab.restore_split(ExperimentLab.copy_state(state), target)
        actual = fetch(start_run(ExperimentLab.algorithm_steps(restored); threaded=false,
            onstep=(_, _) -> error("completed snapshot must not emit another update")))
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
