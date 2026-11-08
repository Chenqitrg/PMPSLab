using Test, ExperimentLab, PMPSLab, TensorKit, LatticeIteratorLab, TensorLinearSolveLab

@testset "ExperimentLab owns split history and snapshot capture" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    state = PMPSLab.initialize_split(target, algorithm)
    mktempdir() do path
        store = DataStore(path)
        config = ExperimentLab.record_algorithm!(store, algorithm)
        execution = ExperimentLab.record_execution!(store, config)
        ref = record_state!(store, execution; step=0)
        captured = capture(ref, state)
        @test !hasproperty(captured.value, :context)
        @test !hasproperty(state, :log)
        @test captured.value !== state
    end
end

@testset "Registered update events are owned by the execution" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=ZigzagSweep(max_sweeps=1), solver=CholeskySolve())
    state = PMPSLab.initialize_split(target, algorithm)
    mktempdir() do path
        store = DataStore(path)
        config = ExperimentLab.record_algorithm!(store, algorithm)
        generator = ExperimentLab.algorithm_steps(state)
        session = start_run(store, config, generator; threaded=false,
            onstep=(session, event) -> begin
                # Read immediately: the log must already be flushed before onstep.
                logpath = joinpath(path, ExperimentLab.info(store, session.execution)["log_path"])
                latest = ExperimentLab.JSON.parse(last(readlines(logpath)))["data"]
                @test event === state.progress
                @test latest["completed_updates"] == event.completed_updates
                @test latest["local_cost"] == event.local_cost
                @test !haskey(latest, "last_error")
                @test !haskey(latest, "stable_comparisons")
                if event.completed_updates == 3
                    ref = record_state!(store, session.execution; step=event.completed_updates)
                    save_snapshot!(store, capture(ref, state))
                end
            end)
        result = fetch(session)
        rows = ExperimentLab.JSON.parse.(readlines(joinpath(path,
            ExperimentLab.info(store, session.execution)["log_path"])))
        @test length(rows) == 6
        @test all(row["event"] == "mps.split.update.v1" for row in rows)
        @test all(row["owner"]["id"] == string(session.execution.id) for row in rows)
        @test [row["data"]["site"] for row in rows] == [1, 2, 3, 4, 3, 2]
        @test rows[end]["data"]["sweep_completed"]
        saved = load_snapshot(store, only(ExperimentLab.records(store, :snapshot)))
        @test saved.progress.completed_updates == 3
        restored = PMPSLab.restore_split(saved, target)
        recovered = fetch(start_run(ExperimentLab.algorithm_steps(restored); threaded=false))
        @test dense(recovered.mps) ≈ dense(result.mps)
        @test recovered.progress.completed_updates == 6
        @test saved.progress.completed_updates == 3
    end
end
