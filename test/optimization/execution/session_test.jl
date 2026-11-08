using PMPSLab: SVDInitialization
using Test, ExperimentLab, PMPSLab, TensorKit, LatticeIteratorLab, TensorLinearSolveLab

@testset "Pause after a selected site prevents the next update" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=CholeskySolve())
    state = PMPSLab.initialize_split(target, algorithm)
    untouched = [copy(state.approximation[i]) for i in 3:4]
    visited = Int[]
    session = start_run(ExperimentLab.algorithm_steps(state); threaded=false,
        onstep=(session, event) -> begin
            push!(visited, event.site)
            if event.sweep == 1 && event.site == 2
                pause!(session)
            end
        end)
    @test timedwait(() -> run_status(session) == ExperimentLab.PAUSED, 20) == :ok
    @test visited == [1, 2]
    @test state.progress.completed_updates == 2
    @test run_status(session) == ExperimentLab.PAUSED
    # Giving the driver another opportunity to run must not advance it.
    yield()
    @test run_status(session) == ExperimentLab.PAUSED
    @test visited == [1, 2]
    @test all(state.approximation[i] == untouched[i - 2] for i in 3:4)
    resume!(session)
    result = fetch(session)
    @test visited == [1, 2, 3, 4, 1, 2, 3, 4]
    @test result.progress.completed_updates == 8
    @test run_status(session) == ExperimentLab.COMPLETED
end

@testset "Split generator uses ExperimentLab controls and storage" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(initialization=SVDInitialization(),
        sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    state = PMPSLab.initialize_split(target, algorithm)
    mktempdir() do path
        store = DataStore(path)
        config = ExperimentLab.record_algorithm!(store, algorithm)
        session = start_run(store, config, ExperimentLab.algorithm_steps(state); threaded=false,
            onstep=(s, entry) -> begin
                if entry.site == 1
                    ref = record_state!(store, s.execution; step=1)
                    save_snapshot!(store, capture(ref, state))
                    pause!(s)
                end
            end)
        @test timedwait(() -> run_status(session) == ExperimentLab.PAUSED, 20) == :ok
        @test state.progress.completed_updates == 1
        resume!(session)
        result = fetch(session)
        @test dense(result.mps) ≈ dense(target)
        @test result.progress.completed_updates == 4
        saved = only(ExperimentLab.records(store, :snapshot))
        restored = PMPSLab.restore_split(load_snapshot(store, saved), target)
        @test restored.progress.completed_updates == 1
        recovered = fetch(start_run(ExperimentLab.algorithm_steps(restored); threaded=false))
        @test dense(recovered.mps) ≈ dense(result.mps)
        @test ExperimentLab.info(store, session.execution)["status"] == "completed"
    end
end
