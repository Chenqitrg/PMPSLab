using PMPSLab
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab

# The algorithm owns numerical state; ExperimentLab owns its records and logs.
target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
algorithm = ALSSplit(sweep=LatticeIteratorLab.CircularSweep(max_sweeps=1), solver=TensorLinearSolveLab.CholeskySolve())
state = PMPSLab.initialize_split(target, algorithm)
store = ExperimentLab.DataStore(mktempdir())
config = ExperimentLab.record_algorithm!(store, algorithm)
session = ExperimentLab.start_run(store, config, ExperimentLab.algorithm_steps(state); threaded=false,
    onstep=(session, event) -> begin
        if event.completed_updates == 2
            ref = ExperimentLab.record_state!(store, session.execution; step=event.completed_updates)
            # Copies the approximation once and omits derived contraction caches.
            ExperimentLab.save_snapshot!(store, ExperimentLab.capture(ref, state))
        end
    end)
result = fetch(session)
@show result.progress.completed_updates # 4; no history is retained in result.
ExperimentLab.showtree(store)
println()

# Every update was written as a registered event in this execution's JSONL file.
logfile = joinpath(store.path, ExperimentLab.info(store, session.execution)["log_path"])
@show logfile
println(first(readlines(logfile)))

saved_ref = only(ExperimentLab.records(store, :snapshot))
saved = ExperimentLab.load_snapshot(store, saved_ref)
restored = PMPSLab.restore_split(saved, target)
@show restored.progress.completed_updates # 2
recovered = fetch(ExperimentLab.start_run(ExperimentLab.algorithm_steps(restored); threaded=false))
@assert dense(recovered.mps) ≈ dense(result.mps)
nothing
