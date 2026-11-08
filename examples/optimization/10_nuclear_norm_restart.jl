# Run from PMPSLab: julia --project=. examples/optimization/10_nuclear_norm_restart.jl
using PMPSLab: NuclearNorm
using PMPSLab
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab
using Test: @test

target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
algorithm = ALSSplit(
    sweep=LatticeIteratorLab.CircularSweep(max_sweeps=2),
    solver=TensorLinearSolveLab.CholeskySolve(),
    regularization=NuclearNorm(),
)
state = PMPSLab.initialize_split(target, algorithm)
store = ExperimentLab.DataStore(mktempdir())
config = ExperimentLab.record_algorithm!(store, algorithm)

# This is the ordinary ExperimentLab path: there is no NNR-specific session.
session = ExperimentLab.start_run(store, config, PMPSLab.split_steps(state);
    threaded=false,
    onstep=(session, progress) -> begin
        if progress.completed_updates == 3
            ref = ExperimentLab.record_state!(store, session.execution; step=3)
            # Captures the MPS AND the auxiliary/dual tensors needed to continue.
            ExperimentLab.save_snapshot!(store, ExperimentLab.capture(ref, state))
        end
    end,
)
result = fetch(session)

snapshot_ref = only(ExperimentLab.records(store, :snapshot))
saved = ExperimentLab.load_snapshot(store, snapshot_ref)
restored = PMPSLab.restore_split(saved, target)
recovered = fetch(ExperimentLab.start_run(PMPSLab.split_steps(restored); threaded=false))
@test dense(recovered.mps) ≈ dense(result.mps)
@test recovered.progress.penalty == result.progress.penalty

# The JSONL records contain duration, site, sweep, local_cost, penalty and
# cost_scale. Each step is flushed before the onstep callback runs.
logfile = joinpath(store.path, ExperimentLab.info(store, session.execution)["log_path"])
@show logfile
println(first(readlines(logfile)))
nothing
