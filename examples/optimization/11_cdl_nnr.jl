using PMPSLab: SVDInitialization, NuclearNorm
# julia --project=. examples/optimization/11_cdl_nnr.jl
# Optional: set CDL_JOURNEY_OUTPUT to a fresh directory for persistent artifacts.
using PMPSLab, Random, LinearAlgebra, Test
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab
include(joinpath(@__DIR__, "journeys", "cdl_diagnostics.jl"))
const JourneyJSON = ExperimentLab.JSON

output = get(ENV, "CDL_JOURNEY_OUTPUT", mktempdir())
store = ExperimentLab.DataStore(output)
fixture = CDLFixture()
Random.seed!(fixture.seed)
target = random_cdl_mps(ComplexF64, fixture.sites, fixture.corner_dimension)
# Put BOTH ALS and NNR on the same scale for their absolute Krylov tolerance.
# This uniform scalar normalization cannot remove an internal correlation loop.
if fixture.normalize
    site_scale = norm(dense(target))^(1 / fixture.sites)
    target = PeriodicMPS([tensor / site_scale for tensor in target])
end
initial = initial_approximation(target, SVDInitialization()) # No gauge filtering.
initial_diagnostics = ExperimentLab.measure(CDLDiagnostics(), (; target, mps=initial))
@test initial_diagnostics.max_second_ratio > 0.1
@test initial_diagnostics.relative_squared_error < 1e-20

# Give the generated input its own provenance and one tensor snapshot.
input_config = ExperimentLab.record_algorithm!(store, fixture; label="CDL fixture")
input_run = ExperimentLab.record_execution!(store, input_config)
input_state = ExperimentLab.record_state!(store, input_run; step=0)
ExperimentLab.save_snapshot!(store, ExperimentLab.capture(input_state, target))
ExperimentLab.finish_execution!(store, input_run)

# Use the SAME solver and SAME initial tensors in all three cases. An exact CDL
# has singular normal equations, so a nonsingular direct-solver assumption is unsafe.
solver = TensorLinearSolveLab.KrylovSolve(
    algorithm=TensorLinearSolveLab.KrylovKit.GMRES(tol=1e-12, maxiter=100))
cases = (
    "als" => PMPSLab.NoRegularization(),
    "nnr_default" => NuclearNorm(),
    "nnr_slow_decay" => NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01, factor=0.95)),
)
measurement_sweeps = (1, 5, 10, 20, 50, 100)

for (label, regularization) in cases
    algorithm = ALSSplit(; sweep=LatticeIteratorLab.CircularSweep(max_sweeps=100), solver, regularization)
    config = ExperimentLab.record_algorithm!(store, algorithm; label,
        metadata=(; fixture=ExperimentLab.describe(input_state), role="comparison"))
    state = PMPSLab.initialize_split(target, initial, algorithm)
    session = ExperimentLab.start_run(store, config, ExperimentLab.algorithm_steps(state);
        label, initial_state=input_state, threaded=false,
        provenance=(; seed=fixture.seed, numerical_threads=Threads.nthreads(),
            blas_threads=LinearAlgebra.BLAS.get_num_threads()),
        onstep=(session, progress) -> begin
            # Read the file DURING the run: the latest site log must be flushed.
            path = joinpath(store.path, ExperimentLab.info(store, session.execution)["log_path"])
            last_row = JourneyJSON.parse(last(readlines(path)))
            @test last_row["data"]["completed_updates"] == progress.completed_updates
            @test last_row["owner"]["id"] == string(session.execution.id)
            if progress.completed_updates == 1
                before = ExperimentLab.record_state!(store, session.execution; step=0, label="initial SVD")
                ExperimentLab.measure!(store, CDLDiagnostics(), (; target, mps=initial); state=before)
            end
            if progress.sweep_completed && progress.completed_sweeps in measurement_sweeps
                ref = ExperimentLab.record_state!(store, session.execution;
                    step=progress.completed_updates, metadata=(; sweep=progress.completed_sweeps))
                ExperimentLab.measure!(store, CDLDiagnostics(), (; target, mps=state.approximation); state=ref)
                if label == "nnr_slow_decay" && progress.completed_sweeps in (50, 100)
                    ExperimentLab.save_snapshot!(store, ExperimentLab.capture(ref, state))
                end
            end
            if label == "nnr_slow_decay" && progress.completed_updates == 3
                ExperimentLab.pause!(session)
            end
        end)
    if label == "nnr_slow_decay"
        @test timedwait(() -> ExperimentLab.run_status(session) == ExperimentLab.PAUSED, 60) == :ok
        # Exporting live Session status is an application responsibility. The
        # persisted ExecutionRef alone stays "running" while paused.
        paused = (; session=ExperimentLab.describe(session),
            execution=ExperimentLab.info(store, session.execution),
            latest=JourneyJSON.parse(last(readlines(joinpath(store.path,
                ExperimentLab.info(store, session.execution)["log_path"])))))
        write(joinpath(store.path, "paused_view.json"), JourneyJSON.json(paused))
        ExperimentLab.resume!(session)
    end
    result = fetch(session)
    final = ExperimentLab.measure(CDLDiagnostics(), (; target, mps=result.mps))
    if label == "nnr_slow_decay"
        @test maximum(final.effective_pair_ranks) == 1
        @test final.relative_squared_error < 1e-12
        @test final.rank1_relative_change < 1e-16
        write(joinpath(store.path, "completed_view.json"), JourneyJSON.json(ExperimentLab.describe(session)))
    end
    println(label, ": error=", final.relative_squared_error, ", max sigma2/sigma1=", final.max_second_ratio)
end

# Restart the slow-decay run from its sweep-50 snapshot, as a new execution.
original_execution = only(ref for ref in ExperimentLab.records(store, :execution)
    if ExperimentLab.info(store, ref)["label"] == "nnr_slow_decay")
original_config = ExperimentLab.parent_ref(store, original_execution)
restart_state = only(ref for ref in ExperimentLab.children(store, original_execution)
    if ExperimentLab.info(store, ref)["step"] == 400)
restart_snapshot = only(ref for ref in ExperimentLab.children(store, restart_state)
    if ExperimentLab.describe(ref).kind == "snapshot")
restored = PMPSLab.restore_split(ExperimentLab.load_snapshot(store, restart_snapshot), target)
restarted = ExperimentLab.start_run(store, original_config, ExperimentLab.algorithm_steps(restored);
    label="nnr_resumed", initial_state=restart_state, threaded=false,
    onstep=(session, progress) -> begin
        if progress.completed_updates == 800
            ref = ExperimentLab.record_state!(store, session.execution; step=800)
            ExperimentLab.measure!(store, CDLDiagnostics(), (; target, mps=restored.approximation); state=ref)
        end
    end)
recovered = fetch(restarted)
final_state = only(ref for ref in ExperimentLab.children(store, original_execution)
    if ExperimentLab.info(store, ref)["step"] == 800)
final_snapshot = only(ref for ref in ExperimentLab.children(store, final_state)
    if ExperimentLab.describe(ref).kind == "snapshot")
expected = ExperimentLab.load_snapshot(store, final_snapshot).approximation
restart_difference = norm(dense(recovered.mps) - dense(expected))^2 / norm(dense(expected))^2
@test restart_difference < 1e-18
write(joinpath(store.path, "restart_check.json"), JourneyJSON.json((; relative_squared_difference=restart_difference)))
write(joinpath(store.path, "tree.json"), JourneyJSON.json(ExperimentLab.describe(store; maxnodes=1000)))
println("JSON-only reader: python3 examples/optimization/journeys/read_cdl_json.py ", repr(store.path))
@show store.path
nothing
