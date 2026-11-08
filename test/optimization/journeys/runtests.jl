using PMPSLab: SVDInitialization, NuclearNorm
using Test, Random, LinearAlgebra, PMPSLab
import ExperimentLab, LatticeIteratorLab, TensorLinearSolveLab
include(joinpath(@__DIR__, "..", "..", "..", "examples", "optimization", "journeys", "cdl_diagnostics.jl"))
include("json_contract.jl")

# Fail if a completed step has not actually been flushed when onstep begins.
# Read only new bytes, so this check does not repeatedly scan an expanding log.
function check_flushed_step(store, session, progress, offset)
    path = joinpath(store.path, ExperimentLab.info(store, session.execution)["log_path"])
    row = open(path) do io
        seek(io, offset[])
        lines = readlines(io)
        offset[] = position(io)
        JSON.parse(last(lines))
    end
    # Throw in the worker task so fetch(session) propagates any failure.
    row["data"]["completed_updates"] == progress.completed_updates ||
        error("The latest site update was not flushed before onstep")
    row["owner"]["id"] == string(session.execution.id) ||
        error("The site log belongs to another execution")
end

function cdl_measurement!(store, session, state, target; step, initial=nothing)
    ref = ExperimentLab.record_state!(store, session.execution; step)
    mps = initial === nothing ? state.approximation : initial
    ExperimentLab.measure!(store, CDLDiagnostics(), (; target, mps); state=ref)
    return ref
end

function test_cdl_restart(store, config, execution, target, expected; steps)
    origin = only(ref for ref in ExperimentLab.children(store, execution)
                  if ExperimentLab.info(store, ref)["step"] == steps ÷ 2)
    saved = only(ref for ref in ExperimentLab.children(store, origin)
                 if ExperimentLab.describe(ref).kind == "snapshot")
    restored = PMPSLab.restore_split(ExperimentLab.load_snapshot(store, saved), target)
    offset = Ref(Int64(0))
    session = ExperimentLab.start_run(store, config, PMPSLab.split_steps(restored);
        label="resumed", initial_state=origin, threaded=false,
        onstep=(session, progress) -> begin
            check_flushed_step(store, session, progress, offset)
            if progress.completed_updates == steps
                cdl_measurement!(store, session, restored, target; step=steps)
            end
        end)
    recovered = fetch(session)
    @test norm(dense(recovered.mps) - dense(expected))^2 / norm(dense(expected))^2 < 1e-18
end

@testset "CDL NNR journey: physical filtering and JSON-only observation" begin
    Random.seed!(716)
    target = random_cdl_mps(ComplexF64, 4, 2)
    scale = norm(dense(target))^(1 / length(target))
    target = PeriodicMPS([tensor / scale for tensor in target])
    original_target = deepcopy(target)
    initial = initial_approximation(target, SVDInitialization())
    original_initial = deepcopy(initial)
    before = ExperimentLab.measure(CDLDiagnostics(), (; target, mps=initial))
    @test before.relative_squared_error < 1e-20
    @test before.max_second_ratio > 0.1
    @test before.rank1_relative_change > 1e-4
    sweeps, sites = 50, length(initial)
    steps = sweeps * sites
    solver = TensorLinearSolveLab.KrylovSolve(
        algorithm=TensorLinearSolveLab.KrylovKit.GMRES(tol=1e-12, maxiter=100))

    mktempdir() do path
        store = ExperimentLab.DataStore(path)
        source_config = ExperimentLab.record_algorithm!(store, CDLFixture())
        source_run = ExperimentLab.record_execution!(store, source_config)
        source = ExperimentLab.record_state!(store, source_run; step=0)
        ExperimentLab.save_snapshot!(store, ExperimentLab.capture(source, target))
        ExperimentLab.finish_execution!(store, source_run)

        for (label, regularization) in (
                "als" => PMPSLab.NoRegularization(),
                "nnr" => NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01, factor=0.95)),
            )
            algorithm = ALSSplit(; sweep=LatticeIteratorLab.CircularSweep(max_sweeps=sweeps), solver, regularization)
            config = ExperimentLab.record_algorithm!(store, algorithm; label)
            state = PMPSLab.initialize_split(target, initial, algorithm)
            offset = Ref(Int64(0))
            session = ExperimentLab.start_run(store, config, PMPSLab.split_steps(state);
                label, initial_state=source, threaded=false,
                onstep=(session, progress) -> begin
                    check_flushed_step(store, session, progress, offset)
                    if progress.completed_updates == 1
                        cdl_measurement!(store, session, state, target; step=0, initial)
                    end
                    if progress.completed_updates in (steps ÷ 2, steps)
                        ref = cdl_measurement!(store, session, state, target; step=progress.completed_updates)
                        label == "nnr" && ExperimentLab.save_snapshot!(store, ExperimentLab.capture(ref, state))
                    end
                    if label == "nnr" && progress.completed_updates == 3
                        ExperimentLab.pause!(session)
                    end
                end)
            if label == "nnr"
                ready = timedwait(() -> ExperimentLab.run_status(session) == ExperimentLab.PAUSED, 30)
                if ready != :ok
                    ExperimentLab.cancel!(session)
                    error("CDL journey did not pause at site 3")
                end
                paused = (; session=ExperimentLab.describe(session),
                    execution=ExperimentLab.info(store, session.execution),
                    completed_updates=state.progress.completed_updates)
                write(joinpath(path, "paused.json"), JSON.json(paused))
                ExperimentLab.resume!(session)
            end
            result = fetch(session)
            after = ExperimentLab.measure(CDLDiagnostics(), (; target, mps=result.mps))
            @test after.bond_dimensions == before.bond_dimensions
            @test after.relative_squared_error < 1e-10
            if label == "nnr"
                @test all(==(1), after.effective_pair_ranks)
                @test after.max_second_ratio < 1e-6
                @test after.rank1_relative_change < 1e-16
                write(joinpath(path, "completed.json"), JSON.json(ExperimentLab.describe(session)))
                test_cdl_restart(store, config, session.execution, target, result.mps; steps)
            else
                @test after.max_second_ratio > 0.1
                @test after.rank1_relative_change > 1e-4
            end
        end
        @test all(target[i] == original_target[i] for i in eachindex(target))
        @test all(initial[i] == original_initial[i] for i in eachindex(initial))
        write(joinpath(path, "tree.json"), JSON.json(ExperimentLab.describe(store; maxnodes=1000)))

        # Exercise the reader on a physically separate directory with no tensors.
        mktempdir() do json_only
            for (directory, _, files) in walkdir(path), file in files
                endswith(file, ".json") || endswith(file, ".jsonl") || continue
                destination = joinpath(json_only, relpath(directory, path), file)
                mkpath(dirname(destination))
                cp(joinpath(directory, file), destination)
            end
            @test all(!endswith(file, ".jld2") for (_, _, files) in walkdir(json_only) for file in files)
            check_cdl_json(json_only; sweeps, sites)
        end
    end
end
