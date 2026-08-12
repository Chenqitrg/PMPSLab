mutable struct SnapshotPayloadResult <: AbstractMeasurementResult
    payload::Vector{Int}
end

struct SnapshotPayloadMeasurement <: AbstractFixedPointMeasurement end
PMPSLab.measure(
    ::SnapshotPayloadMeasurement, cell::AbstractTensorCell
) = SnapshotPayloadResult([length(cell)])

function snapshot_loop_algorithm(
        maximal_rg::Int;
        maximal_sweep::Int = 1,
        measurement::TNRMeasurement = TNRMeasurement(),
    )
    split_algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(8; maximal_sweep),
        CholeskySolve(),
    )
    return LoopTNR(split_algorithm; maximal_rg, measurement)
end

snapshot_loop_cell() = UniformGates(
    ones(ComplexF64, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1)
)

function loop_event_signature(event::LoopTNRCoarseGrained)
    return (:rg, event.rg)
end

function loop_event_signature(event::LoopTNRSplitStepCompleted)
    state = event.event.log.state
    return (:split, event.rg, state.n, state.site)
end

function finish_loop_run!(run::LoopTNRRun)
    events = AlgorithmEvent[]
    while !isfinished(run)
        push!(events, step!(run))
    end
    return events, run_result(run)
end

function assert_same_loop_completion(uninterrupted, restored)
    expected_events, expected = finish_loop_run!(uninterrupted)
    restored_events, actual = finish_loop_run!(restored)

    @test loop_event_signature.(restored_events) ==
          loop_event_signature.(expected_events)
    @test length(actual.log.rg_logs) == length(expected.log.rg_logs)
    @test length(actual.measurement_results) ==
          length(expected.measurement_results)
    @test getfield.(actual.measurement_results, :rg) ==
          getfield.(expected.measurement_results, :rg)
    @test all(
        actual_tensor ≈ expected_tensor
        for (actual_tensor, expected_tensor) in
            zip(collect(actual.cell), collect(expected.cell))
    )
    return nothing
end

@testset "LoopTNRSnapshot — restores every phase boundary" begin
    algorithm = snapshot_loop_algorithm(2)

    initial = initialize_run(snapshot_loop_cell(), algorithm)
    initial_restore = restore_run(snapshot(initial), algorithm)
    @test initial_restore.phase == PMPSLab.TNR_INITIALIZE
    assert_same_loop_completion(initial, initial_restore)

    prepare = initialize_run(snapshot_loop_cell(), algorithm)
    step!(prepare)
    prepare_restore = restore_run(snapshot(prepare), algorithm)
    @test prepare_restore.phase == PMPSLab.TNR_PREPARE_SPLIT
    @test isnothing(prepare_restore.active_split)
    assert_same_loop_completion(prepare, prepare_restore)

    optimize = initialize_run(snapshot_loop_cell(), algorithm)
    step!(optimize)
    for _ in 1:3
        step!(optimize)
    end
    optimize_saved = snapshot(optimize)
    optimize_restore = restore_run(optimize_saved, algorithm)
    @test optimize_restore.phase == PMPSLab.TNR_OPTIMIZE
    @test length(something(optimize_restore.active_split).log.site_logs) == 3
    continuation = step!(optimize_restore)
    @test loop_event_signature(continuation) == (:split, 1, 0, 4)
    step!(optimize)
    assert_same_loop_completion(optimize, optimize_restore)

    coarse = initialize_run(snapshot_loop_cell(), algorithm)
    step!(coarse)
    for _ in 1:8
        step!(coarse)
    end
    coarse_restore = restore_run(snapshot(coarse), algorithm)
    @test coarse_restore.phase == PMPSLab.TNR_COARSE_GRAIN
    @test isfinished(something(coarse_restore.active_split))
    assert_same_loop_completion(coarse, coarse_restore)

    committed = initialize_run(snapshot_loop_cell(), algorithm)
    step!(committed)
    for _ in 1:9
        step!(committed)
    end
    committed_restore = restore_run(snapshot(committed), algorithm)
    @test committed_restore.completed_rg == 1
    @test committed_restore.phase == PMPSLab.TNR_PREPARE_SPLIT
    assert_same_loop_completion(committed, committed_restore)

    finished_algorithm = snapshot_loop_algorithm(1)
    finished = initialize_run(snapshot_loop_cell(), finished_algorithm)
    finish_loop_run!(finished)
    finished_restore = restore_run(snapshot(finished), finished_algorithm)
    @test isfinished(finished_restore)
    @test isnothing(finished_restore.active_split)
    @test length(run_result(finished_restore).measurement_results) == 2
end

@testset "LoopTNRSnapshot — isolates cell, logs, and measurements" begin
    measurement = TNRMeasurement(
        InfinityNormNormalization(), SnapshotPayloadMeasurement()
    )
    algorithm = snapshot_loop_algorithm(1; measurement)
    run = initialize_run(snapshot_loop_cell(), algorithm)
    step!(run)
    step!(run)
    saved = snapshot(run)

    @test saved.cell !== run.cell
    @test all(
        saved_tensor !== run_tensor
        for (saved_tensor, run_tensor) in
            zip(collect(saved.cell), collect(run.cell))
    )
    @test saved.log !== run.log
    @test saved.log.rg_logs !== run.log.rg_logs
    @test saved.measurement_results !== run.measurement_results
    @test saved.measurement_results[1] !== run.measurement_results[1]
    @test saved.measurement_results[1].fixed_point !==
          run.measurement_results[1].fixed_point
    @test saved.active_split !== nothing
    @test saved.active_split.mpsB !== something(run.active_split).context.mpsB

    push!(run.measurement_results[1].fixed_point.payload, 99)
    @test saved.measurement_results[1].fixed_point.payload == [1]

    restored = restore_run(saved, algorithm)
    push!(restored.measurement_results[1].fixed_point.payload, 42)
    @test saved.measurement_results[1].fixed_point.payload == [1]
    @test restored.measurement_results[1].fixed_point.payload == [1, 42]
end

@testset "LoopTNRSnapshot — paused root contains active split" begin
    algorithm = snapshot_loop_algorithm(1)
    run = initialize_run(snapshot_loop_cell(), algorithm)
    session_ref = Ref{LocalRunSession}()
    observer = event -> begin
        if event isa LoopTNRSplitStepCompleted &&
                event.event.log.state.site == 4
            pause!(session_ref[])
        end
        return nothing
    end
    session = LocalRunSession(run; observer, threaded=false)
    session_ref[] = session

    start_run!(session)
    wait(session)
    @test run_status(session) == PMPSLab.LOCAL_PAUSED

    saved = snapshot(session.run)
    @test saved isa LoopTNRSnapshot
    @test saved.completed_rg == 0
    @test saved.phase == PMPSLab.TNR_OPTIMIZE
    @test length(saved.measurement_results) == 1
    @test saved.measurement_results[1].rg == 0
    @test saved.active_split isa SplitSnapshot
    @test length(saved.active_split.log.site_logs) == 4

    restored = restore_run(saved, algorithm)
    next_event = step!(restored)
    @test loop_event_signature(next_event) == (:split, 1, 0, 5)
end

@testset "LoopTNRSnapshot — rejects incompatible or invalid state" begin
    algorithm = snapshot_loop_algorithm(1)
    run = initialize_run(snapshot_loop_cell(), algorithm)
    step!(run)
    saved = snapshot(run)

    @test_throws ArgumentError restore_run(
        saved, snapshot_loop_algorithm(2)
    )

    invalid_phase = LoopTNRSnapshot(
        saved.cell,
        saved.completed_rg,
        PMPSLab.TNR_FINISHED,
        saved.active_split,
        saved.log,
        saved.measurement_results,
        saved.identity,
    )
    @test_throws ArgumentError restore_run(invalid_phase, algorithm)

    invalid_log = deepcopy(saved.log)
    invalid_log.rg_logs[1] = RGLog(1, nothing, 0.0)
    invalid_records = LoopTNRSnapshot(
        saved.cell,
        saved.completed_rg,
        saved.phase,
        saved.active_split,
        invalid_log,
        saved.measurement_results,
        saved.identity,
    )
    @test_throws ArgumentError restore_run(invalid_records, algorithm)
end
