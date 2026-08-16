using Logging
import ProgressLogging

function looptnr_progress_fixture(; max_rg_steps::Int=1)
    gate = ones(ComplexF64, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1)
    split_algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=8, max_sweeps=1),
        solver=CholeskySolve(),
    )
    return UniformGates(gate), LoopTNR(; split=split_algorithm, max_rg_steps)
end

function captured_looptnr_logs(f)
    logger = Test.TestLogger(; min_level=Logging.BelowMinLevel)
    result = Logging.with_logger(f, logger)
    return result, logger.logs
end

mutable struct PausingLoopTNRProgressObserver
    inner::PMPSLab.LoopTNRProgressObserver
    session::Ref{LocalRunSession}
    pause_site::Int
end

PMPSLab._observer_running!(
    observer::PausingLoopTNRProgressObserver, run::LoopTNRRun
) = PMPSLab._observer_running!(observer.inner, run)

PMPSLab._observer_paused!(
    observer::PausingLoopTNRProgressObserver, run::LoopTNRRun
) = PMPSLab._observer_paused!(observer.inner, run)

PMPSLab._observer_cancelled!(
    observer::PausingLoopTNRProgressObserver, run::LoopTNRRun
) = PMPSLab._observer_cancelled!(observer.inner, run)

PMPSLab._observer_completed!(
    observer::PausingLoopTNRProgressObserver, run::LoopTNRRun
) = PMPSLab._observer_completed!(observer.inner, run)

PMPSLab._observer_failed!(
    observer::PausingLoopTNRProgressObserver, run::LoopTNRRun, exception
) = PMPSLab._observer_failed!(observer.inner, run, exception)

function PMPSLab._observer_step!(
        observer::PausingLoopTNRProgressObserver,
        event::AlgorithmEvent,
        run::LoopTNRRun,
    )
    PMPSLab._observer_step!(observer.inner, event, run)
    if event isa LoopTNRSplitStepCompleted &&
            event.event.log.state.site == observer.pause_site
        pause!(observer.session[])
    end
    return nothing
end

@testset "default LoopTNR observer — selection" begin
    cell, algorithm = looptnr_progress_fixture()
    run = initialize_run(cell, algorithm)
    custom = Returns(nothing)

    @test PMPSLab._resolve_observer(
        run, PMPSLab._DEFAULT_OBSERVER, false
    ) === nothing
    @test PMPSLab._resolve_observer(
        run, PMPSLab._DEFAULT_OBSERVER, true
    ) isa PMPSLab.LoopTNRProgressObserver
    @test PMPSLab._resolve_observer(run, nothing, true) === nothing
    @test PMPSLab._resolve_observer(run, custom, true) === custom
end

@testset "default LoopTNR observer — hierarchical progress" begin
    cell, algorithm = looptnr_progress_fixture()
    observer = PMPSLab._default_observer(initialize_run(cell, algorithm))

    result, records = captured_looptnr_logs() do
        renormalize(cell, algorithm; observer)
    end

    progress = filter(
        record -> record.message isa ProgressLogging.Progress,
        records,
    )
    messages = string.(getproperty.(records, :message))

    @test length(result.log.rg_logs) == 2
    @test count(record -> !record.message.done, progress) == 8
    @test count(record -> record.message.done, progress) == 1
    @test getproperty.(
        getproperty.(progress[1:8], :message), :fraction
    ) == collect(1:8) ./ 8
    @test all(record -> record.level == ProgressLogging.ProgressLevel, progress)
    @test "LoopTNR started · RG 0/1 normalization pending" in messages
    @test any(startswith("LoopTNR RG 0/1 initialized · scale "), messages)
    @test any(
        startswith("LoopTNR RG 1/1 · split sweep 1 completed · cost "),
        messages,
    )
    @test any(
        startswith("LoopTNR RG 1/1 completed · cost "),
        messages,
    )
    @test "LoopTNR completed · RG 1/1 · 2 measurements" in messages
end

@testset "default LoopTNR observer — pause position and resume" begin
    cell, algorithm = looptnr_progress_fixture()
    run = initialize_run(cell, algorithm)
    session_ref = Ref{LocalRunSession}()
    observer = PausingLoopTNRProgressObserver(
        PMPSLab._default_observer(run), session_ref, 4
    )
    session = LocalRunSession(run; observer, threaded=false)
    session_ref[] = session

    _, records = captured_looptnr_logs() do
        start_run!(session)
        wait(session)
        @test run_status(session) == PMPSLab.LOCAL_PAUSED
        resume!(session)
        fetch(session)
    end

    messages = string.(getproperty.(records, :message))
    @test "LoopTNR paused · RG 1/1 · split sweep 1 · next site 5/8" in messages
    @test "LoopTNR resumed · RG 1/1 · split sweep 1 · next site 5/8" in messages
    @test "LoopTNR completed · RG 1/1 · 2 measurements" in messages
end

@testset "default LoopTNR observer — terminal lifecycle" begin
    cell, algorithm = looptnr_progress_fixture()
    cancelled_run = initialize_run(cell, algorithm)
    cancelled_observer = PMPSLab._default_observer(cancelled_run)

    _, cancelled_records = captured_looptnr_logs() do
        session = LocalRunSession(
            cancelled_run; observer=cancelled_observer, threaded=false
        )
        cancel!(session)
        @test run_status(session) == PMPSLab.LOCAL_CANCELLED
    end
    @test "LoopTNR cancelled · RG 0/1 normalization pending" in
        string.(getproperty.(cancelled_records, :message))

    failing_measurement = TNRMeasurement(;
        normalization=InfinityNormNormalization(),
        fixed_point=FailingFixedPointMeasurement(),
    )
    failing_algorithm = LoopTNR(;
        split=algorithm.split,
        max_rg_steps=0,
        measurement=failing_measurement,
    )
    failing_run = initialize_run(cell, failing_algorithm)
    failing_observer = PMPSLab._default_observer(failing_run)

    _, failed_records = captured_looptnr_logs() do
        session = LocalRunSession(
            failing_run; observer=failing_observer, threaded=false
        )
        start_run!(session)
        wait(session)
        @test run_status(session) == PMPSLab.LOCAL_FAILED
    end
    @test any(
        startswith("LoopTNR failed · RG 0/0 normalization pending · "),
        string.(getproperty.(failed_records, :message)),
    )
end

@testset "LoopTNR run and result display" begin
    cell, algorithm = looptnr_progress_fixture()
    run = initialize_run(cell, algorithm)

    @test sprint(show, run) ==
        "LoopTNRRun(RG 0/1 normalization pending)"
    @test occursin("measurements: 0", sprint(show, MIME"text/plain"(), run))

    transient = initialize_run(cell, algorithm)
    transient.phase = PMPSLab.TNR_OPTIMIZE
    @test sprint(show, transient) ==
        "LoopTNRRun(RG 0/1 split pending)"

    step!(run)
    step!(run)
    @test sprint(show, run) ==
        "LoopTNRRun(RG 1/1 · split sweep 1 · next site 2/8)"
    @test occursin(
        "active split: 1/8 site updates",
        sprint(show, MIME"text/plain"(), run),
    )

    completed_child = initialize_run(cell, algorithm)
    step!(completed_child)
    for _ in 1:8
        step!(completed_child)
    end
    completed_child.phase = PMPSLab.TNR_OPTIMIZE
    @test sprint(show, completed_child) ==
        "LoopTNRRun(RG 1/1 coarse graining pending)"
    session = LocalRunSession(run; observer=nothing, threaded=false)
    @test sprint(show, session) ==
        "LocalRunSession(LoopTNRRun; status=LOCAL_READY; " *
        "RG 1/1 · split sweep 1 · next site 2/8; threaded=false)"

    saved = snapshot(run)
    @test sprint(show, saved) ==
        "LoopTNRSnapshot(RG 1/1 · split sweep 1 · next site 2/8; " *
        "1 measurements)"
    @test occursin(
        "active split snapshot: 1/8 site updates",
        sprint(show, MIME"text/plain"(), saved),
    )

    while !isfinished(run)
        step!(run)
    end
    result = run_result(run)
    @test sprint(show, result) ==
        "LoopTNRResult(RG 1; 2 measurements; PlaquetteGates)"
    shown = sprint(show, MIME"text/plain"(), result)
    @test occursin("completed RG: 1", shown)
    @test occursin("measurements: 2", shown)
    @test !occursin("TensorMap", shown)
end
