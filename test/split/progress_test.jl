function split_progress_fixture()
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=4, maximal_sweep=1),
        solve=CholeskySolve(),
    )
    return target, algorithm
end

function captured_logs(f)
    logger = Test.TestLogger(; min_level=Logging.BelowMinLevel)
    result = Logging.with_logger(f, logger)
    return result, logger.logs
end

@testset "default split observer — selection" begin
    target, algorithm = split_progress_fixture()
    run = initialize_run(target, algorithm)
    custom = Returns(nothing)

    @test PMPSLab._resolve_observer(
        run, PMPSLab._DEFAULT_OBSERVER, false
    ) === nothing
    @test PMPSLab._resolve_observer(
        run, PMPSLab._DEFAULT_OBSERVER, true
    ) isa PMPSLab.SplitProgressObserver
    @test PMPSLab._resolve_observer(run, nothing, true) === nothing
    @test PMPSLab._resolve_observer(run, custom, true) === custom
end

@testset "default split observer — blocking progress" begin
    target, algorithm = split_progress_fixture()
    observer = PMPSLab._default_observer(initialize_run(target, algorithm))

    (_, log), records = captured_logs() do
        split(target, algorithm; observer)
    end

    progress = filter(
        record -> record.message isa ProgressLogging.Progress,
        records,
    )
    messages = string.(getproperty.(records, :message))

    @test length(log.site_logs) == 4
    @test count(record -> !record.message.done, progress) == 4
    @test count(record -> record.message.done, progress) == 1
    @test all(record -> record.level == ProgressLogging.ProgressLevel, progress)
    @test any(==("ALS split started"), messages)
    @test any(startswith("ALS split sweep 1 completed: cost="), messages)
    @test any(==("ALS split completed after 4 site updates"), messages)
end

@testset "default split observer — local session progress" begin
    target, algorithm = split_progress_fixture()
    run = initialize_run(target, algorithm)
    observer = PMPSLab._default_observer(run)

    session, records = captured_logs() do
        session = PMPSLab.LocalRunSession(run; observer, threaded=false)
        PMPSLab.start_run!(session)
        PMPSLab.pause!(session)
        wait(session)
        @test PMPSLab.run_status(session) == PMPSLab.LOCAL_PAUSED
        PMPSLab.resume!(session)
        wait(session)
    end

    @test PMPSLab.run_status(session) == PMPSLab.LOCAL_COMPLETED
    @test count(
        record -> record.message isa ProgressLogging.Progress,
        records,
    ) == 5
    messages = string.(getproperty.(records, :message))
    @test "ALS split started" in messages
    @test "ALS split paused" in messages
    @test "ALS split resumed" in messages
    @test "ALS split completed after 4 site updates" in messages
end

@testset "split observers — silence and replacement" begin
    target, algorithm = split_progress_fixture()

    (_, default_log), default_records = captured_logs() do
        split(target, algorithm)
    end
    @test length(default_log.site_logs) == 4
    @test isempty(default_records)

    (_, silent_log), silent_records = captured_logs() do
        split(target, algorithm; observer=nothing)
    end
    @test length(silent_log.site_logs) == 4
    @test isempty(silent_records)

    events = SplitSiteCompleted[]
    (_, custom_log), custom_records = captured_logs() do
        split(target, algorithm; observer=event -> push!(events, event))
    end
    @test length(custom_log.site_logs) == 4
    @test length(events) == 4
    @test isempty(custom_records)
end
