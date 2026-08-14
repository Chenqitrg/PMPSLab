using Test
using PMPSLab

struct DummyAlgorithm end

mutable struct DummyContext <: AlgorithmContext{Int,Nothing}
    value::Int
end

struct DummyRun <: AbstractAlgorithmRun
    target::Int
    algorithm::DummyAlgorithm
    context::DummyContext
end

struct DummySnapshot <: AbstractAlgorithmSnapshot
    value::Int
end

struct DummyStepCompleted <: AlgorithmEvent
    value::Int
end

struct FailingRun <: AbstractAlgorithmRun end

mutable struct GatedRun <: AbstractAlgorithmRun
    target::Int
    context::DummyContext
    entered::Channel{Nothing}
    release::Channel{Nothing}
end

PMPSLab._checkpoint_algorithm(::DummySnapshot) = :dummy

PMPSLab.initialize_run(target::Int, algorithm::DummyAlgorithm) =
    DummyRun(target, algorithm, DummyContext(0))

function PMPSLab.step!(run::DummyRun)
    isfinished(run) && throw(ArgumentError("dummy run is already complete"))
    run.context.value += 1
    return DummyStepCompleted(run.context.value)
end

PMPSLab.isfinished(run::DummyRun) = run.context.value >= run.target

function PMPSLab.run_result(run::DummyRun)
    isfinished(run) || throw(ArgumentError("dummy run is not complete"))
    return run.context.value
end

PMPSLab.snapshot(run::DummyRun) = DummySnapshot(run.context.value)

function PMPSLab.restore_run(saved::DummySnapshot, target::Int, algorithm::DummyAlgorithm)
    0 <= saved.value <= target || throw(ArgumentError(
        "snapshot value $(saved.value) is incompatible with target $target"
    ))
    return DummyRun(target, algorithm, DummyContext(saved.value))
end

function PMPSLab.step!(run::GatedRun)
    put!(run.entered, nothing)
    take!(run.release)
    run.context.value += 1
    return DummyStepCompleted(run.context.value)
end

PMPSLab.isfinished(run::GatedRun) = run.context.value >= run.target
PMPSLab.snapshot(run::GatedRun) = DummySnapshot(run.context.value)

PMPSLab.step!(::FailingRun) = error("dummy step failed")
PMPSLab.isfinished(::FailingRun) = false

@testset "resumable execution interface" begin
    algorithm = DummyAlgorithm()
    run = initialize_run(3, algorithm)

    @test run isa AbstractAlgorithmRun
    @test run.context isa AlgorithmContext
    @test !isfinished(run)
    @test_throws ArgumentError run_result(run)

    event = step!(run)
    @test event isa AlgorithmEvent
    @test event.value == 1

    restored = restore_run(snapshot(run), 3, algorithm)
    @test restored !== run
    @test restored.context !== run.context
    @test restored.context.value == run.context.value

    @test step!(restored).value == 2
    @test step!(restored).value == 3
    @test isfinished(restored)
    @test run_result(restored) == 3
    @test_throws ArgumentError step!(restored)

    @test_throws ArgumentError restore_run(DummySnapshot(4), 3, algorithm)
end

@testset "local REPL session pauses and resumes at safe steps" begin
    algorithm = DummyAlgorithm()
    session_ref = Ref{LocalRunSession}()
    observer = event -> begin
        event.value == 1 && pause!(session_ref[])
        nothing
    end
    session = LocalRunSession(
        initialize_run(3, algorithm); observer, threaded=false
    )
    session_ref[] = session

    start_run!(session)
    wait(session)
    @test run_status(session) == PMPSLab.LOCAL_PAUSED
    @test session.run.context.value == 1

    resume!(session)
    wait(session)
    @test run_status(session) == PMPSLab.LOCAL_COMPLETED
    @test run_result(session) == 3
    @test fetch(session) == 3
end

@testset "local REPL session cancels at a safe step" begin
    session_ref = Ref{LocalRunSession}()
    observer = event -> begin
        event.value == 1 && cancel!(session_ref[])
        nothing
    end
    session = LocalRunSession(
        initialize_run(3, DummyAlgorithm()); observer, threaded=false
    )
    session_ref[] = session

    start_run!(session)
    wait(session)
    @test run_status(session) == PMPSLab.LOCAL_CANCELLED
    @test session.run.context.value == 1
    @test_throws ArgumentError run_result(session)
    @test_throws ArgumentError fetch(session)
end

@testset "local session fetch reports terminal outcomes" begin
    ready = LocalRunSession(initialize_run(1, DummyAlgorithm()); threaded=false)
    @test isnothing(ready.observer)
    @test sprint(show, ready) ==
          "LocalRunSession(DummyRun; status=LOCAL_READY, threaded=false)"
    @test sprint(show, MIME"text/plain"(), ready) == sprint(show, ready)
    @test !occursin("DummyContext", sprint(show, MIME"text/plain"(), ready))
    @test_throws ArgumentError fetch(ready)

    failed = start_run(FailingRun(); threaded=true)
    @test wait(failed) === failed
    @test run_status(failed) == PMPSLab.LOCAL_FAILED
    captured = try
        fetch(failed)
    catch err
        err
    end
    @test captured isa CapturedException
    @test captured.ex isa ErrorException
    @test captured.ex.msg == "dummy step failed"
    @test !isempty(captured.processed_bt)
end

@testset "local session durable pause and cancellation" begin
    mktempdir() do directory
        paused_path = joinpath(directory, "paused.jld2")
        paused = LocalRunSession(
            initialize_run(3, DummyAlgorithm()); threaded=false
        )
        start_run!(paused)
        @test pause_and_checkpoint!(paused, paused_path) ==
              PMPSLab.LOCAL_CHECKPOINT_WRITTEN
        @test run_status(paused) == PMPSLab.LOCAL_PAUSED
        @test isfile(paused_path)
        @test read_checkpoint(paused_path).value == paused.run.context.value
        @test_throws ArgumentError fetch(paused)

        missing_path = joinpath(directory, "missing", "paused.jld2")
        @test_throws ArgumentError checkpoint!(paused, missing_path)
        @test run_status(paused) == PMPSLab.LOCAL_PAUSED

        resume!(paused)
        wait(paused)
        @test run_result(paused) == 3

        cancelled_path = joinpath(directory, "cancelled.jld2")
        cancelled = LocalRunSession(
            initialize_run(3, DummyAlgorithm()); threaded=false
        )
        @test cancel_and_checkpoint!(cancelled, cancelled_path) ==
              PMPSLab.LOCAL_CHECKPOINT_WRITTEN
        @test run_status(cancelled) == PMPSLab.LOCAL_CANCELLED
        @test read_checkpoint(cancelled_path).value == 0

        completed_path = joinpath(directory, "completed.jld2")
        completed = start_run(
            initialize_run(1, DummyAlgorithm()); threaded=false
        )
        wait(completed)
        @test cancel_and_checkpoint!(completed, completed_path) ==
              PMPSLab.LOCAL_CHECKPOINT_SKIPPED_COMPLETED
        @test run_status(completed) == PMPSLab.LOCAL_COMPLETED
        @test !ispath(completed_path)
    end
end

@testset "durable helpers wait for an active safe step" begin
    mktempdir() do directory
        paused_path = joinpath(directory, "active-paused.jld2")
        paused_run = GatedRun(
            2, DummyContext(0), Channel{Nothing}(1), Channel{Nothing}(1)
        )
        paused = LocalRunSession(paused_run; threaded=true)
        start_run!(paused)
        take!(paused_run.entered)

        pause_task = @async pause_and_checkpoint!(paused, paused_path)
        yield()
        @test run_status(paused) == PMPSLab.LOCAL_PAUSE_REQUESTED
        @test !ispath(paused_path)

        put!(paused_run.release, nothing)
        @test fetch(pause_task) == PMPSLab.LOCAL_CHECKPOINT_WRITTEN
        @test run_status(paused) == PMPSLab.LOCAL_PAUSED
        @test paused_run.context.value == 1
        @test read_checkpoint(paused_path).value == 1

        cancelled_path = joinpath(directory, "active-cancelled.jld2")
        cancelled_run = GatedRun(
            2, DummyContext(0), Channel{Nothing}(1), Channel{Nothing}(1)
        )
        cancelled = LocalRunSession(cancelled_run; threaded=true)
        start_run!(cancelled)
        take!(cancelled_run.entered)

        cancel_task = @async cancel_and_checkpoint!(cancelled, cancelled_path)
        yield()
        @test run_status(cancelled) == PMPSLab.LOCAL_CANCEL_REQUESTED
        @test !ispath(cancelled_path)

        put!(cancelled_run.release, nothing)
        @test fetch(cancel_task) == PMPSLab.LOCAL_CHECKPOINT_WRITTEN
        @test run_status(cancelled) == PMPSLab.LOCAL_CANCELLED
        @test cancelled_run.context.value == 1
        @test read_checkpoint(cancelled_path).value == 1

        completed_path = joinpath(directory, "active-completed.jld2")
        completed_run = GatedRun(
            1, DummyContext(0), Channel{Nothing}(1), Channel{Nothing}(1)
        )
        completed = LocalRunSession(completed_run; threaded=true)
        start_run!(completed)
        take!(completed_run.entered)

        completed_task = @async cancel_and_checkpoint!(completed, completed_path)
        yield()
        put!(completed_run.release, nothing)
        @test fetch(completed_task) ==
              PMPSLab.LOCAL_CHECKPOINT_SKIPPED_COMPLETED
        @test run_status(completed) == PMPSLab.LOCAL_COMPLETED
        @test !ispath(completed_path)
    end
end

@testset "durable helpers reject calls from the observer task" begin
    mktempdir() do directory
        path = joinpath(directory, "observer.jld2")
        session_ref = Ref{LocalRunSession}()
        failure = Ref{Any}(nothing)
        observer = event -> begin
            if event.value == 1
                try
                    pause_and_checkpoint!(session_ref[], path)
                catch err
                    failure[] = err
                end
                pause!(session_ref[])
            end
            nothing
        end
        session = LocalRunSession(
            initialize_run(3, DummyAlgorithm()); observer, threaded=false
        )
        session_ref[] = session

        start_run!(session)
        wait(session)
        @test failure[] isa ArgumentError
        @test run_status(session) == PMPSLab.LOCAL_PAUSED
        @test !ispath(path)
    end
end
