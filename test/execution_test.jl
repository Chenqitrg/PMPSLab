using Test
using PMPSLab

struct DummyAlgorithm end

mutable struct DummyContext <: AlgorithmContext{Int,Nothing}
    value::Int
end

mutable struct DummyRun <: AbstractAlgorithmRun
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
end
