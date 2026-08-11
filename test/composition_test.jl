using Test
using PMPSLab

struct CompositionChildAlgorithm end

mutable struct CompositionChildRun <: AbstractAlgorithmRun
    target::Int
    value::Int
end

struct CompositionChildSnapshot <: AbstractAlgorithmSnapshot
    value::Int
end

struct CompositionChildCompleted <: AlgorithmEvent
    value::Int
end

struct CompositionAlgorithm
    child::CompositionChildAlgorithm
end

mutable struct CompositionRun{T<:Tuple} <: AbstractAlgorithmRun
    targets::T
    algorithm::CompositionAlgorithm
    child_index::Int
    child::Union{Nothing,CompositionChildRun}
    results::Vector{Int}
end

struct CompositionSnapshot <: AbstractAlgorithmSnapshot
    child_index::Int
    child::Union{Nothing,CompositionChildSnapshot}
    results::Vector{Int}
end

struct CompositionStepCompleted{E<:AlgorithmEvent} <: AlgorithmEvent
    path::Tuple{Int}
    event::E
end

PMPSLab.initialize_run(target::Int, ::CompositionChildAlgorithm) =
    CompositionChildRun(target, 0)

function PMPSLab.step!(run::CompositionChildRun)
    isfinished(run) && throw(ArgumentError("composition child is complete"))
    run.value += 1
    return CompositionChildCompleted(run.value)
end

PMPSLab.isfinished(run::CompositionChildRun) = run.value >= run.target

function PMPSLab.run_result(run::CompositionChildRun)
    isfinished(run) || throw(ArgumentError("composition child is incomplete"))
    return run.value
end

PMPSLab.snapshot(run::CompositionChildRun) = CompositionChildSnapshot(run.value)

function PMPSLab.restore_run(
        saved::CompositionChildSnapshot,
        target::Int,
        ::CompositionChildAlgorithm,
    )
    0 <= saved.value <= target || throw(ArgumentError(
        "composition child snapshot is incompatible with target $target"
    ))
    return CompositionChildRun(target, saved.value)
end

function PMPSLab.initialize_run(
        targets::Tuple{Vararg{Int}},
        algorithm::CompositionAlgorithm,
    )
    isempty(targets) && throw(ArgumentError("composition requires a child"))
    all(>(0), targets) || throw(ArgumentError("child targets must be positive"))
    child = initialize_run(first(targets), algorithm.child)
    return CompositionRun(targets, algorithm, 1, child, Int[])
end

PMPSLab.isfinished(run::CompositionRun) = isnothing(run.child)

function PMPSLab.step!(run::CompositionRun)
    isfinished(run) && throw(ArgumentError("composition run is complete"))
    child_index = run.child_index
    child = something(run.child)
    event = step!(child)

    if isfinished(child)
        push!(run.results, run_result(child))
        run.child_index += 1
        run.child = run.child_index <= length(run.targets) ?
            initialize_run(run.targets[run.child_index], run.algorithm.child) :
            nothing
    end

    return CompositionStepCompleted((child_index,), event)
end

function PMPSLab.run_result(run::CompositionRun)
    isfinished(run) || throw(ArgumentError("composition run is incomplete"))
    return copy(run.results)
end

function PMPSLab.snapshot(run::CompositionRun)
    child = isnothing(run.child) ? nothing : snapshot(run.child)
    return CompositionSnapshot(run.child_index, child, copy(run.results))
end

function PMPSLab.restore_run(
        saved::CompositionSnapshot,
        targets::Tuple{Vararg{Int}},
        algorithm::CompositionAlgorithm,
    )
    completed_children = saved.child_index - 1
    0 <= completed_children <= length(targets) || throw(ArgumentError(
        "composition snapshot has an invalid child cursor"
    ))
    saved.results == collect(targets[1:completed_children]) || throw(
        ArgumentError("composition snapshot has inconsistent committed results")
    )

    if saved.child_index <= length(targets)
        isnothing(saved.child) && throw(ArgumentError(
            "composition snapshot is missing its active child"
        ))
        child = restore_run(
            saved.child, targets[saved.child_index], algorithm.child
        )
    else
        isnothing(saved.child) || throw(ArgumentError(
            "completed composition snapshot retains an active child"
        ))
        child = nothing
    end

    return CompositionRun(
        targets,
        algorithm,
        saved.child_index,
        child,
        copy(saved.results),
    )
end

function drive_composition!(run::CompositionRun)
    events = CompositionStepCompleted[]
    while !isfinished(run)
        push!(events, step!(run))
    end
    return events
end

@testset "resumable parent-child composition" begin
    targets = (2, 3)
    algorithm = CompositionAlgorithm(CompositionChildAlgorithm())

    uninterrupted = initialize_run(targets, algorithm)
    events = drive_composition!(uninterrupted)
    @test getfield.(events, :path) == [(1,), (1,), (2,), (2,), (2,)]
    @test [event.event.value for event in events] == [1, 2, 1, 2, 3]
    @test run_result(uninterrupted) == [2, 3]

    inside_child = initialize_run(targets, algorithm)
    step!(inside_child)
    restored_inside = restore_run(snapshot(inside_child), targets, algorithm)
    @test restored_inside.child_index == 1
    @test something(restored_inside.child).value == 1
    restored_events = drive_composition!(restored_inside)
    @test first(restored_events).path == (1,)
    @test first(restored_events).event.value == 2
    @test run_result(restored_inside) == [2, 3]

    after_child = initialize_run(targets, algorithm)
    step!(after_child)
    step!(after_child)
    @test after_child.child_index == 2
    @test after_child.results == [2]
    restored_after = restore_run(snapshot(after_child), targets, algorithm)
    drive_composition!(restored_after)
    @test run_result(restored_after) == [2, 3]
end

@testset "parent-child pauses preserve composition boundaries" begin
    targets = (2, 3)
    algorithm = CompositionAlgorithm(CompositionChildAlgorithm())
    expected = [2, 3]

    before_ref = Ref{LocalRunSession}()
    before_observer = event -> begin
        event.path == (1,) && event.event.value == 1 && pause!(before_ref[])
        nothing
    end
    before = LocalRunSession(
        initialize_run(targets, algorithm);
        observer=before_observer,
        threaded=false,
    )
    before_ref[] = before
    start_run!(before)
    wait(before)
    @test run_status(before) == PMPSLab.LOCAL_PAUSED
    @test before.run.child_index == 1
    @test something(before.run.child).value == 1
    resume!(before)
    @test fetch(before) == expected

    after_ref = Ref{LocalRunSession}()
    after_observer = event -> begin
        event.path == (1,) && event.event.value == 2 && pause!(after_ref[])
        nothing
    end
    after = LocalRunSession(
        initialize_run(targets, algorithm);
        observer=after_observer,
        threaded=false,
    )
    after_ref[] = after
    start_run!(after)
    wait(after)
    @test run_status(after) == PMPSLab.LOCAL_PAUSED
    @test after.run.child_index == 2
    @test after.run.results == [2]
    @test something(after.run.child).value == 0
    resume!(after)
    @test fetch(after) == expected
end
