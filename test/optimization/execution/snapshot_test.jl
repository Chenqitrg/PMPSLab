using PMPSLab: RandomInitialization, SVDInitialization
@testset "Minimal split snapshots restore every forward and backward boundary" begin
    Random.seed!(306)
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 2)
    for sweep in (CircularSweep(max_sweeps=2), ZigzagSweep(max_sweeps=2))
        algorithm = ALSSplit(sweep=sweep, solver=LUSolve(),
            initialization=SVDInitialization(trunc=truncrank(1)))
        run = initialize_split(target, algorithm)
        snapshots = [PMPSLab.snapshot(run)]
        expected = run_split!(run; onstep=function capture_boundary(::PMPSLab.SplitState, ::PMPSLab.SplitProgress)::Nothing
            push!(snapshots, PMPSLab.snapshot(run))
            return nothing
        end)
        for (index, saved) in enumerate(snapshots)
            boundary = index - 1
            @test saved.progress.completed_updates == boundary
            @test all(saved.approximation[i] !== run.iterate[i] for i in 1:4)
            restored = restore_split(saved, target)
            visited_updates = Int[]
            actual = run_split!(restored; onstep=function collect_updates(::PMPSLab.SplitState, event::PMPSLab.SplitProgress)::Nothing
                push!(visited_updates, event.completed_updates)
                return nothing
            end)
            @test visited_updates == collect(boundary+1:expected.progress.completed_updates)
            @test dense(actual.mps) ≈ dense(expected.mps)
            @test actual.progress.completed_updates == expected.progress.completed_updates
            @test actual.progress.completed_sweeps == 2
            @test saved.progress.completed_updates == boundary
        end
    end
end

@testset "Captured tensors stay detached and incompatible targets are rejected" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    run = initialize_split(target, algorithm)
    snapshots = [PMPSLab.snapshot(run)]
    run_split!(run; onstep=function capture_first(::PMPSLab.SplitState, event::PMPSLab.SplitProgress)::Nothing
        event.completed_updates == 1 && push!(snapshots, PMPSLab.snapshot(run))
        return nothing
    end)
    saved = snapshots[end]
    before = copy(saved.approximation[1])
    run.iterate[1] *= 2
    @test saved.approximation[1] == before
    restored = restore_split(saved, target)
    restored.iterate[1] *= 3
    @test saved.approximation[1] == before
    incompatible = rand(PeriodicMPS, ComplexF64, 2, (3, 3), 1)
    @test_throws DimensionMismatch restore_split(saved, incompatible)
    bad = SplitSnapshot(saved.approximation, saved.algorithm, saved.target_spaces,
        PMPSLab.SplitProgress(completed_updates=1, site=4), saved.stopping_criterion_state, saved.checked_iteration, saved.regularization)
    @test_throws ArgumentError restore_split(bad, target)
end

@testset "Graded split snapshots preserve spaces" begin
    v = Z2Space(0 => 2, 1 => 1)
    p = Z2Space(0 => 2)
    target = rand(PeriodicMPS, ComplexF64, 2, p ⊗ p, v)
    algorithm = ALSSplit(initialization=RandomInitialization(virtual_spaces=fill(v, 4)),
        sweep=CircularSweep(max_sweeps=1), solver=LUSolve())
    state = initialize_split(target, algorithm)
    saved = PMPSLab.snapshot(state)
    restored = restore_split(saved, target)
    @test dense(restored.iterate) == dense(state.iterate)
    @test all(space(restored.iterate[i]) == space(state.iterate[i]) for i in 1:4)
end

@testset "Snapshot progress has one authoritative traversal count" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=1), solver=CholeskySolve())
    state = initialize_split(target, algorithm)
    saved = PMPSLab.snapshot(state)
    @test !hasproperty(saved, :pending_iteration)
    @test restore_split(saved, target).progress === saved.progress
    run_split!(state)
    saved = PMPSLab.snapshot(state)
    for bad in (
        (@set saved.progress.completed_updates = -1),
        (@set saved.progress.completed_updates = 5),
        (@set saved.progress.completed_sweeps = 0),
        (@set saved.progress.site = 1),
        (@set saved.progress.direction = "backward"),
        (@set saved.progress.sweep_completed = false),
        (@set saved.stopping_criterion_state.stable_comparisons = 10),
    )
        @test_throws ArgumentError restore_split(bad, target)
    end
end

# Cursor integers need not equal completed update counts.
struct SpacedCursorSweep <: LatticeIteratorLab.AbstractSweep end
struct SpacedCursorStates{S}
    positions::S
end
LatticeIteratorLab.sweep_states(::SpacedCursorSweep, n::Integer) =
    SpacedCursorStates(LatticeIteratorLab.sweep_states(CircularSweep(max_sweeps=1), n))
Base.length(s::SpacedCursorStates) = length(s.positions)
Base.iterate(s::SpacedCursorStates) = iterate(s, 100)
function Base.iterate(s::SpacedCursorStates, cursor::Int)
    next = iterate(s.positions, div(cursor - 100, 10))
    isnothing(next) && return nothing
    position, next_cursor = next
    return position, 100 + 10next_cursor
end

@testset "Restore delegates nonconsecutive cursor semantics to the sweep" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=SpacedCursorSweep(), solver=CholeskySolve())
    state = initialize_split(target, algorithm)
    snapshots = [PMPSLab.snapshot(state)]
    expected = run_split!(state; onstep=function capture_cursor(::PMPSLab.SplitState, ::PMPSLab.SplitProgress)::Nothing
        push!(snapshots, PMPSLab.snapshot(state))
        return nothing
    end)
    for saved in snapshots
        restored = restore_split(saved, target)
        actual = run_split!(restored)
        @test actual.progress.completed_updates == 4
        @test dense(actual.mps) ≈ dense(expected.mps)
    end
end

@testset "Zero sweeps return an initial summary without events" begin
    target = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=0), solver=CholeskySolve())
    state = initialize_split(target, algorithm)
    restored = restore_split(PMPSLab.snapshot(state), target)
    events = PMPSLab.SplitProgress[]
    result = run_split!(restored; onstep=function collect_events(::PMPSLab.SplitState, event::PMPSLab.SplitProgress)::Nothing
        push!(events, event)
        return nothing
    end)
    @test result isa NamedTuple{(:mps, :progress)}
    @test result.mps === restored.iterate
    @test result.progress.completed_updates == 0
    @test isnothing(result.progress.site)
    @test isempty(events)
end
