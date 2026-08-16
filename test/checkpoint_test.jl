using Test
using JLD2
using PMPSLab
using TensorKit

function write_test_checkpoint(
        path,
        saved;
        magic="PMPSLab checkpoint",
        schema_version=UInt8(1),
        algorithm="split",
    )
    jldopen(path, "w") do file
        file["header/magic"] = magic
        file["header/schema_version"] = schema_version
        file["header/algorithm"] = algorithm
        file["header/package_version"] = string(Base.pkgversion(PMPSLab))
        file["snapshot"] = saved
    end
end

function checkpoint_loop_fixture()
    gate = ones(ComplexF64, ℂ^1 ⊗ ℂ^1, ℂ^1 ⊗ ℂ^1)
    split_algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=8, max_sweeps=1),
        solver=CholeskySolve(),
    )
    algorithm = LoopTNR(; split=split_algorithm, max_rg_steps=1)
    return initialize_run(UniformGates(gate), algorithm), algorithm
end

@testset "SplitSnapshot checkpoint — JLD2 file round trip" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=CholeskySolve(),
        stopping=CostConvergence(; min_sweeps=2),
    )
    run = initialize_run(mpsA, algorithm)
    step!(run)
    saved = snapshot(run)

    mktempdir() do directory
        path = joinpath(directory, "split.jld2")
        @test isnothing(write_checkpoint(path, saved))
        @test isfile(path)
        loaded = read_checkpoint(path)

        @test loaded isa SplitSnapshot
        @test loaded !== saved
        @test loaded.pending_iteration == saved.pending_iteration
        @test loaded.log.site_logs == saved.log.site_logs
        @test all(
            loaded.approximation[i] == saved.approximation[i] for i in eachindex(saved.approximation)
        )

        restored = restore_run(loaded, mpsA, algorithm)
        while !isfinished(run)
            step!(run)
        end
        while !isfinished(restored)
            step!(restored)
        end
        @test dense(run_result(restored).mps) ≈
              dense(run_result(run).mps)
    end
end

@testset "LoopTNRSnapshot checkpoint — active split round trip" begin
    run, algorithm = checkpoint_loop_fixture()
    step!(run)
    for _ in 1:3
        step!(run)
    end
    saved = snapshot(run)

    mktempdir() do directory
        path = joinpath(directory, "looptnr.jld2")
        @test isnothing(write_checkpoint(path, saved))
        @test jldopen(path, "r") do file
            file["header/algorithm"] == "looptnr"
        end
        loaded = read_checkpoint(path)

        @test loaded isa LoopTNRSnapshot
        @test loaded.phase == PMPSLab.TNR_OPTIMIZE
        @test loaded.completed_rg == 0
        @test length(loaded.measurement_results) == 1
        @test loaded.measurement_results[1].rg == 0
        @test loaded.active_split isa SplitSnapshot
        @test length(loaded.active_split.log.site_logs) == 3
        @test isempty(loaded.active_split.measurement_results)

        restored = restore_run(loaded, algorithm)
        next_event = step!(restored)
        @test next_event isa LoopTNRSplitStepCompleted
        @test next_event.event.log.state == SweepState{Forward}(4, 0)

        while !isfinished(run)
            step!(run)
        end
        while !isfinished(restored)
            step!(restored)
        end
        expected = run_result(run)
        actual = run_result(restored)
        @test getfield.(actual.measurement_results, :rg) == [0, 1]
        @test getfield.(actual.measurement_results, :rg) ==
              getfield.(expected.measurement_results, :rg)
        @test all(
            actual_tensor ≈ expected_tensor
            for (actual_tensor, expected_tensor) in
                zip(collect(actual.cell), collect(expected.cell))
        )
    end
end

@testset "LoopTNR local pause-and-checkpoint — split continuation" begin
    run, algorithm = checkpoint_loop_fixture()
    reached_site_four = Channel{Nothing}(1)
    observer = event -> begin
        if event isa LoopTNRSplitStepCompleted &&
                event.event.log.state.site == 4
            put!(reached_site_four, nothing)
        end
        return nothing
    end
    session = LocalRunSession(run; observer, threaded=false)

    mktempdir() do directory
        path = joinpath(directory, "paused-looptnr.jld2")
        start_run!(session)
        take!(reached_site_four)
        @test pause_and_checkpoint!(session, path) ==
              PMPSLab.LOCAL_CHECKPOINT_WRITTEN
        @test run_status(session) == PMPSLab.LOCAL_PAUSED

        loaded = read_checkpoint(path)
        @test loaded isa LoopTNRSnapshot
        @test loaded.completed_rg == 0
        @test length(loaded.measurement_results) == 1
        @test length(loaded.active_split.log.site_logs) == 4

        restored = restore_run(loaded, algorithm)
        next_event = step!(restored)
        @test next_event.event.log.state == SweepState{Forward}(5, 0)
        while !isfinished(restored)
            step!(restored)
        end
        @test getfield.(run_result(restored).measurement_results, :rg) == [0, 1]
    end
end

@testset "Checkpoint metadata validation" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=4, max_sweeps=1),
        solver=CholeskySolve(),
    )
    saved = snapshot(initialize_run(mpsA, algorithm))

    mktempdir() do directory
        path = joinpath(directory, "split.jld2")
        write_test_checkpoint(path, saved; schema_version=UInt8(0xff))
        @test_throws ArgumentError read_checkpoint(path)

        write_test_checkpoint(path, saved; magic="not a PMPSLab checkpoint")
        @test_throws ArgumentError read_checkpoint(path)

        write_test_checkpoint(path, saved; algorithm="unknown")
        @test_throws ArgumentError read_checkpoint(path)

        loop_run, _ = checkpoint_loop_fixture()
        loop_saved = snapshot(loop_run)
        write_test_checkpoint(path, loop_saved; algorithm="split")
        @test_throws ArgumentError read_checkpoint(path)
    end
end
