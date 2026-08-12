using Test
using JLD2
using PMPSLab

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

@testset "SplitSnapshot checkpoint — JLD2 file round trip" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve();
        stopping=CostConvergence(min_sweeps=2),
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
            loaded.mpsB[i] == saved.mpsB[i] for i in eachindex(saved.mpsB)
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

@testset "Checkpoint metadata validation" begin
    mpsA = rand(PeriodicMPS, ComplexF64, 2, (2, 2), 1)
    algorithm = PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(4; maximal_sweep=1),
        CholeskySolve(),
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
    end
end
