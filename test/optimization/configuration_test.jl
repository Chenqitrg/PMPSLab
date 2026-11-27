using Configurations: to_dict, from_dict
using PMPSLab: GaugeInitialization, NuclearNorm, CostConvergence
using JLD2

@testset "Configurations preserves nested options and validation" begin
    original = ALSSplit(initialization=GaugeInitialization(), solver=CholeskySolve(),
        sweep=CircularSweep(max_sweeps=4), regularization=NuclearNorm(),
        stopping_criterion=CostConvergence())
    for include_defaults in (true, false)
        parameters = to_dict(original; include_defaults)
        @test isequal(from_dict(typeof(original), parameters), original)
    end
    parameters = to_dict(original)
    parameters["error_measure"] = "invalid"
    @test_throws ArgumentError from_dict(typeof(original), parameters)
    changed = @set original.regularization.schedule.factor = 0.6
    @test changed.regularization.schedule.factor == 0.6
    @test original.regularization.schedule.factor == 0.8
    @test_throws ArgumentError (@set original.regularization.schedule.factor = 1.1)
end

@testset "Native A.I. criteria survive Julia configuration storage" begin
    original = ALSSplit(sweep=CircularSweep(max_sweeps=4), solver=CholeskySolve(),
        stopping_criterion=AI.StopWhenAny(CostConvergence(), AI.StopAfterIteration(12)))
    mktempdir(function check_storage(path::String)::Nothing
        file = joinpath(path, "algorithm.jld2")
        jldsave(file; algorithm=original)
        restored = load(file, "algorithm")
        @test typeof(restored) === typeof(original)
        @test restored == original
        @test restored.stopping_criterion.criteria[2].max_iterations == 12
        return nothing
    end)
end
