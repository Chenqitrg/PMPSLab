using PMPSLab: GaugeInitialization, NoStopping, CostTolerance
using Accessors: @set, @reset

@testset "Accessors — functional algorithm updates" begin
    algorithm = ALSSplit(;
        initialization=GaugeInitialization(),
        sweep=CircularSweep(; max_sweeps=40),
        solver=LDLTSolve(),
        stopping=CostTolerance(; atol=1e-9, rtol=1e-2, min_sweeps=5),
    )

    modified = algorithm
    @reset modified.sweep.max_sweeps = 60
    @reset modified.stopping.atol = 1e-10
    @test modified.sweep.max_sweeps == 60
    @test modified.stopping.atol == 1e-10
    @test modified.stopping.rtol == algorithm.stopping.rtol
    @test modified.environments === algorithm.environments

    replacement_trunc = algorithm.initialization.gauge_fix.trunc
    retruncated = (@set algorithm.initialization.initialization.trunc = replacement_trunc)
    @test retruncated.initialization.initialization.trunc == replacement_trunc

    @test algorithm.sweep.max_sweeps == 40
    @test algorithm.stopping.atol == 1e-9
end

@testset "Accessors — concrete strategy replacement" begin
    algorithm = ALSSplit(;
        initialization=GaugeInitialization(),
        sweep=CircularSweep(; max_sweeps=40),
        solver=LDLTSolve(),
        stopping=CostTolerance(; atol=1e-9, rtol=1e-2, min_sweeps=5),
    )
    modified = @set algorithm.stopping = NoStopping()

    @test modified.stopping isa NoStopping
    @test typeof(modified) !== typeof(algorithm)
    @test modified.initialization === algorithm.initialization
end

@testset "Accessors — paths and constructors are validated" begin
    algorithm = ALSSplit(;
        initialization=GaugeInitialization(),
        sweep=CircularSweep(; max_sweeps=40),
        solver=LDLTSolve(),
        stopping=CostTolerance(; atol=1e-9, rtol=1e-2, min_sweeps=5),
    )

    @test_throws ArgumentError @set algorithm.unknown = 1
    @test_throws ArgumentError @set algorithm.sweep.max_sweeps = -1
    @test_throws ArgumentError @set algorithm.stopping.min_sweeps = 1

end
