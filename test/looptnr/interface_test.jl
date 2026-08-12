function rg0_split(; sites::Int=8)
    return PeriodicSplit(
        SVDAnsatz(),
        CircularSweep(sites; maximal_sweep=0),
        CholeskySolve(),
    )
end

@testset "LoopTNR configuration" begin
    custom_split = rg0_split()
    measurement = TNRMeasurement()
    algorithm = LoopTNR(
        custom_split; maximal_rg=0, measurement
    )

    @test algorithm.split === custom_split
    @test algorithm.maximal_rg == 0
    @test algorithm.measurement === measurement
    @test algorithm == deepcopy(algorithm)
    @test hash(algorithm) == hash(deepcopy(algorithm))
    @test algorithm.split.ansatz isa SVDAnsatz

    @test_throws ArgumentError LoopTNR(custom_split; maximal_rg=-1)
    @test_throws DimensionMismatch LoopTNR(rg0_split(; sites=7); maximal_rg=0)
end

@testset "LoopTNR default split configuration" begin
    maxrank = 7
    algorithm = LoopTNR(maxrank; maximal_rg=0)
    expected_svd = truncerror(; rtol=1.0e-13)
    expected_projector =
        truncrank(maxrank) & truncerror(; rtol=1.0e-13)

    @test algorithm.split.ansatz isa PeriodicGaugeAnsatz
    @test algorithm.split.ansatz.svd_pre.truncation == expected_svd
    @test algorithm.split.ansatz.canonical.trscheme == expected_projector
    @test algorithm.split.sweep == CircularSweep(8; maximal_sweep=40)
    @test algorithm.split.solve == CholeskySolve()
    @test algorithm.split.stopping == CostConvergence(
        ; atol=0,
        rtol=1.0e-10,
        min_sweeps=5,
        consecutive_sweeps=2,
    )
    @test_throws ArgumentError LoopTNR(0; maximal_rg=0)
end
