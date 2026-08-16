function rg0_split(; sites::Int=8)
    return ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; N=sites, maximal_sweep=0),
        solve=CholeskySolve(),
    )
end

@testset "LoopTNR configuration" begin
    custom_split = rg0_split()
    measurement = TNRMeasurement()
    algorithm = LoopTNR(;
        split=custom_split, maximal_rg=0, measurement
    )

    @test algorithm.split === custom_split
    @test algorithm.maximal_rg == 0
    @test algorithm.measurement === measurement
    @test algorithm == deepcopy(algorithm)
    @test hash(algorithm) == hash(deepcopy(algorithm))
    @test algorithm.split.ansatz isa SVDAnsatz

    @test_throws ArgumentError LoopTNR(; split=custom_split, maximal_rg=-1)
    @test_throws DimensionMismatch LoopTNR(;
        split=rg0_split(; sites=7), maximal_rg=0
    )
end

@testset "LoopTNR default split configuration" begin
    dcut = 7
    algorithm = default_looptnr(; dcut, maximal_rg=3)
    numerical_truncation = PMPSLab.truncerror(; rtol=1.0e-13)
    expected_svd = PMPSLab.truncrank(2dcut) & numerical_truncation
    expected_projector =
        PMPSLab.truncrank(dcut) & numerical_truncation

    @test algorithm.maximal_rg == 3
    @test algorithm.split.ansatz isa PeriodicGaugeAnsatz
    @test algorithm.split.ansatz.svd_pre.trunc == expected_svd
    @test algorithm.split.ansatz.canonical.tol == 1.0e-13
    @test algorithm.split.ansatz.canonical.maxiter == 100
    @test algorithm.split.ansatz.canonical.trunc == expected_projector
    @test algorithm.split.sweep == CircularSweep(; N=8, maximal_sweep=40)
    @test algorithm.split.solve == LDLTSolve()
    @test algorithm.split.environments.norm isa DenseTransferPolicy
    @test algorithm.split.environments.mixed isa DenseTransferPolicy
    @test algorithm.split.stopping == CostTolerance(;
        atol=1.0e-9,
        rtol=1.0e-2,
        min_sweeps=5,
    )
    @test algorithm.measurement.normalization isa InfinityNormNormalization
    @test algorithm.measurement.fixed_point isa NoFixedPointMeasurement
    @test_throws ArgumentError default_looptnr(; dcut=0, maximal_rg=1)
    @test_throws ArgumentError default_looptnr(; dcut=2, maximal_rg=-1)
end
