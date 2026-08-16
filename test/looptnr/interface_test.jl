function rg0_split(; sites::Int=8)
    return ALSSplit(;
        ansatz=SVDAnsatz(),
        sweep=CircularSweep(; site_count=sites, max_sweeps=0),
        solver=CholeskySolve(),
    )
end

@testset "LoopTNR configuration" begin
    custom_split = rg0_split()
    measurement = TNRMeasurement()
    algorithm = LoopTNR(;
        split=custom_split, max_rg_steps=0, measurement
    )

    @test algorithm.split === custom_split
    @test algorithm.max_rg_steps == 0
    @test algorithm.measurement === measurement
    @test algorithm == deepcopy(algorithm)
    @test hash(algorithm) == hash(deepcopy(algorithm))
    @test algorithm.split.ansatz isa SVDAnsatz

    @test_throws ArgumentError LoopTNR(; split=custom_split, max_rg_steps=-1)
    @test_throws DimensionMismatch LoopTNR(;
        split=rg0_split(; sites=7), max_rg_steps=0
    )
end

@testset "LoopTNR default split configuration" begin
    dcut = 7
    algorithm = default_looptnr(; dcut, max_rg_steps=3)
    numerical_truncation = PMPSLab.truncerror(; rtol=1.0e-13)
    expected_svd = PMPSLab.truncrank(2dcut) & numerical_truncation
    expected_projector =
        PMPSLab.truncrank(dcut) & numerical_truncation

    @test algorithm.max_rg_steps == 3
    @test algorithm.split.ansatz isa PeriodicGaugeAnsatz
    @test algorithm.split.ansatz.initial_split.trunc == expected_svd
    @test algorithm.split.ansatz.gauge_fix.tol == 1.0e-13
    @test algorithm.split.ansatz.gauge_fix.maxiter == 100
    @test algorithm.split.ansatz.gauge_fix.trunc == expected_projector
    @test algorithm.split.sweep == CircularSweep(; site_count=8, max_sweeps=40)
    @test algorithm.split.solver == LDLTSolve()
    @test algorithm.split.environments.normal isa DenseTransferPolicy
    @test algorithm.split.environments.mixed isa DenseTransferPolicy
    @test algorithm.split.stopping == CostTolerance(;
        atol=1.0e-9,
        rtol=1.0e-2,
        min_sweeps=5,
    )
    @test algorithm.measurement.normalization isa InfinityNormNormalization
    @test algorithm.measurement.fixed_point isa NoFixedPointMeasurement
    @test_throws ArgumentError default_looptnr(; dcut=0, max_rg_steps=1)
    @test_throws ArgumentError default_looptnr(; dcut=2, max_rg_steps=-1)
end
