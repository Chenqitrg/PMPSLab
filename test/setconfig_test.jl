@testset "@setconfig — functional algorithm updates" begin
    algorithm = default_looptnr(; max_rg_steps=3, dcut=8)

    modified = @setconfig algorithm begin
        max_rg_steps = 5
        split/sweep/max_sweeps = 60
        split/stopping/atol = 1e-10
    end
    @test modified.max_rg_steps == 5
    @test modified.split.sweep.max_sweeps == 60
    @test modified.split.stopping.atol == 1e-10
    @test modified.split.stopping.rtol == algorithm.split.stopping.rtol
    @test modified.measurement === algorithm.measurement

    replacement_trunc = algorithm.split.ansatz.gauge_fix.trunc
    retruncated = @setconfig(
        algorithm,
        split/ansatz/initial_split/trunc = replacement_trunc,
    )
    @test retruncated.split.ansatz.initial_split.trunc == replacement_trunc

    @test algorithm.max_rg_steps == 3
    @test algorithm.split.sweep.max_sweeps == 40
    @test algorithm.split.stopping.atol == 1e-9
end

@testset "@setconfig — concrete strategy replacement" begin
    algorithm = default_looptnr(; max_rg_steps=3, dcut=8)
    modified = @setconfig algorithm split/stopping = NoStopping()

    @test modified.split.stopping isa NoStopping
    @test typeof(modified.split) !== typeof(algorithm.split)
    @test modified.split.ansatz === algorithm.split.ansatz
end

@testset "@setconfig — paths and constructors are validated" begin
    algorithm = default_looptnr(; max_rg_steps=3, dcut=8)

    @test_throws ArgumentError @setconfig algorithm unknown = 1
    @test_throws ArgumentError @setconfig algorithm split/unknown = 1
    @test_throws ArgumentError @setconfig algorithm max_rg_steps/value = 1
    @test_throws ArgumentError @setconfig algorithm max_rg_steps = -1
    @test_throws ArgumentError @setconfig algorithm split/stopping/min_sweeps = 1

    invalid_assignment = :(@setconfig algorithm split + stopping = 1)
    @test_throws ArgumentError macroexpand(@__MODULE__, invalid_assignment)
end
