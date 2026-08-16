@testset "@setconfig — functional algorithm updates" begin
    algorithm = default_looptnr(; maximal_rg=3, dcut=8)

    modified = @setconfig algorithm begin
        maximal_rg = 5
        split/sweep/maximal_sweep = 60
        split/stopping/atol = 1e-10
    end
    @test modified.maximal_rg == 5
    @test modified.split.sweep.maximal_sweep == 60
    @test modified.split.stopping.atol == 1e-10
    @test modified.split.stopping.rtol == algorithm.split.stopping.rtol
    @test modified.measurement === algorithm.measurement

    replacement_trunc = algorithm.split.ansatz.canonical.trunc
    retruncated = @setconfig(
        algorithm,
        split/ansatz/svd_pre/trunc = replacement_trunc,
    )
    @test retruncated.split.ansatz.svd_pre.trunc == replacement_trunc

    @test algorithm.maximal_rg == 3
    @test algorithm.split.sweep.maximal_sweep == 40
    @test algorithm.split.stopping.atol == 1e-9
end

@testset "@setconfig — concrete strategy replacement" begin
    algorithm = default_looptnr(; maximal_rg=3, dcut=8)
    modified = @setconfig algorithm split/stopping = NoStopping()

    @test modified.split.stopping isa NoStopping
    @test typeof(modified.split) !== typeof(algorithm.split)
    @test modified.split.ansatz === algorithm.split.ansatz
end

@testset "@setconfig — paths and constructors are validated" begin
    algorithm = default_looptnr(; maximal_rg=3, dcut=8)

    @test_throws ArgumentError @setconfig algorithm unknown = 1
    @test_throws ArgumentError @setconfig algorithm split/unknown = 1
    @test_throws ArgumentError @setconfig algorithm maximal_rg/value = 1
    @test_throws ArgumentError @setconfig algorithm maximal_rg = -1
    @test_throws ArgumentError @setconfig algorithm split/stopping/min_sweeps = 1

    invalid_assignment = :(@setconfig algorithm split + stopping = 1)
    @test_throws ArgumentError macroexpand(@__MODULE__, invalid_assignment)
end
