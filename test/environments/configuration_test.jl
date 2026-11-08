
import ExperimentLab

@testset "PMPSLab configuration identities" begin
    for config in (DenseTransferPolicy(), TruncatedTransferPolicy(), TruncatedTransferPolicy(trunc=truncrank(3))), defaults in (true, false)
        registry = ExperimentLab.config_registry(typeof(config))
        record = ExperimentLab.config_record(registry, config; include_defaults=defaults)
        @test record["type"] == "PMPSLab." * string(nameof(typeof(config)))
        @test haskey(record, "parameters")
        io = IOBuffer()
        ExperimentLab.writeconfig(io, config; include_defaults=defaults)
        text = String(take!(io))
        @test isequal(ExperimentLab.readconfig(IOBuffer(text), typeof(config)), config)
    end
end
