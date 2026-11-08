using ExperimentLab: @config, configtree
using Configurations: to_dict, from_dict
using TensorKit: MatrixAlgebraKit

@testset "projector configuration interchange" begin
    original = ProjectorGaugeFix(; tol=1e-13, trunc=truncrank(4))
    for mode in (:full, :compact)
        @test isequal(include_string(@__MODULE__, "@config " * repr(configtree(original; mode))), original)
        @test isequal(from_dict(typeof(original), to_dict(original; include_defaults=(mode === :full))), original)
    end
end

import ExperimentLab

@testset "PMPSLab configuration identities" begin
    for config in (ProjectorGaugeFix(), ProjectorGaugeFix(trunc=truncrank(4))), defaults in (true, false)
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
