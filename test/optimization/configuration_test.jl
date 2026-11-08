using PMPSLab: RandomInitialization, IdentityInitialization, SVDInitialization, GaugeInitialization, SplitEnvironmentPolicy, NuclearNorm, NoStopping, CostTolerance, CostConvergence
using TensorLinearSolveLab
using ExperimentLab: @config, configtree
using Configurations: to_dict, from_dict
using TensorKit: MatrixAlgebraKit

@testset "split configuration interchange" begin
    original = ALSSplit(; initialization=GaugeInitialization(), solver=CholeskySolve(),
                        sweep=CircularSweep(; max_sweeps=4))
    for mode in (:full, :compact)
        @test isequal(include_string(@__MODULE__, "@config " * repr(configtree(original; mode))), original)
        @test isequal(from_dict(typeof(original), to_dict(original; include_defaults=(mode === :full))), original)
    end
end

@testset "Every owned configuration has an external name" begin
    packages = (PMPSLab, LatticeIteratorLab, TensorLinearSolveLab)
    registry = ExperimentLab.config_registry(ALSSplit)
    registered = Set(keys(registry.names))
    owned = Set{Type}()
    for package in packages, name in names(package; all=true)
        isdefined(package, name) || continue
        type = getfield(package, name)
        type isa Type && type <: ExperimentLab.AbstractAlgorithmConfig || continue
        isabstracttype(type) && continue
        parentmodule(type) === package || continue
        push!(owned, type)
    end
    @test owned == registered

    # Every branch, including empty strategies, is identified without reading
    # the large Julia config_type string stored alongside the JSON document.
    algorithm = ALSSplit(initialization=GaugeInitialization(),
        sweep=ZigzagSweep(max_sweeps=3), solver=LDLTSolve(),
        environments=SplitEnvironmentPolicy(mixed=TruncatedTransferPolicy()),
        regularization=NuclearNorm(), stopping=CostConvergence())
    function check_config_nodes(config, document)
        @test document["type"] == string(parentmodule(typeof(config)), ".", nameof(typeof(config)))
        for field in fieldnames(typeof(config))
            value = getfield(config, field)
            value isa ExperimentLab.AbstractAlgorithmConfig || continue
            check_config_nodes(value, document["parameters"][string(field)])
        end
    end
    check_config_nodes(algorithm, ExperimentLab.config_record(registry, algorithm))
    samples = (algorithm, RandomInitialization(virtual_spaces=[ℂ^2, ℂ^3]),
        IdentityInitialization(virtual_spaces=[ℂ^2, ℂ^3]), SVDInitialization(),
        NoStopping(), CostTolerance(atol=1e-8, rtol=1e-6), CostConvergence(), PMPSLab.NoRegularization())
    for config in samples
        io = IOBuffer()
        ExperimentLab.writeconfig(io, config)
        @test occursin("\"type\":\"PMPSLab.$(nameof(typeof(config)))\"", String(take!(io)))
    end
    for defaults in (true, false)
        io = IOBuffer()
        ExperimentLab.writeconfig(io, algorithm; include_defaults=defaults)
        @test isequal(ExperimentLab.readconfig(seekstart(io), typeof(algorithm)), algorithm)
    end
    # Inference is possible when the children are named configurations; opaque
    # external fields such as truncation policies need the concrete owner type.
    environments = SplitEnvironmentPolicy()
    io = IOBuffer()
    ExperimentLab.writeconfig(io, environments)
    @test ExperimentLab.readconfig(seekstart(io), registry) == environments
end

@testset "concrete loading preserves split validation" begin
    original = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=CholeskySolve())
    fields = to_dict(original)
    fields["error_measure"] = "invalid"
    @test_throws ArgumentError from_dict(typeof(original), fields)
    @test_throws ArgumentError ALSSplit(sweep=original.sweep, solver=original.solver, error_measure=:invalid)
end

@testset "Nested NNR configuration uses the existing tree and JSON interfaces" begin
    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=4), solver=CholeskySolve(),
        regularization=NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.02, factor=0.7, minimum=1e-6)))
    tree = configtree(algorithm)
    @test isequal(include_string(@__MODULE__, "@config " * repr(tree)), algorithm)
    modified = @set algorithm.regularization.schedule.factor = 0.6
    @test modified.regularization.schedule.factor == 0.6
    @test algorithm.regularization.schedule.factor == 0.7
    io = IOBuffer()
    ExperimentLab.writeconfig(io, algorithm)
    seekstart(io)
    @test ExperimentLab.readconfig(io, typeof(algorithm)) == algorithm
end

@testset "Regularization JSON identifies each configuration node" begin
    none = PMPSLab.NoRegularization()
    rule = NuclearNorm(schedule=PMPSLab.GeometricDecay(initial=0.01, factor=0.95))
    registry = ExperimentLab.config_registry(typeof(rule))
    expected = Dict("type" => "PMPSLab.NuclearNorm", "parameters" => Dict("schedule" => Dict(
        "type" => "PMPSLab.GeometricDecay", "parameters" => Dict("initial" => 0.01,
        "factor" => 0.95, "minimum" => 1e-7))))
    @test fieldnames(typeof(none)) == ()
    @test fieldnames(typeof(rule.schedule)) == (:initial, :factor, :minimum)
    @test ExperimentLab.config_record(registry, none) ==
        Dict("type" => "PMPSLab.NoRegularization", "parameters" => Dict())
    @test ExperimentLab.config_record(registry, rule) == expected
    for config in (none, rule)
        io = IOBuffer()
        ExperimentLab.writeconfig(io, config)
        text = String(take!(io))
        @test ExperimentLab.readconfig(IOBuffer(text), typeof(config)) == config
        @test ExperimentLab.readconfig(IOBuffer(text), registry) == config
        for mode in (:full, :compact)
            tree = configtree(config; mode)
            @test !occursin("type:", tree)
            @test isequal(include_string(@__MODULE__, "@config " * repr(tree)), config)
        end
    end
    changed = @set rule.schedule.factor = 0.8
    @test changed.schedule.factor == 0.8
    @test rule.schedule.factor == 0.95
    @test_throws ArgumentError (@set rule.schedule.factor = 1.1)
    @test_throws ArgumentError (@set rule.schedule.initial = -1.0)
    @test ExperimentLab.config_record(registry, changed)["parameters"]["schedule"]["type"] ==
        "PMPSLab.GeometricDecay"
    @test sprint(show, none; context=:compact=>true) == "NoRegularization()"

    algorithm = ALSSplit(sweep=CircularSweep(max_sweeps=2), solver=CholeskySolve(),
        regularization=rule)
    mktempdir() do path
        store = ExperimentLab.DataStore(path)
        ref = ExperimentLab.record_algorithm!(store, algorithm)
        @test ExperimentLab.info(store, ref)["config"]["parameters"]["regularization"] == expected
        @test ExperimentLab.load_algorithm(store, ref) == algorithm
    end
end
